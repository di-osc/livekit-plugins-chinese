from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

import asyncio
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from livekit import rtc
from livekit.agents import stt, utils, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from .log import logger


@dataclass
class STTOptions:
    api_key: str | None
    language: str | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: str
    smart_format: bool
    endpointing: int | None
    sample_rate: int = 16000


class STT(stt.STT):
    def __init__(
        self,
        *,
        language="zh",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        model: str = "paraformer-realtime-v2",
        api_key: str | None = None,
        min_silence_duration: int = 500,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("DASHSCOPE API key is required")
        self._opts = STTOptions(
            api_key=api_key,
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
            endpointing=min_silence_duration,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("not implemented")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStream":
        return SpeechStream(stt=self, opts=self._opts, conn_options=conn_options)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)

        if opts.language is None:
            raise ValueError("language detection is not supported in streaming mode")
        self._opts: STTOptions = opts
        self._config = opts
        self._speaking = False
        self.recognition = Recognition(
            model=opts.model,
            format="pcm",
            sample_rate=opts.sample_rate,
            callback=Callback(self),
            disfluency_removal_enabled=True,
            semantic_punctuation_enabled=False,
            max_sentence_silence=opts.endpointing,
            language_hints=[opts.language],
        )
        self._closed = False
        self._request_id = utils.shortuuid()
        self._reconnect_event = asyncio.Event()

    async def _run(self) -> None:
        self.recognition.start()
        samples_100ms = self._opts.sample_rate // 10
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=samples_100ms,
        )
        while True:
            try:
                has_ended = False
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())
                        has_ended = True
                    for frame in frames:
                        self.recognition.send_audio_frame(frame.data.tobytes())
                        if has_ended:
                            self.recognition.stop()
            finally:
                self.recognition.stop()


def live_transcription_to_speech_data(
    language: str,
    data,
) -> List[stt.SpeechData]:
    return [
        stt.SpeechData(
            language=language,
            start_time=data["begin_time"],
            end_time=data["end_time"],
            confidence=0.0,
            text=data["text"],
        )
    ]


class Callback(RecognitionCallback):
    def __init__(self, _stt: SpeechStream):
        self._stt = _stt

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        dg_alts = live_transcription_to_speech_data(
            self._stt._config.language, sentence
        )
        if not result.is_sentence_end(sentence):
            interim_event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=dg_alts,
            )
            self._stt._event_ch.send_nowait(interim_event)
            logger.info("transcription start")
        else:
            final_event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=dg_alts,
            )
            self._stt._event_ch.send_nowait(final_event)
            logger.info(
                "transcription end", extra={"text": final_event.alternatives[0].text}
            )
