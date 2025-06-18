import threading
import os
from dataclasses import dataclass
from typing import AsyncIterable, Optional
import time

from dashscope.audio.tts_v2 import ResultCallback, SpeechSynthesizer, AudioFormat
from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, utils
from osc_data.text_stream import TextStreamSentencizer

from .log import logger


STREAM_EOS = "EOS"

@dataclass
class TTSOptions:
    api_key: str
    model: str
    voice: str
    # 合成音频的语速，取值范围：0.5~2。
    speech_rate: int
    # 合成音频的音量，取值范围：0~100。
    volume: int
    sample_rate: int



class TTS(tts.TTS):
    def __init__(
            self,
            *,
            api_key: Optional[str] = None,
            sample_rate: int = 24000,
            voice: str = "longcheng_v2",
            model: str = "cosyvoice-v2",
            speech_rate: int = 1,
            volume: int = 100,
    ) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,)
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")

        self._opts = TTSOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            speech_rate=speech_rate,
            volume=volume,
            sample_rate=sample_rate
        )

    def synthesize(
            self,
            text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        raise NotImplementedError

    def stream(
            self,
            *,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
            self,
            *,
            tts: TTS,
            opts: TTSOptions,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        
        request_id = utils.shortuuid()
        emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            mime_type="audio/pcm",
            stream=True,
            num_channels=1,
            frame_size_ms=200
        )
        complete_event = threading.Event()
        synthesizer = SpeechSynthesizer(
        model=self._opts.model,
        voice=self._opts.voice,
        format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        speech_rate=self._opts.speech_rate,
        volume=self._opts.volume,
        callback=Callback(emitter=emitter, complete_event=complete_event)
)
        
        splitter = TextStreamSentencizer()
        first_sentence_spend = None
        start_time = time.perf_counter()
        async for token in self._input_ch:
            if isinstance(token, self._FlushSentinel):
                sentences = splitter.flush()
            else:
                sentences = splitter.push(text=token)
            for sentence in sentences:
                if first_sentence_spend is None:
                        first_sentence_spend = time.perf_counter() - start_time
                        logger.info(
                            "llm first sentence",
                            extra={"spent": str(first_sentence_spend)},
                        )
                synthesizer.call(text=sentence)
                complete_event.wait()
                self._pushed_text = self._pushed_text.replace(sentence, "")
                # emitter.end_segment()
    

class Callback(ResultCallback):
    def __init__(self, emitter: tts.AudioEmitter, complete_event: threading.Event):
        self.emitter = emitter
        self.complete_event = complete_event
        self.first_response_spend = None
        self.start_time = None
    
    def on_open(self):
        self.emitter.start_segment(segment_id=utils.shortuuid())
        self.start_time = time.perf_counter()
        
    def on_complete(self):
        self.complete_event.set()
        self.emitter.end_segment()

    def on_data(self, data: bytes) -> None:
        if self.first_response_spend is None:
            self.first_response_spend = time.perf_counter() - self.start_time
            logger.info(
                "tts first response",
                extra={"spent": round(self.first_response_spend, 4)},
            )
        logger.info("tts on data",)
        self.emitter.push(data)
