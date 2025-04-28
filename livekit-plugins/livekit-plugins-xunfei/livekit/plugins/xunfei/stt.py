from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional, TypeVar
from urllib.parse import quote

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger


@dataclass
class STTOptions:
    app_id: str
    api_key: str
    end_tag: str = '{"end": true}'
    language: str = "cn"  # "cn"中英混合 "en"英文
    sample_rate: int = 16000
    audio_format: str = "pcm"
    num_channels: int = 1
    base_url: str = "ws://rtasr.xfyun.cn/v1/ws"
    punc: Literal[0, 1] = 0

    def get_ws_url(self):
        ts = str(int(time.time()))
        tt = (self.app_id + ts).encode("utf-8")
        md5 = hashlib.md5()
        md5.update(tt)
        baseString = md5.hexdigest()
        baseString = bytes(baseString, encoding="utf-8")

        apiKey = self.api_key.encode("utf-8")
        signa = hmac.new(apiKey, baseString, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, "utf-8")
        return (
            self.base_url
            + "?appid="
            + self.app_id
            + "&ts="
            + ts
            + "&signa="
            + quote(signa)
            + "&punc="
            + str(self.punc)
        )


class STT(stt.STT):
    def __init__(
        self,
        *,
        app_id: str = None,
        api_key: str = None,
        http_session: aiohttp.ClientSession | None = None,
        interim_results: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )

        self._app_id = app_id or os.environ.get("XUNFEI_STT_APP_ID")
        self._api_key = api_key or os.environ.get("XUNFEI_STT_API_KEY")
        if not self._app_id or not self._api_key:
            raise ValueError("XUNFEI_STT_APP_ID and XUNFEI_STT_API_KEY must be set")

        self._opts = STTOptions(
            api_key=self._api_key,
            app_id=self._app_id,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        self._opts.language = language if is_given(language) else self._opts.language
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate
        )

        self._opts = opts
        self._session = http_session
        self._speaking = False
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

        self._request_id = utils.shortuuid()
        self._reconnect_event = asyncio.Event()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws

            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

                    if has_ended:
                        self._audio_duration_collector.flush()
                        await ws.send_str(self._opts.end_tag)
                        has_ended = False

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    logger.warning("stt connection closed unexpectedly")
                    break

                try:
                    self._process_stream_event(msg.data)
                except Exception:
                    logger.exception("failed to process deepgram message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws = await asyncio.wait_for(
            self._session.ws_connect(self._opts.get_ws_url()),
            self._conn_options.timeout,
        )
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)

    def _process_stream_event(self, data: dict) -> None:
        result = json.loads(data)
        data_str = result.get("data", None)
        if data_str:
            result_data = json.loads(result["data"])
            st = result_data["cn"]["st"]
            begin = st["bg"]
            end = st["ed"]
            seg_type = str(st["type"])
            text = "".join([w["cw"][0]["w"] for w in st["rt"][0]["ws"]])
            if not self._speaking and seg_type == "1":
                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)
                logger.info("transcription start")
            if self._speaking and seg_type == "1":
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=begin,
                        end_time=end,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)
            if self._speaking and seg_type == "0":
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=begin,
                        end_time=end,
                    )
                ]
                final_transcript_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(final_transcript_event)
                end_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                )
                self._event_ch.send_nowait(end_event)
                self._speaking = False
                logger.info("transcription end", extra={"text": text})


def live_transcription_to_speech_data(
    language: str, data: dict
) -> list[stt.SpeechData]:
    dg_alts = data["channel"]["alternatives"]

    return [
        stt.SpeechData(
            language=language,
            start_time=alt["words"][0]["start"] if alt["words"] else 0,
            end_time=alt["words"][-1]["end"] if alt["words"] else 0,
            confidence=alt["confidence"],
            text=alt["transcript"],
        )
        for alt in dg_alts
    ]


T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        """
        Create a new periodic collector that accumulates values and calls the callback
        after the specified duration if there are values to report.

        Args:
            duration: Time in seconds between callback invocations
            callback: Function to call with accumulated value when duration expires
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: Optional[T] = None

    def push(self, value: T) -> None:
        """Add a value to the accumulator"""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Force callback to be called with current total if non-zero"""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()
