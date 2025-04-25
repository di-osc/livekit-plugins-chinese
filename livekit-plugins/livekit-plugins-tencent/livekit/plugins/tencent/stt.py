from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional, TypeVar
from random import randint
import hmac
import base64
import hashlib
from urllib.parse import urlencode

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
from livekit.agents.utils import AudioBuffer, shortuuid

from .log import logger


@dataclass
class STTOptions:
    app_id: int
    secret_id: str
    secret_key: str

    engine_model_type: str = "16k_zh"
    sample_rate: int = 16000
    voice_format: Literal[1, 4, 6, 8, 10, 12, 14, 16] = (
        1  # 1：pcm；4：speex(sp)；6：silk；8：mp3；10：opus；12：wav；14：m4a（每个分片须是一个完整的 m4a 音频）；16：aac
    )
    need_vad: Literal[0, 1] = 1  # 0：不开启；1：开启
    vad_silence_time: int = 500  # 240 - 2000 ms
    noise_threshold: float = 0.5  # 噪音参数阈值，默认为0，取值范围：[-1,1]，对于一些音频片段，取值越大，判定为噪音情况越大。取值越小，判定为人声情况越大。

    base_url: str = "wss://asr.cloud.tencent.com/asr/v2/"

    def get_ws_url(self):
        params = self.get_params()
        url = self.base_url + str(self.app_id) + "?" + urlencode(params)
        hmacstr = hmac.new(
            self.secret_key.encode("utf-8"), url[6:].encode("utf-8"), hashlib.sha1
        ).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode("utf-8")
        urlencoded_s = urlencode({"signature": s})
        return url + "&" + urlencoded_s

    def get_params(self):
        params = {
            "secretid": self.secret_id,
            "timestamp": int(time.time()),
            "expired": int(time.time()) + 24 * 60 * 60,
            "nonce": randint(0, 10000),
            "engine_model_type": self.engine_model_type,
            "voice_id": shortuuid(),
            "needvad": self.need_vad,
            "vad_silence_time": self.vad_silence_time,
            "voice_format": self.voice_format,
        }
        # 返回字典排序
        return dict(sorted(params.items(), key=lambda x: x[0]))

    @property
    def language(self):
        return self.engine_model_type.split("_")[1]


class STT(stt.STT):
    def __init__(
        self,
        *,
        app_id: int = None,
        secret_key: str = None,
        secret_id: str = None,
        http_session: aiohttp.ClientSession | None = None,
        interim_results: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )

        self.app_id = app_id or os.environ.get("TENCENT_STT_APP_ID")
        self.secret_key = secret_key or os.environ.get("TENCENT_STT_SECRET_KEY")
        self.secret_id = secret_id or os.environ.get("TENCENT_STT_SECRET_ID")
        if not self.app_id or not self.secret_key or not self.secret_id:
            raise ValueError(
                "TENCENT_STT_APP_ID, TENCENT_STT_SECRET_KEY, TENCENT_STT_SECRET_ID must be set"
            )

        self._opts = STTOptions(
            app_id=self.app_id,
            secret_id=self.secret_id,
            secret_key=self.secret_key,
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
                num_channels=1,
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
                        await ws.send_str(self._opts.get_finish_params())
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
                    logger.warning("connection closed unexpectedly")
                    break

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process stt message")

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
        logger.info("connected to stt websocket successfully")
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
        result_code = data.get("code", 0)
        result_msg = data.get("message", None)
        if result_code != 0:
            logger.warning(
                "failed to process stt message",
                extra={"result_code": result_code, "result_msg": result_msg},
            )
            return
        result = data.get("result")
        if not result:
            logger.warning(f"no result in stt message {data}")
            result
        else:
            slice_type = result.get(
                "slice_type", None
            )  # 0：一段话开始识别 1：一段话识别中，voice_text_str 为非稳态结果(该段识别结果还可能变化) 2：一段话识别结束，voice_text_str 为稳态结果(该段识别结果不再变化)
            start_time = result.get("start_time", None)
            end_time = result.get("end_time", None)
            text = result.get("voice_text_str", None)
            if slice_type == 0 and not self._speaking:
                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)
                logger.info("transcription start")
                if text:
                    alternatives = [
                        stt.SpeechData(
                            language=self._opts.language,
                            text=text,
                            start_time=start_time,
                            end_time=end_time,
                        )
                    ]
                    interim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=alternatives,
                    )
                    self._event_ch.send_nowait(interim_event)

            if slice_type == 1 and self._speaking:
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)

            elif slice_type == 2 and self._speaking:
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)
                end_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                )
                self._event_ch.send_nowait(end_event)
                self._speaking = False
                logger.info(
                    "transcription end",
                    extra={
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                )
            elif slice_type == 2 and not self._speaking:
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)
                end_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                )
                self._event_ch.send_nowait(end_event)
                self._speaking = False
                logger.info(
                    "transcription end",
                    extra={
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                )


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
