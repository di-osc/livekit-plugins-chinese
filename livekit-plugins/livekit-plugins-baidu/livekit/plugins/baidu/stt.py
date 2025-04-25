from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional, TypeVar

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
    app_key: str

    # 1537	中文普通话	弱标点（逗号，句号）
    # 15372	中文普通话	加强标点（逗号、句号、问号、感叹号）
    # 15376	中文多方言	弱标点（逗号，句号）
    # 1737	英语	无标点
    # 17372	英语	加强标点（逗号、句号、问号）
    dev_pid: Literal[1537, 15372, 15376, 1737, 17372] = 15376
    user: int = 11111111  # 选择中文多方言模型时，必须填写user字段，这个值可以是任意整数

    sample_rate: int = 16000
    audio_format: str = "pcm"

    base_url: str = "wss://vop.baidu.com/realtime_asr"

    @property
    def language(self) -> str:
        if self.dev_pid in (1537, 15376, 15376):
            return "zh-CN"
        elif self.dev_pid in (1737, 17372):
            return "en-US"

    def get_ws_url(self):
        return f"{self.base_url}?sn={shortuuid()}"

    def get_start_params(self):
        req = {
            "type": "START",
            "data": {
                "appid": self.app_id,  # 网页上的appid
                "appkey": self.app_key,  # 网页上的appid对应的appkey
                "dev_pid": self.dev_pid,  # 识别模型
                "user": self.user,  # 选择中文多方言模型时，必须填写user字段，这个值可以是任意整数
                "cuid": shortuuid(),  # 随便填不影响使用。机器的mac或者其它唯一id，百度计算UV用。
                "sample": self.sample_rate,  # 固定参数
                "format": self.audio_format,  # 固定参数
            },
        }
        body = json.dumps(req)
        return body

    def get_cancle_params(self):
        req = {"type": "CANCEL"}
        body = json.dumps(req)
        return body

    def get_finish_params(self):
        req = {"type": "FINISH"}
        body = json.dumps(req)
        return body


class STT(stt.STT):
    def __init__(
        self,
        *,
        app_id: str = None,
        api_key: str = None,
        dev_pid: int = 15372,
        http_session: aiohttp.ClientSession | None = None,
        interim_results: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )

        self.app_id = app_id or os.environ.get("BAIDU_APP_ID")
        self.app_key = api_key or os.environ.get("BAIDU_API_KEY")
        if not self.app_id or not self.app_key:
            raise ValueError(
                "app_id and app_key must be provided or set in the environment variables"
            )

        self._opts = STTOptions(
            app_id=int(self.app_id),
            app_key=self.app_key,
            dev_pid=dev_pid,
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
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})
    _FINALIZE_MSG: str = json.dumps({"type": "Finalize"})

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
        self._opts.user = self._request_id
        self._reconnect_event = asyncio.Event()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            start_params = self._opts.get_start_params()
            await ws.send_str(start_params)
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
        error_msg = data.get("err_msg", "OK")
        if error_msg != "OK":  # silence
            return
        result_type = data.get("type", None)  # "MID_TEXT" or "FIN_TEXT"
        text = data.get("result", None)
        start_time = data.get("start_time", None)
        end_time = data.get("end_time", None)
        if result_type == "MID_TEXT" and not self._speaking:
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

        if result_type == "MID_TEXT" and self._speaking:
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

        elif result_type == "FIN_TEXT" and self._speaking:
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
                extra={"text": text, "start_time": start_time, "end_time": end_time},
            )
        elif result_type == "FIN_TEXT" and not self._speaking:
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
