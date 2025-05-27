from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Literal, Optional, TypeVar

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

# This is the magic number during testing that we use to determine if a frame is loud enough
# to possibly contain speech. It's very conservative.
MAGIC_NUMBER_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(
        self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD
    ):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                # keep speaking during cooldown
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state


PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

PROTOCOL_VERSION_BITS = 4
HEADER_BITS = 4
MESSAGE_TYPE_BITS = 4
MESSAGE_TYPE_SPECIFIC_FLAGS_BITS = 4
MESSAGE_SERIALIZATION_BITS = 4
MESSAGE_COMPRESSION_BITS = 4
RESERVED_BITS = 8

# Message Type:
CLIENT_FULL_REQUEST = 0b0001
CLIENT_AUDIO_ONLY_REQUEST = 0b0010
SERVER_FULL_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
THRIFT = 0b0011
CUSTOM_TYPE = 0b1111

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001
CUSTOM_COMPRESSION = 0b1111


def generate_header(
    version=PROTOCOL_VERSION,
    message_type=CLIENT_FULL_REQUEST,
    message_type_specific_flags=NO_SEQUENCE,
    serial_method=JSON,
    compression_type=GZIP,
    reserved_data=0x00,
    extension_header=b"",
):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    """
    header = bytearray()
    header_size = int(len(extension_header) / 4) + 1
    header.append((version << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    header.extend(extension_header)
    return header


@dataclass
class STTOptions:
    # app
    app_id: str
    cluster: str
    access_token: str

    # audio
    audio_format: Literal["raw", "wav", "mp3", "ogg"] = "raw"
    sample_rate: int = 16000
    bits: int = 16
    num_channels: int = 1
    language: str = "zh-CN"
    codec: Literal["raw", "opus"] = "raw"  # raw(pcm)

    energy_filter: AudioEnergyFilter | bool = False

    # request
    nbest: int = 1
    confidence: int = 0
    workflow: str = "audio_in,resample,partition,vad,fe,decode,nlu_punctuate"
    show_language: bool = True

    # ws
    base_url: str = "wss://openspeech.bytedance.com/api/v2"

    def get_ws_url(self):
        return f"{self.base_url}/asr"

    def get_ws_query_params(self, uid: str | None = None) -> bytearray:
        if uid is None:
            uid = utils.shortuuid()
        if self.app_id is None:
            self.app_id = utils.shortuuid()
        if self.cluster is None:
            raise ValueError("volcengine stt cluster is not set")
        submit_request_json = {
            "app": {
                "appid": self.app_id,
                "token": self.access_token,
                "cluster": self.cluster,
            },
            "user": {"uid": uid},
            "audio": {
                "format": self.audio_format,
                "rate": self.sample_rate,
                "bits": self.bits,
                "channels": self.num_channels,
                "codec": self.codec,
                "language": self.language,
            },
            "request": {
                "reqid": utils.shortuuid(),
                "sequence": 1,
                "nbest": self.nbest,
                "confidence": self.confidence,
                "workflow": self.workflow,
                "show_utterances": True,
                "show_language": self.show_language,
                "result_type": "single",
            },
        }
        payload_bytes = str.encode(json.dumps(submit_request_json))
        payload_bytes = gzip.compress(
            payload_bytes
        )  # if no compression, comment this line
        full_client_request = bytearray(generate_header())
        full_client_request.extend(
            (len(payload_bytes)).to_bytes(4, "big")
        )  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        return full_client_request

    def get_chunk_request(self, chunk: bytes, last: bool = False) -> bytearray:
        payload_bytes = gzip.compress(chunk)
        if last:
            audio_only_request = bytearray(
                generate_header(
                    message_type=CLIENT_AUDIO_ONLY_REQUEST,
                    message_type_specific_flags=NEG_SEQUENCE,
                )
            )
        else:
            audio_only_request = bytearray(
                generate_header(message_type=CLIENT_AUDIO_ONLY_REQUEST)
            )
        audio_only_request.extend(
            (len(payload_bytes)).to_bytes(4, "big")
        )  # payload size(4 bytes)
        audio_only_request.extend(payload_bytes)  # payload
        return audio_only_request

    def get_ws_header(self):
        if self.access_token is None:
            self.access_token = os.environ.get("VOLCENGINE_STT_ACCESS_TOKEN")
        if self.access_token is None:
            raise ValueError("VOLCENGINE_STT_ACCESS_TOKEN is not set")
        return {"Authorization": f"Bearer; {self.access_token}"}


class STT(stt.STT):
    def __init__(
        self,
        *,
        app_id: str,
        cluster: str,
        base_url: str = "wss://openspeech.bytedance.com/api/v2",
        access_token: str | None = None,
        audio_format: Literal["raw", "wav", "mp3", "ogg"] = "raw",
        sample_rate: int = 16000,
        bits: int = 16,
        num_channels: int = 1,
        nbest: int = 1,
        confidence: int = 0,
        workflow: str = "audio_in,resample,partition,vad,fe,decode,nlu_punctuate",
        http_session: aiohttp.ClientSession | None = None,
        interim_results: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )

        self._access_token = access_token or os.environ.get(
            "VOLCENGINE_STT_ACCESS_TOKEN"
        )
        if not self._access_token:
            raise ValueError("VOLCENGINE_STT_ACCESS_TOKEN is not set")

        self._opts = STTOptions(
            base_url=base_url,
            app_id=app_id,
            cluster=cluster,
            access_token=self._access_token,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bits=bits,
            num_channels=num_channels,
            nbest=nbest,
            confidence=confidence,
            workflow=workflow,
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

        self._audio_energy_filter: AudioEnergyFilter | None = None
        if opts.energy_filter:
            if isinstance(opts.energy_filter, AudioEnergyFilter):
                self._audio_energy_filter = opts.energy_filter
            else:
                self._audio_energy_filter = AudioEnergyFilter()

        self._request_id = utils.shortuuid()
        self._reconnect_event = asyncio.Event()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws

            full_client_request = self._opts.get_ws_query_params()
            await ws.send_bytes(full_client_request)

            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            last_frame: rtc.AudioFrame | None = None
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    state = self._check_energy_state(data)
                    if state in (
                        AudioEnergyFilter.State.START,
                        AudioEnergyFilter.State.SPEAKING,
                    ):
                        if last_frame:
                            frames.extend(
                                audio_bstream.write(last_frame.data.tobytes())
                            )
                            last_frame = None
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif state == AudioEnergyFilter.State.END:
                        # no need to buffer as we have cooldown period
                        frames.extend(audio_bstream.flush())
                        has_ended = True
                    elif state == AudioEnergyFilter.State.SILENCE:
                        # buffer the last silence frame, since it could contain beginning of speech
                        # TODO: improve accuracy by using a ring buffer with longer window
                        last_frame = data
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    chunk_request = self._opts.get_chunk_request(
                        frame.data.tobytes(), has_ended
                    )
                    await ws.send_bytes(chunk_request)

                    if has_ended:
                        self._audio_duration_collector.flush()
                        await ws.send_str(SpeechStream._FINALIZE_MSG)
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
                    raise APIStatusError(message="connection closed unexpectedly")

                try:
                    self._process_stream_event(msg.data)
                except Exception:
                    logger.exception("failed to process message")

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
            self._session.ws_connect(
                self._opts.get_ws_url(), headers=self._opts.get_ws_header()
            ),
            self._conn_options.timeout,
        )
        return ws

    def _check_energy_state(self, frame: rtc.AudioFrame) -> AudioEnergyFilter.State:
        if self._audio_energy_filter:
            return self._audio_energy_filter.update(frame)
        return AudioEnergyFilter.State.SPEAKING

    def _on_audio_duration_report(self, duration: float) -> None:
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)

    def _process_stream_event(self, data: dict) -> None:
        assert self._opts.language is not None
        results = parse_response(res=data)["payload_msg"]
        result = results.get("result", None)
        if result is None:
            return
        utterances = result[0].get("utterances", [])
        if len(utterances) == 0:
            return
        definite = result[0]["utterances"][0].get("definite", "False")
        start_time = result[0]["utterances"][0].get("start_time", 0.0)
        end_time = result[0]["utterances"][0].get("end_time", 0.0)
        text = result[0].get("text", "")
        confidence = result[0].get("confidence", 0.0)
        if not definite and not self._speaking:
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
                        confidence=confidence,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)

        elif not definite and self._speaking:
            alternatives = [
                stt.SpeechData(
                    language=self._opts.language,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                )
            ]
            interim_event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=alternatives,
            )
            self._event_ch.send_nowait(interim_event)

        elif definite and self._speaking:
            alternatives = [
                stt.SpeechData(
                    language=self._opts.language,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
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
            logger.info("transcription end", extra={"text": text})


def parse_response(res):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    payload 类似与http 请求体
    """
    # protocol_version = res[0] >> 4
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    # message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    # reserved = res[3]
    # header_extensions = res[4 : header_size * 4]
    payload = res[header_size * 4 :]
    result = {}
    payload_msg = None
    payload_size = 0
    if message_type == SERVER_FULL_RESPONSE:
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result["seq"] = seq
        if len(payload) >= 8:
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
    elif message_type == SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        result["code"] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]
    if payload_msg is None:
        return result
    if message_compression == GZIP:
        payload_msg = gzip.decompress(payload_msg)
    if serialization_method == JSON:
        payload_msg = json.loads(str(payload_msg, "utf-8"))
    elif serialization_method != NO_SERIALIZATION:
        payload_msg = str(payload_msg, "utf-8")
    result["payload_msg"] = payload_msg
    result["payload_size"] = payload_size
    return result


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
