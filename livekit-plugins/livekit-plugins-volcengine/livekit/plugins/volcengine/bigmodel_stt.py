from __future__ import annotations

import asyncio
import gzip
import json
import os
import weakref
from dataclasses import dataclass
from typing import Literal

import aiohttp

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
from livekit.agents.utils import AudioBuffer

from .log import logger

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

# Message Type:
FULL_CLIENT_REQUEST = 0b0001
AUDIO_ONLY_REQUEST = 0b0010
FULL_SERVER_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_WITH_SEQUENCE = 0b0011
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001


def generate_header(
    message_type=FULL_CLIENT_REQUEST,
    message_type_specific_flags=NO_SEQUENCE,
    serial_method=JSON,
    compression_type=GZIP,
    reserved_data=0x00,
):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    """
    header = bytearray()
    header_size = 1
    header.append((PROTOCOL_VERSION << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    return header


def generate_before_payload(sequence: int):
    before_payload = bytearray()
    before_payload.extend(sequence.to_bytes(4, "big", signed=True))  # sequence
    return before_payload


@dataclass
class BigModelSTTOptions:
    # app
    app_id: str | None = None
    access_token: str | None = None
    source_type: Literal["duration", "concurrent"] = "duration"

    # audio
    base_url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    format: Literal["pcm", "wav", "ogg"] = "pcm"
    rate: int = 16000
    bits: int = 16
    num_channels: int = 1
    language: str = "zh-CN"

    # request
    model_name: str = "bigmodel"
    codec: Literal["raw", "opus"] = "raw"
    enable_itn: bool = False
    enable_punc: bool = True
    enable_ddc: bool = False
    show_utterance: bool = True
    result_type: Literal["full", "single"] = "single"
    vad_segment_duration: int = 3000
    end_window_size: int = 500
    force_to_speech_time: int = 1000

    def get_ws_url(self):
        return self.base_url

    def get_ws_query_params(self, uid: str | None = None) -> bytearray:
        if uid is None:
            uid = utils.shortuuid()
        submit_request_json = {
            "user": {"uid": uid},
            "audio": {
                "format": self.format,
                "rate": self.rate,
                "bits": self.bits,
                "channels": self.num_channels,
                "codec": self.codec,
            },
            "request": {
                "model_name": self.model_name,
                "enable_itn": self.enable_itn,
                "enable_punc": self.enable_punc,
                "enable_ddc": self.enable_ddc,
                "show_utterance": self.show_utterance,
                "result_type": self.result_type,
                "vad_segment_duration": self.vad_segment_duration,
                "end_window_size": self.end_window_size,
                "force_to_speech_time": self.force_to_speech_time,
            },
        }
        payload_bytes = str.encode(json.dumps(submit_request_json))
        payload_bytes = gzip.compress(
            payload_bytes
        )  # if no compression, comment this line
        full_client_request = bytearray(
            generate_header(message_type_specific_flags=POS_SEQUENCE)
        )
        seq = 1
        full_client_request.extend(
            generate_before_payload(sequence=seq)
        )  # payload size(4 bytes)
        full_client_request.extend(
            (len(payload_bytes)).to_bytes(4, "big")
        )  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        return full_client_request

    def get_chunk_request(
        self, chunk: bytes, seq: int, last: bool = False
    ) -> bytearray:
        payload_bytes = gzip.compress(chunk)
        audio_only_request = bytearray(
            generate_header(
                message_type=AUDIO_ONLY_REQUEST,
                message_type_specific_flags=POS_SEQUENCE,
            )
        )
        if last:
            audio_only_request = bytearray(
                generate_header(
                    message_type=AUDIO_ONLY_REQUEST,
                    message_type_specific_flags=NEG_WITH_SEQUENCE,
                )
            )
        audio_only_request.extend(generate_before_payload(sequence=seq))
        audio_only_request.extend(
            (len(payload_bytes)).to_bytes(4, "big")
        )  # payload size(4 bytes)
        audio_only_request.extend(payload_bytes)
        return audio_only_request

    def get_ws_header(self, reqid: str | None = None) -> dict[str, str]:
        header = {}
        if reqid is None:
            reqid = utils.shortuuid()
        if self.source_type == "duration":
            header["X-Api-Resource-Id"] = "volc.bigasr.sauc.duration"
        else:
            header["X-Api-Resource-Id"] = "volc.bigasr.sauc.concurrent"
        if self.app_id is None:
            self.app_id = os.environ.get("VOLCENGINE_STT_APP_ID", None)
            if self.app_id is None:
                raise ValueError("VOLCENGINE_STT_APP_ID is not set")
        if self.access_token is None:
            self.access_token = os.environ.get("VOLCENGINE_STT_ACCESS_TOKEN", None)
            if self.access_token is None:
                raise ValueError("VOLCENGINE_STT_ACCESS_TOKEN is not set")
        header["X-Api-Access-Key"] = self.access_token
        header["X-Api-App-Key"] = self.app_id
        header["X-Api-Request-Id"] = reqid
        return header


class BigModelSTT(stt.STT):
    def __init__(
        self,
        *,
        app_id: str | None = None,
        base_url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
        access_token: str | None = None,
        model_name: str = "bigmodel",
        enable_itn: bool = False,
        enable_punc: bool = True,
        enable_ddc: bool = False,
        vad_segment_duration: int = 3000,
        end_window_size: int = 500,
        force_to_speech_time: int = 1000,
        http_session: aiohttp.ClientSession | None = None,
        interim_results: bool = True,
    ) -> None:
        """火山引擎大模型实时语音识别

                Args:
                    app_id (str): 应用ID,可以在控制台查看。
                    base_url (str, optional): 地址. Defaults to "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel".
                    access_token (str | None, optional): 使用火山引擎控制台获取的Access Token. Defaults to None.
                    audio_format (Literal[&quot;raw&quot;, &quot;wav&quot;, &quot;mp3&quot;, &quot;ogg&quot;], optional): 音频容器格式. Defaults to "raw".
                    sample_rate (int, optional): 音频采样率. Defaults to 16000.
                    bits (int, optional): 音频采样点位数. Defaults to 16.
                    num_channels (int, optional): 音频声道数. Defaults to 1.
                    model_name (str, optional): 模型名称，目前只有bigmodel. Defaults to "bigmodel".
                    codec (Literal[&quot;raw&quot;, &quot;opus&quot;], optional): 音频编码格式. Defaults to "raw".
                    enable_itn (bool, optional): 文本规范化 (ITN) 是自动语音识别 (ASR) 后处理管道的一部分。 ITN 的任务是将 ASR 模型的原始语音输出转换为书面形式，以提高文本的可读性。
        例如，“一九七零年”->“1970年”和“一百二十三美元”->“$123”。. Defaults to False.
                    enable_punc (bool, optional): 启用标点. Defaults to True.
                    enable_ddc (bool, optional): **语义顺滑**‌是一种技术，旨在提高自动语音识别（ASR）结果的文本可读性和流畅性。这项技术通过删除或修改ASR结果中的不流畅部分，如停顿词、语气词、语义重复词等，使得文本更加易于阅读和理解。. Defaults to False.
                    vad_segment_duration (int, optional): 单位ms，默认为3000。当静音时间超过该值时，会将文本分为两个句子。不决定判停，所以不会修改definite出现的位置。在end_window_size配置后，该参数失效。. Defaults to 3000.
                    end_window_size (int, optional): 单位ms，默认为800，最小200。静音时长超过该值，会直接判停，输出definite。配置该值，不使用语义分句，根据静音时长来分句。用于实时性要求较高场景，可以提前获得definite句子. Defaults to 500.
                    force_to_speech_time (int, optional): 单位ms，默认为10000，最小1。音频时长超过该值之后，才会判停，根据静音时长输出definite，需配合end_window_size使用。
        用于解决短音频+实时性要求较高场景，不配置该参数，只使用end_window_size时，前10s不会判停。推荐设置1000，可能会影响识别准确率. Defaults to 1000.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )

        self._opts = BigModelSTTOptions(
            base_url=base_url,
            access_token=access_token,
            app_id=app_id,
            model_name=model_name,
            enable_itn=enable_itn,
            enable_punc=enable_punc,
            enable_ddc=enable_ddc,
            vad_segment_duration=vad_segment_duration,
            end_window_size=end_window_size,
            force_to_speech_time=force_to_speech_time,
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
        stt: BigModelSTT,
        opts: BigModelSTTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.rate)

        self._opts = opts
        self._session = http_session
        self._speaking = False

        self._request_id = utils.shortuuid()
        self._reconnect_event = asyncio.Event()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws

            full_client_request = self._opts.get_ws_query_params(uid=self._request_id)
            await ws.send_bytes(full_client_request)

            samples_100ms = self._opts.rate // 10
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_100ms,
            )
            has_ended = False
            seq = 1
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True
                for frame in frames:
                    seq += 1
                    if has_ended:
                        seq = -seq
                    chunk_request = self._opts.get_chunk_request(
                        frame.data.tobytes(), seq=seq, last=has_ended
                    )
                    await ws.send_bytes(chunk_request)

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
                self._opts.get_ws_url(),
                headers=self._opts.get_ws_header(reqid=self._request_id),
                max_msg_size=1000000000,
            ),
            self._conn_options.timeout,
        )
        return ws

    def _process_stream_event(self, data: dict) -> None:
        results = parse_response(res=data)["payload_msg"]
        result = results.get("result", None)
        if result is None:
            return
        text = result.get("text", "")
        if text == "":
            return
        utterances = result.get("utterances", [])
        if len(utterances) == 0:
            return
        language = self._opts.language
        definite = utterances[0].get("definite", "False")
        start_time = utterances[0].get("start_time", 0.0)
        end_time = utterances[0].get("end_time", 0.0)
        confidence = result.get("confidence", 0.0)
        if not definite and not self._speaking:
            self._speaking = True
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_ch.send_nowait(start_event)
            logger.info("transcription start")
            if text:
                alternatives = [
                    stt.SpeechData(
                        language=language,
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
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    language=language,
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
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    language=language,
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
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    payload = res[header_size * 4 :]
    result = {
        "is_last_package": False,
    }
    payload_msg = None
    payload_size = 0
    if message_type_specific_flags & 0x01:
        # receive frame with sequence
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result["payload_sequence"] = seq
        payload = payload[4:]

    if message_type_specific_flags & 0x02:
        # receive last package
        result["is_last_package"] = True

    if message_type == FULL_SERVER_RESPONSE:
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
