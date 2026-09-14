from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import time
import weakref
import gzip
import uuid
import base64
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, Callable

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .log import logger


_TOKENS_PER_MILLION = 1_000_000
_INPUT_TEXT_PRICE_CNY = 10.0
_INPUT_AUDIO_PRICE_CNY = 80.0
_CACHED_INPUT_PRICE_CNY = 5.0
_OUTPUT_TEXT_PRICE_CNY = 80.0
_OUTPUT_AUDIO_PRICE_CNY = 300.0


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@dataclass(frozen=True)
class RealtimeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    cached_input_text_tokens: int = 0
    cached_input_audio_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0


def _token_count(data: dict[str, Any], *names: str) -> int:
    for name in names:
        value = data.get(name)
        if value is not None:
            return int(value or 0)
    return 0


def _parse_realtime_usage(data: dict[str, Any]) -> RealtimeUsage:
    input_details = (
        data.get("input_tokens_details") or data.get("input_token_details") or {}
    )
    output_details = (
        data.get("output_tokens_details") or data.get("output_token_details") or {}
    )
    cached_details = (
        input_details.get("cached_tokens_details")
        or input_details.get("cached_token_details")
        or data.get("cached_tokens_details")
        or {}
    )
    cached_text = _token_count(
        cached_details, "text_tokens", "cached_text_tokens"
    ) or _token_count(input_details, "cached_text_tokens")
    cached_audio = _token_count(
        cached_details, "audio_tokens", "cached_audio_tokens"
    ) or _token_count(input_details, "cached_audio_tokens")
    input_text = _token_count(input_details, "text_tokens", "input_text_tokens")
    input_audio = _token_count(input_details, "audio_tokens", "input_audio_tokens")
    output_text = _token_count(output_details, "text_tokens", "output_text_tokens")
    output_audio = _token_count(output_details, "audio_tokens", "output_audio_tokens")
    input_tokens = _token_count(data, "input_tokens") or input_text + input_audio
    output_tokens = _token_count(data, "output_tokens") or output_text + output_audio
    return RealtimeUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_token_count(data, "total_tokens") or input_tokens + output_tokens,
        input_text_tokens=input_text,
        input_audio_tokens=input_audio,
        cached_input_text_tokens=cached_text,
        cached_input_audio_tokens=cached_audio,
        output_text_tokens=output_text,
        output_audio_tokens=output_audio,
    )


def _estimate_realtime_cost_cny(usage: RealtimeUsage) -> float:
    uncached_text = max(0, usage.input_text_tokens - usage.cached_input_text_tokens)
    uncached_audio = max(0, usage.input_audio_tokens - usage.cached_input_audio_tokens)
    return (
        uncached_text * _INPUT_TEXT_PRICE_CNY
        + uncached_audio * _INPUT_AUDIO_PRICE_CNY
        + usage.cached_input_text_tokens * _CACHED_INPUT_PRICE_CNY
        + usage.cached_input_audio_tokens * _CACHED_INPUT_PRICE_CNY
        + usage.output_text_tokens * _OUTPUT_TEXT_PRICE_CNY
        + usage.output_audio_tokens * _OUTPUT_AUDIO_PRICE_CNY
    ) / _TOKENS_PER_MILLION


def _completed_transcript(event: dict[str, Any], accumulated: str = "") -> str:
    return str(event.get("transcript") or event.get("text") or accumulated)


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

MSG_WITH_EVENT = 0b0100

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
    message_type_specific_flags=MSG_WITH_EVENT,
    serial_method=JSON,
    compression_type=GZIP,
    reserved_data=0x00,
    extension_header=bytes(),
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


def parse_response(res):
    """
    - header
        - (4bytes)header
        - (4bits)version(v1) + (4bits)header_size
        - (4bits)messageType + (4bits)messageTypeFlags
            -- 0001	CompleteClient | -- 0001 hasSequence
            -- 0010	audioonly      | -- 0010 isTailPacket
                                           | -- 0100 hasEvent
        - (4bits)payloadFormat + (4bits)compression
        - (8bits) reserve
    - payload
        - [optional 4 bytes] event
        - [optional] session ID
          -- (4 bytes)session ID len
          -- session ID data
        - (4 bytes)data len
        - data
    """
    if isinstance(res, str):
        return {}
    # protocol_version = res[0] >> 4
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    # reserved = res[3]
    # header_extensions = res[4 : header_size * 4]
    payload = res[header_size * 4 :]
    result = {}
    payload_msg = None
    payload_size = 0
    start = 0
    if message_type == SERVER_FULL_RESPONSE or message_type == SERVER_ACK:
        result["message_type"] = "SERVER_FULL_RESPONSE"
        if message_type == SERVER_ACK:
            result["message_type"] = "SERVER_ACK"
        if message_type_specific_flags & NEG_SEQUENCE > 0:
            result["seq"] = int.from_bytes(payload[:4], "big", signed=False)
            start += 4
        if message_type_specific_flags & MSG_WITH_EVENT > 0:
            result["event"] = int.from_bytes(payload[:4], "big", signed=False)
            start += 4
        payload = payload[start:]
        session_id_size = int.from_bytes(payload[:4], "big", signed=True)
        session_id = payload[4 : session_id_size + 4]
        result["session_id"] = str(session_id)
        payload = payload[4 + session_id_size :]
        payload_size = int.from_bytes(payload[:4], "big", signed=False)
        payload_msg = payload[4:]
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


@dataclass
class _RealtimeOptions:
    app_id: str | None
    access_token: str | None
    bot_name: str
    system_role: str
    max_session_duration: float | None
    conn_options: APIConnectOptions
    modalities: list[Literal["text", "audio"]]
    api_key: str | None = None
    opening: str = "你好啊，今天过得怎么样？"
    speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。"
    speaker: str = "zh_female_vv_jupiter_bigtts"
    sample_rate: int = 24000
    num_channels: int = 1
    format: str = "pcm_s16le"
    model: Literal["O", "SC"] = "O"
    character_manifest: str | None = None
    end_smooth_window_ms: int = 500
    enable_volc_websearch: bool = False
    volc_websearch_type: Literal["web_summary", "web"] = "web_summary"
    volc_websearch_api_key: str | None = None
    volc_websearch_no_result_message: str = "抱歉，我找不到相关信息。"

    @property
    def ws_url(self) -> str:
        return "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"

    def get_ws_headers(self) -> dict:
        connect_id = str(uuid.uuid4())
        headers = {
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-Connect-Id": connect_id,
            "X-Api-Request-Id": connect_id,
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key.strip()
            return headers
        if not self.app_id or not self.access_token:
            raise ValueError(
                "VOLCENGINE_REALTIME_API_KEY or app_id/access_token is required"
            )
        headers.update(
            {
                "X-Api-App-Id": self.app_id,
                "X-Api-Access-Key": self.access_token,
                "X-Api-App-Key": "PlgvMymc7f3tQnJ6",
            }
        )
        return headers

    def get_start_session_reqs(self, dialog_id: str | None) -> dict:
        return {
            "asr": {
                "extra": {
                    "end_smooth_window_ms": self.end_smooth_window_ms,
                }
            },
            "tts": {
                "audio_config": {
                    "channel": self.num_channels,
                    "format": self.format,
                    "sample_rate": self.sample_rate,
                },
                "speaker": self.speaker,
            },
            "dialog": {
                "bot_name": self.bot_name,
                "system_role": self.system_role,
                "dialog_id": dialog_id or str(utils.shortuuid()),
                "speaking_style": self.speaking_style,
                "character_manifest": self.character_manifest,
                "extra": {
                    "strict_audit": False,
                    "enable_volc_websearch": self.enable_volc_websearch,
                    "volc_websearch_type": self.volc_websearch_type,
                    "volc_websearch_api_key": self.volc_websearch_api_key,
                    "volc_websearch_no_result_message": self.volc_websearch_no_result_message,
                },
            },
        }


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    audio_transcript: str = ""
    modalities: asyncio.Future[list[Literal["text", "audio"]]] | None = None


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    messages: dict[str, _MessageGeneration]

    _done_fut: asyncio.Future[None]
    _created_timestamp: float
    """timestamp when the response was created"""
    _first_token_timestamp: float | None = None
    """timestamp when the first token was received"""


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        bot_name: str = "豆包",
        speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。",
        speaker: str = "zh_female_vv_jupiter_bigtts",
        opening: str | None = None,
        app_id: str | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
        system_role: str | None = None,
        character_manifest: str = None,
        model: Literal["O", "SC"] = "O",
        end_smooth_window_ms: int = 500,
        enable_volc_websearch: bool = False,
        volc_websearch_type: Literal["web_summary", "web"] = "web_summary",
        volc_websearch_api_key: str = None,
        volc_websearch_no_result_message: str = "抱歉，我找不到相关信息。",
        rag_fn: Callable[[str], str] = None,
        audio_output: bool = True,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        modalities = modalities if utils.is_given(modalities) else ["text", "audio"]
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=("audio" in modalities),
                manual_function_calls=True,
            )
        )
        logger.info(f"Model: {model}")
        logger.info(f"Character Manifest: {character_manifest}")
        logger.info(f"End Smooth Window MS: {end_smooth_window_ms}")
        logger.info(f"Enable Volc Websearch: {enable_volc_websearch}")
        logger.info(f"Volc Websearch Type: {volc_websearch_type}")
        logger.info(f"Volc Websearch API Key: {volc_websearch_api_key}")
        logger.info(
            f"Volc Websearch No Result Message: {volc_websearch_no_result_message}"
        )
        api_key = _clean_secret(
            api_key or os.environ.get("VOLCENGINE_REALTIME_API_KEY")
        )
        app_id = _clean_secret(app_id or os.environ.get("VOLCENGINE_REALTIME_APP_ID"))
        access_token = _clean_secret(
            access_token or os.environ.get("VOLCENGINE_REALTIME_ACCESS_TOKEN")
        )
        if api_key is None and (app_id is None or access_token is None):
            raise ValueError(
                "VOLCENGINE_REALTIME_API_KEY or VOLCENGINE_REALTIME_APP_ID/"
                "VOLCENGINE_REALTIME_ACCESS_TOKEN is required"
            )
        self._opts = _RealtimeOptions(
            app_id=app_id,
            access_token=access_token,
            api_key=api_key,
            bot_name=bot_name,
            system_role=system_role or "",
            speaker=speaker,
            opening=opening,
            speaking_style=speaking_style,
            character_manifest=character_manifest,
            model=model,
            end_smooth_window_ms=end_smooth_window_ms,
            enable_volc_websearch=enable_volc_websearch,
            volc_websearch_type=volc_websearch_type,
            volc_websearch_api_key=volc_websearch_api_key,
            volc_websearch_no_result_message=volc_websearch_no_result_message,
            modalities=modalities,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )
        self._rag_fn = rag_fn
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def update_options(
        self,
        *,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(max_session_duration):
            self._opts.max_session_duration = max_session_duration

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # The Volcengine dialogue protocol owns turn detection and does not
        # expose a session-level switch. Accept the Agents 1.6.9 keyword so
        # the framework can create the session without a TypeError; the
        # capability remains false, so external turn handling is unsupported.
        sess = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None: ...


class RealtimeSession(
    llm.RealtimeSession[
        Literal["volcengine_server_event_received", "volcengine_client_event_queued"]
    ]
):
    """
    A session for the volcengine Realtime API.

    This class is used to interact with the volcengine Realtime API.
    It is responsible for sending events to the volcengine Realtime API and receiving events from it.

    It exposes two more events:
    - volcengine_server_event_received: expose the raw server events from the OpenAI Realtime API
    - volcengine_client_event_queued: expose the raw client events sent to the OpenAI Realtime API
    """

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        turn_detection_disabled: bool = False,
    ) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        # Keep a session-local copy for API compatibility and future provider
        # support. Volcengine currently has no turn_detection field to alter.
        self._opts = replace(realtime_model._opts)
        self._turn_detection_disabled = turn_detection_disabled
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[rtc.AudioFrame | dict]()
        self._input_resampler: rtc.AudioResampler | None = None
        self.session_id = str(uuid.uuid4())

        self._instructions: str | None = None
        self._main_atask = asyncio.create_task(
            self._main_task(), name="RealtimeSession._main_task"
        )

        self._response_created_futures: dict[
            str, asyncio.Future[llm.GenerationCreatedEvent]
        ] = {}
        self._item_delete_future: dict[str, asyncio.Future] = {}
        self._item_create_future: dict[str, asyncio.Future] = {}

        self._current_generation: _ResponseGeneration | None = None
        self._current_generation_event: llm.GenerationCreatedEvent | None = None
        self._current_item: _MessageGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        self._chat_ctx = llm.ChatContext.empty()
        self._is_opening = False
        self._first_tts_response = True
        self._first_llm_response = True
        self._first_llm_sentence = True

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # The duplex protocol recommends 20 ms packets (640 bytes at 16 kHz).
        self._bstream = utils.audio.AudioByteStream(
            16000,
            self._realtime_model._opts.num_channels,
            samples_per_channel=16000 // 50,
        )
        self._pushed_duration_s: float = (
            0  # duration of audio pushed to the OpenAI Realtime API
        )
        self._audio_muted = False
        self._input_transcripts: dict[str, str] = {}
        self._sent_user_item_ids: set[str] = set()
        self._usage = RealtimeUsage()
        self._estimated_cost_cny = 0.0

    def send_event(self, event: rtc.AudioFrame | dict) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @property
    def usage(self) -> RealtimeUsage:
        return self._usage

    @property
    def estimated_cost_cny(self) -> float:
        return self._estimated_cost_cny

    def _send_json_event(self, event: dict) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        logger.info("start realtime main task")
        # while not self._msg_ch.closed:
        ws_conn = await self._create_ws_conn()

        try:
            await self._run_ws(ws_conn)

        except Exception as e:
            logger.error("realtime main task error", exc_info=e)
            self._emit_error(e, recoverable=False)
            raise e
        logger.info("realtime main task break")
        # break

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = self._realtime_model._opts.get_ws_headers()
        url = self._realtime_model._opts.ws_url
        try:
            return await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(
                    url=url,
                    headers=headers,
                ),
                self._realtime_model._opts.conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as exc:
            if exc.status in (401, 403):
                raise RuntimeError(
                    f"火山引擎 Realtime 握手失败（{exc.status}）。"
                    "请确认 VOLCENGINE_REALTIME_API_KEY 来自豆包语音控制台新版 API Key，"
                    "并已开通端到端实时语音（Seeduplex）资源；"
                    "或改用 VOLCENGINE_REALTIME_APP_ID / VOLCENGINE_REALTIME_ACCESS_TOKEN。"
                ) from exc
            raise

    async def _run_ws_legacy(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False
        logger.info("start connection")
        start_connection_request = bytearray(generate_header())
        start_connection_request.extend(int(1).to_bytes(4, "big"))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        start_connection_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        start_connection_request.extend(payload_bytes)
        await ws_conn.send_bytes(start_connection_request)
        _ = await ws_conn.receive_bytes()

        logger.info("start session")
        await self._start_session(ws_conn=ws_conn, dialog_id=self.session_id)

        if self._realtime_model._opts.opening is not None:
            self._is_opening = True
            payload = {
                "content": self._realtime_model._opts.opening,
            }
            hello_request = bytearray(generate_header())
            hello_request.extend(int(300).to_bytes(4, "big"))
            payload_bytes = str.encode(json.dumps(payload))
            payload_bytes = gzip.compress(payload_bytes)
            hello_request.extend((len(self.session_id)).to_bytes(4, "big"))
            hello_request.extend(str.encode(self.session_id))
            hello_request.extend((len(payload_bytes)).to_bytes(4, "big"))
            hello_request.extend(payload_bytes)
            await ws_conn.send_bytes(hello_request)
            self._is_opening = True
            logger.info("send hello request")

            self._current_generation = _ResponseGeneration(
                message_ch=utils.aio.Chan(),
                function_ch=utils.aio.Chan(),
                messages={},
                _created_timestamp=time.time(),
                _done_fut=asyncio.Future(),
            )

            generation_ev = llm.GenerationCreatedEvent(
                message_stream=self._current_generation.message_ch,
                function_stream=self._current_generation.function_ch,
                user_initiated=False,
            )
            self.emit("generation_created", generation_ev)
            item_id = utils.shortuuid()
            modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = (
                asyncio.Future()
            )
            self._current_item = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
                modalities=modalities_fut,
            )
            if not self._realtime_model.capabilities.audio_output:
                self._current_item.audio_ch.close()
                self._current_item.modalities.set_result(["text"])  # type: ignore[union-attr]
            else:
                self._current_item.modalities.set_result(["audio", "text"])  # type: ignore[union-attr]

            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=self._current_item.text_ch,
                    audio_stream=self._current_item.audio_ch,
                    modalities=self._current_item.modalities,
                )
            )

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for frame in self._msg_ch:
                try:
                    task_request = bytearray(
                        generate_header(
                            message_type=CLIENT_AUDIO_ONLY_REQUEST,
                            serial_method=NO_SERIALIZATION,
                        )
                    )
                    task_request.extend(int(200).to_bytes(4, "big"))
                    task_request.extend((len(self.session_id)).to_bytes(4, "big"))
                    task_request.extend(str.encode(self.session_id))
                    payload_bytes = gzip.compress(frame.data.tobytes())
                    task_request.extend(
                        (len(payload_bytes)).to_bytes(4, "big")
                    )  # payload size(4 bytes)
                    task_request.extend(payload_bytes)
                    await ws_conn.send_bytes(task_request)

                except Exception:
                    logger.error("send task error", exc_info=True)
                    break

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                try:
                    msg = await ws_conn.receive()
                    if msg.data is None:
                        continue
                    response = parse_response(msg.data)
                    event = response.get("event")
                    if event == 450:  # ASRInfo
                        self.emit("input_speech_started", llm.InputSpeechStartedEvent())
                        logger.info("transcription start")
                    elif event == 451:  # ASRResponse
                        response = response["payload_msg"]
                        transcription = response["results"][0]["alternatives"][0][
                            "text"
                        ]
                        is_final = not response["results"][0]["is_interim"]
                        if is_final:
                            item_id = utils.shortuuid()
                            self.emit(
                                "input_audio_transcription_completed",
                                llm.InputTranscriptionCompleted(
                                    item_id=item_id,
                                    transcript=transcription,
                                    is_final=True,
                                ),
                            )
                            if self._current_generation is None:
                                self._current_generation = _ResponseGeneration(
                                    message_ch=utils.aio.Chan(),
                                    function_ch=utils.aio.Chan(),
                                    messages={},
                                    _created_timestamp=time.time(),
                                    _done_fut=asyncio.Future(),
                                )

                                generation_ev = llm.GenerationCreatedEvent(
                                    message_stream=self._current_generation.message_ch,
                                    function_stream=self._current_generation.function_ch,
                                    user_initiated=False,
                                )

                                self.emit("generation_created", generation_ev)
                                item_id = utils.shortuuid()
                                modalities_fut: asyncio.Future[
                                    list[Literal["text", "audio"]]
                                ] = asyncio.Future()
                                self._current_item = _MessageGeneration(
                                    message_id=item_id,
                                    text_ch=utils.aio.Chan(),
                                    audio_ch=utils.aio.Chan(),
                                    modalities=modalities_fut,
                                )
                                if not self._realtime_model.capabilities.audio_output:
                                    self._current_item.audio_ch.close()
                                    self._current_item.modalities.set_result(["text"])  # type: ignore[union-attr]
                                else:
                                    self._current_item.modalities.set_result(
                                        ["audio", "text"]
                                    )  # type: ignore[union-attr]
                                self._current_generation.message_ch.send_nowait(
                                    llm.MessageGeneration(
                                        message_id=item_id,
                                        text_stream=self._current_item.text_ch,
                                        audio_stream=self._current_item.audio_ch,
                                        modalities=self._current_item.modalities,
                                    )
                                )

                    elif event == 459:  # ASREnd
                        logger.info("transcription end")
                        self.emit(
                            "input_speech_stopped",
                            llm.InputSpeechStoppedEvent(
                                user_transcription_enabled=False
                            ),
                        )
                        if self._realtime_model._rag_fn is not None:
                            logger.info("rag start")
                            rag_result = self._realtime_model._rag_fn(transcription)
                            payload = {
                                "external_rag": rag_result,
                            }
                            payload_bytes = str.encode(json.dumps(payload))
                            payload_bytes = gzip.compress(payload_bytes)
                            chat_rag_text_request = bytearray(generate_header())
                            chat_rag_text_request.extend(int(502).to_bytes(4, "big"))
                            chat_rag_text_request.extend(
                                (len(self.session_id)).to_bytes(4, "big")
                            )
                            chat_rag_text_request.extend(str.encode(self.session_id))
                            chat_rag_text_request.extend(
                                (len(payload_bytes)).to_bytes(4, "big")
                            )
                            chat_rag_text_request.extend(payload_bytes)
                            await ws_conn.send_bytes(chat_rag_text_request)
                            logger.info("rag end")
                        logger.info("llm start")
                        logger.info("tts start")

                    elif event == 352:  # TTSResponse
                        if self._first_tts_response:
                            logger.info("llm first sentence")
                            logger.info("tts first response")
                            self._first_tts_response = False
                        audio_bytes = response[
                            "payload_msg"
                        ]  # 原始为float32，需要转为int16
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)
                        # 裁剪到 [-1.0, 1.0]，避免溢出
                        audio = np.clip(audio, -1.0, 1.0)
                        audio = (audio * 32767).astype(np.int16)
                        audio_bytes = audio.tobytes()
                        self._current_item.audio_ch.send_nowait(
                            rtc.AudioFrame(
                                data=audio_bytes,
                                sample_rate=self._realtime_model._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(audio_bytes) // 2,
                            )
                        )
                    elif event == 350:  # TTSSentenceStart
                        pass
                    elif event == 351:  # TTSSentenceEnd
                        pass
                    elif event == 359:  # TTSEnded
                        logger.info("tts end")
                        self._current_item.audio_ch.close()
                        if self._is_opening:
                            self._current_item.text_ch.send_nowait(
                                self._realtime_model._opts.opening
                            )
                            self._current_item.text_ch.close()
                            self._is_opening = False
                        self._current_generation.message_ch.close()
                        self._current_generation.function_ch.close()
                        self._current_generation = None
                        self._first_tts_response = True
                    elif event == 550:  # 模型回复的文本内容
                        if self._first_llm_response:
                            logger.info("llm first response")
                            self._first_llm_response = False
                        text = response["payload_msg"]["content"]
                        self._current_item.text_ch.send_nowait(text)
                    elif event == 559:  # 模型回复文本结束事件
                        logger.info("llm end")
                        self._current_item.text_ch.close()
                        self._first_llm_response = True
                    else:
                        pass
                except Exception:
                    logger.error("recv task error", exc_info=True)
                    break

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()

        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        async def send(event: dict) -> None:
            await ws_conn.send_str(
                json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            )

        opts = self._realtime_model._opts
        await send(
            {
                "type": "session.create",
                "event_id": utils.shortuuid("event_"),
                "session": {
                    "id": self.session_id,
                    "model": "1.2.6.1",
                    "instructions": opts.system_role or "",
                    "audio": {
                        "input": {"format": {"type": "pcm", "rate": 16000}},
                        "output": {
                            "format": {"type": opts.format, "rate": 24000},
                            "voice": opts.speaker,
                        },
                    },
                },
                "extension": opts.get_start_session_reqs(self.session_id),
            }
        )
        while True:
            msg = await ws_conn.receive()
            if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                continue
            event = json.loads(
                msg.data.decode() if isinstance(msg.data, bytes) else msg.data
            )
            if event.get("type") == "session.created":
                self.session_id = event.get("session", {}).get("id", self.session_id)
                break
            if event.get("type") == "error":
                raise RuntimeError(event)

        if opts.opening:
            self._ensure_generation()
            await send(
                {
                    "type": "speech_text_buffer.commit",
                    "event_id": utils.shortuuid("event_"),
                    "text": opts.opening,
                }
            )

        async def recv_task() -> None:
            async for msg in ws_conn:
                if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    continue
                event = json.loads(
                    msg.data.decode() if isinstance(msg.data, bytes) else msg.data
                )
                kind = event.get("type", "")
                if kind == "error":
                    raise RuntimeError(event)
                if kind == "conversation.item.input_audio_transcription.started":
                    item_id = str(event.get("item_id") or utils.shortuuid("item_"))
                    self._input_transcripts[item_id] = ""
                    self.emit("input_speech_started", llm.InputSpeechStartedEvent())
                elif kind == "conversation.item.input_audio_transcription.delta":
                    item_id = str(event.get("item_id") or "")
                    if item_id:
                        self._input_transcripts[item_id] = self._input_transcripts.get(
                            item_id, ""
                        ) + str(event.get("delta") or "")
                elif kind == "conversation.item.input_audio_transcription.completed":
                    item_id = str(event.get("item_id") or utils.shortuuid("item_"))
                    transcript = _completed_transcript(
                        event,
                        self._input_transcripts.get(item_id, ""),
                    )
                    self._input_transcripts.pop(item_id, None)
                    logger.debug(
                        "Volcengine Realtime transcription completed",
                        extra={"item_id": item_id, "transcript": transcript},
                    )
                    self.emit(
                        "input_audio_transcription_completed",
                        llm.InputTranscriptionCompleted(
                            item_id=item_id,
                            transcript=transcript,
                            is_final=True,
                        ),
                    )
                    self.emit(
                        "input_speech_stopped",
                        llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
                    )
                elif kind == "response.output_text.delta":
                    self._ensure_generation()
                    if self._current_generation._first_token_timestamp is None:
                        self._current_generation._first_token_timestamp = time.time()
                    self._current_item.text_ch.send_nowait(event.get("delta", ""))
                elif kind == "response.output_text.done" and self._current_item:
                    self._current_item.text_ch.close()
                elif kind == "response.output_audio.delta":
                    self._ensure_generation()
                    if self._current_generation._first_token_timestamp is None:
                        self._current_generation._first_token_timestamp = time.time()
                    data = base64.b64decode(event.get("delta", ""))
                    self._current_item.audio_ch.send_nowait(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=24000,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )
                elif kind == "response.function_call_arguments.done":
                    self._ensure_generation()
                    for item in event.get("items", []):
                        self._current_generation.function_ch.send_nowait(
                            llm.FunctionCall(
                                call_id=str(item.get("call_id", "")),
                                name=str(item.get("name", "")),
                                arguments=str(item.get("arguments", "")),
                            )
                        )
                elif kind == "response.output_audio.done":
                    if self._current_item:
                        # Some 3.0 responses do not emit
                        # response.output_text.done. Audio is the authoritative
                        # end-of-turn signal, so close both streams here.
                        self._current_item.text_ch.close()
                        self._current_item.audio_ch.close()
                    if self._current_generation:
                        self._current_generation.message_ch.close()
                        self._current_generation.function_ch.close()
                        self._current_generation = None
                        self._current_generation_event = None
                        self._current_item = None
                elif kind == "response.done":
                    self._handle_response_done(event)

        async def send_task() -> None:
            async for item in self._msg_ch:
                if isinstance(item, rtc.AudioFrame):
                    await send(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(item.data.tobytes()).decode(
                                "ascii"
                            ),
                        }
                    )
                else:
                    await send(item)

        tasks = [asyncio.create_task(recv_task()), asyncio.create_task(send_task())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _ensure_generation(self) -> llm.GenerationCreatedEvent:
        if self._current_generation is not None:
            assert self._current_generation_event is not None
            return self._current_generation_event
        generation = _ResponseGeneration(
            utils.aio.Chan(), utils.aio.Chan(), {}, asyncio.Future(), time.time()
        )
        self._current_generation = generation
        generation_event = llm.GenerationCreatedEvent(
            message_stream=generation.message_ch,
            function_stream=generation.function_ch,
            user_initiated=False,
        )
        self._current_generation_event = generation_event
        self.emit("generation_created", generation_event)
        modalities = asyncio.Future[list[Literal["text", "audio"]]]()
        self._current_item = _MessageGeneration(
            utils.shortuuid(), utils.aio.Chan(), utils.aio.Chan(), modalities=modalities
        )
        if self._realtime_model.capabilities.audio_output:
            modalities.set_result(["audio", "text"])
        else:
            self._current_item.audio_ch.close()
            modalities.set_result(["text"])
        generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=self._current_item.message_id,
                text_stream=self._current_item.text_ch,
                audio_stream=self._current_item.audio_ch,
                modalities=modalities,
            )
        )
        return generation_event

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response") or event
        usage = _parse_realtime_usage(response.get("usage") or event.get("usage") or {})
        response_id = str(
            response.get("id")
            or event.get("response_id")
            or utils.shortuuid("response_")
        )
        generation = self._current_generation
        created = generation._created_timestamp if generation else time.time()
        duration = max(time.time() - created, 1e-6)
        response_cost = _estimate_realtime_cost_cny(usage)
        self._usage = RealtimeUsage(
            **{
                field: getattr(self._usage, field) + getattr(usage, field)
                for field in RealtimeUsage.__dataclass_fields__
            }
        )
        self._estimated_cost_cny += response_cost
        logger.info(
            "Volcengine Realtime usage cost",
            extra={
                "response_id": response_id,
                **usage.__dict__,
                "estimated_response_cost_cny": round(response_cost, 6),
                "estimated_session_cost_cny": round(self._estimated_cost_cny, 6),
                "pricing_basis": (
                    "Volcengine China public pay-as-you-go list price; free quota, "
                    "discounts and negotiated pricing excluded"
                ),
            },
        )
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id=response_id,
                timestamp=created,
                duration=duration,
                ttft=(
                    generation._first_token_timestamp - created
                    if generation and generation._first_token_timestamp is not None
                    else -1
                ),
                cancelled=False,
                label=self._realtime_model.label,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                tokens_per_second=usage.output_tokens / duration,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    text_tokens=usage.input_text_tokens,
                    audio_tokens=usage.input_audio_tokens,
                    image_tokens=0,
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=usage.output_text_tokens,
                    audio_tokens=usage.output_audio_tokens,
                    image_tokens=0,
                ),
                metadata=Metadata(
                    model_name="1.2.6.1",
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )
        if self._current_item:
            with contextlib.suppress(Exception):
                self._current_item.text_ch.close()
            with contextlib.suppress(Exception):
                self._current_item.audio_ch.close()
        if generation:
            generation.message_ch.close()
            generation.function_ch.close()
        self._current_item = None
        self._current_generation = None
        self._current_generation_event = None

    def _create_session_update_event(self):
        pass

    async def chat_tts_text(
        self,
        start: bool,
        end: bool,
        content: str,
        ws_conn: aiohttp.ClientWebSocketResponse,
    ) -> None:
        """发送Chat TTS Text消息"""
        payload = {
            "start": start,
            "end": end,
            "content": content,
        }
        logger.info("ChatTTSTextRequest")
        payload_bytes = str.encode(json.dumps(payload))
        payload_bytes = gzip.compress(payload_bytes)

        chat_tts_text_request = bytearray(generate_header())
        chat_tts_text_request.extend(int(500).to_bytes(4, "big"))
        chat_tts_text_request.extend((len(self.session_id)).to_bytes(4, "big"))
        chat_tts_text_request.extend(str.encode(self.session_id))
        chat_tts_text_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        chat_tts_text_request.extend(payload_bytes)
        await ws_conn.send_bytes(chat_tts_text_request)

    async def _start_session(
        self, ws_conn: aiohttp.ClientWebSocketResponse, dialog_id: str
    ) -> None:
        request_params = self._realtime_model._opts.get_start_session_reqs(
            dialog_id=dialog_id
        )
        payload_bytes = str.encode(json.dumps(request_params))
        payload_bytes = gzip.compress(payload_bytes)
        start_session_request = bytearray(generate_header())
        start_session_request.extend(int(100).to_bytes(4, "big"))
        start_session_request.extend((len(self.session_id)).to_bytes(4, "big"))
        start_session_request.extend(str.encode(self.session_id))
        start_session_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        start_session_request.extend(payload_bytes)
        await ws_conn.send_bytes(start_session_request)
        _ = await ws_conn.receive_bytes()

    async def _finish_session(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        finish_session_request = bytearray(generate_header())
        finish_session_request.extend(int(102).to_bytes(4, "big"))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        finish_session_request.extend((len(self.session_id)).to_bytes(4, "big"))
        finish_session_request.extend(str.encode(self.session_id))
        finish_session_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        finish_session_request.extend(payload_bytes)
        await ws_conn.send_bytes(finish_session_request)

    async def _finish_connection(
        self, ws_conn: aiohttp.ClientWebSocketResponse
    ) -> None:
        finish_connection_request = bytearray(generate_header())
        finish_connection_request.extend(int(2).to_bytes(4, "big"))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        finish_connection_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        finish_connection_request.extend(payload_bytes)
        await ws_conn.send_bytes(finish_connection_request)
        _ = await ws_conn.receive_bytes()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(voice):
            self._opts.speaker = voice
            self._send_json_event(
                {
                    "type": "session.update",
                    "event_id": utils.shortuuid("event_"),
                    "session": {"audio": {"output": {"voice": voice}}},
                }
            )

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)
        self._send_json_event(
            {
                "type": "session.update",
                "event_id": utils.shortuuid("event_"),
                "session": {"tools": self._tools.parse_function_tools("openai")},
            }
        )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx.copy()
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        items = []
        for item in chat_ctx.items:
            if item.type != "message":
                continue
            role = item.role
            text = " ".join(part for part in item.content if isinstance(part, str))
            if text:
                item_id = str(item.id or utils.shortuuid("item_"))
                items.append(
                    {
                        "id": item_id,
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
                if role == "user":
                    self._sent_user_item_ids.add(item_id)
        if items:
            self._send_json_event(
                {
                    "type": "conversation.item.create",
                    "event_id": utils.shortuuid("event_"),
                    "items": items,
                }
            )

    def _create_update_chat_ctx_events(self, chat_ctx: llm.ChatContext):
        events = []

        return events

    async def update_instructions(self, instructions: str) -> None:
        self._opts.system_role = instructions
        self._send_json_event(
            {
                "type": "session.update",
                "event_id": utils.shortuuid("event_"),
                "session": {"instructions": instructions},
            }
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_muted:
            self._audio_muted = False
            self._send_json_event(
                {
                    "type": "input_audio_unmute.commit",
                    "event_id": utils.shortuuid("event_"),
                }
            )
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.send_event(nf)
                self._pushed_duration_s += nf.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        if self._pushed_duration_s > 0.1:
            self._pushed_duration_s = 0
            self._send_json_event(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": utils.shortuuid("event_"),
                }
            )

    def clear_audio(self) -> None:
        self._pushed_duration_s = 0
        if not self._audio_muted:
            self._audio_muted = True
            self._send_json_event(
                {
                    "type": "input_audio_mute.commit",
                    "event_id": utils.shortuuid("event_"),
                }
            )

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """使用 3.0 Realtime 文本事件请求一轮回复。"""
        future = asyncio.get_running_loop().create_future()
        if utils.is_given(tools):
            self._tools = llm.ToolContext(tools)
            self._send_json_event(
                {
                    "type": "session.update",
                    "event_id": utils.shortuuid("event_"),
                    "session": {"tools": self._tools.parse_function_tools("openai")},
                }
            )
        text = instructions
        user_item_id: str | None = None
        if not utils.is_given(text):
            for item in reversed(self._chat_ctx.items):
                if item.type == "message" and item.role == "user" and item.text_content:
                    item_id = str(item.id or "")
                    if item_id and item_id in self._sent_user_item_ids:
                        continue
                    text = item.text_content
                    user_item_id = item_id or None
                    break
        if utils.is_given(text) and text:
            user_item_id = user_item_id or utils.shortuuid("item_")
            # speech_text_buffer.commit is a direct-TTS event; input_text must
            # be inserted as a user conversation item so the model answers it.
            self._send_json_event(
                {
                    "type": "conversation.item.create",
                    "event_id": utils.shortuuid("event_"),
                    "items": [
                        {
                            "id": user_item_id,
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": text}],
                        }
                    ],
                }
            )
            self._sent_user_item_ids.add(user_item_id)
        future.set_result(self._ensure_generation())
        return future

    def interrupt(self) -> None:
        self._send_json_event(
            {"type": "response.cancel", "event_id": utils.shortuuid("event_")}
        )

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if "audio" in modalities:
            # 当前 volcengine 实时接口未暴露远端音频截断事件；占位以对齐接口
            pass
        elif utils.is_given(audio_transcript):
            # 同步转写文本到远端会话上下文
            chat_ctx = self.chat_ctx.copy()
            if (idx := chat_ctx.index_by_id(message_id)) is not None:
                new_item = copy.copy(chat_ctx.items[idx])
                assert new_item.type == "message"

                new_item.content = [audio_transcript]
                chat_ctx.items[idx] = new_item
                events = self._create_update_chat_ctx_events(chat_ctx)
                for ev in events:
                    self.send_event(ev)

    async def aclose(self) -> None:
        self._send_json_event(
            {"type": "session.close", "event_id": utils.shortuuid("event_")}
        )
        self._msg_ch.close()
        await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != 16000
            or frame.num_channels != self._realtime_model._opts.num_channels
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=16000,
                num_channels=self._realtime_model._opts.num_channels,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )
