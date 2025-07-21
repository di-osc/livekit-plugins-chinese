from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import os
import time
import weakref
import gzip
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast, overload
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from pydantic import BaseModel, ValidationError

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, io, llm, utils
from livekit.agents.llm.tool_context import (
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from openai.types.beta.realtime import (
    ConversationItem,
    ConversationItemContent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemTruncateEvent,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeClientEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionUpdateEvent,
    session_update_event,
)
from openai.types.beta.realtime.response_create_event import Response
from openai.types.beta.realtime.session import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    Tracing,
    TracingTracingConfiguration,
    TurnDetection,
)


from .log import logger

# When a response is created with the OpenAI Realtime API, those events are sent in this order:
# 1. response.created (contains resp_id)
# 2. response.output_item.added (contains item_id)
# 3. conversation.item.created
# 4. response.content_part.added (type audio/text)
# 5. response.audio_transcript.delta (x2, x3, x4, etc)
# 6. response.audio.delta (x2, x3, x4, etc)
# 7. response.content_part.done
# 8. response.output_item.done (contains item_status: "completed/incomplete")
# 9. response.done (contains status_details for cancelled/failed/turn_detected/content_filter)
#
# Ourcode assumes a response will generate only one item with type "message"


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
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    reserved = res[3]
    header_extensions = res[4 : header_size * 4]
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
    app_id: str
    access_token: str
    bot_name: str
    system_role: str
    max_session_duration: float | None
    conn_options: APIConnectOptions
    opening: str = "你好啊，今天过得怎么样？"
    speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。"
    sample_rate: int = 24000
    num_channels: int = 1
    format: str = "pcm"

    @property
    def ws_url(self) -> str:
        return "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"

    def get_ws_headers(self) -> dict:
        headers = {
            "X-Api-App-ID": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": "volc.speech.dialog",  # 固定值
            "X-Api-App-Key": "PlgvMymc7f3tQnJ6",  # 固定值
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        return headers

    def get_start_session_reqs(self):
        start_session_req = {
            "tts": {
                "audio_config": {
                    "channel": self.num_channels,
                    "format": self.format,
                    "sample_rate": self.sample_rate,
                },
            },
            "dialog": {
                "bot_name": self.bot_name,
                "system_role": self.system_role,
                "speaking_style": self.speaking_style,
                "extra": {"strict_audit": False},
            },
        }
        return start_session_req


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    audio_transcript: str = ""


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
        bot_name: str = "小助手",
        system_role: str = "you are a helpful assistant",
        opening: str = "你好啊，今天过得怎么样？",
        speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。",
        app_id: str = None,
        access_token: str = None,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
            )
        )

        app_id = app_id or os.environ.get("VOLCENGINE_REALTIME_APP_ID")
        if app_id is None:
            raise ValueError("VOLCENGINE_REALTIME_APP_ID is required")
        access_token = access_token or os.environ.get(
            "VOLCENGINE_REALTIME_ACCESS_TOKEN"
        )
        if access_token is None:
            raise ValueError("VOLCENGINE_REALTIME_ACCESS_TOKEN is required")
        self._opts = _RealtimeOptions(
            app_id=app_id,
            access_token=access_token,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
            bot_name=bot_name,
            system_role=system_role,
            opening=opening,
            speaking_style=speaking_style,
        )
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def update_options(
        self,
        *,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        pass

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
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

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[Union[RealtimeClientEvent, dict[str, Any]]]()
        self._input_resampler: rtc.AudioResampler | None = None
        self.session_id = str(uuid.uuid4())

        self._instructions: str | None = None
        self._main_atask = asyncio.create_task(
            self._main_task(), name="RealtimeSession._main_task"
        )
        self.send_event(self._create_session_update_event())

        self._response_created_futures: dict[
            str, asyncio.Future[llm.GenerationCreatedEvent]
        ] = {}
        self._item_delete_future: dict[str, asyncio.Future] = {}
        self._item_create_future: dict[str, asyncio.Future] = {}

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # 100ms chunks
        self._bstream = utils.audio.AudioByteStream(
            self._realtime_model._opts.sample_rate,
            self._realtime_model._opts.num_channels,
            samples_per_channel=self._realtime_model._opts.sample_rate // 10,
        )
        self._pushed_duration_s: float = (
            0  # duration of audio pushed to the OpenAI Realtime API
        )

    def send_event(self, event: RealtimeClientEvent | dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        logger.info("run realtime main task")
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        async def _reconnect() -> None:
            logger.debug(
                "reconnecting to Volcengine Realtime API",
                extra={
                    "max_session_duration": self._realtime_model._opts.max_session_duration
                },
            )

            events: list[RealtimeClientEvent] = []

            # options and instructions
            events.append(self._create_session_update_event())

            # tools
            tools = list(self._tools.function_tools.values())
            if tools:
                events.append(self._create_tools_update_event(tools))

            # chat context
            chat_ctx = self.chat_ctx.copy(
                exclude_function_call=True,
                exclude_instructions=True,
                exclude_empty_message=True,
            )
            old_chat_ctx_copy = copy.deepcopy(self._remote_chat_ctx)
            self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
            events.extend(self._create_update_chat_ctx_events(chat_ctx))

            try:
                for ev in events:
                    msg = ev.model_dump(
                        by_alias=True, exclude_unset=True, exclude_defaults=False
                    )
                    self.emit("volcengine_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))
            except Exception as e:
                self._remote_chat_ctx = (
                    old_chat_ctx_copy  # restore the old chat context
                )
                raise APIConnectionError(
                    message=(
                        "Failed to send message to OpenAI Realtime API during session re-connection"
                    ),
                ) from e

            logger.debug("reconnected to OpenAI Realtime API")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        reconnecting = False
        while not self._msg_ch.closed:
            ws_conn = await self._create_ws_conn()

            try:
                if reconnecting:
                    await _reconnect()
                    num_retries = 0  # reset the retry counter
                await self._run_ws(ws_conn)

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"OpenAI Realtime API connection failed after {num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = (
                        self._realtime_model._opts.conn_options._interval_for_retry(
                            num_retries
                        )
                    )
                    logger.warning(
                        f"OpenAI Realtime API connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

            reconnecting = True

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = self._realtime_model._opts.get_ws_headers()
        url = self._realtime_model._opts.ws_url
        return await asyncio.wait_for(
            self._realtime_model._ensure_http_session().ws_connect(
                url=url, headers=headers
            ),
            self._realtime_model._opts.conn_options.timeout,
        )

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False
        start_connection_request = bytearray(generate_header())
        start_connection_request.extend(int(1).to_bytes(4, "big"))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        start_connection_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        start_connection_request.extend(payload_bytes)
        await ws_conn.send_bytes(start_connection_request)
        response = await ws_conn.receive_bytes()
        logger.info(f"StartConnection response: {parse_response(response)}")

        request_params = self._realtime_model._opts.get_start_session_reqs()
        payload_bytes = str.encode(json.dumps(request_params))
        payload_bytes = gzip.compress(payload_bytes)
        start_session_request = bytearray(generate_header())
        start_session_request.extend(int(100).to_bytes(4, "big"))
        start_session_request.extend((len(self.session_id)).to_bytes(4, "big"))
        start_session_request.extend(str.encode(self.session_id))
        start_session_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        start_session_request.extend(payload_bytes)
        await ws_conn.send_bytes(start_session_request)
        response = await ws_conn.receive_bytes()
        logger.info(f"StartSession response: {parse_response(response)}")

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                print(msg)
                try:
                    if isinstance(msg, BaseModel):
                        msg = msg.model_dump(
                            by_alias=True, exclude_unset=True, exclude_defaults=False
                        )
                    task_request = bytearray(
                        generate_header(
                            message_type=CLIENT_AUDIO_ONLY_REQUEST,
                            serial_method=NO_SERIALIZATION,
                        )
                    )
                    task_request.extend(int(200).to_bytes(4, "big"))
                    task_request.extend((len(self.session_id)).to_bytes(4, "big"))
                    task_request.extend(str.encode(self.session_id))
                    payload_bytes = gzip.compress(audio)
                    task_request.extend(
                        (len(payload_bytes)).to_bytes(4, "big")
                    )  # payload size(4 bytes)
                    task_request.extend(payload_bytes)
                    await self.ws.send(task_request)
                    self.emit("volcengine_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))

                except Exception:
                    break

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:  # closing is expected, see _send_task
                        return

                    # this will trigger a reconnection
                    raise APIConnectionError(
                        message="OpenAI S2S connection closed unexpectedly"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)

                # emit the raw json dictionary instead of the BaseModel because different
                # providers can have different event types that are not part of the OpenAI Realtime API  # noqa: E501
                self.emit("volcengine_server_event_received", event)

                try:
                    if event["type"] == "input_audio_buffer.speech_started":
                        self._handle_input_audio_buffer_speech_started(
                            InputAudioBufferSpeechStartedEvent.construct(**event)
                        )
                    elif event["type"] == "input_audio_buffer.speech_stopped":
                        self._handle_input_audio_buffer_speech_stopped(
                            InputAudioBufferSpeechStoppedEvent.construct(**event)
                        )
                    elif event["type"] == "response.created":
                        self._handle_response_created(
                            ResponseCreatedEvent.construct(**event)
                        )
                    elif event["type"] == "response.output_item.added":
                        self._handle_response_output_item_added(
                            ResponseOutputItemAddedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.created":
                        self._handle_conversion_item_created(
                            ConversationItemCreatedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.deleted":
                        self._handle_conversion_item_deleted(
                            ConversationItemDeletedEvent.construct(**event)
                        )
                    elif (
                        event["type"]
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        self._handle_conversion_item_input_audio_transcription_completed(
                            ConversationItemInputAudioTranscriptionCompletedEvent.construct(
                                **event
                            )
                        )
                    elif (
                        event["type"]
                        == "conversation.item.input_audio_transcription.failed"
                    ):
                        self._handle_conversion_item_input_audio_transcription_failed(
                            ConversationItemInputAudioTranscriptionFailedEvent.construct(
                                **event
                            )
                        )
                    elif event["type"] == "response.content_part.done":
                        self._handle_response_content_part_done(
                            ResponseContentPartDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.text.delta":
                        self._handle_response_text_delta(
                            ResponseTextDeltaEvent.construct(**event)
                        )
                    elif event["type"] == "response.text.done":
                        self._handle_response_text_done(
                            ResponseTextDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.delta":
                        self._handle_response_audio_transcript_delta(event)
                    elif event["type"] == "response.audio.delta":
                        self._handle_response_audio_delta(
                            ResponseAudioDeltaEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.done":
                        self._handle_response_audio_transcript_done(
                            ResponseAudioTranscriptDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio.done":
                        self._handle_response_audio_done(
                            ResponseAudioDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.output_item.done":
                        self._handle_response_output_item_done(
                            ResponseOutputItemDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.done":
                        self._handle_response_done(ResponseDoneEvent.construct(**event))
                    elif event["type"] == "error":
                        self._handle_error(ErrorEvent.construct(**event))
                except Exception:
                    if event["type"] == "response.audio.delta":
                        event["delta"] = event["delta"][:10] + "..."
                    logger.exception("failed to handle event", extra={"event": event})

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        wait_reconnect_task: asyncio.Task | None = None
        if self._realtime_model._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._realtime_model._opts.max_session_duration),
                name="_timeout_task",
            )
            tasks.append(wait_reconnect_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # propagate exceptions from completed tasks
            for task in done:
                if task != wait_reconnect_task:
                    task.result()

            if (
                wait_reconnect_task
                and wait_reconnect_task in done
                and self._current_generation
            ):
                # wait for the current generation to complete before reconnecting
                await self._current_generation._done_fut

        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _create_session_update_event(self) -> SessionUpdateEvent:
        input_audio_transcription_opts = (
            self._realtime_model._opts.input_audio_transcription
        )
        input_audio_transcription = (
            session_update_event.SessionInputAudioTranscription.model_validate(
                input_audio_transcription_opts.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
            )
            if input_audio_transcription_opts
            else None
        )

        turn_detection_opts = self._realtime_model._opts.turn_detection
        turn_detection = (
            session_update_event.SessionTurnDetection.model_validate(
                turn_detection_opts.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
            )
            if turn_detection_opts
            else None
        )

        tracing_opts = self._realtime_model._opts.tracing
        if isinstance(tracing_opts, TracingTracingConfiguration):
            tracing: session_update_event.SessionTracing | None = (
                session_update_event.SessionTracingTracingConfiguration.model_validate(
                    tracing_opts.model_dump(
                        by_alias=True,
                        exclude_unset=True,
                        exclude_defaults=True,
                    )
                )
            )
        else:
            tracing = tracing_opts

        kwargs: dict[str, Any] = {
            "model": self._realtime_model._opts.model,
            "voice": self._realtime_model._opts.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "modalities": self._realtime_model._opts.modalities,
            "turn_detection": turn_detection,
            "input_audio_transcription": input_audio_transcription,
            "input_audio_noise_reduction": self._realtime_model._opts.input_audio_noise_reduction,
            "temperature": self._realtime_model._opts.temperature,
            "tool_choice": _to_oai_tool_choice(self._realtime_model._opts.tool_choice),
        }
        if self._instructions is not None:
            kwargs["instructions"] = self._instructions

        if self._realtime_model._opts.speed is not None:
            kwargs["speed"] = self._realtime_model._opts.speed

        if tracing:
            kwargs["tracing"] = tracing

        # initial session update
        return SessionUpdateEvent(
            type="session.update",
            # Using model_construct since OpenAI restricts voices to those defined in the BaseModel.  # noqa: E501
            # Other providers support different voices, so we need to accommodate that.
            session=session_update_event.Session.model_construct(**kwargs),
            event_id=utils.shortuuid("session_update_"),
        )

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
        temperature: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[
            InputAudioTranscription | None
        ] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            events = self._create_update_chat_ctx_events(chat_ctx)
            futs: list[asyncio.Future[None]] = []

            for ev in events:
                futs.append(f := asyncio.Future[None]())
                if isinstance(ev, ConversationItemDeleteEvent):
                    self._item_delete_future[ev.item_id] = f
                elif isinstance(ev, ConversationItemCreateEvent):
                    assert ev.item.id is not None
                    self._item_create_future[ev.item.id] = f
                self.send_event(ev)

            if not futs:
                return
            try:
                await asyncio.wait_for(
                    asyncio.gather(*futs, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                raise llm.RealtimeError("update_chat_ctx timed out.") from None

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        diff_ops = llm.utils.compute_chat_ctx_diff(
            self._remote_chat_ctx.to_chat_ctx(), chat_ctx
        )

        def _delete_item(msg_id: str) -> None:
            events.append(
                ConversationItemDeleteEvent(
                    type="conversation.item.delete",
                    item_id=msg_id,
                    event_id=utils.shortuuid("chat_ctx_delete_"),
                )
            )

        def _create_item(previous_msg_id: str | None, msg_id: str) -> None:
            chat_item = chat_ctx.get_by_id(msg_id)
            assert chat_item is not None
            events.append(
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item=_livekit_item_to_openai_item(chat_item),
                    previous_item_id=(
                        "root" if previous_msg_id is None else previous_msg_id
                    ),
                    event_id=utils.shortuuid("chat_ctx_create_"),
                )
            )

        for msg_id in diff_ops.to_remove:
            _delete_item(msg_id)

        for previous_msg_id, msg_id in diff_ops.to_create:
            _create_item(previous_msg_id, msg_id)

        # update the items with the same id but different content
        for previous_msg_id, msg_id in diff_ops.to_update:
            _delete_item(msg_id)
            _create_item(previous_msg_id, msg_id)

        return events

    async def update_instructions(self, instructions: str) -> None:
        event_id = utils.shortuuid("instructions_update_")
        # f = asyncio.Future()
        # self._response_futures[event_id] = f
        self.send_event(
            SessionUpdateEvent(
                type="session.update",
                session=session_update_event.Session.model_construct(
                    instructions=instructions
                ),
                event_id=event_id,
            )
        )
        self._instructions = instructions

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.send_event(
                    InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(nf.data).decode("utf-8"),
                    )
                )
                self._pushed_duration_s += nf.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        if self._pushed_duration_s > 0.1:  # OpenAI requires at least 100ms of audio
            self.send_event(
                InputAudioBufferCommitEvent(type="input_audio_buffer.commit")
            )
            self._pushed_duration_s = 0

    def clear_audio(self) -> None:
        self.send_event(InputAudioBufferClearEvent(type="input_audio_buffer.clear"))
        self._pushed_duration_s = 0

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut
        self.send_event(
            ResponseCreateEvent(
                type="response.create",
                event_id=event_id,
                response=Response(
                    instructions=instructions or None,
                    metadata={"client_event_id": event_id},
                ),
            )
        )

        def _on_timeout() -> None:
            if fut and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())
        return fut

    def interrupt(self) -> None:
        self.send_event(ResponseCancelEvent(type="response.cancel"))

    def truncate(
        self,
        *,
        message_id: str,
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._realtime_model.capabilities.audio_output:
            self.send_event(
                ConversationItemTruncateEvent(
                    type="conversation.item.truncate",
                    content_index=0,
                    item_id=message_id,
                    audio_end_ms=audio_end_ms,
                )
            )
        elif utils.is_given(audio_transcript):
            # sync the forwarded text to the remote chat ctx
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
        self._msg_ch.close()
        await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        user_transcription_enabled = (
            self._realtime_model._opts.input_audio_transcription is not None
        )
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(
                user_transcription_enabled=user_transcription_enabled
            ),
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        assert event.response.id is not None, "response.id is None"

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

        if (
            isinstance(event.response.metadata, dict)
            and (client_event_id := event.response.metadata.get("client_event_id"))
            and (fut := self._response_created_futures.pop(client_event_id, None))
        ):
            generation_ev.user_initiated = True
            fut.set_result(generation_ev)

        self.emit("generation_created", generation_ev)

    def _handle_response_output_item_added(
        self, event: ResponseOutputItemAddedEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "message":
            item_generation = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
            )
            if not self._realtime_model.capabilities.audio_output:
                item_generation.audio_ch.close()

            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_generation.text_ch,
                    audio_stream=item_generation.audio_ch,
                )
            )
            self._current_generation.messages[item_id] = item_generation

    def _handle_response_content_part_done(
        self, event: ResponseContentPartDoneEvent
    ) -> None:
        if event.part.type == "text" and self._realtime_model.capabilities.audio_output:
            logger.error(
                "Text response received from OpenAI Realtime API in audio modality. "
                "This usually happens when text chat context is synced to the API. "
                "Try to use text modality with TTS instead."
            )
            self._emit_error(
                llm.RealtimeError(
                    "Text response received from OpenAI Realtime API in audio modality."
                ),
                recoverable=False,
            )

    def _handle_conversion_item_created(
        self, event: ConversationItemCreatedEvent
    ) -> None:
        assert event.item.id is not None, "item.id is None"

        try:
            self._remote_chat_ctx.insert(
                event.previous_item_id, _openai_item_to_livekit_item(event.item)
            )
        except ValueError as e:
            logger.warning(
                f"failed to insert item `{event.item.id}`: {str(e)}",
            )

        if fut := self._item_create_future.pop(event.item.id, None):
            fut.set_result(None)

    def _handle_conversion_item_deleted(
        self, event: ConversationItemDeletedEvent
    ) -> None:
        assert event.item_id is not None, "item_id is None"

        try:
            self._remote_chat_ctx.delete(event.item_id)
        except ValueError as e:
            logger.warning(
                f"failed to delete item `{event.item_id}`: {str(e)}",
            )

        if fut := self._item_delete_future.pop(event.item_id, None):
            fut.set_result(None)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            remote_item.item.content.append(event.transcript)

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.transcript,
                is_final=True,
            ),
        )

    def _handle_conversion_item_input_audio_transcription_failed(
        self, event: ConversationItemInputAudioTranscriptionFailedEvent
    ) -> None:
        logger.error(
            "OpenAI Realtime API failed to transcribe input audio",
            extra={"error": event.error},
        )

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]

        item_generation.text_ch.send_nowait(event.delta)
        item_generation.audio_transcript += event.delta

    def _handle_response_text_done(self, event: ResponseTextDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_transcript_delta(self, event: dict[str, Any]) -> None:
        assert self._current_generation is not None, "current_generation is None"

        item_id = event["item_id"]
        delta = event["delta"]

        if (start_time := event.get("start_time")) is not None:
            delta = io.TimedString(delta, start_time=start_time)

        item_generation = self._current_generation.messages[item_id]
        item_generation.text_ch.send_nowait(delta)
        item_generation.audio_transcript += delta

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]

        data = base64.b64decode(event.delta)
        item_generation.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
        )

    def _handle_response_audio_transcript_done(
        self, _: ResponseAudioTranscriptDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_output_item_done(
        self, event: ResponseOutputItemDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "function_call":
            item = event.item
            assert item.call_id is not None, "call_id is None"
            assert item.name is not None, "name is None"
            assert item.arguments is not None, "arguments is None"

            self._current_generation.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                )
            )
        elif item_type == "message":
            item_generation = self._current_generation.messages[item_id]
            item_generation.text_ch.close()
            item_generation.audio_ch.close()

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        if self._current_generation is None:
            return  # OpenAI has a race condition where we could receive response.done without any previous response.created (This happens generally during interruption)  # noqa: E501

        assert self._current_generation is not None, "current_generation is None"

        created_timestamp = self._current_generation._created_timestamp
        first_token_timestamp = self._current_generation._first_token_timestamp

        for generation in self._current_generation.messages.values():
            # close all messages that haven't been closed yet
            if not generation.text_ch.closed:
                generation.text_ch.close()
            if not generation.audio_ch.closed:
                generation.audio_ch.close()

        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        for item_id, item_generation in self._current_generation.messages.items():
            if (remote_item := self._remote_chat_ctx.get(item_id)) and isinstance(
                remote_item.item, llm.ChatMessage
            ):
                remote_item.item.content.append(item_generation.audio_transcript)

        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation._done_fut.set_result(None)
        self._current_generation = None

        # calculate metrics
        usage = (
            event.response.usage.model_dump(exclude_defaults=True)
            if event.response.usage
            else {}
        )
        ttft = (
            first_token_timestamp - created_timestamp if first_token_timestamp else -1
        )
        duration = time.time() - created_timestamp
        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id=event.response.id or "",
            ttft=ttft,
            duration=duration,
            cancelled=event.response.status == "cancelled",
            label=self._realtime_model._label,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tokens_per_second=usage.get("output_tokens", 0) / duration,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=usage.get("input_token_details", {}).get(
                    "audio_tokens", 0
                ),
                cached_tokens=usage.get("input_token_details", {}).get(
                    "cached_tokens", 0
                ),
                text_tokens=usage.get("input_token_details", {}).get("text_tokens", 0),
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    text_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("text_tokens", 0),
                    audio_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("audio_tokens", 0),
                    image_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("image_tokens", 0),
                ),
                image_tokens=0,
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=usage.get("output_token_details", {}).get("text_tokens", 0),
                audio_tokens=usage.get("output_token_details", {}).get(
                    "audio_tokens", 0
                ),
                image_tokens=0,
            ),
        )
        self.emit("metrics_collected", metrics)

    def _handle_error(self, event: ErrorEvent) -> None:
        if event.error.message.startswith("Cancellation failed"):
            return

        logger.error(
            "OpenAI Realtime API returned an error",
            extra={"error": event.error},
        )
        self._emit_error(
            APIError(
                message="OpenAI Realtime API returned an error",
                body=event.error,
                retryable=True,
            ),
            recoverable=True,
        )

        # TODO: set exception for the response future if it exists

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


def _livekit_item_to_openai_item(item: llm.ChatItem) -> ConversationItem:
    conversation_item = ConversationItem(
        id=item.id,
    )

    if item.type == "function_call":
        conversation_item.type = "function_call"
        conversation_item.call_id = item.call_id
        conversation_item.name = item.name
        conversation_item.arguments = item.arguments

    elif item.type == "function_call_output":
        conversation_item.type = "function_call_output"
        conversation_item.call_id = item.call_id
        conversation_item.output = item.output

    elif item.type == "message":
        role = "system" if item.role == "developer" else item.role
        conversation_item.type = "message"
        conversation_item.role = role

        content_list: list[ConversationItemContent] = []
        for c in item.content:
            if isinstance(c, str):
                content_list.append(
                    ConversationItemContent(
                        type=("text" if role == "assistant" else "input_text"),
                        text=c,
                    )
                )

            elif isinstance(c, llm.ImageContent):
                continue  # not supported for now
            elif isinstance(c, llm.AudioContent):
                if conversation_item.role == "user":
                    encoded_audio = base64.b64encode(
                        rtc.combine_audio_frames(c.frame).data
                    ).decode("utf-8")

                    content_list.append(
                        ConversationItemContent(
                            type="input_audio",
                            audio=encoded_audio,
                            transcript=c.transcript,
                        )
                    )

        conversation_item.content = content_list

    return conversation_item


def _openai_item_to_livekit_item(item: ConversationItem) -> llm.ChatItem:
    assert item.id is not None, "id is None"

    if item.type == "function_call":
        assert item.call_id is not None, "call_id is None"
        assert item.name is not None, "name is None"
        assert item.arguments is not None, "arguments is None"

        return llm.FunctionCall(
            id=item.id,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    if item.type == "function_call_output":
        assert item.call_id is not None, "call_id is None"
        assert item.output is not None, "output is None"

        return llm.FunctionCallOutput(
            id=item.id,
            call_id=item.call_id,
            output=item.output,
            is_error=False,
        )

    if item.type == "message":
        assert item.role is not None, "role is None"
        assert item.content is not None, "content is None"

        content: list[llm.ChatContent] = []

        for c in item.content:
            if c.type == "text" or c.type == "input_text":
                assert c.text is not None, "text is None"
                content.append(c.text)

        return llm.ChatMessage(
            id=item.id,
            role=item.role,
            content=content,
        )

    raise ValueError(f"unsupported item type: {item.type}")


def _to_oai_tool_choice(tool_choice: llm.ToolChoice | None) -> str:
    if isinstance(tool_choice, str):
        return tool_choice

    elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
        return tool_choice["function"]["name"]

    return "auto"
