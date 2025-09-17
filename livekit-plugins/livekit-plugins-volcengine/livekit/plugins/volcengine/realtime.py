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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Callable

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .log import logger


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
    app_id: str
    access_token: str
    bot_name: str
    system_role: str
    max_session_duration: float | None
    conn_options: APIConnectOptions
    modalities: list[Literal["text", "audio"]]
    opening: str = "你好啊，今天过得怎么样？"
    speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。"
    speaker: str = "zh_female_vv_jupiter_bigtts"
    sample_rate: int = 24000
    num_channels: int = 1
    format: str = "pcm"
    model: Literal["O", "SC"] = "O"
    character_manifest: str = None
    end_smooth_window_ms: int = 500
    enable_volc_websearch: bool = False
    volc_websearch_type: Literal["web_summary", "web"] = "web_summary"
    volc_websearch_api_key: str = None
    volc_websearch_no_result_message: str = "抱歉，我找不到相关信息。"

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

    def get_start_session_reqs(self, dialog_id: str | None) -> dict:
        start_session_req = {
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
                    "model": self.model,
                },
            },
        }
        return start_session_req


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
            bot_name=bot_name,
            system_role=system_role,
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
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[rtc.AudioFrame]()
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
        self._current_item: _MessageGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        self._is_opening = False
        self._first_tts_response = True
        self._first_llm_response = True
        self._first_llm_sentence = True

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

    def send_event(self, event: rtc.AudioFrame) -> None:
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
        return await asyncio.wait_for(
            self._realtime_model._ensure_http_session().ws_connect(
                url=url,
                headers=headers,
            ),
            self._realtime_model._opts.conn_options.timeout,
        )

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
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
        pass

    async def update_tools(self, tools):
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass

    def _create_update_chat_ctx_events(self, chat_ctx: llm.ChatContext):
        events = []

        return events

    async def update_instructions(self, instructions: str) -> None:
        self._opts.system_role = instructions

    def push_audio(self, frame: rtc.AudioFrame) -> None:
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

    def clear_audio(self) -> None:
        self._pushed_duration_s = 0

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """仅文字输入"""
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut

        def _on_timeout() -> None:
            if fut and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())
        return fut

    def interrupt(self) -> None:
        pass

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
        self._msg_ch.close()
        await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != self._realtime_model._opts.sample_rate
            or frame.num_channels != self._realtime_model._opts.num_channels
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._realtime_model._opts.sample_rate,
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
