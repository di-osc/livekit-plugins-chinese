from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterable
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import function_tool, llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.aliyun import realtime as aliyun_realtime
from livekit.plugins.aliyun.realtime import (
    ClonedVoiceId,
    DEFAULT_MODEL,
    RealtimeModel,
    RealtimeSession,
    _build_say_instruction,
    _estimate_response_cost_cny,
    _livekit_item_to_qwen_item,
    _process_base_url,
    _qwen_item_to_livekit_item,
)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: asyncio.Queue[SimpleNamespace] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self) -> SimpleNamespace:
        return await self.incoming.get()

    async def close(self) -> None:
        return None


class _FakeHTTPSession:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    async def ws_connect(self, *, url: str, headers: dict[str, str]) -> _FakeWebSocket:
        self.url = url
        self.headers = headers
        return self.ws


async def _wait_until(predicate: object, *, attempts: int = 20) -> None:
    """Yield to background WebSocket tasks until a synchronous predicate is true."""
    assert callable(predicate)
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached by the background task")


async def _initialize_qwen_session(
    ws: _FakeWebSocket,
    session: RealtimeSession,
) -> None:
    """Complete Qwen's session.created -> session.update -> session.updated handshake."""
    await asyncio.sleep(0)
    await ws.incoming.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"type": "session.created", "session": {"id": "sess-1"}}),
        )
    )
    await _wait_until(lambda: len(ws.sent) == 1)
    await ws.incoming.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"type": "session.updated", "session": {"id": "sess-1"}}),
        )
    )
    await _wait_until(session._server_session_updated.is_set)


def _sent_event_types(ws: _FakeWebSocket) -> list[str]:
    """Return protocol event types already written to the fake WebSocket."""
    return [json.loads(event)["type"] for event in ws.sent]


def test_process_base_url_preserves_query_and_sets_model() -> None:
    assert (
        _process_base_url(
            "https://example.aliyuncs.com/api-ws/v1/realtime?workspace=test",
            "qwen-audio-3.0-realtime-flash",
        )
        == "wss://example.aliyuncs.com/api-ws/v1/realtime"
        "?workspace=test&model=qwen-audio-3.0-realtime-flash"
    )


def test_model_defaults_and_workspace_url() -> None:
    model = RealtimeModel(api_key="test", workspace_id="ws-id")

    assert model.model == DEFAULT_MODEL
    assert model.capabilities.audio_output is True
    assert model.capabilities.mutable_tools is True
    assert model.capabilities.message_truncation is False
    assert model.capabilities.per_response_tool_choice is False
    assert model.capabilities.supports_say is True
    assert (
        model._opts.base_url
        == "wss://ws-id.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
    )


def test_model_rejects_audio_only_modality() -> None:
    with pytest.raises(ValueError, match="modalities"):
        RealtimeModel(api_key="test", modalities=["audio"])


def test_text_only_model_does_not_advertise_say() -> None:
    model = RealtimeModel(api_key="test", modalities=["text"])
    assert model.capabilities.supports_say is False


@pytest.mark.parametrize(
    "voice",
    [
        "longanqian",
        "longanlingxin",
        "longanlingxi",
        "longanxiaoxin",
        "longanlufeng",
    ],
)
def test_model_accepts_every_builtin_voice(voice: str) -> None:
    model = RealtimeModel(api_key="test", voice=voice)  # type: ignore[arg-type]
    assert model._opts.voice == voice


def test_model_accepts_matching_cloned_voice() -> None:
    voice = ClonedVoiceId("qwen-audio-3.0-realtime-plus-myvoice-0123456789")
    model = RealtimeModel(api_key="test", voice=voice)
    assert model._opts.voice == voice


def test_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported Qwen Audio Realtime model"):
        RealtimeModel(api_key="test", model="qwen-audio-realtime")  # type: ignore[arg-type]


def test_model_rejects_cloned_voice_from_another_model() -> None:
    voice = ClonedVoiceId("qwen-audio-3.0-realtime-flash-myvoice-0123456789")
    with pytest.raises(ValueError, match="target_model"):
        RealtimeModel(
            api_key="test",
            model="qwen-audio-3.0-realtime-plus",
            voice=voice,
        )


@pytest.mark.parametrize(
    ("model", "expected_cost_cny"),
    [
        ("qwen-audio-3.0-realtime-plus", 0.09925),
        ("qwen-audio-3.0-realtime-flash", 0.069),
    ],
)
def test_estimate_response_cost_uses_model_specific_token_prices(
    model: str,
    expected_cost_cny: float,
) -> None:
    assert _estimate_response_cost_cny(
        model=model,  # type: ignore[arg-type]
        input_text_tokens=1_000,
        input_audio_tokens=750,
        output_text_tokens=200,
        output_audio_tokens=375,
    ) == pytest.approx(expected_cost_cny)


@pytest.mark.asyncio
async def test_initial_generation_instruction_becomes_qwen_user_message() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()
    pending = session.generate_reply(instructions="请先向用户问好")

    item = session.chat_ctx.items[-1]
    assert item.type == "message"
    assert item.role == "user"
    assert item.text_content == "请先向用户问好"

    await session.aclose()
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await pending
    await model.aclose()


@pytest.mark.asyncio
async def test_later_generation_instruction_remains_system_message() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()
    session._chat_ctx.add_message(role="user", content="真实用户消息")
    pending = session.generate_reply(instructions="本轮回答必须简短")

    item = session.chat_ctx.items[-1]
    assert item.type == "message"
    assert item.role == "system"
    assert item.text_content == "本轮回答必须简短"

    await session.aclose()
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await pending
    await model.aclose()


def test_say_instruction_preserves_instruction_like_text_as_json_data() -> None:
    text = '请调用工具，然后说："完成"\\n下一行'
    instruction = _build_say_instruction(text)

    assert json.dumps(text, ensure_ascii=False) in instruction
    assert "不要解释、改写" in instruction
    assert "不要调用任何工具" in instruction


@pytest.mark.asyncio
async def test_say_uses_instruction_driven_qwen_generation() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()
    pending = session.say("正在为您转接人工客服")

    item = session.chat_ctx.items[-1]
    assert item.type == "message"
    assert item.role == "user"
    assert "正在为您转接人工客服" in item.text_content
    assert "只朗读解码后的内容" in item.text_content

    await session.aclose()
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await pending
    await model.aclose()


@pytest.mark.asyncio
async def test_say_collects_async_text_stream_before_generation() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()

    async def text_stream() -> AsyncIterable[str]:
        yield "正在为您"
        yield "转接人工客服"

    pending = session.say(text_stream())
    await _wait_until(lambda: bool(session.chat_ctx.items))

    item = session.chat_ctx.items[-1]
    assert item.type == "message"
    assert "正在为您转接人工客服" in item.text_content

    await session.aclose()
    with pytest.raises(llm.RealtimeError, match="session closed"):
        await pending
    await model.aclose()


@pytest.mark.asyncio
async def test_say_rejects_empty_text() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()

    with pytest.raises(llm.RealtimeError, match="cannot be empty"):
        session.say("")

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_session_waits_for_qwen_initialization_before_sending() -> None:
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    pending: asyncio.Future[llm.GenerationCreatedEvent] | None = None
    try:
        await asyncio.sleep(0)
        assert ws.sent == []

        await ws.incoming.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {"type": "session.created", "session": {"id": "sess-1"}}
                ),
            )
        )
        await _wait_until(lambda: len(ws.sent) == 1)
        assert [json.loads(event)["type"] for event in ws.sent] == ["session.update"]

        await ws.incoming.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {"type": "session.updated", "session": {"id": "sess-1"}}
                ),
            )
        )
        await _wait_until(session._server_session_updated.is_set)

        pending = session.generate_reply()
        await _wait_until(lambda: len(ws.sent) == 2)
        assert [json.loads(event)["type"] for event in ws.sent] == [
            "session.update",
            "response.create",
        ]
    finally:
        await session.aclose()
        if pending is not None:
            with pytest.raises(llm.RealtimeError, match="session closed"):
                await pending
        await model.aclose()


@pytest.mark.asyncio
async def test_provider_error_logs_details_and_fails_pending_generation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    pending = session.generate_reply()

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.aliyun"):
        session._handle_server_event(
            {
                "type": "error",
                "event_id": "provider-event-1",
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "message": "invalid session configuration",
                    "param": "session.turn_detection",
                },
            },
        )

    with pytest.raises(llm.RealtimeError, match="invalid session configuration"):
        await pending

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Qwen Audio Realtime returned a provider error"
    )
    assert record.provider_error_type == "invalid_request_error"  # type: ignore[attr-defined]
    assert record.provider_error_code == "invalid_value"  # type: ignore[attr-defined]
    assert record.provider_error_param == "session.turn_detection"  # type: ignore[attr-defined]
    assert record.provider_event_id == "provider-event-1"  # type: ignore[attr-defined]

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_websocket_close_logs_code_reason_and_retry_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        conn_options=APIConnectOptions(max_retry=0),
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()

    with caplog.at_level(logging.ERROR, logger="livekit.plugins.aliyun"):
        await ws.incoming.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.CLOSE,
                data=1006,
                extra="upstream reset",
            )
        )
        await _wait_until(
            lambda: any(
                record.getMessage()
                == "Qwen Audio Realtime connection failed permanently"
                for record in caplog.records
            )
        )

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Qwen Audio Realtime connection failed permanently"
    )
    assert record.error_type == "APIConnectionError"  # type: ignore[attr-defined]
    assert record.attempt == 1  # type: ignore[attr-defined]
    assert record.max_retry == 0  # type: ignore[attr-defined]
    assert record.error_details == {  # type: ignore[attr-defined]
        "phase": "receive",
        "ws_message_type": "CLOSE",
        "close_code": 1006,
        "close_reason": "upstream reset",
    }

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_smart_turn_marks_invalid_stopped_turns_as_not_transcribed() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()
    stopped_events: list[llm.InputSpeechStoppedEvent] = []
    session.on("input_speech_stopped", stopped_events.append)

    session._handle_server_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "invalid-user-turn",
            "reason": "turn_invalid",
        }
    )
    session._handle_server_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "valid-user-turn",
        }
    )
    assert [event.user_transcription_enabled for event in stopped_events] == [
        False,
        True,
    ]

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_speech_started_closes_active_response_and_ignores_late_events() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)

    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "interrupted-response", "status": "in_progress"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.output_item.added",
            "response_id": "interrupted-response",
            "item": {"id": "interrupted-message", "type": "message"},
        }
    )
    message = await anext(generations[0].message_stream.__aiter__())

    session._handle_server_event({"type": "input_audio_buffer.speech_started"})

    assert session._current_generation is None
    assert "interrupted-response" in session._cancelled_response_ids
    with pytest.raises(StopAsyncIteration):
        await anext(message.text_stream.__aiter__())
    with pytest.raises(StopAsyncIteration):
        await anext(message.audio_stream.__aiter__())

    # Buffered events from the cancelled response must not be attached to the
    # next generation or reopen the interrupted output streams.
    session._handle_server_event(
        {
            "type": "response.audio.delta",
            "response_id": "interrupted-response",
            "item_id": "interrupted-message",
            "delta": base64.b64encode(b"\x01\x00").decode(),
        }
    )
    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "next-response", "status": "in_progress"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.done",
            "response": {
                "id": "interrupted-response",
                "status": "cancelled",
                "usage": {
                    "input_tokens_details": {
                        "text_tokens": 0,
                        "audio_tokens": 10,
                    },
                    "output_tokens_details": {
                        "text_tokens": 0,
                        "audio_tokens": 10,
                    },
                },
                "status_details": {
                    "type": "cancelled",
                    "reason": "turn_detected",
                },
            },
        }
    )
    assert session._current_generation is not None
    assert session._current_generation.response_id == "next-response"
    assert "interrupted-response" not in session._cancelled_response_ids
    assert session._estimated_cost_cny == pytest.approx(0.0019)

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_late_output_after_response_done_is_ignored() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()

    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "completed-response", "status": "in_progress"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.done",
            "response": {"id": "completed-response", "status": "completed"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.audio_transcript.delta",
            "response_id": "completed-response",
            "item_id": "late-message",
            "delta": "迟到的字幕",
        }
    )

    assert session._current_generation is None

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_stale_output_is_not_attached_to_new_response() -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()

    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "new-response", "status": "in_progress"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.audio_transcript.delta",
            "response_id": "old-response",
            "item_id": "old-message",
            "delta": "上一轮的迟到字幕",
        }
    )

    generation = session._current_generation
    assert generation is not None
    assert generation.response_id == "new-response"
    assert generation.messages == {}

    session._handle_server_event(
        {
            "type": "response.audio_transcript.delta",
            "response_id": "new-response",
            "item_id": "new-message",
            "delta": "当前轮字幕",
        }
    )
    assert set(generation.messages) == {"new-message"}

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_response_create_waits_for_valid_smart_turn_response_done() -> None:
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    await _initialize_qwen_session(ws, session)

    session._handle_server_event({"type": "input_audio_buffer.speech_started"})
    pending = session.generate_reply()
    await asyncio.sleep(0)
    assert "response.create" not in _sent_event_types(ws)

    session._handle_server_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "valid-user-turn",
        }
    )
    await asyncio.sleep(0)
    assert "response.create" not in _sent_event_types(ws)

    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "automatic-response", "status": "in_progress"},
        }
    )
    assert pending.done() is False
    session._handle_server_event(
        {
            "type": "response.done",
            "response": {"id": "automatic-response", "status": "completed"},
        }
    )
    await _wait_until(lambda: _sent_event_types(ws).count("response.create") == 1)

    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "deferred-response", "status": "in_progress"},
        }
    )
    generation = await pending
    assert generation.response_id == "deferred-response"
    assert generation.user_initiated is True

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_invalid_smart_turn_releases_queued_response_create() -> None:
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    await _initialize_qwen_session(ws, session)

    session._handle_server_event({"type": "input_audio_buffer.speech_started"})
    pending = session.generate_reply()
    await asyncio.sleep(0)
    assert "response.create" not in _sent_event_types(ws)

    session._handle_server_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "invalid-user-turn",
            "reason": "turn_invalid",
        }
    )
    await _wait_until(lambda: _sent_event_types(ws).count("response.create") == 1)
    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "response-after-invalid-turn"},
        }
    )
    assert (await pending).response_id == "response-after-invalid-turn"

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_user_speaking_response_create_error_is_requeued() -> None:
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    await _initialize_qwen_session(ws, session)

    pending = session.generate_reply()
    await _wait_until(lambda: _sent_event_types(ws).count("response.create") == 1)
    session._handle_server_event(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": "Cannot create response while user is speaking.",
                "param": "response.create",
            },
        }
    )
    await asyncio.sleep(0)
    assert pending.done() is False

    session._handle_server_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "valid-user-turn",
        }
    )
    await asyncio.sleep(0)
    assert _sent_event_types(ws).count("response.create") == 1
    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "automatic-user-response"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.done",
            "response": {
                "id": "automatic-user-response",
                "status": "completed",
            },
        }
    )
    await _wait_until(lambda: _sent_event_types(ws).count("response.create") == 2)
    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "retried-response"},
        }
    )
    assert (await pending).response_id == "retried-response"

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_response_create_timeout_starts_only_after_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aliyun_realtime, "_RESPONSE_CREATE_TIMEOUT", 0.01)
    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    await _initialize_qwen_session(ws, session)

    session._handle_server_event({"type": "input_audio_buffer.speech_started"})
    pending = session.generate_reply()
    await asyncio.sleep(0.03)
    assert pending.done() is False

    session._handle_server_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "invalid-user-turn",
            "reason": "turn_invalid",
        }
    )
    await _wait_until(lambda: _sent_event_types(ws).count("response.create") == 1)
    with pytest.raises(llm.RealtimeError, match="generate_reply timed out"):
        await pending

    await session.aclose()
    await model.aclose()


def test_chat_item_conversion_round_trip() -> None:
    message = llm.ChatMessage(
        id="item-1",
        role="user",
        content=["你好"],
    )

    qwen_item = _livekit_item_to_qwen_item(message)
    converted = _qwen_item_to_livekit_item(qwen_item)

    assert qwen_item == {
        "id": "item-1",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "你好"}],
    }
    assert converted.id == message.id
    assert converted.type == message.type
    assert converted.role == message.role
    assert converted.content == message.content

    assistant = llm.ChatMessage(
        id="item-2",
        role="assistant",
        content=["您好"],
    )
    assert _livekit_item_to_qwen_item(assistant)["content"] == [
        {"type": "output_text", "text": "您好"}
    ]


@pytest.mark.asyncio
async def test_session_maps_qwen_audio_and_function_events() -> None:
    ws = _FakeWebSocket()
    http_session = _FakeHTTPSession(ws)
    model = RealtimeModel(api_key="test", http_session=http_session)  # type: ignore[arg-type]
    session = model.session()
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)

    await asyncio.sleep(0)
    session._handle_server_event(
        {
            "type": "response.created",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    session._handle_server_event(
        {
            "type": "response.output_item.added",
            "item": {"id": "message-1", "type": "message"},
        }
    )

    message = await anext(generations[0].message_stream.__aiter__())
    session._handle_server_event(
        {
            "type": "response.audio_transcript.delta",
            "item_id": "message-1",
            "delta": "你好",
        }
    )
    pcm = b"\x01\x00\x02\x00"
    session._handle_server_event(
        {
            "type": "response.audio.delta",
            "item_id": "message-1",
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    session._handle_server_event(
        {
            "type": "response.function_call_arguments.done",
            "item_id": "call-item-1",
            "call_id": "call-1",
            "name": "weather",
            "arguments": '{"city":"杭州"}',
        }
    )

    assert await anext(message.text_stream.__aiter__()) == "你好"
    audio = await anext(message.audio_stream.__aiter__())
    assert audio.sample_rate == 24000
    assert audio.data.tobytes() == pcm
    function_call = await anext(generations[0].function_stream.__aiter__())
    assert function_call.call_id == "call-1"
    assert function_call.name == "weather"

    session._handle_server_event(
        {
            "type": "response.done",
            "response": {
                "id": "response-1",
                "status": "completed",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
            },
        }
    )
    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_response_done_logs_response_and_session_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(_FakeWebSocket()),  # type: ignore[arg-type]
    )
    session = model.session()

    with caplog.at_level(logging.INFO, logger="livekit.plugins.aliyun"):
        for response_id in ("cost-response-1", "cost-response-2"):
            session._handle_server_event(
                {
                    "type": "response.created",
                    "response": {"id": response_id, "status": "in_progress"},
                }
            )
            session._handle_server_event(
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "usage": {
                            "input_tokens": 850,
                            "output_tokens": 425,
                            "total_tokens": 1_275,
                            "input_tokens_details": {
                                "text_tokens": 100,
                                "audio_tokens": 750,
                            },
                            "output_tokens_details": {
                                "text_tokens": 50,
                                "audio_tokens": 375,
                            },
                        },
                    },
                }
            )

    records = [
        record
        for record in caplog.records
        if record.getMessage() == "Qwen Audio Realtime usage cost"
    ]
    assert len(records) == 2
    assert records[0].estimated_response_cost_cny == pytest.approx(  # type: ignore[attr-defined]
        0.08875
    )
    assert records[0].estimated_session_cost_cny == pytest.approx(  # type: ignore[attr-defined]
        0.08875
    )
    assert records[1].estimated_session_cost_cny == pytest.approx(  # type: ignore[attr-defined]
        0.1775
    )
    assert records[1].input_audio_tokens == 750  # type: ignore[attr-defined]
    assert records[1].output_audio_tokens == 375  # type: ignore[attr-defined]

    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_tools_use_qwen_function_schema() -> None:
    @function_tool
    async def weather(city: str) -> str:
        """查询天气。"""
        return city

    ws = _FakeWebSocket()
    model = RealtimeModel(
        api_key="test",
        http_session=_FakeHTTPSession(ws),  # type: ignore[arg-type]
    )
    session = model.session()
    await session.update_tools([weather])

    event = session._create_session_update_event()
    assert event["session"]["tools"][0]["function"]["name"] == "weather"

    await session.aclose()
    await model.aclose()
