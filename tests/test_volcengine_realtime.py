import pytest
from livekit.agents import llm

from livekit.plugins.volcengine.realtime import (
    RealtimeModel,
    _RealtimeOptions,
    _completed_transcript,
    _estimate_realtime_cost_cny,
    _parse_realtime_usage,
    RealtimeSession,
)


def test_volcengine_realtime_uses_duplex_v3_endpoint() -> None:
    model = RealtimeModel(api_key="api-key")

    assert model._opts.ws_url == (
        "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
    )
    assert model._opts.get_ws_headers() == {"X-Api-Key": "api-key"}


def test_volcengine_realtime_legacy_auth_remains_supported() -> None:
    opts = _RealtimeOptions(
        app_id="app",
        access_token="token",
        api_key=None,
        bot_name="bot",
        system_role="system",
        max_session_duration=None,
        conn_options=object(),  # type: ignore[arg-type]
        modalities=["audio", "text"],
    )

    headers = opts.get_ws_headers()
    assert headers["X-Api-App-Id"] == "app"
    assert headers["X-Api-Access-Key"] == "token"
    assert headers["X-Api-Resource-Id"] == "volc.speech.dialog"
    assert headers["X-Api-App-Key"] == "PlgvMymc7f3tQnJ6"
    assert headers["X-Api-Request-Id"] == headers["X-Api-Connect-Id"]


def test_volcengine_realtime_session_config_is_v3_duplex() -> None:
    model = RealtimeModel(api_key="api-key", system_role="be concise")
    opts = model._opts

    assert opts.format == "pcm_s16le"
    assert opts.get_start_session_reqs("dialog")["dialog"]["dialog_id"] == "dialog"
    assert opts.get_start_session_reqs("dialog")["tts"]["audio_config"] == {
        "channel": 1,
        "format": "pcm_s16le",
        "sample_rate": 24000,
    }


def test_volcengine_realtime_parses_usage_and_estimates_cost() -> None:
    usage = _parse_realtime_usage(
        {
            "input_tokens": 300,
            "output_tokens": 150,
            "total_tokens": 450,
            "input_tokens_details": {
                "text_tokens": 100,
                "audio_tokens": 200,
                "cached_tokens_details": {
                    "text_tokens": 20,
                    "audio_tokens": 50,
                },
            },
            "output_tokens_details": {
                "text_tokens": 50,
                "audio_tokens": 100,
            },
        }
    )

    assert usage.input_tokens == 300
    assert usage.output_tokens == 150
    assert usage.total_tokens == 450
    assert usage.cached_input_text_tokens == 20
    assert usage.cached_input_audio_tokens == 50
    assert _estimate_realtime_cost_cny(usage) == pytest.approx(0.04715)


def test_volcengine_realtime_supports_singular_usage_detail_aliases() -> None:
    usage = _parse_realtime_usage(
        {
            "input_token_details": {
                "text_tokens": 10,
                "audio_tokens": 20,
                "cached_text_tokens": 2,
                "cached_audio_tokens": 3,
            },
            "output_token_details": {"text_tokens": 4, "audio_tokens": 5},
        }
    )

    assert usage.input_tokens == 30
    assert usage.output_tokens == 9
    assert usage.total_tokens == 39


@pytest.mark.asyncio
async def test_volcengine_realtime_text_input_is_a_user_message() -> None:
    session = object.__new__(RealtimeSession)
    session._tools = llm.ToolContext.empty()
    session._chat_ctx = llm.ChatContext.empty()
    session._sent_user_item_ids = set()
    events: list[dict] = []
    session._send_json_event = events.append
    session._ensure_generation = lambda: object()

    session.generate_reply(instructions="今天天气怎么样")

    assert events[0]["type"] == "conversation.item.create"
    item = events[0]["items"][0]
    assert item["role"] == "user"
    assert item["content"] == [{"type": "input_text", "text": "今天天气怎么样"}]


@pytest.mark.parametrize(
    ("event", "accumulated", "expected"),
    [
        ({"transcript": "你好"}, "", "你好"),
        ({"text": "火山引擎"}, "", "火山引擎"),
        ({}, "增量识别结果", "增量识别结果"),
    ],
)
def test_volcengine_realtime_extracts_completed_transcript(
    event: dict, accumulated: str, expected: str
) -> None:
    assert _completed_transcript(event, accumulated) == expected
