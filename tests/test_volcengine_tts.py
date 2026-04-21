import base64

import pytest

from livekit.agents import APIStatusError
from livekit.plugins.volcengine.tts import (
    TTS,
    _TTSOptions,
    parse_http_stream_event,
    parse_response,
)


class _DummyStream:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_volcengine_tts_aclose_closes_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = TTS(app_id="app", access_token="token", http_session=object())
    dummy_stream = _DummyStream()

    def fake_stream(**kwargs):
        return dummy_stream

    monkeypatch.setattr("livekit.plugins.volcengine.tts.SynthesizeStream", fake_stream)

    stream = tts.stream(conn_options=object())
    assert stream is dummy_stream

    await tts.aclose()

    assert dummy_stream.closed is True


def test_volcengine_tts_parse_response_raises_on_error_message() -> None:
    error_text = "permission denied".encode("utf-8")
    payload = bytearray(8)
    payload.extend(error_text)
    packet = bytearray(b"\x11\xf0\x00\x00")
    packet.extend(payload)

    with pytest.raises(APIStatusError, match="permission denied"):
        parse_response(bytes(packet))


def test_volcengine_tts_parse_response_raises_on_unknown_message_type() -> None:
    packet = bytearray(b"\x11\xd0\x00\x00")

    with pytest.raises(APIStatusError, match="unsupported message type"):
        parse_response(bytes(packet))


def test_volcengine_tts_uses_v3_endpoint() -> None:
    opts = _TTSOptions(app_id="app", access_token="token")

    assert opts.get_http_url() == "https://openspeech.bytedance.com/api/v3/tts/unidirectional"


def test_volcengine_tts_http_header_uses_v3_auth() -> None:
    headers = _TTSOptions(
        app_id="app",
        access_token="token",
        resource_id="seed-tts-2.0",
    ).get_http_header(reqid="request-id")

    assert headers["X-Api-App-Key"] == "app"
    assert headers["X-Api-Access-Key"] == "token"
    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    assert headers["X-Api-Request-Id"] == "request-id"
    assert headers["Content-Type"] == "application/json"


def test_volcengine_tts_http_payload_uses_submit_operation() -> None:
    payload = _TTSOptions(
        app_id="app",
        access_token="token",
        voice="voice",
        sample_rate=24000,
    ).get_http_request("hello", reqid="request-id", uid="user-id")

    assert payload["user"]["uid"] == "user-id"
    assert payload["req_params"]["text"] == "hello"
    assert payload["req_params"]["speaker"] == "voice"
    assert payload["req_params"]["audio_params"]["format"] == "pcm"
    assert payload["req_params"]["audio_params"]["sample_rate"] == 24000


def test_volcengine_tts_defaults_to_v3_bigtts_voice() -> None:
    opts = _TTSOptions(app_id="app", access_token="token")

    assert opts.voice == "zh_female_xiaohe_uranus_bigtts"


def test_volcengine_tts_defaults_to_test_resource() -> None:
    headers = _TTSOptions(app_id="app", access_token="token").get_http_header()

    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"


def test_volcengine_tts_parses_v3_audio_chunk() -> None:
    audio = b"\x00\x01\x02\x03"
    line = (
        '{"code":0,"message":"Success","data":"'
        + base64.b64encode(audio).decode("ascii")
        + '"}\n'
    ).encode("utf-8")

    done, chunk = parse_http_stream_event(line)

    assert done is False
    assert chunk == audio


def test_volcengine_tts_parses_v3_done_chunk() -> None:
    done, chunk = parse_http_stream_event(b'{"code":20000000,"message":"Success"}\n')

    assert done is True
    assert chunk is None


def test_volcengine_tts_raises_on_v3_stream_error() -> None:
    with pytest.raises(APIStatusError, match="voice not found"):
        parse_http_stream_event(b'{"code":3050,"message":"voice not found"}\n')
