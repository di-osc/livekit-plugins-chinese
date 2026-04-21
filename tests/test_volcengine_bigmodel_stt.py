import pytest

from livekit.plugins import volcengine
from livekit.plugins.volcengine.stt import STTOptions


def test_bigmodel_stt_ws_header_includes_connect_id() -> None:
    headers = STTOptions(app_id="app", access_token="token").get_ws_header(
        reqid="request-id"
    )

    assert headers["X-Api-Connect-Id"] == "request-id"


def test_bigmodel_stt_ws_header_accepts_custom_resource_id() -> None:
    headers = STTOptions(
        app_id="app",
        access_token="token",
        resource_id="volc.seedasr.sauc.duration",
    ).get_ws_header(reqid="request-id")

    assert headers["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"


def test_volcengine_exports_stt_but_not_bigmodel_stt() -> None:
    assert hasattr(volcengine, "STT")
    assert not hasattr(volcengine, "BigModelSTT")

    with pytest.raises(ImportError):
        exec("from livekit.plugins.volcengine.bigmodel_stt import BigModelSTT")
