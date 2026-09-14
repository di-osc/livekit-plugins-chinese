import pytest
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

from livekit.plugins.aliyun.stt import STT, SpeechStream


class _EventChannel:
    def __init__(self) -> None:
        self.events = []

    def send_nowait(self, event: object) -> None:
        self.events.append(event)


def test_stream_normalizes_missing_provider_timestamps() -> None:
    plugin = STT(api_key="test")
    stream = object.__new__(SpeechStream)
    stream._opts = plugin._opts
    stream._speaking = False
    stream._event_ch = _EventChannel()
    stream._request_id = "request-1"

    stream._process_stream_event(
        {
            "header": {"event": "result-generated"},
            "payload": {
                "output": {
                    "sentence": {
                        "sentence_end": False,
                        "begin_time": 100,
                        "end_time": None,
                        "text": "你好",
                    }
                }
            },
        }
    )

    transcript = stream._event_ch.events[-1]
    assert transcript.alternatives[0].start_time == 100.0
    assert transcript.alternatives[0].end_time == 0.0


def test_run_task_uses_pcm_at_16khz() -> None:
    params = STT(api_key="test")._opts.get_run_task_params(task_id="task-1")

    assert params["payload"]["parameters"]["format"] == "pcm"
    assert params["payload"]["parameters"]["sample_rate"] == 16000


@pytest.mark.asyncio
async def test_stream_requests_16khz_resample() -> None:
    plugin = STT(api_key="test")
    stream = SpeechStream(
        stt=plugin,
        opts=plugin._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        http_session=object(),
    )
    try:
        assert stream._needed_sr == 16000
    finally:
        await stream.aclose()
