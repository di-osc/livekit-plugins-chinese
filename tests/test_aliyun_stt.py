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
