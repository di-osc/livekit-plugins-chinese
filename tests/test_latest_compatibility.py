import inspect

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.aliyun.tts import TTS as AliyunTTS
from livekit.plugins.stepfun.realtime import (
    RealtimeModel as StepFunRealtimeModel,
    RealtimeSession as StepFunRealtimeSession,
)
from livekit.plugins.volcengine.llm import LLM as VolcengineLLM


def test_aliyun_tts_synthesize_accepts_livekit_connection_options() -> None:
    signature = inspect.signature(AliyunTTS.synthesize)

    assert "conn_options" in signature.parameters
    assert signature.parameters["conn_options"].kind is inspect.Parameter.KEYWORD_ONLY

    instance = AliyunTTS(api_key="test")
    with pytest.raises(NotImplementedError):
        instance.synthesize(
            "你好",
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )


def test_volcengine_llm_defaults_to_current_doubao_model() -> None:
    instance = VolcengineLLM(api_key="test")

    assert instance._opts.model == "doubao-seed-2-0-lite-260215"


def test_stepfun_realtime_uses_current_model_without_openai_transcriber() -> None:
    instance = StepFunRealtimeModel(api_key="test")

    assert instance._opts.model == "stepaudio-2.5-realtime"
    assert instance._opts.input_audio_transcription is None
    assert instance.capabilities.user_transcription is True


def test_stepfun_session_update_omits_transcriber_by_default() -> None:
    instance = StepFunRealtimeModel(api_key="test")
    session = object.__new__(StepFunRealtimeSession)
    session._realtime_model = instance
    session._instructions = None

    event = session._create_session_update_event()
    payload = event.model_dump(
        by_alias=True,
        exclude_unset=True,
        exclude_defaults=False,
    )

    assert "input_audio_transcription" not in payload["session"]
