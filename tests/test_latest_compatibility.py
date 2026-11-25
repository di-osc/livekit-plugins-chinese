import inspect

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.aliyun.tts import TTS as AliyunTTS
from livekit.plugins.volcengine.realtime import RealtimeModel as VolcengineRealtimeModel
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


def test_volcengine_realtime_accepts_agents_turn_detection_keyword() -> None:
    signature = inspect.signature(VolcengineRealtimeModel.session)

    assert "turn_detection_disabled" in signature.parameters
    assert (
        signature.parameters["turn_detection_disabled"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
