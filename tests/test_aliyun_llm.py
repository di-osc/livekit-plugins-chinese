import sys

from livekit.agents.llm.chat_context import (
    AgentConfigUpdate,
    AgentHandoff,
    ChatContext,
    ChatMessage,
)
from openai.resources.chat import AsyncChat

from livekit.plugins.aliyun.llm import LLM
from livekit.plugins.aliyun.utils import to_chat_ctx


def test_to_chat_ctx_ignores_livekit_internal_items() -> None:
    chat_ctx = ChatContext(
        [
            ChatMessage(role="system", content=["You are helpful."]),
            AgentConfigUpdate(instructions="updated instructions"),
            AgentHandoff(new_agent_id="agent-2"),
            ChatMessage(role="user", content=["你好"]),
        ]
    )

    assert to_chat_ctx(chat_ctx, 1) == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "你好"},
    ]


def test_llm_materializes_openai_chat_client_on_init() -> None:
    plugin = LLM(api_key="test")

    assert isinstance(plugin._client.chat, AsyncChat)
    assert "openai.resources.chat" in sys.modules
    assert "openai.types.beta.realtime.input_audio_buffer_cleared_event" in sys.modules
