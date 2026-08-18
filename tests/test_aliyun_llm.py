from livekit.agents.llm.chat_context import (
    AgentConfigUpdate,
    AgentHandoff,
    ChatContext,
    ChatMessage,
)
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
