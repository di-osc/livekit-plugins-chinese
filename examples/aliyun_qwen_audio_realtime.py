"""阿里云 Qwen Audio Realtime 示例。"""

import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import aliyun


load_dotenv()


async def entrypoint(ctx: agents.JobContext) -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY")

    session = AgentSession(
        llm=aliyun.QwenAudioRealtime(
            model="qwen-audio-3.1-realtime-plus",
            voice="longanqian_v3.1",
            # server_vad、smart_turn、smart_turn_v2；None 为手动模式。
            turn_detection=aliyun.QwenAudioTurnDetection.smart_turn_v2(),
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="你是一个简洁、友好的中文语音助手。"),
    )
    await session.generate_reply(instructions="请先用中文向用户问好。")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="aliyun-qwen-audio-realtime",
        )
    )
