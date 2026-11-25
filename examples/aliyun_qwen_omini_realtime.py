"""阿里云 Qwen-Omni Realtime 示例，支持语音和摄像头画面。"""

import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, room_io
from livekit.plugins import aliyun


load_dotenv()


async def entrypoint(ctx: agents.JobContext) -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY")

    session = AgentSession(
        llm=aliyun.QwenOmniRealtime(
            model="qwen3.5-omni-flash-realtime",
            voice="Tina",
            # server_vad、semantic_vad；semantic_vad_ec 仅 qwen3.8。None 为手动模式。
            turn_detection=aliyun.QwenOmniTurnDetection.semantic_vad(),
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="你是一个简洁、友好的中文语音助手，可以看到用户的摄像头画面。"
        ),
        room_options=room_io.RoomOptions(video_input=True),
    )
    await session.generate_reply(instructions="请先用中文向用户问好。")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="aliyun-qwen-omni-realtime",
        )
    )
