# 阿里云插件用法示例

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
        stt=aliyun.STT(model="paraformer-realtime-v2", language="zh"),
        llm=aliyun.LLM(model="qwen3.6-flash"),
        tts=aliyun.TTS(model="cosyvoice-v3-flash", voice="longanyang"),
        turn_detection="stt"
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="你是一个简洁、友好的中文语音助手。"),
    )
    await session.generate_reply(instructions="请先用中文向用户问好。")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
