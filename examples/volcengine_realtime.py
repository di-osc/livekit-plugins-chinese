"""火山引擎豆包实时语音模型 3.0（Seeduplex）示例。"""

import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import volcengine


load_dotenv()


async def entrypoint(ctx: agents.JobContext) -> None:
    api_key = os.getenv("VOLCENGINE_REALTIME_API_KEY")
    app_id = os.getenv("VOLCENGINE_REALTIME_APP_ID")
    access_token = os.getenv("VOLCENGINE_REALTIME_ACCESS_TOKEN")
    if not api_key and not (app_id and access_token):
        raise RuntimeError(
            "请设置 VOLCENGINE_REALTIME_API_KEY，或同时设置 "
            "VOLCENGINE_REALTIME_APP_ID 和 VOLCENGINE_REALTIME_ACCESS_TOKEN"
        )

    session = AgentSession(
        llm=volcengine.RealtimeModel(
            api_key=api_key,
            app_id=app_id,
            access_token=access_token,
            model="O",
            speaker="zh_female_vv_jupiter_bigtts",
            system_role="你是一个简洁、友好的中文语音助手。",
            opening="你好，请问有什么可以帮你？",
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="你是一个简洁、友好的中文语音助手。"),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="volcengine-realtime"))
