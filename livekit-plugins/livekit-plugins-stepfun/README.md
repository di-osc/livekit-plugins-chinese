# livekit-plugins-stepfun

阶跃星辰[livekit-agent](https://github.com/livekit/agents)插件。目前支持[Realtime](https://platform.stepfun.com/docs/api-reference/realtime/chat)

## 安装
```python
pip install livekit-plugins-stepfun
```

## 环境变量

- Stepfun Realtime: `STEPFUN_REALTIME_API_KEY`

## 使用示例


```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import stepfun
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        llm=stepfun.RealtimeModel(voice="ganliannvsheng")
    )
    
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

