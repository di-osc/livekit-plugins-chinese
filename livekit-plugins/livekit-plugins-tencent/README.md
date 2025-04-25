# LiveKit Plugins Tencent

Agent Framework plugin for services from Tencent. currently supports: [STT](https://cloud.tencent.com/document/product/1093/48982#signature).

## Installation
```python
pip install livekit-plugins-tencent
```

## Pre-requisites

- tencent STT environment variable: `TENCENT_STT_APP_ID`, `TENCENT_STT_SECRET_KEY`, `TENCENT_STT_ID`

## Usage


This example shows how to use the Tencent STT plugin.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import tencent
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # app_id can be found in the tencent cloud console. https://console.cloud.tencent.com/
        stt=tencent.STT(app_id=xxx, secret_key=xxx, secret_id=xxx),
    )
    
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

