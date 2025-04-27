# LiveKit Plugins Xunfei

Agent Framework plugin for services from Xunfei. Currently supports [STT](https://console.xfyun.cn/services/rta).

## Installation
```python
pip install livekit-plugins-xunfei
```

## Pre-requisites

- XunFei STT environment variable: `XUNFEI_STT_APP_ID`, `XUNFEI_STT_API_KEY`.

## Usage


This example shows how to use the xunfei plugin to create a voice agent.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import xunfei
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # app_id and api_key can be found in the xfyun console.
        stt=xunfei.STT(app_id="xxx", api_key="xxx"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

