# LiveKit Plugins Baidu

Agent Framework plugin for services from Baidu. currently supports: [STT](https://cloud.baidu.com/doc/SPEECH/s/jlbxejt2i).

## Installation
```python
pip install livekit-plugins-baidu
```

## Pre-requisites

- Volcengine STT environment variable: `BAIDU_API_KEY`

## Usage


This example shows how to use the Baidu STT plugin.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import baidu
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # app_id can be found in the baidu cloud console. https://console.bce.baidu.com/ai-engine/old/#/ai/speech/app/detail~appId=6752989
        stt=volcengine.STT(app_id=1000000),
    )
    
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

