# LiveKit Plugins Minimax

Agent Framework plugin for services from minimax(海螺AI). currently supports: [TTS](https://platform.minimaxi.com/document/Price?key=66701c7e1d57f38758d5818c).

## Installation
```python
pip install livekit-plugins-minimax
```

## Pre-requisites

- Minimax environment variable: `MINIMAX_API_KEY`, `MINIMAX_GROUP_ID` you can find [here](https://platform.minimaxi.com/user-center/basic-information/interface-key)

## Usage


This example shows how to use the minimax TTS plugin.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import minimax
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # you can find model and voice id at https://platform.minimaxi.com/document/T2A%20V2?key=66719005a427f0c8a5701643
        stt=minimax.STT(model="xxx", voice_id="xxx"),
    )
    
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

