# LiveKit Plugins Zhipu(智谱)

Agent Framework plugin for services from Zhipu. Currently supports [LLM](https://bigmodel.cn/dev/api/normal-model/glm-4)

## Installation
```python
pip install livekit-plugins-zhipu
```

## Pre-requisites

- Zhipu LLM environment variable: `ZHIPU_LLM_API_KEY`

## Usage


This example shows how to use the Zhipu plugin to create a voice agent.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import zhipu
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        llm=zhipu.LLM(model="glm-4-flashx"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

