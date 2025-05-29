# LiveKit Plugins Dify

Agent Framework plugin for Dify.

## Installation
```python
pip install livekit-plugins-difyai
```

## Pre-requisites

- Dify API Key environment variables: `DIFY_API_KEY`
- Dify API Base URL environment variables: `DIFY_API_BASE`. defaults to **https://api.dify.ai/v1**.

## Usage


This example shows how to use the Dify plugin.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import dify
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")
    llm = dify.LLM(user="xxx")
    session = AgentSession(
        stt = xxx,
        llm = llm
        tts = xxx,
    )
    
    await session.start(agent=agent, room=ctx.room)
    ## 支持dify开场词
    opening_words = await llm.get_opening_words()
    if opening_words:
        await session.say(opening_words)


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

