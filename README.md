# LiveKit Plugins Volcengine

Agent Framework plugin for services from Volcengine(火山引擎). Currently supports [TTS](https://www.volcengine.com/docs/6561/79817)

## Installation
```python
pip install livekit-plugins-volcengine
```

## Pre-requisites

- Volcengine TTS environment variable: `VOLCENGINE_TTS_ACCESS_TOKEN`

## Usage

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import openai, volcengine, deepgram, silero
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(language="zh"),
        ## app_id and cluster can be found in the Volcengine TTS console, and you can find voice type at https://www.volcengine.com/docs/6561/97465
        tts=volcengine.TTS(app_id="xxx", cluster="xxx", streaming=True, vioce_type="BV001_V2_streaming"),
        llm=openai.LLM(model="gpt-4o-mini"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

