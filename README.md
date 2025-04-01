# LiveKit Plugins Volcengine

Agent Framework plugin for services from Volcengine(火山引擎). Currently supports [TTS](https://www.volcengine.com/docs/6561/79817)

## Installation
```python
pip install livekit-plugins-volcengine
```

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
        ## app_id and cluster can be found in the Volcengine TTS console
        tts=volcengine.TTS(app_id="xxx", cluster="xxx", streaming=True),
        llm=openai.LLM(model="gpt-4o-mini"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply(instructions="向用户问好", allow_interruptions=False)
```

