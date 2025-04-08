# LiveKit Plugins Volcengine

Agent Framework plugin for services from Volcengine(火山引擎). Currently supports [TTS](https://www.volcengine.com/docs/6561/79817), [LLM](https://www.volcengine.com/docs/82379/1298454#%E6%B5%81%E5%BC%8F%E8%B0%83%E7%94%A8), [STT](https://www.volcengine.com/docs/6561/80818#python).

## Installation
```python
pip install livekit-plugins-volcengine
```

## Pre-requisites

- Volcengine TTS environment variable: `VOLCENGINE_TTS_ACCESS_TOKEN`
- Volcengine STT environment variable: `VOLCENGINE_STT_ACCESS_TOKEN`
- Volcengine LLM environment variable: `VOLCENGINE_LLM_API_KEY`

## Usage


This example shows how to use the Volcengine plugin to create a voice agent that achieved 1 second of latency.

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine, deepgram, silero
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        vad=silero.VAD.load(),
        # app_id and cluster can be found in the Volcengine STT console. https://console.volcengine.com/speech/service/16?AppID=5909198413
        stt=volcengine.STT(app_id="xxx", cluster="xxx"),
        # app_id and cluster can be found in the Volcengine TTS console, and you can find voice type at https://www.volcengine.com/docs/6561/97465
        tts=volcengine.TTS(app_id="xxx", cluster="xxx", vioce_type="BV001_V2_streaming"),
        # model can be endpoint id or model id, you can find it at https://www.volcengine.com/docs/82379/1513689
        llm=volcengine.LLM(model="doubao-1-5-lite-32k-250115"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

