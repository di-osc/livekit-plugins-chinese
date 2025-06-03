# livekit-plugins-volcengine

适配火山引擎服务的[livekit-agent](https://github.com/livekit/agents)框架插件。目前支持[TTS](https://www.volcengine.com/docs/6561/79817), [LLM](https://www.volcengine.com/docs/82379/1298454#%E6%B5%81%E5%BC%8F%E8%B0%83%E7%94%A8), [STT](https://www.volcengine.com/docs/6561/80818#python)，[BigModelSTT](https://www.volcengine.com/docs/6561/1354869)。

## 安装
```python
pip install livekit-plugins-volcengine
```

## 环境变量

- Volcengine TTS: `VOLCENGINE_TTS_ACCESS_TOKEN`
- Volcengine STT: `VOLCENGINE_STT_ACCESS_TOKEN`
- Volcengine LLM: `VOLCENGINE_LLM_API_KEY`

## 使用示例

以下代码展示了如何在`livekit-agent`中使用`livekit-plugins-volcengine`插件。

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine, deepgram, silero
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别：https://console.volcengine.com/speech/service/16
        stt=volcengine.STT(app_id="xxx", cluster="xxx"),
        # 大模型语音识别：https://console.volcengine.com/speech/service/10011
        stt = volcengine.BigModelSTT(app_id="xxx"),
        # 语音识别：https://www.volcengine.com/docs/6561/97465
        tts=volcengine.TTS(app_id="xxx", cluster="xxx", vioce_type="BV001_V2_streaming"),
        # 大语言模型：https://www.volcengine.com/docs/82379/1513689
        llm=volcengine.LLM(model="doubao-1-5-lite-32k-250115"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

