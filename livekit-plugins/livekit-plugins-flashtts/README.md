# 简介

[FlashTTS](https://github.com/HuiResearch/FlashTTS)是一个开源的TTS推理框架，你可以使用它部署Spark-TTS，MegaTTS等开源模型。

## 安装
```python
pip install livekit-plugins-flashtts
```

## 环境变量

- `FLASHTTS_API_URL`，FlashTTS的API地址，默认值为`http://localhost:8000`。
- `FLASHTTS_API_KEY`，FlashTTS的API密钥，默认值为空。

## 使用


以下是一个使用FlashTTS插件的示例：

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import flashtts
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        tts=flashtts.TTS(voice="female"),
    )
    
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

