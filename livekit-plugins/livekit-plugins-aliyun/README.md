# livekit-plugins-aliyun

适配[阿里云百炼](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.6b887b08kVpe2w&tab=model#/model-market)的[livekit-agent](https://github.com/livekit/agents)框架插件。目前支持[TTS](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.6b887b08kVpe2w&tab=model#/model-market?capabilities=%5B%22TTS%22%5D&z_type_=%7B%22capabilities%22%3A%22array%22%7D), [LLM](https://bailian.console.aliyun.com/?tab=model#/model-market), [STT](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.6b887b08kVpe2w&tab=model#/model-market?capabilities=%5B%22ASR%22%5D&z_type_=%7B%22capabilities%22%3A%22array%22%7D)。

## 安装
```python
pip install livekit-plugins-aliyun
```

## 环境变量

- LLM, STT, TTS: `DASHSCOPE_API_KEY`

## 使用示例

以下代码展示了如何在`livekit-agent`中使用`livekit-plugins-aliyun`插件。

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import aliyun
from dotenv import load_dotenv


async def entry_point(ctx: JobContext):
    
    await ctx.connect()
    
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        stt=aliyun.STT(model="paraformer-realtime-v2"),
        tts=aliyun.TTS(model="cosyvoice-v2", voice="longcheng_v2"),
        llm=aliyun.LLM(model="qwen-plus"),
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

若需要使用STT热词功能，需要在`STT`插件中配置`vocabulary_id`参数。
```python
    session = AgentSession(
        stt=aliyun.STT(model="paraformer-realtime-v2", vocabulary_id="your_vocabulary_id"),
        tts=aliyun.TTS(model="cosyvoice-v2", voice="longcheng_v2"),
        llm=aliyun.LLM(model="qwen-plus"),
    )
```