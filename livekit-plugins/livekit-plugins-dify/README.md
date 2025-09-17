# livekit-plugins-dify

[![PyPI version](https://badge.fury.io/py/livekit-plugins-dify.svg)](https://pypi.org/project/livekit-plugins-dify/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Dify.ai服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供AI应用集成解决方案。

## ✨ 特性

- 🤖 **大语言模型 (LLM)** - 支持Dify.ai平台的应用集成
- 🔧 **灵活配置** - 支持自定义API端点和密钥
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| LLM | 大语言模型 | [Dify.ai文档](https://docs.dify.ai/)

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-dify
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-dify
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `DIFY_API_KEY` | Dify API密钥 | 必需 |
| `DIFY_API_BASE` | Dify API基础URL | https://api.dify.ai/v1 |

### .env 文件示例

```bash
# .env
DIFY_API_KEY=your_api_key_here
DIFY_API_BASE=https://api.dify.ai/v1
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import dify
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # 创建Dify LLM实例
    llm = dify.LLM(user="your_user_id")

    session = AgentSession(llm=llm)

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

    # 支持Dify开场词
    opening_words = await llm.get_opening_words()
    if opening_words:
        await session.say(opening_words)

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 完整配置示例

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import dify, volcengine
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # Dify LLM配置
    llm = dify.LLM(
        user="your_user_id",
        api_key="your_api_key",      # 可选，从环境变量获取
        api_base="your_api_base"     # 可选，从环境变量获取
    )

    session = AgentSession(
        stt=volcengine.STT(app_id="your_stt_app_id", cluster="your_cluster"),
        llm=llm,
        tts=volcengine.TTS(app_id="your_tts_app_id", cluster="your_cluster")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

## 🔧 API 参考

### LLM (大语言模型)

```python
dify.LLM(
    user: str,                     # 用户ID
    api_key: str = None,           # API密钥 (从环境变量获取)
    api_base: str = None,          # API基础URL (从环境变量获取)
    conversation_id: str = None    # 会话ID (可选)
)
```

#### 特殊方法

- `get_opening_words()` - 获取开场词
- `say(text)` - 发送语音消息

## ❓ 常见问题

### Q: 如何获取Dify API密钥？

A: 请访问[Dify.ai平台](https://dify.ai/)，创建应用并获取API密钥。

### Q: 如何配置自定义API端点？

A: 可以通过环境变量 `DIFY_API_BASE` 或参数 `api_base` 配置自定义API端点。

### Q: 如何使用开场词功能？

A: Dify插件支持自动获取和播放开场词：
```python
opening_words = await llm.get_opening_words()
if opening_words:
    await session.say(opening_words)
```

## 📝 更新日志

### v1.2.9
- 支持Dify.ai平台的应用集成
- 支持开场词功能
- 支持自定义API端点配置
- 完善的API文档和使用示例

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 Apache 2.0 许可证。

## 🙏 致谢

- [LiveKit](https://github.com/livekit/agents) - 优秀的实时通信框架
- [Dify.ai](https://dify.ai/) - 强大的AI应用平台

