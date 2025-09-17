# livekit-plugins-flashtts

[![PyPI version](https://badge.fury.io/py/livekit-plugins-flashtts.svg)](https://pypi.org/project/livekit-plugins-flashtts/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

[FlashTTS](https://github.com/HuiResearch/FlashTTS) 开源TTS推理框架专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，支持Spark-TTS、MegaTTS等开源模型部署。

## ✨ 特性

- 🗣️ **语音合成 (TTS)** - 支持FlashTTS框架的多种开源TTS模型
- 🚀 **高性能** - 快速的TTS推理速度
- 🔧 **灵活部署** - 支持本地和远程API部署
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| TTS | 语音合成 | [FlashTTS](https://github.com/HuiResearch/FlashTTS) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-flashtts
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-flashtts
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `FLASHTTS_API_URL` | FlashTTS API地址 | http://localhost:8000 |
| `FLASHTTS_API_KEY` | FlashTTS API密钥 | 空 |

### .env 文件示例

```bash
# .env
FLASHTTS_API_URL=http://localhost:8000
FLASHTTS_API_KEY=your_api_key_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import flashtts
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音合成
        tts=flashtts.TTS(voice="female")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import flashtts

# 自定义TTS配置
tts = flashtts.TTS(
    voice="female",              # 语音类型
    api_url="http://localhost:8000",  # API地址 (可选，从环境变量获取)
    api_key="your_api_key"       # API密钥 (可选，从环境变量获取)
)
```

## 🔧 API 参考

### TTS (语音合成)

```python
flashtts.TTS(
    voice: str = "female",          # 语音类型
    api_url: str = None,           # API地址 (从环境变量获取)
    api_key: str = None            # API密钥 (从环境变量获取)
)
```

## ❓ 常见问题

### Q: 如何部署FlashTTS服务？

A: 请参考[FlashTTS官方文档](https://github.com/HuiResearch/FlashTTS)进行部署。部署完成后，将API地址配置到环境变量中。

### Q: 支持哪些TTS模型？

A: FlashTTS支持多种开源TTS模型，包括：
- Spark-TTS
- MegaTTS
- 其他兼容的开源TTS模型

### Q: 如何自定义语音类型？

A: 语音类型参数 `voice` 取决于您部署的FlashTTS服务支持的语音模型。可用选项请参考您的FlashTTS服务配置。

### Q: 如何提高TTS性能？

A: 可以通过以下方式优化性能：
- 使用高性能GPU部署FlashTTS服务
- 选择合适的模型大小
- 优化网络连接质量
- 使用本地部署减少网络延迟

## 📝 更新日志

### v1.2.9
- 支持FlashTTS框架的多种开源TTS模型
- 支持本地和远程API部署
- 完善的API文档和使用示例

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 Apache 2.0 许可证。

## 🙏 致谢

- [LiveKit](https://github.com/livekit/agents) - 优秀的实时通信框架
- [FlashTTS](https://github.com/HuiResearch/FlashTTS) - 优秀的开源TTS框架

