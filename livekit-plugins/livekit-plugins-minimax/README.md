# livekit-plugins-minimax

[![PyPI version](https://badge.fury.io/py/livekit-plugins-minimax.svg)](https://pypi.org/project/livekit-plugins-minimax/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

MiniMax (海螺AI) 服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供完整的语音和语言模型集成解决方案。

## ✨ 特性

- 🗣️ **语音合成 (TTS)** - 支持MiniMax语音合成服务
- 🤖 **大语言模型 (LLM)** - 支持MiniMax对话模型
- 🎵 **高品质语音** - 支持多种音色和情感表达
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| TTS | 语音合成 | [MiniMax TTS](https://platform.minimaxi.com/document/Price?key=66701c7e1d57f38758d5818c) |
| LLM | 大语言模型 | [MiniMax对话](https://platform.minimaxi.com/document/%E5%AF%B9%E8%AF%9D?key=66701d281d57f38758d581d0) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-minimax
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-minimax
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `MINIMAX_API_KEY` | MiniMax API密钥 | [MiniMax控制台](https://platform.minimaxi.com/user-center/basic-information/interface-key) |
| `MINIMAX_GROUP_ID` | MiniMax Group ID | [MiniMax控制台](https://platform.minimaxi.com/user-center/basic-information/interface-key) |

### .env 文件示例

```bash
# .env
MINIMAX_API_KEY=your_api_key_here
MINIMAX_GROUP_ID=your_group_id_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import minimax
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别 - 参数可在MiniMax控制台获取
        stt=minimax.STT(model="speech-01-turbo", voice_id="female-tianmei"),
        # 大语言模型
        llm=minimax.LLM(model="MiniMax-Text-01")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import minimax

# 自定义TTS配置
tts = minimax.TTS(
    model="speech-01-turbo",     # 模型名称
    voice_id="female-tianmei",   # 语音ID
    speed=1.0,                   # 语速 (0.5-2.0)
    vol=1.0,                     # 音量 (0.0-2.0)
    pitch=0,                     # 音调 (-12到12)
    emotion="neutral"            # 情感 (neutral, happy, sad, angry)
)

# 自定义LLM配置
llm = minimax.LLM(
    model="MiniMax-Text-01",     # 模型名称
    temperature=0.7,             # 温度
    max_tokens=2000,             # 最大token数
    system_prompt="你是一个有帮助的AI助手"  # 系统提示词
)
```

## 🔧 API 参考

### TTS (语音合成)

```python
minimax.TTS(
    model: str = "speech-01-turbo",     # 模型名称
    voice_id: str = "female-tianmei",   # 语音ID
    speed: float = 1.0,                # 语速 (0.5-2.0)
    vol: float = 1.0,                  # 音量 (0.0-2.0)
    pitch: int = 0,                    # 音调 (-12到12)
    emotion: str = "neutral"           # 情感
)
```

### LLM (大语言模型)

```python
minimax.LLM(
    model: str = "MiniMax-Text-01",     # 模型名称
    temperature: float = 0.7,          # 温度
    max_tokens: int = 2000,            # 最大token数
    system_prompt: str = None          # 系统提示词
)
```

## ❓ 常见问题

### Q: 如何获取MiniMax的认证信息？

A: 请访问[MiniMax控制台](https://platform.minimaxi.com/user-center/basic-information/interface-key)获取API密钥和Group ID。

### Q: 支持哪些语音模型？

A: MiniMax支持多种语音模型，包括：
- `speech-01-turbo` - 高品质语音合成
- `speech-01` - 标准语音合成
- 其他MiniMax支持的语音模型

### Q: 如何选择合适的语音ID？

A: 语音ID可在[MiniMax文档](https://platform.minimaxi.com/document/T2A%20V2?key=66719005a427f0c8a5701643)中查看，包括：
- `female-tianmei` - 女声天美
- `male-qn` - 男声青年
- 其他多种音色选择

### Q: 支持哪些对话模型？

A: MiniMax支持多种对话模型：
- `MiniMax-Text-01` - 通用对话模型
- 其他MiniMax系列对话模型

### Q: 如何调整语音参数？

A: 可以通过以下参数调整语音效果：
- `speed`: 控制语速，范围0.5-2.0
- `vol`: 控制音量，范围0.0-2.0
- `pitch`: 控制音调，范围-12到12
- `emotion`: 设置情感，如neutral、happy、sad、angry

## 📝 更新日志

### v1.2.9
- 支持MiniMax语音合成和对话模型
- 支持多种音色和情感表达
- 完善的API文档和使用示例

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 Apache 2.0 许可证。

## 🙏 致谢

- [LiveKit](https://github.com/livekit/agents) - 优秀的实时通信框架
- [MiniMax](https://platform.minimaxi.com/) - 强大的AI服务提供商

