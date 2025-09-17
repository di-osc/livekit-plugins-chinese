# livekit-plugins-stepfun

[![PyPI version](https://badge.fury.io/py/livekit-plugins-stepfun.svg)](https://pypi.org/project/livekit-plugins-stepfun/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

阶跃星辰服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供实时语音交互解决方案。

## ✨ 特性

- ⚡ **实时语音模型** - 支持阶跃星辰实时语音交互
- 🎤 **语音合成** - 支持多种语音选择
- 🧠 **对话生成** - 实时对话处理
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| Realtime | 实时语音模型 | [阶跃星辰实时语音](https://platform.stepfun.com/docs/api-reference/realtime/chat) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-stepfun
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-stepfun
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `STEPFUN_REALTIME_API_KEY` | 阶跃星辰API密钥 | [阶跃星辰控制台](https://platform.stepfun.com/) |

### .env 文件示例

```bash
# .env
STEPFUN_REALTIME_API_KEY=your_stepfun_api_key_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import stepfun
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # 使用实时语音模型
    realtime_llm = stepfun.RealtimeModel(
        voice="ganliannvsheng"  # 语音类型
    )

    session = AgentSession(llm=realtime_llm)

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import stepfun

# 高级配置 - 自定义各种参数
realtime_llm = stepfun.RealtimeModel(
    voice="ganliannvsheng",      # 语音类型
    model="step-1o-audio",       # 模型名称
    temperature=0.8,             # 温度
    max_response_output_tokens="inf",  # 最大输出token数
    turn_detection=None,         # 轮次检测
    speed=1.0,                   # 语速
    api_key="your_api_key"       # API密钥 (可选，从环境变量获取)
)
```

## 🔧 API 参考

### RealtimeModel (实时语音模型)

```python
stepfun.RealtimeModel(
    voice: str = "ganliannvsheng",        # 语音类型
    model: str = "step-1o-audio",         # 模型名称
    temperature: float = 0.8,            # 温度 (0.0-2.0)
    max_response_output_tokens: Union[int, Literal["inf"]] = "inf",  # 最大输出token数
    turn_detection: TurnDetection = None,  # 轮次检测配置
    speed: float = None,                  # 语速
    api_key: str = None,                  # API密钥 (从环境变量获取)
    modalities: list[Literal["text", "audio"]] = ["text", "audio"],  # 输出模态
    http_session: aiohttp.ClientSession = None,  # HTTP会话
    max_session_duration: float = None,   # 最大会话持续时间
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS  # 连接选项
)
```

## ❓ 常见问题

### Q: 如何获取阶跃星辰API密钥？

A: 请访问[阶跃星辰平台](https://platform.stepfun.com/)，注册账号并获取API密钥。

### Q: 支持哪些语音类型？

A: 支持多种语音类型，包括：
- `ganliannvsheng` - 甘栗女声
- 其他阶跃星辰支持的语音类型

### Q: 如何调整对话参数？

A: 可以通过以下参数调整对话行为：
- `temperature`: 控制生成文本的随机性 (0.0-2.0)
- `max_response_output_tokens`: 限制输出长度
- `turn_detection`: 配置轮次检测行为
- `speed`: 调整语音播放速度

### Q: 支持哪些模型？

A: 主要支持 `step-1o-audio` 模型，该模型针对实时语音交互进行了优化。

## 📝 更新日志

### v1.2.9
- 支持阶跃星辰实时语音模型
- 支持多种语音类型选择
- 完善的API文档和使用示例

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 📞 联系我们

- 项目主页: [GitHub](https://github.com/your-repo/livekit-plugins-volcengine)
- 问题反馈: [Issues](https://github.com/your-repo/livekit-plugins-volcengine/issues)
- 邮箱: 790990241@qq.com

## 🙏 致谢

- [LiveKit](https://github.com/livekit/agents) - 优秀的实时通信框架
- [阶跃星辰](https://platform.stepfun.com/) - 强大的AI服务提供商
