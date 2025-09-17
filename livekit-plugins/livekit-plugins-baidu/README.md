# livekit-plugins-baidu

[![PyPI version](https://badge.fury.io/py/livekit-plugins-baidu.svg)](https://pypi.org/project/livekit-plugins-baidu/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

百度云服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供语音识别集成解决方案。

## ✨ 特性

- 🎤 **语音识别 (STT)** - 支持百度云语音识别服务
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| STT | 语音识别 | [百度云语音识别](https://cloud.baidu.com/doc/SPEECH/s/jlbxejt2i) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-baidu
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-baidu
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `BAIDU_API_KEY` | 百度云API密钥 | [百度云控制台](https://console.bce.baidu.com/) |

### .env 文件示例

```bash
# .env
BAIDU_API_KEY=your_baidu_api_key_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import baidu
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别 - app_id可在百度云控制台获取
        stt=baidu.STT(app_id=1000000)
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import baidu

# 自定义STT配置
stt = baidu.STT(
    app_id=1000000,        # 应用ID
    secret_key="your_secret_key",  # 密钥
    dev_pid=1537,          # 语言模型 (1537:普通话, 1737:英语, etc.)
    cuid="your_cuid"       # 用户唯一标识
)
```

## 🔧 API 参考

### STT (语音识别)

```python
baidu.STT(
    app_id: int,                    # 应用ID
    secret_key: str = None,         # 密钥 (可选，从环境变量获取)
    dev_pid: int = 1537,           # 语言模型 (1537:普通话)
    cuid: str = "default"          # 用户唯一标识
)
```

## ❓ 常见问题

### Q: 如何获取百度云API密钥？

A: 请访问[百度云控制台](https://console.bce.baidu.com/)，创建语音识别应用并获取API密钥和应用ID。

### Q: 支持哪些语言？

A: 支持多种语言和方言：
- `1537` - 普通话(支持简单的英文识别)
- `1737` - 英语
- `1637` - 粤语
- `1837` - 四川话

### Q: 如何提高识别准确率？

A: 可以通过以下方式提高识别准确率：
- 使用更专业的语言模型(dev_pid)
- 确保音频质量清晰
- 使用单声道音频
- 控制音频时长在60秒以内

## 📝 更新日志

### v1.2.9
- 支持百度云语音识别服务
- 支持多种语言和方言
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
- [百度云](https://cloud.baidu.com/) - 强大的AI服务提供商

