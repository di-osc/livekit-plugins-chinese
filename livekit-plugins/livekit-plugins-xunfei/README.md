# livekit-plugins-xunfei

[![PyPI version](https://badge.fury.io/py/livekit-plugins-xunfei.svg)](https://pypi.org/project/livekit-plugins-xunfei/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

科大讯飞服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供语音识别集成解决方案。

## ✨ 特性

- 🎤 **语音识别 (STT)** - 支持科大讯飞实时语音识别服务
- ⚡ **实时处理** - 支持实时语音转文字
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| STT | 实时语音识别 | [讯飞实时语音转写](https://console.xfyun.cn/services/rta) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-xunfei
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-xunfei
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `XUNFEI_STT_APP_ID` | 讯飞应用ID | [讯飞开放平台](https://console.xfyun.cn/) |
| `XUNFEI_STT_API_KEY` | 讯飞API密钥 | [讯飞开放平台](https://console.xfyun.cn/) |

### .env 文件示例

```bash
# .env
XUNFEI_STT_APP_ID=your_app_id
XUNFEI_STT_API_KEY=your_api_key
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import xunfei
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别 - 参数可从讯飞开放平台获取
        stt=xunfei.STT(
            app_id="your_app_id",
            api_key="your_api_key"
        )
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import xunfei

# 自定义STT配置
stt = xunfei.STT(
    app_id="your_app_id",        # 应用ID
    api_key="your_api_key",      # API密钥
    api_secret="your_api_secret", # API密钥 (可选)
    domain="iat",                # 领域 (iat: 实时语音识别)
    language="zh_cn",            # 语言 (zh_cn: 中文, en_us: 英文)
    accent="mandarin",           # 方言 (mandarin: 普通话)
    sample_rate=16000,           # 采样率
    format="wav"                 # 音频格式
)
```

## 🔧 API 参考

### STT (语音识别)

```python
xunfei.STT(
    app_id: str,                    # 应用ID
    api_key: str,                   # API密钥
    api_secret: str = None,         # API密钥 (可选)
    domain: str = "iat",            # 领域
    language: str = "zh_cn",        # 语言
    accent: str = "mandarin",       # 方言
    sample_rate: int = 16000,       # 采样率
    format: str = "wav"             # 音频格式
)
```

## ❓ 常见问题

### Q: 如何获取讯飞的认证信息？

A: 请访问[讯飞开放平台](https://console.xfyun.cn/)，创建语音识别应用并获取：
- App ID: 应用ID
- API Key: API密钥
- API Secret: API密钥 (可选)

### Q: 支持哪些语言和方言？

A: 支持多种语言和方言：
- **中文**: `zh_cn` (普通话、四川话、粤语等)
- **英文**: `en_us`
- **其他语言**: 根据讯飞平台支持情况

### Q: 如何优化识别效果？

A: 可以通过以下方式优化识别效果：
- 选择合适的语言和方言设置
- 确保音频采样率匹配 (推荐16000Hz)
- 使用高质量的音频输入
- 根据应用场景选择合适的领域参数

## 📝 更新日志

### v1.2.9
- 支持科大讯飞实时语音识别服务
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
- [科大讯飞](https://www.xfyun.cn/) - 强大的AI服务提供商

