# livekit-plugins-tencent

[![PyPI version](https://badge.fury.io/py/livekit-plugins-tencent.svg)](https://pypi.org/project/livekit-plugins-tencent/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

腾讯云服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供语音识别集成解决方案。

## ✨ 特性

- 🎤 **语音识别 (STT)** - 支持腾讯云语音识别服务
- 🔒 **安全认证** - 支持腾讯云标准的密钥认证方式
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| STT | 语音识别 | [腾讯云语音识别](https://cloud.tencent.com/document/product/1093/48982) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-tencent
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-tencent
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `TENCENT_STT_APP_ID` | 腾讯云应用ID | [腾讯云控制台](https://console.cloud.tencent.com/) |
| `TENCENT_STT_SECRET_KEY` | 腾讯云密钥 | [腾讯云控制台](https://console.cloud.tencent.com/) |
| `TENCENT_STT_ID` | 腾讯云Secret ID | [腾讯云控制台](https://console.cloud.tencent.com/) |

### .env 文件示例

```bash
# .env
TENCENT_STT_APP_ID=your_app_id
TENCENT_STT_SECRET_KEY=your_secret_key
TENCENT_STT_ID=your_secret_id
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import tencent
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别 - 参数可从腾讯云控制台获取
        stt=tencent.STT(
            app_id="your_app_id",
            secret_key="your_secret_key",
            secret_id="your_secret_id"
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
from livekit.plugins import tencent

# 自定义STT配置
stt = tencent.STT(
    app_id="your_app_id",        # 应用ID
    secret_key="your_secret_key", # 密钥
    secret_id="your_secret_id",   # Secret ID
    region="ap-beijing",         # 地域 (默认: ap-beijing)
    engine_model_type="16k_zh",  # 引擎模型类型
    voice_format="wav",          # 音频格式
    filter_dirty=1,              # 是否过滤脏话 (0: 不过滤, 1: 过滤)
    filter_modal=1,              # 是否过滤语气词 (0: 不过滤, 1: 过滤)
    convert_num_mode=1           # 数字转换模式 (0: 不转换, 1: 转换为阿拉伯数字)
)
```

## 🔧 API 参考

### STT (语音识别)

```python
tencent.STT(
    app_id: str,                    # 应用ID
    secret_key: str,                # 密钥
    secret_id: str,                 # Secret ID
    region: str = "ap-beijing",     # 地域
    engine_model_type: str = "16k_zh",  # 引擎模型类型
    voice_format: str = "wav",      # 音频格式
    filter_dirty: int = 1,          # 是否过滤脏话
    filter_modal: int = 1,          # 是否过滤语气词
    convert_num_mode: int = 1       # 数字转换模式
)
```

## ❓ 常见问题

### Q: 如何获取腾讯云的认证信息？

A: 请访问[腾讯云控制台](https://console.cloud.tencent.com/)，创建语音识别应用并获取以下信息：
- App ID: 应用ID
- Secret Key: 密钥
- Secret ID: Secret ID

### Q: 支持哪些音频格式？

A: 支持多种音频格式，包括：
- `wav` - WAV格式
- `mp3` - MP3格式
- `m4a` - M4A格式
- 其他腾讯云支持的音频格式

### Q: 如何配置语音识别参数？

A: 可以通过以下参数优化识别效果：
- `engine_model_type`: 选择合适的引擎模型 (16k_zh, 8k_zh等)
- `filter_dirty`: 过滤敏感词汇
- `filter_modal`: 过滤语气词
- `convert_num_mode`: 数字转换设置

### Q: 支持哪些地域？

A: 支持腾讯云的各个地域，包括：
- `ap-beijing` - 北京
- `ap-shanghai` - 上海
- `ap-guangzhou` - 广州
- 其他腾讯云支持的地域

## 📝 更新日志

### v1.2.9
- 支持腾讯云语音识别服务
- 支持多种音频格式和地域
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
- [腾讯云](https://cloud.tencent.com/) - 强大的AI服务提供商

