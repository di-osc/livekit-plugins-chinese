# livekit-plugins-zhipu

[![PyPI version](https://badge.fury.io/py/livekit-plugins-zhipu.svg)](https://pypi.org/project/livekit-plugins-zhipu/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

智谱AI服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供大语言模型集成解决方案。

## ✨ 特性

- 🤖 **大语言模型 (LLM)** - 支持智谱GLM系列大模型
- 🧠 **对话生成** - 支持多轮对话和上下文理解
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| LLM | 大语言模型 | [智谱GLM模型](https://bigmodel.cn/dev/api/normal-model/glm-4)

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-zhipu
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e ./livekit-plugins/livekit-plugins-zhipu
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `ZHIPU_LLM_API_KEY` | 智谱API密钥 | [智谱AI开放平台](https://bigmodel.cn/) |

### .env 文件示例

```bash
# .env
ZHIPU_LLM_API_KEY=your_api_key_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import zhipu
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 大语言模型
        llm=zhipu.LLM(model="glm-4")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 高级配置

```python
from livekit.plugins import zhipu

# 自定义LLM配置
llm = zhipu.LLM(
    model="glm-4-plus",     # 模型名称 (glm-4, glm-4-plus, glm-3-turbo等)
    temperature=0.7,        # 温度 (0.0-1.0)
    max_tokens=2000,        # 最大token数
    top_p=0.9,             # 核采样参数
    api_key="your_api_key"  # API密钥 (可选，从环境变量获取)
)
```

## 🔧 API 参考

### LLM (大语言模型)

```python
zhipu.LLM(
    model: str = "glm-4",           # 模型名称
    temperature: float = 0.7,       # 温度
    max_tokens: int = 2000,         # 最大token数
    top_p: float = 0.9,             # 核采样参数
    api_key: str = None             # API密钥 (从环境变量获取)
)
```

## ❓ 常见问题

### Q: 如何获取智谱API密钥？

A: 请访问[智谱AI开放平台](https://bigmodel.cn/)，注册账号并获取API密钥。

### Q: 支持哪些GLM模型？

A: 支持智谱GLM系列模型：
- `glm-4` - GLM-4 基础版
- `glm-4-plus` - GLM-4 增强版
- `glm-3-turbo` - GLM-3 Turbo版
- 其他GLM系列模型

### Q: 如何调整模型参数？

A: 可以通过以下参数调整生成效果：
- `temperature`: 控制生成文本的随机性 (0.0-1.0)
- `max_tokens`: 限制生成文本的最大长度
- `top_p`: 控制生成文本的多样性

## 📝 更新日志

### v1.2.9
- 支持智谱GLM系列大语言模型
- 支持多种模型参数配置
- 完善的API文档和使用示例

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 Apache 2.0 许可证。

## 🙏 致谢

- [LiveKit](https://github.com/livekit/agents) - 优秀的实时通信框架
- [智谱AI](https://bigmodel.cn/) - 强大的AI服务提供商

