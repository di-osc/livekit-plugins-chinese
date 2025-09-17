# livekit-plugins-volcengine

[![PyPI version](https://badge.fury.io/py/livekit-plugins-volcengine.svg)](https://pypi.org/project/livekit-plugins-volcengine/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

火山引擎服务专用的 [LiveKit Agents](https://github.com/livekit/agents) 插件，提供完整的语音和语言模型集成解决方案。

## ✨ 特性

- 🎤 **语音识别 (STT)** - 支持火山引擎语音识别服务
- 🗣️ **语音合成 (TTS)** - 支持火山引擎文本转语音服务
- 🤖 **大语言模型 (LLM)** - 支持豆包大模型系列
- 🎯 **大模型语音识别 (BigModelSTT)** - 增强版语音识别服务
- ⚡ **实时语音模型 (Realtime)** - 端到端实时语音交互
- 🔧 **简单集成** - 与 LiveKit Agents 框架无缝集成
- 📦 **开箱即用** - 完整的 Python 包支持

## 📋 支持的服务

| 服务 | 描述 | 文档链接 |
|------|------|----------|
| TTS | 文本转语音 | [语音合成](https://www.volcengine.com/docs/6561/79817) |
| STT | 语音识别 | [语音识别](https://www.volcengine.com/docs/6561/80818) |
| BigModelSTT | 大模型语音识别 | [大模型语音识别](https://www.volcengine.com/docs/6561/1354869) |
| LLM | 大语言模型 | [流式调用](https://www.volcengine.com/docs/82379/1298454) |
| Realtime | 实时语音模型 | [实时语音](https://www.volcengine.com/docs/6561/1594356) |

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install livekit-plugins-volcengine
```

### 从源码安装

```bash
git clone https://github.com/your-repo/livekit-plugins-volcengine.git
cd livekit-plugins-volcengine
pip install -e .
```

### 系统要求

- Python >= 3.9
- LiveKit Agents >= 1.2.9

## ⚙️ 配置

### 环境变量

在使用插件前，请配置以下环境变量：

| 环境变量 | 描述 | 获取方式 |
|----------|------|----------|
| `VOLCENGINE_TTS_ACCESS_TOKEN` | TTS 服务的访问令牌 | [语音合成控制台](https://console.volcengine.com/speech/service/16) |
| `VOLCENGINE_STT_ACCESS_TOKEN` | STT 服务的访问令牌 | [语音识别控制台](https://console.volcengine.com/speech/service/16) |
| `VOLCENGINE_LLM_API_KEY` | LLM 服务的 API 密钥 | [大模型控制台](https://console.volcengine.com/ark/) |
| `VOLCENGINE_REALTIME_ACCESS_TOKEN` | 实时服务的访问令牌 | [实时语音控制台](https://console.volcengine.com/speech/service/10011) |

### .env 文件示例

```bash
# .env
VOLCENGINE_TTS_ACCESS_TOKEN=your_tts_token_here
VOLCENGINE_STT_ACCESS_TOKEN=your_stt_token_here
VOLCENGINE_LLM_API_KEY=your_llm_api_key_here
VOLCENGINE_REALTIME_ACCESS_TOKEN=your_realtime_token_here
```

## 📖 使用指南

### 基础使用

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # 使用实时语音模型
    llm = volcengine.RealtimeModel(
        app_id="your_app_id",
        access_token="your_access_token",
        bot_name="智能助手",
        model="O"  # 或 "SC"
    )

    session = AgentSession(llm=llm)

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 大模型语音识别

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # 使用大模型语音识别
    stt = volcengine.BigModelSTT(app_id="your_app_id")

    session = AgentSession(
        stt=stt,
        tts=volcengine.TTS(app_id="your_tts_app_id", cluster="your_cluster"),
        llm=volcengine.LLM(model="doubao-1-5-pro-32k-250115")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 实时语音模型

实时语音模型提供端到端的全双工语音交互体验，支持实时语音识别、对话生成和语音合成。

#### 基础配置

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # 基础实时语音模型配置
    realtime_llm = volcengine.RealtimeModel(
        app_id="your_app_id",
        access_token="your_access_token",
        bot_name="豆包",  # 机器人名称
        model="O"  # 默认使用O版本，功能全面
    )

    session = AgentSession(llm=realtime_llm)

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

#### 高级配置

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine
from dotenv import load_dotenv

# 自定义RAG函数
def rag_function(transcription: str) -> str:
    """自定义RAG函数，用于增强模型回复"""
    # 根据用户输入返回相关的上下文信息
    return '[{"title":"相关信息","content":"这是相关的背景知识"}]'

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # 高级配置 - 包含网络搜索和RAG功能
    realtime_llm = volcengine.RealtimeModel(
        app_id="your_app_id",
        access_token="your_access_token",
        bot_name="智能助手",
        model="O",  # O版本支持联网搜索和RAG，SC版本专注角色扮演
        enable_volc_websearch=True,  # 启用联网搜索
        volc_websearch_api_key="your_websearch_api_key",
        rag_fn=rag_function,  # 自定义RAG函数
        speaking_style="专业、友好、耐心",
        system_role="你是一个专业的AI助手，擅长解答各种问题。"
    )

    session = AgentSession(llm=realtime_llm)

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

#### 模型版本对比

火山引擎实时语音模型提供两个不同的版本，各有独特的功能特点：

##### O版本 (标准功能版)
```python
realtime_llm = volcengine.RealtimeModel(
    model="O",  # 标准功能版，功能全面
    # ... 其他配置
)
```

**核心功能：**
- 🌐 **内置联网搜索**：支持实时网络信息检索和查询
- 🧠 **外部RAG支持**：支持外部知识库检索、总结和口语化改写
- 🎵 **精品音色**：支持 vv、xiaohe、yunzhou、xiaotian 等高品质音色
- ⚙️ **SP开放配置**：支持 bot_name、system_role、speaking_style 等参数配置
- 🚫 **角色描述**：不支持 character_manifest 参数
- 🚫 **克隆音色**：不支持 ICL_ 或 S_ 开头的音色名称

**适用场景：**
- 需要联网搜索的应用
- 需要外部知识库集成的场景
- 对音质要求较高的应用
- 需要灵活参数配置的场景

##### SC版本 (角色增强版)
```python
realtime_llm = volcengine.RealtimeModel(
    model="SC",  # 角色增强版，专注角色扮演
    # ... 其他配置
)
```

**核心功能：**
- 🚫 **内置联网搜索**：不支持联网搜索功能
- 🚫 **外部RAG支持**：不支持外部RAG功能
- 🚫 **精品音色**：不支持 vv、xiaohe、yunzhou、xiaotian 等音色
- 🚫 **SP开放配置**：不支持 bot_name、system_role、speaking_style 参数
- 🎭 **角色描述**：支持 character_manifest 参数进行角色配置
- 🎤 **克隆音色**：支持 ICL_ 或 S_ 开头的克隆音色

**适用场景：**
- 专注于角色扮演的应用
- 需要使用克隆音色的场景
- 对角色个性化要求较高的应用
- 需要 character_manifest 配置的场景

#### 功能对比表格

| 功能特性 | O版本 | SC版本 |
|----------|-------|--------|
| 🌐 内置联网搜索 | ✅ | ❌ |
| 🧠 外部RAG总结和口语化改写 | ✅ | ❌ |
| 🎵 精品音色（vv、xiaohe、yunzhou、xiaotian） | ✅ | ❌ |
| ⚙️ SP开放配置（bot_name、system_role、speaking_style） | ✅ | ❌ |
| 🎭 角色描述（character_manifest） | ❌ | ✅ |
| 🎤 克隆音色（ICL_或S_开头） | ❌ | ✅ |

#### 版本选择建议

| 应用场景 | 推荐版本 | 主要原因 |
|----------|----------|----------|
| 智能客服 | O版本 | 需要联网搜索和灵活配置 |
| 教育辅导 | O版本 | 需要联网搜索和精品音质 |
| 角色扮演 | SC版本 | 专注于角色特性和克隆音色 |
| 娱乐应用 | O版本 | 需要联网搜索和多样音色 |
| 企业培训 | SC版本 | 需要角色定制和个性化 |
| 内容创作 | O版本 | 需要RAG和联网搜索功能 |

**使用建议：**
- **O版本**：功能更全面，适合大多数应用场景
- **SC版本**：专注于角色扮演，适合特定个性化需求
- 根据您的具体需求选择合适的版本
- 建议先测试两种版本，选择最符合需求的那一个

#### 角色配置 (SC版本)

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    # SC版本支持角色配置和克隆音色
    realtime_llm = volcengine.RealtimeModel(
        app_id="your_app_id",
        access_token="your_access_token",
        bot_name="小智",
        model="SC",  # SC版本支持character_manifest
        character_manifest='{  # JSON字符串格式的角色配置
            "name": "小智",
            "personality": "活泼开朗、聪明机智",
            "background": "一个充满好奇心的AI助手"
        }',
        speaker="ICL_custom_voice",  # SC版本支持克隆音色
        speaking_style="活泼开朗，像个小伙伴一样和你聊天",
        system_role="你叫小智，是一个活泼开朗的AI助手，喜欢用emoji表情，回复简洁有趣。"
    )

    session = AgentSession(llm=realtime_llm)

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

### 单独组件使用

如果您需要分别使用各个组件（STT、TTS、LLM），而不是使用集成的实时语音模型：

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import volcengine
from dotenv import load_dotenv

async def entry_point(ctx: JobContext):
    agent = Agent(instructions="You are a helpful assistant.")

    session = AgentSession(
        # 语音识别
        stt=volcengine.STT(
            app_id="your_stt_app_id",
            cluster="your_cluster"
        ),
        # 语音合成
        tts=volcengine.TTS(
            app_id="your_tts_app_id",
            cluster="your_cluster",
            voice_type="BV001_V2_streaming"
        ),
        # 单独的LLM (非实时)
        llm=volcengine.LLM(model="doubao-1-5-pro-32k-250115")
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
```

## 🔧 API 参考

### TTS (文本转语音)

```python
volcengine.TTS(
    app_id: str,           # 应用ID
    cluster: str,          # 集群ID
    voice_type: str = "BV001_V2_streaming",  # 语音类型
    speed_ratio: float = 1.0,   # 语速 (0.5-2.0)
    volume_ratio: float = 1.0,  # 音量 (0.1-3.0)
    pitch_ratio: float = 1.0    # 音调 (0.5-2.0)
)
```

### STT (语音识别)

```python
volcengine.STT(
    app_id: str,           # 应用ID
    cluster: str,          # 集群ID
    language: str = "zh-CN"  # 语言
)
```

### BigModelSTT (大模型语音识别)

```python
volcengine.BigModelSTT(
    app_id: str,           # 应用ID
    cluster: str = "volcengine_streaming_common",  # 集群
    protocol: str = "http",  # 协议
    language: str = "zh-CN"  # 语言
)
```

### LLM (大语言模型)

```python
volcengine.LLM(
    model: str,            # 模型名称
    temperature: float = 0.7,    # 温度
    max_tokens: int = 2000,      # 最大token数
    system_message: str = None   # 系统消息
)
```

### RealtimeModel (实时语音模型)

```python
volcengine.RealtimeModel(
    # 必需参数
    app_id: str = None,                    # 应用ID，从环境变量 VOLCENGINE_REALTIME_APP_ID 获取
    access_token: str = None,              # 访问令牌，从环境变量 VOLCENGINE_REALTIME_ACCESS_TOKEN 获取

    # 基本配置
    bot_name: str = "豆包",                # 机器人名称
    speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。",  # 说话风格描述
    speaker: str = "zh_female_vv_jupiter_bigtts",  # 语音类型
    opening: str = "你好啊，今天过得怎么样？",     # 开场白内容
    system_role: str = None,               # 系统角色提示词

    # 模型配置
    model: Literal["O", "SC"] = "O",       # 模型版本：O(标准功能版) 或 SC(角色增强版)
    character_manifest: str = None,        # 角色配置，JSON字符串格式

    # 音频配置
    sample_rate: int = 24000,              # 采样率
    num_channels: int = 1,                 # 声道数
    format: str = "pcm",                   # 音频格式

    # ASR配置
    end_smooth_window_ms: int = 500,       # 结束平滑窗口毫秒数

    # 网络搜索配置
    enable_volc_websearch: bool = False,   # 是否启用网络搜索
    volc_websearch_type: Literal["web_summary", "web"] = "web_summary",  # 搜索类型
    volc_websearch_api_key: str = None,    # 网络搜索API密钥
    volc_websearch_no_result_message: str = "抱歉，我找不到相关信息。",  # 无结果提示

    # RAG配置
    rag_fn: Callable[[str], str] = None,   # RAG回调函数，输入查询返回增强内容

    # 其他配置
    audio_output: bool = True,             # 是否输出音频
    modalities: List[Literal["text", "audio"]] = NOT_GIVEN,  # 输出模态
    http_session: aiohttp.ClientSession = None,  # HTTP会话
    max_session_duration: float = None,    # 最大会话持续时间
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS  # 连接选项
)
```

## ❓ 常见问题

### Q: 如何获取访问令牌？

A: 请访问[火山引擎控制台](https://console.volcengine.com/)，在相应服务页面创建应用并获取访问令牌。

### Q: 支持哪些语音类型？

A: 支持多种语音类型，包括 BV001_V2_streaming、BV002_V2_streaming 等，详见[语音合成文档](https://www.volcengine.com/docs/6561/97465)。

### Q: 如何处理连接错误？

A: 请检查网络连接和访问令牌是否正确配置。建议使用环境变量管理敏感信息。

### Q: 支持哪些大语言模型？

A: 支持豆包系列模型，包括 doubao-1-5-lite、doubao-1-5-pro 等，详见[大模型文档](https://www.volcengine.com/docs/82379/1513689)。

### Q: 实时语音模型支持哪些语音类型？

A: 实时语音模型支持多种预训练语音，包括：
- `zh_female_vv_jupiter_bigtts` - 女声 Jupiter 大模型语音
- `zh_male_vv_jupiter_bigtts` - 男声 Jupiter 大模型语音
- 其他火山引擎TTS支持的语音类型

### Q: 如何配置网络搜索功能？

A: 要启用网络搜索功能，需要设置以下参数：
```python
realtime_llm = volcengine.RealtimeModel(
    enable_volc_websearch=True,
    volc_websearch_type="web_summary",  # 或 "web"
    volc_websearch_api_key="your_websearch_key"
)
```

### Q: 什么是RAG功能，如何使用？

A: RAG（Retrieval-Augmented Generation）可以为模型提供额外的上下文信息：
```python
def my_rag_function(query: str) -> str:
    # 根据用户查询返回相关的上下文信息
    return "相关的背景知识和信息"

realtime_llm = volcengine.RealtimeModel(
    rag_fn=my_rag_function
)
```

### Q: 实时语音模型和普通LLM/TTS的区别是什么？

A: 实时语音模型是端到端的解决方案：
- **集成度高**：语音识别、对话生成、语音合成一体化
- **实时性强**：支持全双工实时对话
- **智能化**：内置角色配置、网络搜索、RAG等高级功能
- **简单配置**：一个模型即可完成语音交互全流程

普通LLM+TTS组合需要分别配置和处理各个组件的数据流，而实时模型内部集成了这些功能。

### Q: 如何选择O版本和SC版本？

A: 根据您的应用需求选择合适的版本：

**选择O版本的情况：**
- 需要联网搜索功能的应用
- 需要外部RAG集成的场景
- 对音质要求较高，需要精品音色的应用
- 需要灵活配置bot_name、system_role、speaking_style的场景
- 大多数通用应用场景

**选择SC版本的情况：**
- 专注于角色扮演的应用
- 需要使用克隆音色（ICL_或S_开头）的场景
- 需要character_manifest进行角色配置的场景
- 对角色个性化要求较高的应用

**版本对比：**
- **O版本**：功能全面，适合大多数应用场景，支持联网搜索、RAG、精品音色等
- **SC版本**：专注于角色扮演，支持角色描述和克隆音色，但不支持联网搜索等功能

**使用建议：**
- 如果您的应用需要联网搜索、RAG或精品音色，选择O版本
- 如果您的应用专注于角色扮演和个性化音色，选择SC版本
- 建议根据具体需求测试两个版本，选择最合适的

## 📝 更新日志

### v1.2.9
- ✨ **新增实时语音模型**：支持端到端全双工语音交互
- 🎯 **增强大模型语音识别**：优化BigModelSTT性能和稳定性
- 🔧 **改进API设计**：RealtimeModel支持丰富的配置选项
- 🌐 **集成网络搜索**：支持实时网络信息检索和摘要
- 🧠 **RAG功能支持**：可配置检索增强生成回调函数
- 🎭 **角色配置系统**：支持自定义AI助手角色和性格
- 🔊 **多语音类型**：支持多种预训练语音模型
- 📚 **完善文档**：详细的使用指南和API参考
- 🐛 **修复已知问题**：提升整体稳定性和兼容性

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
- [火山引擎](https://www.volcengine.com/) - 强大的AI服务提供商

