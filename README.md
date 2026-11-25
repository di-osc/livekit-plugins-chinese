# 简介

该项目为[livekit-agents](https://github.com/livekit/agents)**实时语音智能体**框架提供中文插件支持。

📚 在线文档：[https://di-osc.github.io/livekit-plugins-chinese/](https://di-osc.github.io/livekit-plugins-chinese/)

<details>
<summary><strong>📌 近期更新与支持计划</strong></summary>

### ✨ 新增

- 阿里云 Qwen Audio Realtime，支持 3.0 与 3.1。Audio 与 Omni 默认轮次检测都是 `server_vad`
- 阿里云 Qwen-Omni Realtime，支持语音以及图片、视频输入
- 基于 GitHub Pages 的[在线插件文档](https://di-osc.github.io/livekit-plugins-chinese/)

### 📦 支持调整

以下厂商插件将逐步停止维护：

- 百度
- MiniMax（建议迁移至 LiveKit 官方插件）
- 腾讯
- 讯飞
- 智谱

### 🗑 移除

- 移除阶跃星辰插件 `livekit-plugins-stepfun` 及其文档。端到端实时语音请改用阿里云 Qwen Audio / Omni，或火山引擎豆包 Realtime。

</details>

## 环境变量

复制 [`.env.example`](.env.example) 为 `.env`，再填入要用的密钥。示例通过 `python-dotenv` 加载这个文件。各变量的申请入口见[密钥与环境变量](https://di-osc.github.io/livekit-plugins-chinese/getting-started/credentials)。

## 示例

```bash
uv sync
uv run examples/aliyun_pipeline.py console
uv run examples/aliyun_qwen_audio_realtime.py console
uv run examples/aliyun_qwen_omini_realtime.py console
uv run examples/volcengine_realtime.py console
```

## 插件列表

| 厂商 | STT | TTS | LLM | Realtime | 使用说明 |
| ---- | --- | --- | --- | --- | --- |
| 火山引擎 | ✅  | ✅  | ✅  |✅ |[点击这里](livekit-plugins/livekit-plugins-volcengine) |
| 阿里云 | ✅ | ✅  | ✅ | ✅ |[点击这里](livekit-plugins/livekit-plugins-aliyun) |
| Dify | ❎ | ❎  | ✅  | ❎|[点击这里](livekit-plugins/livekit-plugins-dify) |
| FlashTTS | ❎ | ✅  | ❎ | ❎ |[点击这里](livekit-plugins/livekit-plugins-flashtts) |
