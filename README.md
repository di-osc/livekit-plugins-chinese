# 简介

该项目为[livekit-agents](https://github.com/livekit/agents)**实时语音智能体**框架提供中文插件支持。

📚 在线文档：[https://di-osc.github.io/livekit-plugins-chinese/](https://di-osc.github.io/livekit-plugins-chinese/)

## 近期修改

后续将逐步取消以下厂商插件的支持：
- 百度
- MiniMax（迁移到Livekit官方插件支持）
- 腾讯
- 讯飞
- 智谱

新增：
- 阿里云Realtime模型支持
- 基于 GitHub Pages 的在线插件文档


## 插件列表

| 厂商 | STT | TTS | LLM | Realtime | 使用说明 |
| ---- | --- | --- | --- | --- | --- |
| 火山引擎 | ✅  | ✅  | ✅  |✅ |[点击这里](livekit-plugins/livekit-plugins-volcengine) |
| 阿里云 | ✅ | ✅  | ✅ | ✅ |[点击这里](livekit-plugins/livekit-plugins-aliyun) |
| 阶跃星辰 | ❎ | ❎  | ❎ | ✅ |[点击这里](livekit-plugins/livekit-plugins-stepfun) |
| Dify | ❎ | ❎  | ✅  | ❎|[点击这里](livekit-plugins/livekit-plugins-dify) |
| FlashTTS | ❎ | ✅  | ❎ | ❎ |[点击这里](livekit-plugins/livekit-plugins-flashtts) |

## 创建 GitHub Release

发布前，将五个插件的 `version.py` 更新为同一版本并提交，然后推送对应的 `v<版本号>` 标签：

```bash
git tag v1.6.7
git push origin v1.6.7
```

GitHub Actions 会自动校验版本、构建所有插件的 wheel 和源码包、生成 SHA-256
校验文件，并创建附带这些构建产物的 GitHub Release。该流程不会发布到 PyPI。
