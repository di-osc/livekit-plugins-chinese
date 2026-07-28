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

## 发布 PyPI 包和创建 GitHub Release

发布流程使用 PyPI Trusted Publishing，不需要在 GitHub 保存长期 PyPI Token。
首次发布前，需要完成一次授权配置：

1. 在 GitHub 仓库的 `Settings → Environments` 中创建名为 `pypi` 的
   Environment，建议配置 Required reviewers。
2. 在以下五个 PyPI 项目的 `Manage → Publishing` 页面分别添加同一个
   GitHub Trusted Publisher：
   - Owner：`di-osc`
   - Repository：`livekit-plugins-chinese`
   - Workflow：`release.yml`
   - Environment：`pypi`
3. 需要配置的 PyPI 项目是 `livekit-plugins-aliyun`、
   `livekit-plugins-difyai`、`livekit-plugins-flashtts`、
   `livekit-plugins-stepfun` 和 `livekit-plugins-volcengine`。

如果某个项目尚未在 PyPI 创建，请先为该项目配置 Pending Publisher，并使用
相同的 GitHub 仓库、工作流和 Environment 信息。

发布前，将五个插件的 `version.py` 更新为同一版本并提交，然后推送对应的 `v<版本号>` 标签：

```bash
git tag v1.6.7
git push origin v1.6.7
```

GitHub Actions 会自动校验依赖和版本，构建所有插件的 wheel 和源码包，通过
OIDC 发布到 PyPI，生成 SHA-256 校验文件，并创建附带构建产物的 GitHub
Release。PyPI 发布成功后才会创建 GitHub Release。
