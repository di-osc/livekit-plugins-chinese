# LiveKit 中文插件文档站

文档站基于 [di-osc/libraries](https://github.com/di-osc/libraries) 模板构建，使用
Next.js、MDX 和 Pagefind，并通过 GitHub Actions 发布到 GitHub Pages。

```bash
cd website
npm ci
npm run dev
```

完整生产构建：

```bash
npm run build
```

构建结果位于 `website/out/`。推送 `main` 分支中与 `website/` 有关的改动后，
`.github/workflows/deploy-docs.yml` 会自动发布文档。

首次启用时，请在仓库 **Settings → Pages → Build and deployment → Source** 中选择
**GitHub Actions**。发布地址为：

```text
https://di-osc.github.io/livekit-plugins-chinese/
```
