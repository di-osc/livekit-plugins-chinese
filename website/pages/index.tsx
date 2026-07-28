import React from 'react'

import Layout from '../src/templates'
import {
    LandingHeader,
    LandingTitle,
    LandingSubtitle,
    LandingGrid,
    LandingCard,
} from '../src/components/landing'

export default function Home() {
    return (
        <Layout>
            <LandingHeader>
                <LandingTitle>LiveKit 中文插件</LandingTitle>
                <LandingSubtitle>
                    阿里云、火山引擎、阶跃星辰、Dify 与 FlashTTS 的 LiveKit Agents 集成指南
                </LandingSubtitle>
            </LandingHeader>
            <LandingGrid blocks>
                <LandingCard title="五个插件，一个入口" url="/plugins" button="查看支持矩阵">
                    按 STT、TTS、LLM 和 Realtime 能力选择插件，并直接复制可运行配置。
                </LandingCard>
                <LandingCard
                    title="密钥申请不再迷路"
                    url="/getting-started/credentials"
                    button="配置凭证"
                >
                    每个云服务都给出官方申请入口、控制台操作路径和对应环境变量。
                </LandingCard>
                <LandingCard title="端到端实时语音" url="/plugins/aliyun" button="接入 Qwen Audio">
                    使用 Qwen Audio 3.0 Realtime 直接处理语音、轮次判断和工具调用。
                </LandingCard>
            </LandingGrid>
        </Layout>
    )
}
