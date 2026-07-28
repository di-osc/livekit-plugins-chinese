import React from 'react'

import Layout from '../src/templates'
import Link from '../src/components/link'
import classes from '../src/styles/landing.module.sass'
import patternDefault from '../src/images/pattern_blue.png'
import overlayDefault from '../src/images/pattern_landing.png'

const plugins = [
    {
        name: '阿里云',
        href: '/plugins/aliyun',
        accent: 'cyan',
        capabilities: ['Paraformer', 'CosyVoice', 'Qwen', 'Qwen Audio 3.0'],
    },
    {
        name: '火山引擎',
        href: '/plugins/volcengine',
        accent: 'violet',
        capabilities: ['豆包 ASR', 'Seed TTS', '豆包 Seed', '豆包 O / SC'],
    },
    {
        name: '阶跃星辰',
        href: '/plugins/stepfun',
        accent: 'blue',
        capabilities: [null, null, null, 'Step Audio 2.5'],
    },
    {
        name: 'Dify',
        href: '/plugins/dify',
        accent: 'green',
        capabilities: [null, null, 'Chat App', null],
    },
    {
        name: 'FlashTTS',
        href: '/plugins/flashtts',
        accent: 'orange',
        capabilities: [null, '自托管 TTS', null, null],
    },
]

const capabilityNames = ['STT', 'TTS', 'LLM', 'REALTIME']

export default function Home() {
    const backgroundStyle = {
        '--landing-pattern': `url(${patternDefault.src})`,
        '--landing-overlay': `url(${overlayDefault.src})`,
    } as React.CSSProperties

    return (
        <Layout>
            <main className={classes['capability-home']} style={backgroundStyle}>
                <header className={classes['capability-heading']}>
                    <div className={classes['capability-kicker']}>
                        <span />
                        LIVEKIT PLUGIN ECOSYSTEM
                    </div>
                    <h1>中文模型能力矩阵</h1>
                    <p>五个插件，覆盖语音识别、语音合成、文本模型与端到端实时语音。</p>
                </header>

                <section
                    className={classes['matrix-shell']}
                    aria-labelledby="capability-matrix-title"
                >
                    <h2 id="capability-matrix-title" className={classes['sr-only']}>
                        插件模型能力
                    </h2>
                    <div
                        className={classes['matrix-scroll']}
                        role="region"
                        aria-label="插件模型能力矩阵，可横向滚动"
                        tabIndex={0}
                    >
                        <div className={classes['matrix']} role="table">
                            <div
                                className={`${classes['matrix-row']} ${classes['matrix-header']}`}
                                role="row"
                            >
                                <div role="columnheader">PLUGIN</div>
                                {capabilityNames.map((capability) => (
                                    <div role="columnheader" key={capability}>
                                        {capability}
                                    </div>
                                ))}
                            </div>

                            {plugins.map((plugin, index) => (
                                <div
                                    className={`${classes['matrix-row']} ${
                                        classes[`accent-${plugin.accent}`]
                                    }`}
                                    role="row"
                                    key={plugin.name}
                                    style={{ '--row-index': index } as React.CSSProperties}
                                >
                                    <div className={classes['plugin-cell']} role="rowheader">
                                        <span className={classes['plugin-index']}>
                                            {String(index + 1).padStart(2, '0')}
                                        </span>
                                        <Link
                                            to={plugin.href}
                                            noLinkLayout
                                            className={classes['plugin-link']}
                                        >
                                            {plugin.name}
                                            <span aria-hidden="true">↗</span>
                                        </Link>
                                    </div>

                                    {plugin.capabilities.map((capability, capabilityIndex) => (
                                        <div className={classes['capability-cell']} role="cell" key={capabilityNames[capabilityIndex]}>
                                            {capability ? (
                                                <span className={classes['capability-pill']}>
                                                    <i aria-hidden="true" />
                                                    {capability}
                                                </span>
                                            ) : (
                                                <span
                                                    className={classes['capability-empty']}
                                                    aria-label="不支持"
                                                >
                                                    —
                                                </span>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className={classes['matrix-footer']}>
                        <span>
                            <i aria-hidden="true" /> 已支持
                        </span>
                        <span>点击服务商名称查看安装与配置</span>
                    </div>
                </section>
            </main>
        </Layout>
    )
}
