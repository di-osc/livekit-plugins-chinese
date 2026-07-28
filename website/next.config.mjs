import MDX from '@next/mdx'
import PWA from 'next-pwa'

import remarkPlugins from './plugins/index.mjs'

const normalizeBasePath = value => {
    const path = typeof value === 'string' ? value.trim() : ''

    if (!path || path === '/') {
        return ''
    }

    return `/${path.replace(/^\/+|\/+$/g, '')}`
}

const withMDX = MDX({
    extension: /\.mdx?$/,
    options: {
        remarkPlugins,
        providerImportSource: '@mdx-js/react',
    },
    experimental: {
        mdxRs: true,
    },
})

const withPWA = PWA({
    dest: 'public',
    disable: process.env.NODE_ENV === 'development',
})

/** @type {import('next').NextConfig} */
const nextConfig = withPWA(
    withMDX({
        reactStrictMode: true,
        swcMinify: true,
        basePath: normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH),
        pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
        eslint: {
            ignoreDuringBuilds: true,
        },
        typescript: {
            ignoreBuildErrors: true,
        },
        images: { unoptimized: true },
    })
)

export default nextConfig
