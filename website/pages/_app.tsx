import '../src/styles/layout.sass'

import type { AppProps } from 'next/app'
import Head from 'next/head'
import { MDXProvider } from '@mdx-js/react'
import { remarkComponents } from '../src/remark'
import { withBasePath } from '../src/basePath'
import site from '../meta/site.json'

export default function App({ Component, pageProps }: AppProps) {
    return (
        <>
            <Head>
                <link rel="sitemap" type="application/xml" href={withBasePath('/sitemap.xml')} />
                <link rel="manifest" href={withBasePath('/manifest.webmanifest')} />
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1.0, minimum-scale=1, maximum-scale=5.0, shrink-to-fit=no, viewport-fit=cover"
                />
                <meta name="theme-color" content={site.theme} />
            </Head>
            <MDXProvider components={remarkComponents}>
                <Component {...pageProps} />
            </MDXProvider>
        </>
    )
}
