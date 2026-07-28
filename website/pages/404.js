import React from 'react'

import Layout from '../src/templates'
import { LandingHeader, LandingTitle, LandingSubtitle } from '../src/components/landing'
import Button from '../src/components/button'

export default function Page404() {
    return (
        <Layout title="页面不存在" searchExclude>
            <LandingHeader style={{ minHeight: 400 }}>
                <LandingTitle>404</LandingTitle>
                <LandingSubtitle>没有找到这个页面</LandingSubtitle>
                <p>
                    <Button to="/" variant="tertiary" large>
                        返回首页
                    </Button>
                </p>
            </LandingHeader>
        </Layout>
    )
}
