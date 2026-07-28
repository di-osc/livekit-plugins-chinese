import React from 'react'
import PropTypes from 'prop-types'

import ReadNext from '../components/readnext'
import Button from '../components/button'
import Grid from '../components/grid'
import Title from '../components/title'
import Footer from '../components/footer'
import Sidebar from '../components/sidebar'
import Main from '../components/main'
import { getCurrentSource, github } from '../components/util'

import sidebars from '../../meta/sidebars.json'

export default function Docs({ pageContext, children }) {
    const {
        slug,
        title,
        section,
        teaser,
        source,
        tag,
        isIndex,
        next,
        menu,
        theme,
        version,
        apiDetails,
    } = pageContext
    const sidebar = pageContext.sidebar
        ? { items: pageContext.sidebar }
        : sidebars.find((item) => item.section === section)
    const pageMenu = menu ? menu.map(([text, id]) => ({ text, id })) : []
    const sourcePath = source ? github(source) : null

    const subFooter = (
        <div data-pagefind-ignore="">
            <Grid cols={2}>
                <div style={{ marginTop: 'var(--spacing-lg)' }}>
                    <Button to={getCurrentSource(slug, isIndex)} icon="code">
                        建议修改
                    </Button>
                </div>
                {next && <ReadNext title={next.title} to={next.slug} />}
            </Grid>
        </div>
    )

    return (
        <>
            {sidebar && <Sidebar items={sidebar.items} pageMenu={pageMenu} slug={slug} />}
            <Main
                section={section}
                theme={theme || 'blue'}
                sidebar
                asides
                wrapContent
                footer={<Footer />}
            >
                <Title
                    title={title}
                    teaser={teaser}
                    source={sourcePath}
                    tag={tag}
                    version={version}
                    id="_title"
                    apiDetails={apiDetails}
                />
                {children}
                {subFooter}
            </Main>
        </>
    )
}

Docs.propTypes = {
    pageContext: PropTypes.object.isRequired,
    children: PropTypes.node,
}
