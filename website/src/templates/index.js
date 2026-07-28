import React from 'react'
import PropTypes from 'prop-types'
import classNames from 'classnames'

import Navigation from '../components/navigation'
import Progress from '../components/progress'
import Search from '../components/search'
import Footer from '../components/footer'
import SEO from '../components/seo'
import Docs from './docs'
import siteMetadata from '../../meta/site.json'

export default function Layout({ children, ...pageContext }) {
    const { title, section, sectionTitle, teaser, theme, searchExclude } = pageContext
    const bodyClass = classNames(`theme-${theme || 'blue'}`, {
        'search-exclude': Boolean(searchExclude),
    })

    return (
        <div className={bodyClass}>
            <SEO title={title} description={teaser} sectionTitle={sectionTitle} />
            <Navigation
                title={siteMetadata.title}
                items={siteMetadata.navigation}
                section={section}
            >
                <>
                    <Search />
                    <Progress />
                </>
            </Navigation>
            {section ? (
                <Docs pageContext={pageContext}>{children}</Docs>
            ) : (
                <>
                    {children}
                    <Footer wide />
                </>
            )}
        </div>
    )
}

Layout.propTypes = {
    children: PropTypes.node,
}
