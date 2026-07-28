import React from 'react'
import PropTypes from 'prop-types'
import Head from 'next/head'

import siteMetadata from '../../meta/site.json'

export default function SEO({ description, title, sectionTitle }) {
    const pageTitle = title
        ? `${title} · ${sectionTitle || siteMetadata.title}`
        : `${siteMetadata.title} · ${siteMetadata.slogan}`
    const metaDescription = description || siteMetadata.description

    return (
        <Head>
            <title>{pageTitle}</title>
            <meta name="description" content={metaDescription} />
            <meta property="og:title" content={pageTitle} />
            <meta property="og:description" content={metaDescription} />
            <meta property="og:type" content="website" />
            <meta property="og:site_name" content={siteMetadata.title} />
        </Head>
    )
}

SEO.propTypes = {
    description: PropTypes.string,
    title: PropTypes.string,
    sectionTitle: PropTypes.string,
}
