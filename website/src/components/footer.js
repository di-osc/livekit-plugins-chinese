import React from 'react'
import PropTypes from 'prop-types'

import Link from './link'
import Grid from './grid'
import classes from '../styles/footer.module.sass'
import siteMetadata from '../../meta/site.json'

export default function Footer({ wide = false }) {
    const { companyUrl, company, footer } = siteMetadata
    return (
        <footer className={classes.root}>
            <Grid cols={wide ? 3 : 2} narrow className={classes.content}>
                {footer.map(({ label, items }) => (
                    <section key={label}>
                        <ul className={classes.column}>
                            <li className={classes.label}>{label}</li>
                            {items.map(({ text, url }) => (
                                <li key={url}>
                                    <Link to={url} noLinkLayout>
                                        {text}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </section>
                ))}
            </Grid>
            <div className={`${classes.content} ${classes.copy}`}>
                &copy; {new Date().getFullYear()}{' '}
                <Link to={companyUrl} noLinkLayout>
                    {company}
                </Link>
            </div>
        </footer>
    )
}

Footer.propTypes = {
    wide: PropTypes.bool,
}
