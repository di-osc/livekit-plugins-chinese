import React from 'react'
import PropTypes from 'prop-types'
import classNames from 'classnames'
import Prism from 'prismjs'

import 'prismjs/components/prism-bash.min.js'
import 'prismjs/components/prism-diff.min.js'
import 'prismjs/components/prism-docker.min.js'
import 'prismjs/components/prism-ini.min.js'
import 'prismjs/components/prism-javascript.min.js'
import 'prismjs/components/prism-jsx.min.js'
import 'prismjs/components/prism-json.min.js'
import 'prismjs/components/prism-markdown.min.js'
import 'prismjs/components/prism-python.min.js'
import 'prismjs/components/prism-rust.min.js'
import 'prismjs/components/prism-yaml.min.js'

import classes from '../styles/code.module.sass'

export default function Code({ lang = 'none', title, wrap, className, children }) {
    const language = lang === 'sh' ? 'bash' : lang
    const text = typeof children === 'string' ? children : String(children ?? '')
    const highlighted =
        language in Prism.languages
            ? Prism.highlight(text, Prism.languages[language], language)
            : null
    const languageClassName = `language-${language}`
    const hasLanguageClass = className?.split(/\s+/).includes(languageClassName)
    const codeClassNames = classNames(classes.code, className, {
        [languageClassName]: !hasLanguageClass,
        [classes.wrap]: Boolean(wrap),
    })

    return (
        <>
            {title && <h4 className={classes.title}>{title}</h4>}
            <code
                className={codeClassNames}
                {...(highlighted
                    ? { dangerouslySetInnerHTML: { __html: highlighted } }
                    : { children: text })}
            />
        </>
    )
}

Code.propTypes = {
    lang: PropTypes.string,
    title: PropTypes.string,
    wrap: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
    className: PropTypes.string,
    children: PropTypes.node,
}
