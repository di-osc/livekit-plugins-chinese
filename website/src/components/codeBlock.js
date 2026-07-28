import React from 'react'
import classNames from 'classnames'
import Code from './codeDynamic'
import classes from '../styles/code.module.sass'

export const Pre = (props) => {
    const childLanguageClass = React.Children.toArray(props.children)
        .flatMap((child) => child.props?.className?.split(/\s+/) ?? [])
        .find((name) => name.startsWith('language-'))
    const hasLanguageClass = props.className?.split(/\s+/).includes(childLanguageClass)

    return (
        <pre
            className={classNames(classes['pre'], props.className, {
                [childLanguageClass]: childLanguageClass && !hasLanguageClass,
            })}
            data-pagefind-ignore=""
        >
            {props.children}
        </pre>
    )
}

const CodeBlock = (props) => (
    <Pre>
        <Code {...props} />
    </Pre>
)
export default CodeBlock
