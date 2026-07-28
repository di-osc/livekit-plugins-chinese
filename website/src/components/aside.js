import React, { useEffect, useRef } from 'react'
import PropTypes from 'prop-types'

import classes from '../styles/aside.module.sass'

export default function Aside({ title, children }) {
    const rootRef = useRef(null)
    const contentRef = useRef(null)

    useEffect(() => {
        const root = rootRef.current
        const content = contentRef.current
        const section = root?.closest('section')

        if (!section || !content) return undefined

        const fitSectionToExamples = () => {
            if (window.getComputedStyle(content).position !== 'absolute') {
                section.style.minHeight = ''
                return
            }

            section.style.minHeight = ''
            const sectionTop = section.getBoundingClientRect().top
            const examples = section.querySelectorAll('[data-docs-aside-content]')
            const exampleBottom = Math.max(
                ...Array.from(examples, (example) => example.getBoundingClientRect().bottom)
            )
            const requiredHeight = Math.ceil(exampleBottom - sectionTop + 24)

            if (requiredHeight > section.getBoundingClientRect().height) {
                section.style.minHeight = `${requiredHeight}px`
            }
        }

        fitSectionToExamples()
        const observer = window.ResizeObserver
            ? new window.ResizeObserver(fitSectionToExamples)
            : null
        observer?.observe(content)
        window.addEventListener('resize', fitSectionToExamples)

        return () => {
            observer?.disconnect()
            window.removeEventListener('resize', fitSectionToExamples)
            section.style.minHeight = ''
        }
    }, [])

    return (
        <aside className={classes.root} ref={rootRef}>
            <div
                className={classes.content}
                data-docs-aside-content
                ref={contentRef}
                role="complementary"
            >
                <div className={classes.text}>
                    {title && <h4 className={classes.title}>{title}</h4>}
                    {children}
                </div>
            </div>
        </aside>
    )
}

Aside.propTypes = {
    title: PropTypes.string,
    children: PropTypes.node.isRequired,
}
