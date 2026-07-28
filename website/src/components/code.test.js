import React from 'react'
import { render } from '@testing-library/react'

import Code from './code'
import { Pre } from './codeBlock'

describe('Code', () => {
    test('does not duplicate the Markdown language class during hydration', () => {
        const { container } = render(
            <Code lang="text" className="language-text">
                example
            </Code>
        )

        const languageClasses = container
            .querySelector('code')
            .getAttribute('class')
            .split(/\s+/)
            .filter((name) => name === 'language-text')
        expect(languageClasses).toHaveLength(1)
    })
})

describe('Pre', () => {
    test('mirrors the child language class before Prism initializes', () => {
        const { container } = render(
            <Pre>
                <code className="language-text">example</code>
            </Pre>
        )

        expect(container.querySelector('pre')).toHaveClass('language-text')
    })
})
