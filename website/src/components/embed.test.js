import React from 'react'
import { render } from '@testing-library/react'

import { Image } from './embed'

describe('Image', () => {
    test('uses phrasing elements so Markdown images can render inside paragraphs', () => {
        const { container } = render(
            <p>
                <Image src="/images/example.svg" alt="Example" title="Example caption" />
            </p>
        )

        expect(container.querySelector('p > span.gatsby-resp-image-figure')).toBeInTheDocument()
        expect(container.querySelector('span.gatsby-resp-image-figcaption')).toBeInTheDocument()
        expect(container.querySelector('figure')).not.toBeInTheDocument()
        expect(container.querySelector('figcaption')).not.toBeInTheDocument()
    })
})
