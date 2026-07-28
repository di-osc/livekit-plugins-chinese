import React from 'react'
import { render, screen } from '@testing-library/react'

import Aside from './aside'
import { remarkComponents } from '../remark'

test('registers Aside for direct use in MDX examples', () => {
    expect(remarkComponents.Aside).toBe(Aside)
})

test('renders a titled Example as complementary content', () => {
    render(
        <Aside title="Example">
            <code>npm run dev</code>
        </Aside>
    )

    expect(screen.getAllByRole('complementary')[0]).toHaveTextContent('npm run dev')
    expect(screen.getByText('Example')).toBeInTheDocument()
})
