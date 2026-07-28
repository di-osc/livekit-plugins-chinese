import React from 'react'
import { render, screen } from '@testing-library/react'

import Main from './main'
import { Pre } from './codeBlock'
import { InlineCode } from './inlineCode'
import Title from './title'
import { H1 } from './typography'
import Docs from '../templates/docs'
import Layout from '../templates'

jest.mock('./sidebar', () => function MockSidebar() {
    return <aside data-testid="site-sidebar">Sidebar</aside>
})

jest.mock('./footer', () => function MockFooter() {
    return <footer data-testid="site-footer">Footer</footer>
})

jest.mock('./readnext', () => function MockReadNext({ title }) {
    return <span>Read next: {title}</span>
})

jest.mock('./button', () => function MockButton({ children, ...props }) {
    return <span data-pagefind-ignore={props['data-pagefind-ignore']}>{children}</span>
})

jest.mock('./seo', () => function MockSeo() {
    return null
})

jest.mock('./navigation', () => function MockNavigation({ children }) {
    return <nav data-testid="site-navigation">{children}</nav>
})

test('Main marks only wrapped article content as the Pagefind body', () => {
    const { container } = render(
        <Main wrapContent footer={<footer data-testid="main-chrome">Repeated footer</footer>}>
            <p>Searchable article</p>
        </Main>
    )

    const article = container.querySelector('article')
    const footer = screen.getByTestId('main-chrome')
    expect(article).toHaveAttribute('data-pagefind-body', '')
    expect(article).toContainElement(screen.getByText('Searchable article'))
    expect(article).not.toContainElement(footer)
})

test('Pagefind excludes fenced code but keeps inline code searchable', () => {
    const { container } = render(
        <>
            <Pre>const answer = 42</Pre>
            <InlineCode>answer</InlineCode>
        </>
    )

    expect(container.querySelector('pre')).toHaveAttribute('data-pagefind-ignore', '')
    expect(container.querySelector('code')).not.toHaveAttribute('data-pagefind-ignore')
})

test('Title exposes Pagefind title metadata on its final h1', () => {
    render(<Title title="Concise document title" data-track-heading="document" />)

    const heading = screen.getByRole('heading', { level: 1 })
    expect(heading).toHaveAttribute('data-pagefind-meta', 'title')
    expect(heading).toHaveAttribute('data-track-heading', 'document')
})

test('Title excludes its source action from the Pagefind index', () => {
    render(<Title title="Source document" source="module.py" />)

    expect(screen.getByText('Source')).toHaveAttribute('data-pagefind-ignore', '')
})

test('heading DOM attributes are forwarded without leaking custom heading props', () => {
    render(
        <H1
            id="api-heading"
            source="module.py"
            tag="stable"
            version="1.0"
            data-heading-kind="api"
        >
            API heading
        </H1>
    )

    const heading = screen.getByRole('heading', { level: 1 })
    expect(heading).toHaveAttribute('data-heading-kind', 'api')
    expect(heading).not.toHaveAttribute('source')
    expect(heading).not.toHaveAttribute('tag')
    expect(heading).not.toHaveAttribute('version')
    expect(heading).not.toHaveAttribute('permalink')
})

test('Docs excludes its entire subfooter while keeping document content searchable', () => {
    const { container } = render(
        <Docs
            pageContext={{
                slug: '/guide/current',
                title: 'Current guide',
                section: 'guide',
                sidebar: [{ title: 'Guide' }],
                next: { title: 'Following guide', slug: '/guide/next' },
            }}
        >
            <p>Searchable documentation</p>
        </Docs>
    )

    const article = container.querySelector('article[data-pagefind-body=""]')
    const sidebar = screen.getByTestId('site-sidebar')
    const siteFooter = screen.getByTestId('site-footer')
    const ignoredSubFooter = screen.getByText('建议修改').closest('[data-pagefind-ignore]')
    expect(article).toContainElement(screen.getByText('Searchable documentation'))
    expect(article).not.toContainElement(sidebar)
    expect(article).not.toContainElement(siteFooter)
    expect(ignoredSubFooter).toHaveAttribute('data-pagefind-ignore', '')
    expect(article).toContainElement(ignoredSubFooter)
    expect(ignoredSubFooter).toContainElement(screen.getByText('建议修改'))
    expect(ignoredSubFooter).toContainElement(screen.getByText('Read next: Following guide'))
})

test('Layout renders one Search directly before Progress in navigation', () => {
    render(<Layout title="Home">Home content</Layout>)

    const navigation = screen.getByTestId('site-navigation')
    const searchTriggers = screen.getAllByRole('button', { name: '搜索文档' })
    const searchTrigger = searchTriggers[0]
    const progress = screen.getByRole('progressbar')
    expect(searchTriggers).toHaveLength(1)
    expect(searchTrigger.parentElement).toBe(navigation)
    expect(progress.parentElement).toBe(navigation)
    expect(searchTrigger.nextElementSibling).toBe(progress)
})
