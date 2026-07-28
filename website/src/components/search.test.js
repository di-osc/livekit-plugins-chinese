import React from 'react'
import PropTypes from 'prop-types'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { TextEncoder } from 'util'

import Search from './search'

global.TextEncoder = TextEncoder
const { renderToString } = require('react-dom/server')

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props)
        this.state = { failed: false }
    }

    static getDerivedStateFromError() {
        return { failed: true }
    }

    render() {
        return this.state.failed ? <p>Unexpected render failure</p> : this.props.children
    }
}

ErrorBoundary.propTypes = {
    children: PropTypes.node.isRequired,
}

const createClient = overrides => ({
    search: jest.fn().mockResolvedValue([]),
    retry: jest.fn(),
    ...overrides,
})

const deferred = () => {
    let resolve
    let reject
    const promise = new Promise((resolvePromise, rejectPromise) => {
        resolve = resolvePromise
        reject = rejectPromise
    })

    return { promise, resolve, reject }
}

const renderSearch = element => {
    const root = document.createElement('div')
    root.id = '__next'
    document.body.appendChild(root)
    return { root, ...render(element, { container: root }) }
}

const openSearch = async (user, client = createClient()) => {
    renderSearch(<Search client={client} />)
    await user.click(screen.getByRole('button', { name: '搜索文档' }))
    return client
}

afterEach(() => {
    jest.restoreAllMocks()
    document.body.style.overflow = ''
    document.querySelectorAll('#__next').forEach(root => root.remove())
    document.querySelectorAll('[data-search-test-opener]').forEach(opener => opener.remove())
})

test('opens with the idle prompt, focuses the input, and Escape restores trigger focus', async () => {
    const user = userEvent.setup()
    const client = await openSearch(user)

    const trigger = screen.getByRole('button', { name: '搜索文档' })
    const dialog = screen.getByRole('dialog', { name: '搜索文档' })
    const input = screen.getByRole('combobox')

    expect(dialog).toHaveAttribute('aria-modal', 'true')
    const listbox = screen.getByRole('listbox', { name: '搜索结果' })
    const status = screen.getByRole('status')
    expect(listbox).toBeEmptyDOMElement()
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveTextContent('输入关键词搜索文档')
    expect(input).toHaveAttribute('aria-controls', listbox.id)
    expect(input).toHaveAttribute('aria-expanded', 'true')
    expect(input).toHaveAttribute('aria-autocomplete', 'list')
    expect(input).toHaveAttribute('aria-busy', 'false')
    expect(input).toHaveFocus()
    expect(client.search).not.toHaveBeenCalled()

    await user.keyboard('{Escape}')

    expect(screen.queryByRole('dialog', { name: '搜索文档' })).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
})

test('Ctrl+K opens the dialog and renders the non-Mac shortcut', async () => {
    const platform = jest.spyOn(window.navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
    const user = userEvent.setup()
    renderSearch(<Search client={createClient()} />)

    expect(screen.getByText('Ctrl K')).toBeInTheDocument()
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true, shiftKey: true })
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await user.keyboard('{Control>}k{/Control}')

    expect(screen.getByRole('dialog', { name: '搜索文档' })).toBeInTheDocument()
    expect(screen.getByRole('combobox')).toHaveFocus()
    platform.mockRestore()
})

test('server and first client markup use Ctrl before hydrating the Mac shortcut', async () => {
    const platform = jest.spyOn(window.navigator, 'platform', 'get').mockReturnValue('MacIntel')
    const client = createClient()
    const markup = renderToString(<Search client={client} />)

    expect(markup).toContain('Ctrl K')

    const container = document.createElement('div')
    container.innerHTML = markup
    document.body.appendChild(container)
    const error = jest.spyOn(console, 'error').mockImplementation(() => {})

    render(<Search client={client} />, { container, hydrate: true })

    expect(await screen.findByText('⌘ K')).toBeInTheDocument()
    expect(error).not.toHaveBeenCalled()
    fireEvent.keyDown(window, { key: 'k', metaKey: true, altKey: true })
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.getByRole('dialog', { name: '搜索文档' })).toBeInTheDocument()
})

test('multiple Search instances use unique combobox, listbox, and option IDs', async () => {
    const user = userEvent.setup()
    renderSearch(
        <>
            <Search client={createClient({
                search: jest.fn().mockResolvedValue([{ id: 'first', url: '/first', title: '第一项' }]),
            })} />
            <Search client={createClient({
                search: jest.fn().mockResolvedValue([{ id: 'second', url: '/second', title: '第二项' }]),
            })} />
        </>
    )
    const triggers = screen.getAllByRole('button', { name: '搜索文档' })

    await user.click(triggers[0])
    await user.type(screen.getByRole('combobox'), '一')
    const firstIds = {
        combobox: screen.getByRole('combobox').id,
        listbox: screen.getByRole('listbox').id,
        option: (await screen.findByRole('option', { name: '第一项' })).id,
    }
    await user.click(screen.getByRole('button', { name: '关闭搜索' }))

    await user.click(triggers[1])
    await user.type(screen.getByRole('combobox'), '二')
    const secondIds = {
        combobox: screen.getByRole('combobox').id,
        listbox: screen.getByRole('listbox').id,
        option: (await screen.findByRole('option', { name: '第二项' })).id,
    }

    expect(firstIds.combobox).not.toBe(secondIds.combobox)
    expect(firstIds.listbox).not.toBe(secondIds.listbox)
    expect(firstIds.option).not.toBe(secondIds.option)
})

test('a global shortcut restores the previously focused opener when closed', async () => {
    jest.spyOn(window.navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
    const opener = document.createElement('button')
    opener.textContent = '外部操作'
    opener.dataset.searchTestOpener = 'true'
    document.body.appendChild(opener)
    renderSearch(<Search client={createClient()} />)
    opener.focus()

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true })
    expect(screen.getByRole('dialog').parentElement.parentElement).toBe(document.body)
    await userEvent.setup().click(screen.getByRole('button', { name: '关闭搜索' }))

    expect(opener).toHaveFocus()
    opener.remove()
})

test('Tab and Shift+Tab stay trapped inside the modal', async () => {
    const user = userEvent.setup()
    await openSearch(user)
    const input = screen.getByRole('combobox')
    const close = screen.getByRole('button', { name: '关闭搜索' })

    expect(input).toHaveFocus()
    await user.tab({ shift: true })
    expect(close).toHaveFocus()
    await user.tab()
    expect(input).toHaveFocus()
})

test('opening locks the page and close or unmount restores prior page state', async () => {
    document.body.style.overflow = 'clip'
    const user = userEvent.setup()
    const { root, unmount } = renderSearch(<Search client={createClient()} />)

    await user.click(screen.getByRole('button', { name: '搜索文档' }))
    expect(root).toHaveAttribute('inert')
    expect(document.body.style.overflow).toBe('hidden')
    await user.click(screen.getByRole('button', { name: '关闭搜索' }))
    expect(root).not.toHaveAttribute('inert')
    expect(document.body.style.overflow).toBe('clip')

    root.setAttribute('inert', '')
    await user.click(screen.getByRole('button', { name: '搜索文档' }))
    unmount()
    expect(root).toHaveAttribute('inert')
    expect(document.body.style.overflow).toBe('clip')
})

test('searches and renders result links with path and highlighted excerpt', async () => {
    const client = createClient({
        search: jest.fn().mockResolvedValue([{
            id: 'guide-install',
            url: '/guide#install',
            title: '安装指南',
            section: '快速开始',
            excerpt: '运行 <mark>安装</mark> 命令',
        }]),
    })
    const user = userEvent.setup()
    await openSearch(user, client)

    await user.type(screen.getByRole('combobox'), '安装')

    const result = await screen.findByRole('option', { name: /安装指南/ })
    const input = screen.getByRole('combobox')
    const listbox = screen.getByRole('listbox')
    expect(client.search).toHaveBeenCalledWith('安装')
    expect(result).toHaveAttribute('href', '/guide#install')
    expect(result).toHaveAttribute('tabindex', '-1')
    expect(input).toHaveAttribute('aria-controls', listbox.id)
    expect(input).toHaveAttribute('aria-activedescendant', result.id)
    expect(within(listbox).getByRole('option')).toBe(result)
    expect(screen.getByText('快速开始')).toBeInTheDocument()
    expect(screen.getByText('安装', { selector: 'mark' })).toBeInTheDocument()
})

test('clicking a same-page result closes the modal and restores page state without cancelling navigation', async () => {
    document.body.style.overflow = 'clip'
    const user = userEvent.setup()
    await openSearch(user, createClient({
        search: jest.fn().mockResolvedValue([{
            id: 'same-page-section',
            url: '#same-page-section',
            title: '当前页面章节',
        }]),
    }))
    await user.type(screen.getByRole('combobox'), '章节')
    const result = await screen.findByRole('option', { name: '当前页面章节' })
    const root = document.getElementById('__next')

    const navigationAllowed = fireEvent.click(result)

    expect(navigationAllowed).toBe(true)
    expect(screen.queryByRole('dialog', { name: '搜索文档' })).not.toBeInTheDocument()
    expect(root).not.toHaveAttribute('inert')
    expect(document.body.style.overflow).toBe('clip')
})

test('shows loading and then the no-results state', async () => {
    const pending = deferred()
    const client = createClient({ search: jest.fn().mockReturnValue(pending.promise) })
    const user = userEvent.setup()
    await openSearch(user, client)

    await user.type(screen.getByRole('combobox'), 'missing')
    expect(screen.getByRole('status')).toHaveTextContent('正在搜索…')
    expect(screen.getByRole('combobox')).toHaveAttribute('aria-busy', 'true')

    await act(async () => pending.resolve([]))

    expect(await screen.findByRole('status')).toHaveTextContent('未找到相关文档')
    expect(screen.getByRole('combobox')).toHaveAttribute('aria-busy', 'false')
})

test('shows an error and retry reruns the same query', async () => {
    let shouldFail = true
    const client = createClient({
        search: jest.fn(() => shouldFail
            ? Promise.reject(new Error('missing index'))
            : Promise.resolve([{ id: 'recovered', url: '/recovered', title: '恢复成功' }])),
        retry: jest.fn(() => { shouldFail = false }),
    })
    const user = userEvent.setup()
    await openSearch(user, client)

    await user.type(screen.getByRole('combobox'), '索')

    expect(await screen.findByRole('status')).toHaveTextContent('搜索索引暂时不可用。开发环境请运行 npm run preview:search。')
    const listbox = screen.getByRole('listbox', { name: '搜索结果' })
    expect(listbox).toBeEmptyDOMElement()
    await user.click(screen.getByRole('button', { name: '重新加载' }))

    expect(client.retry).toHaveBeenCalledTimes(1)
    expect(client.search).toHaveBeenNthCalledWith(2, '索')
    expect(await screen.findByRole('option', { name: '恢复成功' })).toBeInTheDocument()
})

test('retry focuses the combobox before entering the loading state', async () => {
    const pending = deferred()
    const client = createClient({
        search: jest.fn()
            .mockRejectedValueOnce(new Error('missing index'))
            .mockReturnValueOnce(pending.promise),
    })
    const user = userEvent.setup()
    await openSearch(user, client)
    await user.type(screen.getByRole('combobox'), '索')
    const retry = await screen.findByRole('button', { name: '重新加载' })

    await user.click(retry)

    expect(screen.getByRole('combobox')).toHaveFocus()
    expect(screen.getByRole('status')).toHaveTextContent('正在搜索…')
    expect(screen.getByRole('dialog')).toContainElement(document.activeElement)
})

test('a synchronous retry exception keeps the error modal usable and focused', async () => {
    const client = createClient({
        search: jest.fn().mockRejectedValue(new Error('missing index')),
        retry: jest.fn(() => { throw new Error('retry failure') }),
    })
    const user = userEvent.setup()
    await openSearch(user, client)
    await user.type(screen.getByRole('combobox'), '索')
    const retry = await screen.findByRole('button', { name: '重新加载' })
    const suppressExpectedError = event => event.preventDefault()
    window.addEventListener('error', suppressExpectedError)

    await user.click(retry)
    window.removeEventListener('error', suppressExpectedError)

    expect(screen.getByRole('status')).toHaveTextContent('搜索索引暂时不可用')
    expect(screen.getByRole('combobox')).toHaveFocus()
    expect(screen.getByRole('dialog')).toBeInTheDocument()
})

test('results start selected and Arrow keys move and wrap selection', async () => {
    const client = createClient({
        search: jest.fn().mockResolvedValue([
            { id: 'one', url: '/one', title: '第一个结果' },
            { id: 'two', url: '/two', title: '第二个结果' },
        ]),
    })
    const user = userEvent.setup()
    await openSearch(user, client)
    const input = screen.getByRole('combobox')

    await user.type(input, '结果')
    const options = await screen.findAllByRole('option')
    const combobox = screen.getByRole('combobox')

    expect(options[0]).toHaveAttribute('aria-selected', 'true')
    expect(combobox).toHaveAttribute('aria-activedescendant', options[0].id)

    await user.keyboard('{ArrowDown}')
    expect(options[1]).toHaveAttribute('aria-selected', 'true')
    expect(combobox).toHaveAttribute('aria-activedescendant', options[1].id)

    await user.keyboard('{ArrowDown}')
    expect(options[0]).toHaveAttribute('aria-selected', 'true')

    await user.keyboard('{ArrowUp}')
    expect(options[1]).toHaveAttribute('aria-selected', 'true')
})

test('Arrow navigation scrolls the newly active option into view', async () => {
    const results = Array.from({ length: 8 }, (_, index) => ({
        id: `result-${index}`,
        url: `/result-${index}`,
        title: `结果 ${index}`,
    }))
    const user = userEvent.setup()
    await openSearch(user, createClient({ search: jest.fn().mockResolvedValue(results) }))
    await user.type(screen.getByRole('combobox'), '结果')
    const options = await screen.findAllByRole('option')
    const scrollIntoView = jest.fn()
    options[1].scrollIntoView = scrollIntoView

    await user.keyboard('{ArrowDown}')

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' })
})

test('Enter activates a same-page result after closing the modal and restoring page state', async () => {
    document.body.style.overflow = 'clip'
    const client = createClient({
        search: jest.fn().mockResolvedValue([
            { id: 'one', url: '#one', title: '第一个结果' },
            { id: 'two', url: '/two', title: '第二个结果' },
        ]),
    })
    const click = jest.spyOn(HTMLAnchorElement.prototype, 'click')
    const user = userEvent.setup()
    await openSearch(user, client)

    await user.type(screen.getByRole('combobox'), '结果')
    const options = await screen.findAllByRole('option')
    const root = document.getElementById('__next')
    await user.keyboard('{Enter}')

    expect(click).toHaveBeenCalledTimes(1)
    expect(click.mock.instances[0]).toBe(options[0])
    expect(screen.queryByRole('dialog', { name: '搜索文档' })).not.toBeInTheDocument()
    expect(root).not.toHaveAttribute('inert')
    expect(document.body.style.overflow).toBe('clip')
    click.mockRestore()
})

test('backdrop clicks close the dialog while clicks inside do not', async () => {
    const user = userEvent.setup()
    await openSearch(user)

    const trigger = screen.getByRole('button', { name: '搜索文档' })
    const dialog = screen.getByRole('dialog', { name: '搜索文档' })
    await user.click(dialog)
    expect(dialog).toBeInTheDocument()

    await user.click(dialog.parentElement)
    expect(screen.queryByRole('dialog', { name: '搜索文档' })).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
})

test('closing after resolved results resets search state before reopening', async () => {
    const client = createClient({
        search: jest.fn().mockResolvedValue([{ id: 'done', url: '/done', title: '已完成' }]),
    })
    const user = userEvent.setup()
    await openSearch(user, client)
    await user.type(screen.getByRole('combobox'), '完')
    expect(await screen.findByRole('option', { name: '已完成' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '关闭搜索' }))
    await user.click(screen.getByRole('button', { name: '搜索文档' }))

    expect(screen.getByRole('combobox')).toHaveValue('')
    expect(screen.getByRole('status')).toHaveTextContent('输入关键词搜索文档')
    expect(screen.queryByRole('option')).not.toBeInTheDocument()
    expect(client.search).toHaveBeenCalledTimes(1)
})

test('closing a pending search resets it and ignores its later result on reopen', async () => {
    const pending = deferred()
    const client = createClient({ search: jest.fn().mockReturnValue(pending.promise) })
    const user = userEvent.setup()
    await openSearch(user, client)
    await user.type(screen.getByRole('combobox'), '等')

    await user.click(screen.getByRole('button', { name: '关闭搜索' }))
    await user.click(screen.getByRole('button', { name: '搜索文档' }))
    await act(async () => pending.resolve([{ id: 'late', url: '/late', title: '迟到结果' }]))

    expect(screen.getByRole('combobox')).toHaveValue('')
    expect(screen.getByRole('status')).toHaveTextContent('输入关键词搜索文档')
    expect(screen.queryByText('迟到结果')).not.toBeInTheDocument()
    expect(client.search).toHaveBeenCalledTimes(1)
})

test('a synchronous client exception enters the error state', async () => {
    const client = createClient({ search: jest.fn(() => { throw new Error('sync failure') }) })
    jest.spyOn(console, 'error').mockImplementation(() => {})
    const user = userEvent.setup()
    renderSearch(<ErrorBoundary><Search client={client} /></ErrorBoundary>)
    await user.click(screen.getByRole('button', { name: '搜索文档' }))

    await user.type(screen.getByRole('combobox'), '错')

    expect(await screen.findByRole('status')).toHaveTextContent('搜索索引暂时不可用')
})

test('a stale result cannot replace the newer query result', async () => {
    const first = deferred()
    const second = deferred()
    const client = createClient({
        search: jest.fn(query => query === 'first' ? first.promise : second.promise),
    })
    const user = userEvent.setup()
    await openSearch(user, client)
    const input = screen.getByRole('combobox')

    await user.type(input, 'first')
    await user.clear(input)
    await user.type(input, 'second')
    await act(async () => second.resolve([{ id: 'new', url: '/new', title: 'New result' }]))
    expect(await screen.findByRole('option', { name: 'New result' })).toBeInTheDocument()

    await act(async () => first.resolve([{ id: 'old', url: '/old', title: 'Old result' }]))
    expect(screen.queryByText('Old result')).not.toBeInTheDocument()
    expect(screen.getByText('New result')).toBeInTheDocument()
})

test('a stale rejection cannot replace the newer successful query', async () => {
    const first = deferred()
    const second = deferred()
    const client = createClient({
        search: jest.fn(query => query === 'first' ? first.promise : second.promise),
    })
    const user = userEvent.setup()
    await openSearch(user, client)
    const input = screen.getByRole('combobox')

    await user.type(input, 'first')
    await user.clear(input)
    await user.type(input, 'second')
    await act(async () => second.resolve([{ id: 'new', url: '/new', title: 'New result' }]))
    expect(await screen.findByRole('option', { name: 'New result' })).toBeInTheDocument()

    await act(async () => first.reject(new Error('stale failure')))
    expect(screen.queryByText('搜索索引暂时不可用。开发环境请运行 npm run preview:search。')).not.toBeInTheDocument()
    expect(screen.getByText('New result')).toBeInTheDocument()
})

test('unmounting with a pending query does not update state', async () => {
    const pending = deferred()
    const client = createClient({ search: jest.fn().mockReturnValue(pending.promise) })
    const error = jest.spyOn(console, 'error').mockImplementation(() => {})
    const user = userEvent.setup()
    const { unmount } = renderSearch(<Search client={client} />)

    await user.click(screen.getByRole('button', { name: '搜索文档' }))
    await user.type(screen.getByRole('combobox'), 'pending')
    unmount()
    await act(async () => pending.resolve([{ id: 'late', url: '/late', title: 'Late result' }]))

    expect(error).not.toHaveBeenCalled()
    error.mockRestore()
})
