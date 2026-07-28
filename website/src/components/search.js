import React, { useCallback, useEffect, useId, useRef, useState } from 'react'
import { createPortal, flushSync } from 'react-dom'
import PropTypes from 'prop-types'
import classNames from 'classnames'

import { pagefindClient } from '../search/pagefind'
import classes from '../styles/search.module.sass'

const focusableSelector = [
    'input:not([disabled])',
    'button:not([disabled])',
    'a[href]:not([tabindex="-1"])',
    '[tabindex]:not([tabindex="-1"])',
].join(',')

export default function Search({ client }) {
    const componentId = useId()
    const searchInputId = `${componentId}-input`
    const searchResultsId = `${componentId}-results`
    const optionId = index => `${searchResultsId}-option-${index}`
    const [isOpen, setIsOpen] = useState(false)
    const [query, setQuery] = useState('')
    const [results, setResults] = useState([])
    const [status, setStatus] = useState('idle')
    const [selectedIndex, setSelectedIndex] = useState(0)
    const [retryVersion, setRetryVersion] = useState(0)
    const [isMac, setIsMac] = useState(false)
    const requestId = useRef(0)
    const triggerRef = useRef(null)
    const inputRef = useRef(null)
    const dialogRef = useRef(null)
    const resultRefs = useRef([])
    const openerRef = useRef(null)
    const mountedRef = useRef(true)

    const open = useCallback(() => {
        if (!isOpen) {
            openerRef.current = document.activeElement
        }
        setIsOpen(true)
    }, [isOpen])
    const close = useCallback(() => {
        requestId.current += 1
        setQuery('')
        setResults([])
        setStatus('idle')
        setSelectedIndex(0)
        setIsOpen(false)
    }, [])

    useEffect(() => {
        mountedRef.current = true
        return () => {
            mountedRef.current = false
            requestId.current += 1
        }
    }, [])

    useEffect(() => {
        if (/Mac|iPhone|iPad|iPod/.test(navigator.platform)) {
            setIsMac(true)
        }
    }, [])

    useEffect(() => {
        const handleGlobalKeyDown = event => {
            const hasExactModifier = isMac
                ? event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey
                : event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey

            if (hasExactModifier && event.key.toLowerCase() === 'k') {
                event.preventDefault()
                open()
            } else if (event.key === 'Escape' && isOpen) {
                event.preventDefault()
                close()
            }
        }

        window.addEventListener('keydown', handleGlobalKeyDown)
        return () => window.removeEventListener('keydown', handleGlobalKeyDown)
    }, [close, isMac, isOpen, open])

    useEffect(() => {
        if (!isOpen) {
            return undefined
        }

        const appRoot = document.getElementById('__next')
        const fallbackTrigger = triggerRef.current
        const hadInertAttribute = appRoot?.hasAttribute('inert')
        const previousInert = appRoot && 'inert' in appRoot ? appRoot.inert : undefined
        const previousOverflow = document.body.style.overflow

        document.body.style.overflow = 'hidden'
        if (appRoot) {
            appRoot.setAttribute('inert', '')
            if ('inert' in appRoot) {
                appRoot.inert = true
            }
        }
        inputRef.current?.focus()

        return () => {
            document.body.style.overflow = previousOverflow
            if (appRoot) {
                if (hadInertAttribute) {
                    appRoot.setAttribute('inert', '')
                } else {
                    appRoot.removeAttribute('inert')
                }
                if ('inert' in appRoot) {
                    appRoot.inert = previousInert
                }
            }

            const savedOpener = openerRef.current
            const canFocusOpener = savedOpener?.isConnected && typeof savedOpener.focus === 'function'
            const focusTarget = canFocusOpener ? savedOpener : fallbackTrigger
            if (focusTarget?.isConnected && typeof focusTarget.focus === 'function') {
                focusTarget.focus()
            }
        }
    }, [isOpen])

    useEffect(() => {
        if (!isOpen) {
            return undefined
        }

        const normalizedQuery = query.trim()
        const currentRequest = requestId.current + 1
        requestId.current = currentRequest

        if (!normalizedQuery) {
            setStatus('idle')
            setResults([])
            setSelectedIndex(0)
            return undefined
        }

        let cancelled = false
        setStatus('loading')
        setResults([])
        setSelectedIndex(0)

        Promise.resolve().then(() => client.search(normalizedQuery)).then(nextResults => {
            if (cancelled || !mountedRef.current || requestId.current !== currentRequest) {
                return
            }

            const normalizedResults = Array.isArray(nextResults) ? nextResults : []
            setResults(normalizedResults)
            setSelectedIndex(0)
            setStatus('success')
        }).catch(() => {
            if (cancelled || !mountedRef.current || requestId.current !== currentRequest) {
                return
            }

            setResults([])
            setSelectedIndex(0)
            setStatus('error')
        })

        return () => {
            cancelled = true
        }
    }, [client, isOpen, query, retryVersion])

    useEffect(() => {
        resultRefs.current = resultRefs.current.slice(0, results.length)
        setSelectedIndex(index => {
            if (!results.length) {
                return 0
            }
            return Math.min(Math.max(index, 0), results.length - 1)
        })
    }, [results])

    useEffect(() => {
        if (!results.length) {
            return
        }

        resultRefs.current[selectedIndex]?.scrollIntoView?.({ block: 'nearest' })
    }, [results, selectedIndex])

    const handleRetry = () => {
        inputRef.current?.focus()
        try {
            client.retry()
        } catch {
            setStatus('error')
            return
        }
        setRetryVersion(version => version + 1)
    }

    const handleInputKeyDown = event => {
        if (!results.length) {
            return
        }

        if (event.key === 'ArrowDown') {
            event.preventDefault()
            setSelectedIndex(index => index < 0 ? 0 : (index + 1) % results.length)
        } else if (event.key === 'ArrowUp') {
            event.preventDefault()
            setSelectedIndex(index => index < 0 ? results.length - 1 : (index - 1 + results.length) % results.length)
        } else if (event.key === 'Enter' && selectedIndex >= 0) {
            event.preventDefault()
            resultRefs.current[selectedIndex]?.click()
        }
    }

    const handleDialogKeyDown = event => {
        if (event.key !== 'Tab') {
            return
        }

        const focusableElements = Array.from(dialogRef.current?.querySelectorAll(focusableSelector) || [])
        if (!focusableElements.length) {
            event.preventDefault()
            return
        }

        const first = focusableElements[0]
        const last = focusableElements[focusableElements.length - 1]
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault()
            last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault()
            first.focus()
        } else if (!dialogRef.current?.contains(document.activeElement)) {
            event.preventDefault()
            first.focus()
        }
    }

    const activeOptionId = status === 'success' && results.length
        ? optionId(selectedIndex)
        : undefined

    const modal = (
        <div
            className={classes.backdrop}
            onPointerDown={event => {
                if (event.target === event.currentTarget) {
                    event.preventDefault()
                    close()
                }
            }}
        >
            <section
                ref={dialogRef}
                className={classes.dialog}
                role="dialog"
                aria-modal="true"
                aria-label="搜索文档"
                onKeyDown={handleDialogKeyDown}
            >
                <div className={classes['search-row']}>
                    <svg aria-hidden="true" viewBox="0 0 24 24" width="20" height="20">
                        <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2" />
                        <path d="m16 16 5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                    <input
                        ref={inputRef}
                        id={searchInputId}
                        className={classes.input}
                        type="search"
                        role="combobox"
                        aria-label="搜索关键词"
                        aria-controls={searchResultsId}
                        aria-expanded="true"
                        aria-activedescendant={activeOptionId}
                        aria-autocomplete="list"
                        aria-busy={status === 'loading'}
                        placeholder="搜索文档"
                        value={query}
                        onChange={event => setQuery(event.target.value)}
                        onKeyDown={handleInputKeyDown}
                    />
                    <button
                        type="button"
                        className={classes.close}
                        aria-label="关闭搜索"
                        onClick={close}
                    >
                        <span aria-hidden="true">×</span>
                    </button>
                </div>

                <div className={classes.results} id={searchResultsId} role="listbox" aria-label="搜索结果">
                    {status === 'success' && results.map((result, index) => (
                        <a
                            key={result.id || result.url}
                            id={optionId(index)}
                            ref={element => { resultRefs.current[index] = element }}
                            className={classNames(classes.result, {
                                [classes['is-selected']]: index === selectedIndex,
                            })}
                            role="option"
                            aria-selected={index === selectedIndex}
                            href={result.url}
                            tabIndex={-1}
                            onClick={() => flushSync(close)}
                            onMouseEnter={() => setSelectedIndex(index)}
                        >
                            <strong>{result.title}</strong>
                            {result.section && <span className={classes['result-path']}>{result.section}</span>}
                            {result.excerpt && (
                                <span
                                    className={classes.excerpt}
                                    dangerouslySetInnerHTML={{ __html: result.excerpt }}
                                />
                            )}
                        </a>
                    ))}
                </div>

                <div className={classes.state} role="status" aria-live="polite">
                    {status === 'idle' && <p>输入关键词搜索文档</p>}
                    {status === 'loading' && <p>正在搜索…</p>}
                    {status === 'success' && !results.length && <p>未找到相关文档</p>}
                    {status === 'error' && (
                        <>
                            <p>搜索索引暂时不可用。开发环境请运行 npm run preview:search。</p>
                            <button type="button" className={classes.retry} onClick={handleRetry}>重新加载</button>
                        </>
                    )}
                </div>

                <footer className={classes.footer}>↑↓ 选择 · Enter 打开 · Esc 关闭</footer>
            </section>
        </div>
    )

    return (
        <>
            <button
                ref={triggerRef}
                type="button"
                className={classes.trigger}
                aria-label="搜索文档"
                onClick={open}
            >
                <svg aria-hidden="true" viewBox="0 0 24 24" width="20" height="20">
                    <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2" />
                    <path d="m16 16 5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                <span className={classes['trigger-label']}>搜索文档</span>
                <span className={classes.shortcut} aria-hidden="true">{isMac ? '⌘ K' : 'Ctrl K'}</span>
            </button>

            {isOpen && createPortal(modal, document.body)}
        </>
    )
}

Search.propTypes = {
    client: PropTypes.shape({
        search: PropTypes.func.isRequired,
        retry: PropTypes.func.isRequired,
    }),
}

Search.defaultProps = {
    client: pagefindClient,
}
