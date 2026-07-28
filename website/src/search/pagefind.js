import { basePath, normalizeBasePath, withBasePath } from '../basePath'

const importGeneratedPagefind = path => import(/* webpackIgnore: true */ path)

export const createGeneratedPagefindLoader = (
    deploymentBasePath = basePath,
    importModule = importGeneratedPagefind
) => attempt => {
    const bundleUrl = withBasePath('/pagefind/pagefind.js', deploymentBasePath)

    return importModule(attempt === 0 ? bundleUrl : `${bundleUrl}?retry=${attempt}`)
}

const cleanUrl = url => {
    if (typeof url !== 'string' || !url.trim()) {
        return null
    }

    return url.trim().replace(/\.html(?=$|[?#])/, '')
}

const normalizeResult = (result, page) => {
    const url = cleanUrl(page?.url)

    if (!url) {
        return []
    }

    const title = page?.meta?.title || url
    const subResults = Array.isArray(page?.sub_results) ? page.sub_results : []

    if (!subResults.length) {
        return [{
            id: url,
            url,
            title,
            section: null,
            excerpt: page?.excerpt,
            score: result.score,
        }]
    }

    return subResults.flatMap(subResult => {
        if (!subResult || typeof subResult !== 'object') {
            return []
        }

        const subResultUrl = cleanUrl(subResult.url)

        if (!subResultUrl) {
            return []
        }

        const isPageHeading = subResult.anchor?.element === 'h1'
        const normalizedUrl = isPageHeading ? url : subResultUrl

        return {
            id: normalizedUrl,
            url: normalizedUrl,
            title,
            section: subResult.anchor && !isPageHeading ? subResult.title : null,
            excerpt: subResult.excerpt,
            score: result.score,
        }
    })
}

export const createPagefindClient = (
    load = createGeneratedPagefindLoader(),
    deploymentBasePath = basePath
) => {
    let modulePromise = null
    let attempt = 0
    const searchBaseUrl = `${normalizeBasePath(deploymentBasePath)}/`

    const getPagefind = () => {
        if (!modulePromise) {
            const loadAttempt = attempt

            modulePromise = Promise.resolve()
                .then(() => load(loadAttempt))
                .then(module => module.default || module)
                .then(async pagefind => {
                    if (typeof pagefind.options === 'function') {
                        await pagefind.options({ baseUrl: searchBaseUrl })
                    }

                    return pagefind
                })
        }

        return modulePromise
    }

    return {
        async search(query) {
            const normalizedQuery = query.trim()

            if (!normalizedQuery) {
                return []
            }

            const pagefind = await getPagefind()
            const response = await pagefind.search(normalizedQuery)
            const pages = await Promise.all(response.results.map(result => result.data()))

            return response.results.flatMap((result, index) => normalizeResult(result, pages[index]))
        },
        retry() {
            modulePromise = null
            attempt += 1
        },
    }
}

export const pagefindClient = createPagefindClient()
