export const normalizeBasePath = (value) => {
    const path = typeof value === 'string' ? value.trim() : ''

    if (!path || path === '/') {
        return ''
    }

    return `/${path.replace(/^\/+|\/+$/g, '')}`
}

export const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH)

export const withBasePath = (path, deploymentBasePath = basePath) => {
    const normalizedPath = path.startsWith('/') ? path : `/${path}`

    return `${normalizeBasePath(deploymentBasePath)}${normalizedPath}`
}

export const withBasePathForUrl = (value, deploymentBasePath = basePath) => {
    if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//')) {
        return value
    }

    const normalizedBasePath = normalizeBasePath(deploymentBasePath)
    if (
        normalizedBasePath &&
        (value === normalizedBasePath || value.startsWith(`${normalizedBasePath}/`))
    ) {
        return value
    }

    return `${normalizedBasePath}${value}`
}
