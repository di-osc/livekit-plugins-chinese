import { normalizeBasePath, withBasePath, withBasePathForUrl } from './basePath'

test.each([
    [undefined, ''],
    ['', ''],
    ['/', ''],
    ['libraries', '/libraries'],
    ['/libraries/', '/libraries'],
    [' /team/docs/ ', '/team/docs'],
])('normalizes a deployment base path from %p', (value, expected) => {
    expect(normalizeBasePath(value)).toBe(expected)
})

test('prefixes root-relative public files without changing the root deployment', () => {
    expect(withBasePath('/manifest.webmanifest', '')).toBe('/manifest.webmanifest')
    expect(withBasePath('/manifest.webmanifest', '/libraries')).toBe(
        '/libraries/manifest.webmanifest'
    )
})

test.each([
    ['/images/example.png', '/libraries', '/libraries/images/example.png'],
    ['/images/example.png', '', '/images/example.png'],
    ['/libraries/images/example.png', '/libraries', '/libraries/images/example.png'],
    ['images/example.png', '/libraries', 'images/example.png'],
    ['https://example.com/image.png', '/libraries', 'https://example.com/image.png'],
    ['//cdn.example.com/image.png', '/libraries', '//cdn.example.com/image.png'],
    ['data:image/svg+xml;base64,PHN2Zz4=', '/libraries', 'data:image/svg+xml;base64,PHN2Zz4='],
])('only prefixes local root-relative URLs: %p', (value, deploymentBasePath, expected) => {
    expect(withBasePathForUrl(value, deploymentBasePath)).toBe(expected)
})
