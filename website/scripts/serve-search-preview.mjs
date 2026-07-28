import { createServer } from 'node:http'

import serveHandler from 'serve-handler'

const port = Number(process.env.PORT || 3000)
const options = {
    public: 'out',
    cleanUrls: true,
    trailingSlash: false,
}

const server = createServer(async (request, response) => {
    response.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate')

    try {
        await serveHandler(request, response, options)
    } catch (error) {
        console.error(error)
        response.statusCode = 500
        response.end('Internal Server Error')
    }
})

server.listen(port, () => {
    console.log(`Search preview available at http://localhost:${port}`)
})

let shuttingDown = false

const shutdown = (signal) => {
    if (shuttingDown) return
    shuttingDown = true

    server.close((error) => {
        if (error) {
            console.error(error)
            process.exitCode = 1
        }

        console.log(`Search preview stopped (${signal})`)
    })
}

process.once('SIGINT', () => shutdown('SIGINT'))
process.once('SIGTERM', () => shutdown('SIGTERM'))
