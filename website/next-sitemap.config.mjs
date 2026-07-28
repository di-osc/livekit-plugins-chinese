const deploymentUrl =
    process.env.SITE_URL ||
    process.env.VERCEL_PROJECT_PRODUCTION_URL ||
    process.env.VERCEL_URL ||
    'http://localhost:3000'
const siteUrl = deploymentUrl.startsWith('http') ? deploymentUrl : `https://${deploymentUrl}`

/** @type {import('next-sitemap').IConfig} */
const config = {
    siteUrl,
    generateRobotsTxt: true,
    autoLastmod: false,
}

export default config
