import remarkGfm from 'remark-gfm'
import remarkSmartypants from 'remark-smartypants'

import remarkCustomAttrs from './remarkCustomAttrs.mjs'
import remarkWrapSections from './remarkWrapSections.mjs'
import remarkCodeBlocks from './remarkCodeBlocks.mjs'

const remarkPlugins = [
    remarkGfm,
    remarkSmartypants,
    remarkCustomAttrs,
    remarkCodeBlocks,
    remarkWrapSections,
]

export default remarkPlugins
