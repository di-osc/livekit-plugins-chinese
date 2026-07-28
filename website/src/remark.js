import Link from './components/link'
import Section, { Hr } from './components/section'
import { Table, Tr, Th, Tx, Td } from './components/table'
import Code from './components/codeDynamic'
import { TypeAnnotation } from './components/typeAnnotation'
import { InlineCode } from './components/inlineCode'
import CodeBlock, { Pre } from './components/codeBlock'
import { Ol, Ul, Li } from './components/list'
import { H2, H3, H4, H5, P, Abbr, Help, Label } from './components/typography'
import Accordion from './components/accordion'
import Infobox from './components/infobox'
import Aside from './components/aside'
import Button from './components/button'
import Tag from './components/tag'
import Grid from './components/grid'
import {
    YouTube,
    SoundCloud,
    Iframe,
    Image,
    ImageScrollable,
    GoogleSheet,
    Standalone,
} from './components/embed'

export const remarkComponents = {
    a: Link,
    p: P,
    pre: Pre,
    code: Code,
    del: TypeAnnotation,
    table: Table,
    img: Image,
    tr: Tr,
    th: Th,
    td: Td,
    ol: Ol,
    ul: Ul,
    li: Li,
    h2: H2,
    h3: H3,
    h4: H4,
    h5: H5,
    blockquote: Aside,
    section: Section,
    wrapper: ({ children }) => children,
    hr: Hr,
    H2,
    H3,
    H4,
    H5,
    Infobox,
    Table,
    Tr,
    Tx,
    Th,
    Td,
    Help,
    Button,
    YouTube,
    SoundCloud,
    Iframe,
    GoogleSheet,
    Abbr,
    Tag,
    Accordion,
    Aside,
    Grid,
    CodeBlock,
    InlineCode,
    Image,
    ImageScrollable,
    Standalone,
    Label,
}
