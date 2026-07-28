import React from 'react'
import PropTypes from 'prop-types'
import classNames from 'classnames'

import { isString } from './util'
import Icon from './icon'
import classes from '../styles/tag.module.sass'

export default function Tag({ spaced = false, variant, tooltip, children }) {
    if (variant === 'new') {
        const version = isString(children) && !isNaN(children) ? Number(children).toFixed(1) : children
        return <TagTemplate spaced={spaced} tooltip={tooltip}>v{version}</TagTemplate>
    }
    if (variant === 'model') {
        return <TagTemplate spaced={spaced} tooltip={tooltip}>{children}</TagTemplate>
    }
    return (
        <TagTemplate spaced={spaced} tooltip={tooltip}>
            {children}
        </TagTemplate>
    )
}

const TagTemplate = ({ spaced, tooltip, children }) => {
    const tagClassNames = classNames(classes.root, {
        [classes.spaced]: spaced,
    })
    return (
        <span className={tagClassNames} data-tooltip={tooltip}>
            {children}
            {tooltip && <Icon name="help" width={12} className={classes.icon} />}
        </span>
    )
}

Tag.propTypes = {
    spaced: PropTypes.bool,
    tooltip: PropTypes.string,
    children: PropTypes.node.isRequired,
}
