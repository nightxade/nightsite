import type { Element, ElementContent, Root, Text } from 'hast'
import type { Plugin } from 'unified'
import { visit } from 'unist-util-visit'

const CALLOUT_RE = /^\[!([\w-]+)\]([+-]?)\s*/

// Map callout type to an accent CSS variable suffix
const TYPE_STYLES: Record<string, string> = {
  info: 'primary',
  tip: 'primary',
  abstract: 'primary',
  note: 'primary',
  question: 'reading-paper',
  warning: 'destructive',
  danger: 'destructive',
  failure: 'destructive',
  bug: 'destructive',
  success: 'reading-article',
  check: 'reading-article',
  done: 'reading-article',
}

function isBr(node: ElementContent): boolean {
  return node.type === 'element' && (node as Element).tagName === 'br'
}

function isEmptyText(node: ElementContent): boolean {
  return node.type === 'text' && (node as Text).value.trim() === ''
}

const rehypeCallouts: Plugin<[], Root> = function () {
  return (tree: Root) => {
    visit(
      tree,
      'element',
      (node: Element, index: number | undefined, parent: Element | Root | undefined) => {
        if (node.tagName !== 'blockquote') return
        if (!parent || index === undefined) return

        // Find first <p> element
        const firstP = node.children.find(
          (c): c is Element => c.type === 'element' && (c as Element).tagName === 'p',
        )
        if (!firstP) return

        // Find first text node in the <p>
        const firstText = firstP.children.find((c): c is Text => c.type === 'text')
        if (!firstText) return

        const match = firstText.value.match(CALLOUT_RE)
        if (!match) return

        const [fullMatch, type] = match
        const accentVar = TYPE_STYLES[type.toLowerCase()] ?? 'primary'

        // Strip the [!type] prefix from the first text node
        firstText.value = firstText.value.slice(fullMatch.length)

        // Split firstP.children at the first <br> to separate title from body.
        // Adjacent blockquote lines (title + body on consecutive lines) are joined
        // into one <p> by the markdown parser, with remarkBreaks inserting a <br>.
        const brIdx = firstP.children.findIndex(isBr)

        let titleChildren: ElementContent[]
        let bodyFirstPChildren: ElementContent[] | null = null

        if (brIdx !== -1) {
          titleChildren = firstP.children.slice(0, brIdx)
          const after = firstP.children.slice(brIdx + 1)
          // Drop any leading whitespace/breaks from the body portion
          while (after.length > 0 && (isBr(after[0]) || isEmptyText(after[0]))) after.shift()
          if (after.length > 0) bodyFirstPChildren = after
        } else {
          // No <br>: entire firstP is the title (no inline body content)
          titleChildren = firstP.children.slice()
        }

        // Clean up empty text nodes at the edges of the title
        while (titleChildren.length > 0 && isEmptyText(titleChildren[0])) titleChildren.shift()
        while (titleChildren.length > 0 && isEmptyText(titleChildren[titleChildren.length - 1]))
          titleChildren.pop()

        // Fall back to the type name if the title has no content
        if (titleChildren.length === 0) {
          titleChildren = [{ type: 'text', value: type }]
        }

        // Build body children: optional inline body from the title line, then remaining blockquote children
        const bodyChildren: ElementContent[] = []
        if (bodyFirstPChildren) {
          bodyChildren.push({
            type: 'element',
            tagName: 'p',
            properties: {},
            children: bodyFirstPChildren,
          } as Element)
        }
        // Add remaining blockquote children (skip the firstP we already processed)
        for (const child of node.children) {
          if (child !== firstP) bodyChildren.push(child)
        }

        const calloutDiv: Element = {
          type: 'element',
          tagName: 'div',
          properties: {
            className: ['callout', `callout-${type.toLowerCase()}`],
            style: `--callout-accent: var(--${accentVar})`,
          },
          children: [
            {
              type: 'element',
              tagName: 'div',
              properties: { className: ['callout-title'] },
              children: titleChildren,
            },
            {
              type: 'element',
              tagName: 'div',
              properties: { className: ['callout-body', 'not-prose'] },
              children: bodyChildren,
            },
          ],
        }

        ;(parent as Element).children[index] = calloutDiv
      },
    )
  }
}

export default rehypeCallouts
