import { readFile } from 'node:fs/promises'
import type { Element, ElementContent, Root } from 'hast'
import type { Plugin } from 'unified'
import { visit } from 'unist-util-visit'

const isBr = (n: ElementContent) => n.type === 'element' && (n as Element).tagName === 'br'
const isImg = (n: ElementContent) => n.type === 'element' && (n as Element).tagName === 'img'

function readPngDimensions(buf: Buffer): { width: number; height: number } | null {
  // PNG: 8-byte signature, then IHDR chunk: 4-len 4-type 4-width 4-height
  if (buf.length < 24) return null
  if (buf.readUInt32BE(0) !== 0x89504e47) return null // PNG signature check
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) }
}

const rehypeImageSize: Plugin<[], Root> = function () {
  return async (tree: Root) => {
    const tasks: Promise<void>[] = []

    visit(tree, 'element', (node: Element) => {
      if (node.tagName !== 'img') return
      const src = node.properties?.src as string | undefined
      if (!src?.startsWith('/notes/')) return

      tasks.push(
        (async () => {
          try {
            const buf = await readFile(`./public${src}`)
            const dims = readPngDimensions(buf)
            if (!dims) return
            const ar = dims.width / dims.height
            const existing = (node.properties.style as string | undefined) ?? ''
            node.properties.style = existing ? `${existing};--aspect-ratio:${ar}` : `--aspect-ratio:${ar}`
          } catch {
            // Image not found or unreadable — leave margin at CSS default
          }
        })(),
      )
    })

    await Promise.all(tasks)

    // Strip <br> nodes directly adjacent to <img> inside <p> elements.
    // These come from remarkBreaks converting the newline after an image embed.
    visit(tree, 'element', (node: Element) => {
      if (node.tagName !== 'p') return
      node.children = node.children.filter((child, i) => {
        if (!isBr(child)) return true
        return !isImg(node.children[i - 1]) && !isImg(node.children[i + 1])
      })
    })
  }
}

export default rehypeImageSize
