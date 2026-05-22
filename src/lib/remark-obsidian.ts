import type { Root } from 'mdast'
import { findAndReplace } from 'mdast-util-find-and-replace'
import type { Plugin } from 'unified'
import { visit } from 'unist-util-visit'

interface Options {
  collection: string
}

const remarkObsidian: Plugin<[Options], Root> = function ({ collection }) {
  return (tree: Root) => {
    // Strip %% inline Obsidian comments (e.g. "%% fold %%")
    findAndReplace(tree, [[/%%.+?%%/g, '']])

    // Image embeds: ![[filename.png|750]] or ![[filename.png]]
    findAndReplace(tree, [
      [
        /!\[\[([^\]|]+?)(?:\|(\d+))?\]\]/g,
        (_: string, filename: string, size: string | undefined) => ({
          type: 'image' as const,
          url: `/notes/${collection}/${filename.trim()}`,
          alt: filename.trim(),
          data: size
            ? { hProperties: { style: `max-width:${size}px;width:100%` } }
            : { hProperties: { style: 'max-width:100%' } },
        }),
      ],
    ])

    // Wiki-links with heading anchor: [[link#heading|text]] or [[link#heading]]
    findAndReplace(tree, [
      [
        /\[\[([^\]|#]+)#([^\]|]+?)(?:\|([^\]]+?))?\]\]/g,
        (_: string, _link: string, heading: string, text: string | undefined) => ({
          type: 'text' as const,
          value: text || heading,
        }),
      ],
    ])

    // Plain wiki-links: [[link|text]] or [[link]]
    findAndReplace(tree, [
      [
        /\[\[([^\]|#]+?)(?:\|([^\]]+?))?\]\]/g,
        (_: string, link: string, text: string | undefined) => ({
          type: 'text' as const,
          value: text || link,
        }),
      ],
    ])

    // Transform ```tikz code blocks into <script type="text/tikz"> for TikZJax
    visit(tree, 'code', (node, index, parent) => {
      if (node.lang !== 'tikz' || !parent || index === null || index === undefined) return
      // Strip \begin{document}/\end{document} wrappers Obsidian plugin adds
      const content = node.value
        .replace(/\\begin\{document\}/g, '')
        .replace(/\\end\{document\}/g, '')
        .trim()
      ;(parent.children as any)[index] = {
        type: 'html',
        value: `<script type="text/tikz">\n${content}\n</script>`,
      }
    })

    // Strip page-break divs so they don't clutter the rendered output
    visit(tree, 'html', (node, index, parent) => {
      if (
        node.value.includes('page-break') &&
        parent &&
        index !== null &&
        index !== undefined
      ) {
        parent.children.splice(index, 1)
        return index
      }
    })
  }
}

export default remarkObsidian
