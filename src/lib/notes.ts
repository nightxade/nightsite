import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import rehypeCallouts from '@/lib/rehype-callouts'
import rehypeImageSize from '@/lib/rehype-image-size'
import remarkObsidian from '@/lib/remark-obsidian'
import type { Root } from 'mdast'
import rehypeKatex from 'rehype-katex'
import rehypeStringify from 'rehype-stringify'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import type { Plugin } from 'unified'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'

// Treat single newlines as hard line breaks (matches Obsidian default behavior)
const remarkBreaks: Plugin<[], Root> = () => (tree) => {
  visit(tree, 'text', (node, index, parent) => {
    if (!node.value.includes('\n') || !parent || index === undefined) return
    const parts = node.value.split('\n')
    const nodes: (typeof node | { type: 'break' })[] = []
    for (let i = 0; i < parts.length; i++) {
      if (parts[i]) nodes.push({ type: 'text', value: parts[i] })
      if (i < parts.length - 1) nodes.push({ type: 'break' })
    }
    parent.children.splice(index, 1, ...(nodes as any))
    return index + nodes.length
  })
}

const NOTES_DIR = './src/content/notes'

export interface CollectionMeta {
  name: string
  description: string
  about?: string
}

export const COLLECTION_META: Record<string, CollectionMeta> = {
  cs162: {
    name: 'CS 162',
    description: 'Operating Systems and System Programming',
    about: 'Textbook: "Operating Systems: Principles and Practice" — Dahlin & Anderson'
  },
  cs170: {
    name: 'CS 170',
    description: 'Efficient Algorithms and Intractable Problems',
    about: 'Textbook: "Algorithms" — Papadimitriou, Dasgupta, & Vazirani'
  },
  cs185: {
    name: 'CS 185',
    description: 'Deep Reinforcement Learning',
  },
  deeplearningbook: {
    name: 'Deep Learning',
    description: 'Goodfellow, Bengio & Courville',
  },
  reinforcementlearningbook: {
    name: 'Reinforcement Learning',
    description: 'Sutton & Barto',
  },
  math113: {
    name: 'MATH 113',
    description: 'Abstract Algebra',
    about: 'Textbook: "Abstract Algebra" — Dummit & Foote'
  },
  math53: {
    name: 'MATH 53',
    description: 'Multivariable Calculus',
    about: 'Textbook: "Calculus, 8th Edition" — James Stewart'
  },
}

export interface NoteInfo {
  slug: string
  filename: string
  collection: string
  title: string
  path: string
}

export function toSlug(filename: string): string {
  return filename
    .replace(/\.md$/, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

export async function getNoteCollections(): Promise<string[]> {
  return Object.keys(COLLECTION_META)
}

function extractTitle(content: string): string {
  const withoutFrontmatter = content.replace(/^---[\s\S]*?---\n/, '')
  const match = withoutFrontmatter.match(/^#\s+(.+)$/m)
  if (!match) return ''
  return match[1].replace(/%%.+?%%/g, '').trim()
}

function naturalSort(a: string, b: string): number {
  const numA = parseInt(a.match(/(\d+)/)?.[1] ?? '', 10)
  const numB = parseInt(b.match(/(\d+)/)?.[1] ?? '', 10)
  if (!isNaN(numA) && !isNaN(numB) && numA !== numB) return numA - numB
  return a.localeCompare(b)
}

export async function getNotesForCollection(collection: string): Promise<NoteInfo[]> {
  const dir = join(NOTES_DIR, collection)
  const entries = await readdir(dir, { withFileTypes: true })

  const mdFiles = entries
    .filter((e) => e.isFile() && e.name.endsWith('.md'))
    .filter((e) => e.name !== 'notes-export.md')
    .map((e) => e.name)

  const hasIndividualNotes = mdFiles.some((f) => f !== 'notes.md')
  const filesToUse = hasIndividualNotes ? mdFiles.filter((f) => f !== 'notes.md') : mdFiles

  const notes = await Promise.all(
    filesToUse.map(async (filename) => {
      const path = join(dir, filename)
      const content = await readFile(path, 'utf-8')
      const fileSlug = filename.replace(/\.md$/, '')
      return {
        slug: toSlug(fileSlug),
        filename: fileSlug,
        collection,
        title: extractTitle(content) || fileSlug,
        path,
      }
    }),
  )

  notes.sort((a, b) => naturalSort(a.filename, b.filename))
  return notes
}

export async function processNote(filePath: string, collection: string): Promise<string> {
  const content = await readFile(filePath, 'utf-8')

  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkObsidian, { collection })
    .use(remarkBreaks)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeCallouts)
    .use(rehypeImageSize)
    .use(rehypeKatex)
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(content)

  return String(result)
}
