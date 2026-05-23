import { getCollection, render, type CollectionEntry } from 'astro:content'
import { readingTime, calculateWordCountFromHtml } from '@/lib/utils'

export async function getAllAuthors(): Promise<never[]> {
  return []
}

async function safeBlogCollection(): Promise<CollectionEntry<'blog'>[]> {
  try {
    return await getCollection('blog')
  } catch {
    return []
  }
}

export async function getAllPosts(): Promise<
  (CollectionEntry<'blog'> | CollectionEntry<'writeups'>)[]
> {
  const [blogPosts, writeupPosts] = await Promise.all([
    safeBlogCollection(),
    getCollection('writeups'),
  ])
  return [...blogPosts, ...writeupPosts]
    .filter((post) => !post.data.draft && !isSubpost(post.id))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
}

export async function getAllPostsAndSubposts(): Promise<
  CollectionEntry<'blog'>[]
> {
  const posts = await safeBlogCollection()
  return posts
    .filter((post) => !post.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
}

export async function getAllTags(): Promise<Map<string, number>> {
  const [blogPosts, writeups] = await Promise.all([
    safeBlogCollection(),
    getAllWriteups(),
  ])
  const acc = new Map<string, number>()

  // Blog post frontmatter tags
  blogPosts
    .filter((p) => !p.data.draft && !isSubpost(p.id))
    .forEach((post) =>
      post.data.tags?.forEach((tag) => acc.set(tag, (acc.get(tag) ?? 0) + 1)),
    )

  // Writeup virtual tags (derived from subposts)
  for (const writeup of writeups) {
    const tags = await getSubwriteupTags(writeup.id)
    tags.forEach((tag) => acc.set(tag, (acc.get(tag) ?? 0) + 1))
  }

  return acc
}

export async function getAdjacentPosts(currentId: string): Promise<{
  newer: CollectionEntry<'blog'> | null
  older: CollectionEntry<'blog'> | null
  parent: CollectionEntry<'blog'> | null
}> {
  const blogPosts = await getCollection('blog').then((posts) =>
    posts
      .filter((post) => !post.data.draft && !isSubpost(post.id))
      .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf()),
  )

  if (isSubpost(currentId)) {
    const parentId = getParentId(currentId)
    const parent = blogPosts.find((post) => post.id === parentId) || null

    const posts = await getCollection('blog')
    const subposts = posts
      .filter(
        (post) =>
          isSubpost(post.id) &&
          getParentId(post.id) === parentId &&
          !post.data.draft,
      )
      .sort((a, b) => {
        const dateDiff = a.data.date.valueOf() - b.data.date.valueOf()
        if (dateDiff !== 0) return dateDiff

        const orderA = a.data.order ?? 0
        const orderB = b.data.order ?? 0
        return orderA - orderB
      })

    const currentIndex = subposts.findIndex((post) => post.id === currentId)
    if (currentIndex === -1) {
      return { newer: null, older: null, parent }
    }

    return {
      newer:
        currentIndex < subposts.length - 1 ? subposts[currentIndex + 1] : null,
      older: currentIndex > 0 ? subposts[currentIndex - 1] : null,
      parent,
    }
  }

  const parentPosts = blogPosts.filter((post) => !isSubpost(post.id))
  const currentIndex = parentPosts.findIndex((post) => post.id === currentId)

  if (currentIndex === -1) {
    return { newer: null, older: null, parent: null }
  }

  return {
    newer: currentIndex > 0 ? parentPosts[currentIndex - 1] : null,
    older:
      currentIndex < parentPosts.length - 1
        ? parentPosts[currentIndex + 1]
        : null,
    parent: null,
  }
}

export async function getPostsByTag(
  tag: string,
): Promise<(CollectionEntry<'blog'> | CollectionEntry<'writeups'>)[]> {
  const [blogPosts, writeups] = await Promise.all([
    safeBlogCollection(),
    getAllWriteups(),
  ])

  const results: (CollectionEntry<'blog'> | CollectionEntry<'writeups'>)[] = []

  // Blog posts with this tag in frontmatter
  blogPosts
    .filter((p) => !p.data.draft && !isSubpost(p.id) && p.data.tags?.includes(tag))
    .forEach((p) => results.push(p))

  // Writeup parents whose subposts include this tag
  for (const writeup of writeups) {
    const tags = await getSubwriteupTags(writeup.id)
    if (tags.includes(tag)) results.push(writeup)
  }

  return results.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
}

export async function getRecentPosts(
  count: number,
): Promise<(CollectionEntry<'blog'> | CollectionEntry<'writeups'>)[]> {
  const posts = await getAllPosts()
  return posts.slice(0, count)
}

export async function getSortedTags(): Promise<
  { tag: string; count: number }[]
> {
  const tagCounts = await getAllTags()
  return [...tagCounts.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => {
      const countDiff = b.count - a.count
      return countDiff !== 0 ? countDiff : a.tag.localeCompare(b.tag)
    })
}

export function getParentId(subpostId: string): string {
  return subpostId.split('/')[0]
}

export async function getSubpostsForParent(
  parentId: string,
): Promise<CollectionEntry<'blog'>[]> {
  const posts = await getCollection('blog')
  return posts
    .filter(
      (post) =>
        !post.data.draft &&
        isSubpost(post.id) &&
        getParentId(post.id) === parentId,
    )
    .sort((a, b) => {
      const dateDiff = a.data.date.valueOf() - b.data.date.valueOf()
      if (dateDiff !== 0) return dateDiff

      const orderA = a.data.order ?? 0
      const orderB = b.data.order ?? 0
      return orderA - orderB
    })
}

export function groupPostsByYear(
  posts: CollectionEntry<'blog'>[],
): Record<string, CollectionEntry<'blog'>[]> {
  return posts.reduce(
    (acc: Record<string, CollectionEntry<'blog'>[]>, post) => {
      const year = post.data.date.getFullYear().toString()
      ;(acc[year] ??= []).push(post)
      return acc
    },
    {},
  )
}

export async function hasSubposts(postId: string): Promise<boolean> {
  const subposts = await getSubpostsForParent(postId)
  return subposts.length > 0
}

export function isSubpost(postId: string): boolean {
  return postId.includes('/')
}

export async function getParentPost(
  subpostId: string,
): Promise<CollectionEntry<'blog'> | null> {
  if (!isSubpost(subpostId)) {
    return null
  }

  const parentId = getParentId(subpostId)
  const allPosts = await getCollection('blog')
  return allPosts.find((post) => post.id === parentId) || null
}

export async function parseAuthors(authorIds: string[] = []) {
  if (!authorIds.length) return []

  const allAuthors = await getAllAuthors()
  const authorMap = new Map(allAuthors.map((author) => [author.id, author]))

  return authorIds.map((id) => {
    const author = authorMap.get(id)
    return {
      id,
      name: author?.data?.name || id,
      avatar: author?.data?.avatar || '/static/logo.png',
      isRegistered: !!author,
    }
  })
}

export async function getPostById(
  postId: string,
): Promise<CollectionEntry<'blog'> | null> {
  const allPosts = await getAllPostsAndSubposts()
  return allPosts.find((post) => post.id === postId) || null
}

export async function getSubpostCount(parentId: string): Promise<number> {
  const subposts = await getSubpostsForParent(parentId)
  return subposts.length
}

export async function getCombinedReadingTime(postId: string): Promise<string> {
  const post = await getPostById(postId)
  if (!post) return readingTime(0)

  let totalWords = calculateWordCountFromHtml(post.body)

  if (!isSubpost(postId)) {
    const subposts = await getSubpostsForParent(postId)
    for (const subpost of subposts) {
      totalWords += calculateWordCountFromHtml(subpost.body)
    }
  }

  return readingTime(totalWords)
}

export async function getPostReadingTime(postId: string): Promise<string> {
  const post = await getPostById(postId)
  if (!post) return readingTime(0)

  const wordCount = calculateWordCountFromHtml(post.body)
  return readingTime(wordCount)
}

export type TOCHeading = {
  slug: string
  text: string
  depth: number
  isSubpostTitle?: boolean
}

export type TOCSection = {
  type: 'parent' | 'subpost'
  title: string
  headings: TOCHeading[]
  subpostId?: string
}

export async function getAllProjects(): Promise<CollectionEntry<'projects'>[]> {
  try {
    const projects = await getCollection('projects')
    return projects
      .filter((p) => !p.data.draft)
      .sort((a, b) => {
        if (a.data.pinned !== b.data.pinned) return a.data.pinned ? -1 : 1
        return b.data.startDate.valueOf() - a.data.startDate.valueOf()
      })
  } catch {
    return []
  }
}


export async function getAllWriteups(): Promise<CollectionEntry<'writeups'>[]> {
  const writeups = await getCollection('writeups')
  return writeups
    .filter((w) => !w.data.draft && !isSubpost(w.id))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
}

export async function getAllWriteupsAndSubwriteups(): Promise<
  CollectionEntry<'writeups'>[]
> {
  const writeups = await getCollection('writeups')
  return writeups
    .filter((w) => !w.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
}

export function groupWriteupsByYear(
  writeups: CollectionEntry<'writeups'>[],
): Record<string, CollectionEntry<'writeups'>[]> {
  return writeups.reduce(
    (acc: Record<string, CollectionEntry<'writeups'>[]>, w) => {
      const year = w.data.date.getFullYear().toString()
      ;(acc[year] ??= []).push(w)
      return acc
    },
    {},
  )
}

export async function getSubwriteupsForParent(
  parentId: string,
): Promise<CollectionEntry<'writeups'>[]> {
  const writeups = await getCollection('writeups')
  return writeups
    .filter(
      (w) =>
        !w.data.draft && isSubpost(w.id) && getParentId(w.id) === parentId,
    )
    .sort((a, b) => {
      const dateDiff = a.data.date.valueOf() - b.data.date.valueOf()
      if (dateDiff !== 0) return dateDiff
      return (a.data.order ?? 0) - (b.data.order ?? 0)
    })
}

export async function getWriteupById(
  writeupId: string,
): Promise<CollectionEntry<'writeups'> | null> {
  const all = await getAllWriteupsAndSubwriteups()
  return all.find((w) => w.id === writeupId) || null
}

export async function hasWriteupSubposts(writeupId: string): Promise<boolean> {
  return (await getSubwriteupsForParent(writeupId)).length > 0
}

export async function getParentWriteup(
  subwriteupId: string,
): Promise<CollectionEntry<'writeups'> | null> {
  if (!isSubpost(subwriteupId)) return null
  const parentId = getParentId(subwriteupId)
  const all = await getAllWriteups()
  return all.find((w) => w.id === parentId) || null
}

export async function getWriteupSubpostCount(parentId: string): Promise<number> {
  return (await getSubwriteupsForParent(parentId)).length
}

export const CTF_CATEGORIES = new Set([
  'crypto', 'pwn', 'misc', 'rev', 'forensics', 'web',
  'osint', 'network', 'algo', 'stego', 'jail',
])

export async function getSubwriteupTags(parentId: string): Promise<string[]> {
  const subwriteups = await getSubwriteupsForParent(parentId)
  const seen = new Set<string>()
  for (const sub of subwriteups) {
    sub.data.tags?.filter((tag) => CTF_CATEGORIES.has(tag)).forEach((tag) => seen.add(tag))
  }
  return Array.from(seen)
}

export async function getWriteupsByTag(tag: string): Promise<
  { parent: CollectionEntry<'writeups'>; subposts: CollectionEntry<'writeups'>[] }[]
> {
  const writeups = await getAllWriteups()
  const results: { parent: CollectionEntry<'writeups'>; subposts: CollectionEntry<'writeups'>[] }[] = []
  for (const writeup of writeups) {
    const subs = await getSubwriteupsForParent(writeup.id)
    const matching = subs.filter((s) => s.data.tags?.includes(tag))
    if (matching.length > 0) results.push({ parent: writeup, subposts: matching })
  }
  return results.sort((a, b) => b.parent.data.date.valueOf() - a.parent.data.date.valueOf())
}

export async function getAdjacentWriteups(currentId: string): Promise<{
  newer: CollectionEntry<'writeups'> | null
  older: CollectionEntry<'writeups'> | null
  parent: CollectionEntry<'writeups'> | null
}> {
  const allWriteups = await getAllWriteups()

  if (isSubpost(currentId)) {
    const parentId = getParentId(currentId)
    const parent = allWriteups.find((w) => w.id === parentId) || null
    const writeups = await getCollection('writeups')
    const subwriteups = writeups
      .filter(
        (w) => isSubpost(w.id) && getParentId(w.id) === parentId && !w.data.draft,
      )
      .sort((a, b) => {
        const dateDiff = a.data.date.valueOf() - b.data.date.valueOf()
        if (dateDiff !== 0) return dateDiff
        return (a.data.order ?? 0) - (b.data.order ?? 0)
      })
    const idx = subwriteups.findIndex((w) => w.id === currentId)
    if (idx === -1) return { newer: null, older: null, parent }
    return {
      newer: idx < subwriteups.length - 1 ? subwriteups[idx + 1] : null,
      older: idx > 0 ? subwriteups[idx - 1] : null,
      parent,
    }
  }

  const parents = allWriteups.filter((w) => !isSubpost(w.id))
  const idx = parents.findIndex((w) => w.id === currentId)
  if (idx === -1) return { newer: null, older: null, parent: null }
  return {
    newer: idx > 0 ? parents[idx - 1] : null,
    older: idx < parents.length - 1 ? parents[idx + 1] : null,
    parent: null,
  }
}

export async function getWriteupReadingTime(writeupId: string): Promise<string> {
  const writeup = await getWriteupById(writeupId)
  if (!writeup) return readingTime(0)
  return readingTime(calculateWordCountFromHtml(writeup.body))
}

export async function getCombinedWriteupReadingTime(
  writeupId: string,
): Promise<string> {
  const writeup = await getWriteupById(writeupId)
  if (!writeup) return readingTime(0)
  let totalWords = calculateWordCountFromHtml(writeup.body)
  if (!isSubpost(writeupId)) {
    const subwriteups = await getSubwriteupsForParent(writeupId)
    for (const sub of subwriteups) {
      totalWords += calculateWordCountFromHtml(sub.body)
    }
  }
  return readingTime(totalWords)
}

export async function getWriteupTOCSections(
  writeupId: string,
): Promise<TOCSection[]> {
  const writeup = await getWriteupById(writeupId)
  if (!writeup) return []

  const parentId = isSubpost(writeupId) ? getParentId(writeupId) : writeupId
  const parentWriteup = isSubpost(writeupId)
    ? await getWriteupById(parentId)
    : writeup
  if (!parentWriteup) return []

  const sections: TOCSection[] = []

  const { headings: parentHeadings } = await render(parentWriteup)
  if (parentHeadings.length > 0) {
    sections.push({
      type: 'parent',
      title: 'Overview',
      headings: parentHeadings.map((h) => ({
        slug: h.slug,
        text: h.text,
        depth: h.depth,
      })),
    })
  }

  const subwriteups = await getSubwriteupsForParent(parentId)
  for (const sub of subwriteups) {
    const { headings: subHeadings } = await render(sub)
    if (subHeadings.length > 0) {
      sections.push({
        type: 'subpost',
        title: sub.data.title,
        headings: subHeadings.map((h, i) => ({
          slug: h.slug,
          text: h.text,
          depth: h.depth,
          isSubpostTitle: i === 0,
        })),
        subpostId: sub.id,
      })
    }
  }

  return sections
}

export async function getTOCSections(postId: string): Promise<TOCSection[]> {
  const post = await getPostById(postId)
  if (!post) return []

  const parentId = isSubpost(postId) ? getParentId(postId) : postId
  const parentPost = isSubpost(postId) ? await getPostById(parentId) : post

  if (!parentPost) return []

  const sections: TOCSection[] = []

  const { headings: parentHeadings } = await render(parentPost)
  if (parentHeadings.length > 0) {
    sections.push({
      type: 'parent',
      title: 'Overview',
      headings: parentHeadings.map((heading) => ({
        slug: heading.slug,
        text: heading.text,
        depth: heading.depth,
      })),
    })
  }

  const subposts = await getSubpostsForParent(parentId)
  for (const subpost of subposts) {
    const { headings: subpostHeadings } = await render(subpost)
    if (subpostHeadings.length > 0) {
      sections.push({
        type: 'subpost',
        title: subpost.data.title,
        headings: subpostHeadings.map((heading, index) => ({
          slug: heading.slug,
          text: heading.text,
          depth: heading.depth,
          isSubpostTitle: index === 0,
        })),
        subpostId: subpost.id,
      })
    }
  }

  return sections
}
