import type { IconMap, SocialLink, Site } from '@/types'

export const SITE: Site = {
  title: 'nightxade',
  description:
    'a poisonous flower',
  href: 'https://nightxade.dev',
  author: 'nightxade',
  locale: 'en-US',
  featuredPostCount: 3,
  postsPerPage: 5,
}

export const NAV_LINKS: SocialLink[] = [
  { href: '/', label: 'home' },
  { href: '/blog', label: 'blog' },
  { href: '/writeups', label: 'writeups' },
  { href: '/reading', label: 'reading' },
  { href: '/notes', label: 'notes' },
  { href: '/projects', label: 'projects' },
  { href: '/resume', label: 'resume' },
  { href: '/about', label: 'about' },
]

export const SOCIAL_LINKS: SocialLink[] = [
  {
    href: 'https://github.com/nightxade',
    label: 'GitHub',
  },
  {
    href: 'https://www.linkedin.com/in/matthew-cai1337/',
    label: 'LinkedIn'
  },
  {
    href: 'mailto:nightxade@nightxade.dev',
    label: 'Email',
  },
  {
    href: '/rss.xml',
    label: 'RSS',
  },
]

export const ICON_MAP: IconMap = {
  Website: 'lucide:globe',
  GitHub: 'lucide:github',
  LinkedIn: 'lucide:linkedin',
  Twitter: 'lucide:twitter',
  Email: 'lucide:mail',
  RSS: 'lucide:rss',
}
