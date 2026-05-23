import { useMemo, useState, useEffect } from 'react'
import { Skeleton } from '@/components/ui/skeleton'
import AvatarComponent from '@/components/ui/avatar'
import { cn } from '@/lib/utils'

type DiscordStatus = 'online' | 'idle' | 'dnd' | 'offline'

interface Activity {
  type: number
  application_id?: string
  name?: string
  details?: string
  state?: string
  emoji?: { name?: string; id?: string | null; animated?: boolean }
  timestamps?: { start?: number; end?: number }
  assets?: { large_image?: string; small_image?: string }
}

interface LanyardResponse {
  success: boolean
  data?: {
    discord_status: DiscordStatus
    activities: Activity[]
    discord_user: {
      id: string
      username: string
      display_name?: string
      global_name?: string
      avatar: string | null
    }
  }
  error?: { message?: string }
}

const DISCORD_USER_ID = '484182924762284054'
const LANYARD_API_URL = `https://api.lanyard.rest/v1/users/${DISCORD_USER_ID}`
const LANYARD_REFRESH_MS = 10_000

const STATUS_CONFIGS: Record<
  DiscordStatus,
  { label: string; bgClass: string; indicator?: React.ReactNode }
> = {
  online: {
    label: 'Online',
    bgClass: 'bg-green-500',
  },
  idle: {
    label: 'Idle',
    bgClass: 'bg-yellow-500',
    indicator: <div className="bg-background size-2 rounded-full" />,
  },
  dnd: {
    label: 'Do Not Disturb',
    bgClass: 'bg-red-500',
    indicator: <div className="bg-background h-[3px] w-[9px] rounded-full" />,
  },
  offline: {
    label: 'Offline',
    bgClass: 'bg-muted-foreground',
    indicator: <div className="bg-background size-1.5 rounded-full" />,
  },
}

const DiscordIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 127.14 96.36"
    fill="currentColor"
    className="size-7"
    aria-hidden="true"
  >
    <path d="M107.7,8.07A105.15,105.15,0,0,0,81.47,0a72.06,72.06,0,0,0-3.36,6.83A97.68,97.68,0,0,0,49,6.83,72.37,72.37,0,0,0,45.64,0,105.89,105.89,0,0,0,19.39,8.09C2.79,32.65-1.71,56.6.54,80.21h0A105.73,105.73,0,0,0,32.71,96.36,77.7,77.7,0,0,0,39.6,85.25a68.42,68.42,0,0,1-10.85-5.18c.91-.66,1.8-1.34,2.66-2a75.57,75.57,0,0,0,64.32,0c.87.71,1.76,1.39,2.66,2a68.68,68.68,0,0,1-10.87,5.19,77,77,0,0,0,6.89,11.1A105.25,105.25,0,0,0,126.6,80.22h0C129.24,52.84,122.09,29.11,107.7,8.07ZM42.45,65.69C36.18,65.69,31,60,31,53s5-12.74,11.43-12.74S54,46,53.89,53,48.84,65.69,42.45,65.69Zm42.24,0C78.41,65.69,73.25,60,73.25,53s5-12.74,11.44-12.74S96.23,46,96.12,53,91.08,65.69,84.69,65.69Z" />
  </svg>
)

const StatusIndicator = ({ status }: { status: DiscordStatus }) => {
  const config = STATUS_CONFIGS[status]
  return (
    <div
      className={cn(
        'ring-background absolute right-0 bottom-0 flex size-3.5 items-center justify-center rounded-full ring-2',
        config.bgClass,
      )}
    >
      {config.indicator && (
        <div className="scale-[0.55]">{config.indicator}</div>
      )}
    </div>
  )
}

const ActivityDisplay = ({ activity }: { activity: Activity }) => {
  const [elapsedTime, setElapsedTime] = useState('')
  const [spotifyProgress, setSpotifyProgress] = useState<{
    currentTime: string
    totalTime: string
    progress: number
  } | null>(null)

  const isSpotify = activity.name === 'Spotify' && activity.type === 2

  useEffect(() => {
    if (!activity?.timestamps?.start) return

    const formatTime = (ms: number) => {
      const s = Math.floor(ms / 1000)
      const m = Math.floor(s / 60)
      return `${m}:${(s % 60).toString().padStart(2, '0')}`
    }

    const update = () => {
      if (isSpotify && activity.timestamps?.end) {
        const now = Date.now()
        const elapsed = now - activity.timestamps.start!
        const duration = activity.timestamps.end - activity.timestamps.start!
        const progress = Math.max(0, Math.min(1, elapsed / duration))
        setSpotifyProgress({
          currentTime: formatTime(Math.min(elapsed, duration)),
          totalTime: formatTime(duration),
          progress: progress * 100,
        })
      } else if (activity.timestamps?.start) {
        const diff = Date.now() - activity.timestamps.start
        const h = Math.floor(diff / 3600000)
        const m = Math.floor((diff % 3600000) / 60000)
        const s = Math.floor((diff % 60000) / 1000)
        setElapsedTime(
          `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')} elapsed`,
        )
      }
    }

    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [activity, isSpotify])

  const getImageUrl = (key: 'large_image' | 'small_image') => {
    const val = activity.assets?.[key]
    if (!val) return ''
    if (val.startsWith('spotify:'))
      return `https://i.scdn.co/image/${val.replace('spotify:', '')}`
    if (val.startsWith('mp:external/')) {
      const path = val.replace('mp:external/', '').split('/').slice(1).join('/')
      if (path.startsWith('https/'))
        return 'https://' + decodeURIComponent(path.slice(6))
      if (path.startsWith('http/'))
        return 'http://' + decodeURIComponent(path.slice(5))
    }
    if (val.startsWith('mp:'))
      return `https://media.discordapp.net/${val.replace('mp:', '')}`
    return `https://cdn.discordapp.com/app-assets/${activity.application_id}/${val}`
  }

  const largeImageUrl = getImageUrl('large_image')
  const smallImageUrl = getImageUrl('small_image')

  return (
    <div className="flex w-full items-center gap-x-2.5">
      {largeImageUrl && (
        <div
          className="relative size-14 shrink-0 rounded-md bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url('${largeImageUrl}')` }}
        >
          {smallImageUrl && (
            <img
              src={smallImageUrl}
              alt=""
              width={18}
              height={18}
              className="ring-background absolute -right-1 -bottom-1 size-[18px] rounded-full ring-2"
            />
          )}
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-y-1 overflow-hidden">
        {activity.name && (
          <div className="truncate text-xs leading-none font-medium">
            {activity.name}
          </div>
        )}
        {activity.details && (
          <div className="text-muted-foreground truncate text-[10px] leading-none">
            {activity.details}
          </div>
        )}
        {activity.state && (
          <div className="text-muted-foreground truncate text-[10px] leading-none">
            {activity.state}
          </div>
        )}
        {isSpotify && spotifyProgress ? (
          <div className="flex items-center gap-x-2">
            <span className="text-muted-foreground font-mono text-[9px] leading-none">
              {spotifyProgress.currentTime}
            </span>
            <div className="bg-muted-foreground/20 h-1 flex-1 overflow-hidden rounded-full">
              <div
                className="bg-primary h-full transition-all duration-1000 ease-linear"
                style={{ width: `${spotifyProgress.progress}%` }}
              />
            </div>
            <span className="text-muted-foreground font-mono text-[9px] leading-none">
              {spotifyProgress.totalTime}
            </span>
          </div>
        ) : (
          elapsedTime && (
            <div className="text-muted-foreground text-[10px] leading-none">
              {elapsedTime}
            </div>
          )
        )}
      </div>
    </div>
  )
}

const DiscordSkeleton = () => (
  <div className="relative overflow-hidden rounded-lg">
    <Skeleton className="h-14" />
    <div className="flex flex-col gap-2.5 p-3">
      <Skeleton className="-mt-10 size-16 rounded-full" />
      <Skeleton className="h-14 rounded-md" />
      <Skeleton className="h-16 rounded-md" />
    </div>
  </div>
)

const DiscordPresence = () => {
  const [lanyard, setLanyard] = useState<LanyardResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let mounted = true

    const fetchPresence = async () => {
      try {
        const res = await fetch(LANYARD_API_URL)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const body = (await res.json()) as LanyardResponse
        if (!body.data) throw new Error('missing data field')
        if (mounted) setLanyard(body)
      } catch (err) {
        console.error('[DiscordPresence] fetch failed:', err)
      } finally {
        if (mounted) setIsLoading(false)
      }
    }

    const refreshWhenVisible = () => {
      if (document.visibilityState === 'visible') void fetchPresence()
    }

    void fetchPresence()
    const id = window.setInterval(refreshWhenVisible, LANYARD_REFRESH_MS)
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      mounted = false
      window.clearInterval(id)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [])

  const mainActivity = useMemo(() => {
    const activities = lanyard?.data?.activities ?? []
    return (
      activities.find((a) => a.type === 0 && a.assets) ??
      activities.find((a) => a.type === 2 && a.name === 'Spotify') ??
      null
    )
  }, [lanyard?.data?.activities])

  if (isLoading) return <DiscordSkeleton />
  if (!lanyard?.data) return null

  const { discord_status, discord_user, activities } = lanyard.data
  const status = discord_status as DiscordStatus
  const customStatusActivity = activities.find((a) => a.type === 4)
  const customStatusEmoji = customStatusActivity?.emoji?.name ?? null
  const customStatusText = customStatusActivity?.state ?? null
  const customStatus =
    customStatusEmoji || customStatusText
      ? [customStatusEmoji, customStatusText].filter(Boolean).join(' ')
      : null

  const avatarUrl = discord_user.avatar
    ? `https://cdn.discordapp.com/avatars/${discord_user.id}/${discord_user.avatar}.png?size=128`
    : '/jett-bread-pfp.jpg'
  const displayName =
    discord_user.global_name ??
    discord_user.display_name ??
    discord_user.username

  return (
    <div className="relative flex h-full flex-col overflow-hidden rounded-lg">
      {/* Banner */}
      <div className="bg-muted h-16 shrink-0" />

      {/* Content */}
      <div className="flex flex-1 flex-col gap-2.5 p-3">
          {/* Avatar */}
          <div className="relative w-fit">
            <AvatarComponent
              src={avatarUrl}
              alt={displayName}
              fallback="n"
              className="-mt-10 size-16 rounded-full"
            />
            <StatusIndicator status={status} />
          </div>

          {/* User info */}
          <div className="bg-muted flex flex-col gap-y-1 rounded-md px-3 py-2">
            <div className="flex items-center justify-between gap-x-2">
              <span className="text-sm leading-none font-medium shrink-0">
                {displayName}
              </span>
              {customStatus && (
                <span className="text-muted-foreground truncate text-[12px] leading-none italic">
                  {customStatus}
                </span>
              )}
            </div>
            <span className="text-muted-foreground text-xs leading-none">
              @{discord_user.username}
            </span>
          </div>

          {/* Activity */}
          <div className="bg-muted flex h-[72px] items-center rounded-md px-3 py-2">
            {mainActivity ? (
              <ActivityDisplay activity={mainActivity} />
            ) : (
              <span className="text-muted-foreground text-xs">No activity</span>
            )}
          </div>

          {/* Credit */}
          <a
            href="https://x.com/mioryyn"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground mt-auto text-right text-[10px] leading-none hover:underline"
          >
            profile picture by @mioryyn
          </a>
      </div>

      {/* Discord logo badge */}
      <div className="bg-primary text-background absolute top-0 right-0 m-3 flex size-11 items-center justify-center rounded-full">
        <DiscordIcon />
      </div>
    </div>
  )
}

export default DiscordPresence
