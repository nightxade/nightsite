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
const LANYARD_REFRESH_MS = 60_000

const DiscordSkeleton = () => (
  <div className="relative overflow-hidden sm:aspect-square">
    <div className="grid size-full grid-rows-4">
      <Skeleton className="bg-secondary/50" />
      <div className="row-span-3 flex flex-col gap-3 p-3">
        <div className="flex gap-x-2">
          <Skeleton className="-mt-12 size-20 rounded-full" />
        </div>
        <Skeleton className="h-14 rounded-xl" />
        <Skeleton className="min-h-20 grow rounded-xl" />
      </div>
    </div>
  </div>
)

const StatusIndicator = ({ status }: { status: DiscordStatus }) => {
  const statusClasses: Record<DiscordStatus, string> = {
    online: 'bg-green-500',
    idle: 'bg-yellow-500',
    dnd: 'bg-red-500 flex items-center justify-center',
    offline: 'bg-muted-foreground flex items-center justify-center',
  }

  return (
    <div
      className={cn(
        'border-background absolute right-0 bottom-0 size-5 rounded-full border-[3px]',
        statusClasses[status],
      )}
    >
      {status === 'idle' && (
        <div className="bg-background size-2 rounded-full" />
      )}
      {status === 'dnd' && (
        <div className="bg-background h-[3px] w-[9px] rounded-full" />
      )}
      {status === 'offline' && (
        <div className="bg-background size-1.5 rounded-full" />
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

  return (
    <div className="flex w-full items-center gap-x-3">
      <div
        className="relative aspect-square h-full w-auto shrink-0 rounded-md bg-contain"
        style={{ backgroundImage: `url('${getImageUrl('large_image')}')` }}
      >
        {activity.assets?.small_image && (
          <img
            src={getImageUrl('small_image')}
            alt=""
            width={18}
            height={18}
            className="absolute -right-1 -bottom-1 rounded-full border-2"
          />
        )}
      </div>
      <div className="my-2 flex min-w-0 flex-1 flex-col gap-y-1 overflow-hidden">
        {activity.name && (
          <div className="truncate text-xs leading-none">{activity.name}</div>
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

const DiscordPresence = () => {
  const [lanyard, setLanyard] = useState<LanyardResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    let inFlight = false
    let controller: AbortController | null = null

    const fetchPresence = async (showLoading = false) => {
      if (inFlight) return
      inFlight = true
      if (showLoading) setIsLoading(true)
      const ac = new AbortController()
      controller = ac
      try {
        const res = await fetch(LANYARD_API_URL, { signal: ac.signal })
        if (!res.ok) throw new Error(`${res.status}`)
        const body = (await res.json()) as LanyardResponse
        if (!body.data) throw new Error('invalid response')
        if (mounted) setLanyard(body)
      } catch {
        if (ac.signal.aborted) return
      } finally {
        if (controller === ac) controller = null
        inFlight = false
        if (mounted) setIsLoading(false)
      }
    }

    const refreshWhenVisible = () => {
      if (document.visibilityState === 'visible') void fetchPresence()
    }

    void fetchPresence(true)
    const id = window.setInterval(refreshWhenVisible, LANYARD_REFRESH_MS)
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      mounted = false
      controller?.abort()
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

  const { discord_status, discord_user } = lanyard.data
  const status = discord_status as DiscordStatus
  const avatarUrl = discord_user.avatar
    ? `https://cdn.discordapp.com/avatars/${discord_user.id}/${discord_user.avatar}.png?size=128`
    : '/jett-bread-pfp.jpg'
  const displayName =
    discord_user.global_name ?? discord_user.display_name ?? discord_user.username

  return (
    <div className="relative overflow-hidden sm:aspect-square">
      <div className="grid size-full grid-rows-4">
        <div className="bg-secondary/50" />
        <div className="row-span-3 flex flex-col gap-3 p-3">
          <div className="relative w-fit">
            <AvatarComponent
              src={avatarUrl}
              alt={displayName}
              fallback="n"
              className="-mt-12 size-20 rounded-full"
            />
            <StatusIndicator status={status} />
          </div>

          <div className="bg-secondary/50 flex flex-col gap-y-1 rounded-xl p-3">
            <span className="text-sm leading-none">{displayName}</span>
            <span className="text-muted-foreground text-xs leading-none">
              @{discord_user.username}
            </span>
          </div>

          <div className="bg-secondary/50 flex min-h-20 grow rounded-xl px-3 py-2">
            {mainActivity ? (
              <ActivityDisplay activity={mainActivity} />
            ) : (
              <div className="flex size-full items-center justify-center">
                <span className="text-muted-foreground text-xs">
                  No activity
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DiscordPresence
