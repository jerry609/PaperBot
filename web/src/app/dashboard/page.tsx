import Link from "next/link"
import { unstable_noStore as noStore } from "next/cache"
import {
  ArrowRight,
  BellDot,
  BookOpen,
  CalendarDays,
  Clock3,
  Layers,
  type LucideIcon,
  TrendingUp,
} from "lucide-react"
import { auth } from "@/auth"

import DashboardReadingQueuePanel from "@/components/dashboard/DashboardReadingQueuePanel"
import { fetchDeadlineRadar, fetchPapers } from "@/lib/api"
import { fetchLatestDashboardBrief } from "@/lib/dashboard-brief"
import {
  buildDashboardIntelligenceCards,
  type DashboardIntelligenceCard,
} from "@/lib/dashboard-intelligence"
import type { DashboardReadingQueueItem, DashboardReadingQueuePriority } from "@/lib/dashboard-reading-queue"
import {
  fetchDashboardReadingQueue,
  fetchDashboardTrackFeed,
  fetchDashboardTracks,
  fetchIntelligenceFeed,
} from "@/lib/dashboard-api"
import { safeHref, safeInternalHref } from "@/lib/utils"
import type {
  DeadlineRadarItem,
  Paper,
  ReadingQueueItem,
  ResearchTrackSummary,
  TrackFeedItem,
} from "@/lib/types"

export const dynamic = "force-dynamic"
export const revalidate = 0

type DashboardRecommendationCardData = {
  id: string
  paperRef: string | null
  internalPaperId: number | null
  title: string
  href: string
  meta: string
  summary: string
  tags: string[]
  metric: string
  recommendation?: string | null
  authors: string[]
  year?: number | null
  paperSource?: "arxiv" | "semantic_scholar" | "openalex" | null
  isSaved?: boolean
}

type TrackSpotlightItem = {
  id: number
  name: string
  updateCount: number
  latestPaper: string
  href: string
}

function getGreeting(): string {
  const hour = new Date().getHours()
  if (hour < 12) return "Good morning"
  if (hour < 18) return "Good afternoon"
  return "Good evening"
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return "just now"

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value

  const diffMs = Date.now() - parsed.getTime()
  const diffMinutes = Math.max(1, Math.floor(diffMs / 60_000))
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays === 1) return "yesterday"
  if (diffDays < 7) return `${diffDays}d ago`

  return parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}

function formatRecommendationLabel(value?: string | null): string | null {
  const normalized = String(value || "").trim()
  if (!normalized) return null

  switch (normalized) {
    case "must_read":
      return "Must read"
    case "worth_reading":
      return "Worth reading"
    case "skim":
      return "Skim"
    case "skip":
      return "Skip"
    default:
      return normalized
  }
}

function getQueuePriority(item: ReadingQueueItem, index: number): DashboardReadingQueuePriority {
  if (typeof item.priority === "number") {
    if (item.priority <= 2) return "high"
    if (item.priority <= 4) return "medium"
    return "low"
  }

  if (index < 2) return "high"
  if (index < 4) return "medium"
  return "low"
}

function getRecommendationPriority(
  recommendation?: string | null,
  index = 0,
): DashboardReadingQueuePriority {
  switch (String(recommendation || "").trim().toLowerCase()) {
    case "must read":
    case "must_read":
      return "high"
    case "worth reading":
    case "worth_reading":
      return "medium"
    case "skim":
    case "skip":
      return "low"
    default:
      return index < 2 ? "high" : "medium"
  }
}

function isExternalUrl(value: string): boolean {
  return /^https?:\/\//.test(value)
}

function normalizePaperRef(value: unknown): string | null {
  const normalized = String(value ?? "").trim()
  return normalized || null
}

function parseInternalPaperId(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isInteger(value) && value >= 0 ? value : null
  }

  if (typeof value !== "string") return null

  const normalized = value.trim()
  if (!normalized) return null

  const parsed = Number(normalized)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null
}

function buildRecommendationCards(args: {
  trackFeedItems: TrackFeedItem[]
  activeTrack: ResearchTrackSummary | null
}): DashboardRecommendationCardData[] {
  const { trackFeedItems, activeTrack } = args

  return trackFeedItems.slice(0, 4).map((item, index) => {
    const recommendation = formatRecommendationLabel(item.latest_judge?.recommendation)
    const judgeScore = item.latest_judge?.overall
    const metric = judgeScore != null
      ? `Judge ${Number(judgeScore).toFixed(1)}`
      : `Feed ${item.feed_score.toFixed(1)}`
    const paperRef = normalizePaperRef(item.paper.id)

    return {
      id: paperRef || `track-feed-${index}`,
      paperRef,
      internalPaperId: parseInternalPaperId(item.paper.id),
      title: item.paper.title,
      href: `/papers/${item.paper.id}`,
      meta: item.paper.venue || activeTrack?.name || "Recommendation",
      summary: recommendation
        ? `${recommendation} in the active track feed.`
        : "Pulled into today’s shortlist.",
      tags: item.matched_terms.slice(0, 3),
      metric,
      recommendation,
      authors: [],
      year: item.paper.year ?? null,
      paperSource: null,
      isSaved: String(item.latest_feedback_action || "").trim().toLowerCase() === "save",
    }
  })
}

function buildReadingQueueCards(args: {
  recommendations: DashboardRecommendationCardData[]
  readingQueue: ReadingQueueItem[]
  paperMap: Map<string, Paper>
  activeTrack: ResearchTrackSummary | null
  latestBriefTime?: string | null
  latestBriefSource?: string | null
}): DashboardReadingQueueItem[] {
  const {
    recommendations,
    readingQueue,
    paperMap,
    activeTrack,
    latestBriefTime,
    latestBriefSource,
  } = args

  if (recommendations.length > 0) {
    const sourceLabel = latestBriefSource ? `${latestBriefSource} brief` : "Daily brief"
    const timeLabel = formatRelativeTime(latestBriefTime)

    return recommendations.slice(0, 3).map((item, index) => ({
      id: item.id,
      paperRef: item.paperRef,
      internalPaperId: item.internalPaperId,
      title: item.title,
      venue: item.meta,
      summary: item.summary,
      tags: item.tags,
      sourceLabel,
      priority: getRecommendationPriority(item.recommendation, index),
      timeLabel,
      href: item.href,
      researchHref: activeTrack ? `/research?track_id=${activeTrack.id}` : "/research",
      isExternal: isExternalUrl(item.href),
      metric: item.metric,
      recommendation: item.recommendation,
      authors: item.authors,
      year: item.year,
      paperSource: item.paperSource,
      isSaved: Boolean(item.isSaved),
      canSave: !item.isSaved && (item.internalPaperId !== null || Boolean(item.paperRef)),
    }))
  }

  const fallbackTags = (activeTrack?.keywords || []).slice(0, 2)

  return readingQueue.slice(0, 3).map((item, index) => {
    const paperRef = normalizePaperRef(item.paper_id)
    const paper = paperMap.get(paperRef || String(item.id))
    const internalPaperId = parseInternalPaperId(item.paper_id)
    const tags = (
      paper?.tags?.length
        ? paper.tags
        : fallbackTags.length
          ? fallbackTags
          : ["Reading queue"]
    ).slice(0, 3)

    return {
      id: item.id,
      paperRef,
      internalPaperId,
      title: item.title,
      venue:
        paper?.venue ||
        (item.authors ? item.authors.split(",").slice(0, 2).join(", ") : "Paper library"),
      summary: activeTrack
        ? `Queued for ${activeTrack.name}.`
        : "Queued for review.",
      tags,
      sourceLabel: activeTrack ? `${activeTrack.name} queue` : "Reading queue",
      priority: getQueuePriority(item, index),
      timeLabel: formatRelativeTime(item.saved_at),
      href: paperRef ? `/papers/${paperRef}` : "/papers",
      researchHref: activeTrack ? `/research?track_id=${activeTrack.id}` : "/research",
      authors: item.authors ? item.authors.split(",").map((value) => value.trim()).filter(Boolean) : [],
      year: null,
      paperSource: null,
      isSaved: true,
      canSave: false,
    }
  })
}

function SectionHeader({
  title,
  actionLabel,
  actionHref,
}: {
  title: string
  actionLabel?: string
  actionHref?: string
}) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <h2 className="text-sm font-bold uppercase tracking-[0.18em] text-slate-500">
        {title}
      </h2>
      {actionLabel && actionHref ? (
        <Link
          href={actionHref}
          className="inline-flex items-center gap-1 text-sm font-medium text-indigo-600 transition-colors hover:text-indigo-700"
        >
          {actionLabel}
          <ArrowRight className="size-4" />
        </Link>
      ) : null}
    </div>
  )
}

function OverviewStat({
  label,
  value,
  helper,
  alert = false,
  icon: Icon,
}: {
  label: string
  value: string | number
  helper: string
  alert?: boolean
  icon: LucideIcon
}) {
  return (
    <div className="group flex items-center justify-between rounded-2xl border border-slate-100 bg-white p-5 shadow-sm transition-all hover:shadow-md">
      <div>
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <div className="mt-1 flex items-baseline gap-2">
          <span className="text-2xl font-bold text-slate-800">{value}</span>
          <span className={`text-xs font-medium ${alert ? "text-rose-600" : "text-emerald-600"}`}>
            {helper}
          </span>
        </div>
      </div>
      <div className={`rounded-xl p-3 transition-colors ${alert ? "bg-rose-50 text-rose-600 group-hover:bg-rose-100" : "bg-indigo-50 text-indigo-600 group-hover:bg-indigo-100"}`}>
        <Icon className="size-5" />
      </div>
    </div>
  )
}

function ActionLink({
  href,
  label,
  primary = false,
  external = false,
}: {
  href: string
  label: string
  primary?: boolean
  external?: boolean
}) {
  const safeLink = external ? safeHref(href) : safeInternalHref(href)
  const className = primary
    ? "inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700"
    : "inline-flex items-center rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900"

  if (!safeLink) {
    return (
      <span className={`${className} cursor-not-allowed opacity-60`}>
        {label}
      </span>
    )
  }

  if (external) {
    return (
      <a href={safeLink} target="_blank" rel="noreferrer" className={className}>
        {label}
      </a>
    )
  }

  return (
    <Link href={safeLink} className={className}>
      {label}
    </Link>
  )
}

function TrackSpotlightCard({ item }: { item: TrackSpotlightItem }) {
  return (
    <Link
      href={item.href}
      className="group block rounded-2xl border border-slate-100 bg-white p-5 shadow-sm transition-all hover:border-indigo-200 hover:shadow-md"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="rounded-xl bg-violet-50 p-2 text-violet-600 transition-colors group-hover:bg-violet-100">
          <Layers className="size-4.5" />
        </div>
        <span className="rounded-full border border-rose-100 bg-rose-50 px-2.5 py-1 text-[11px] font-medium text-rose-700">
          {item.updateCount > 0 ? `${item.updateCount} updates` : "Watching"}
        </span>
      </div>
      <h3 className="mt-4 text-base font-bold text-slate-800 transition-colors group-hover:text-indigo-600">
        {item.name}
      </h3>
      <div className="mt-4 border-t border-slate-50 pt-3">
        <span className="block text-xs text-slate-400">Latest capture</span>
        <span className="mt-1 block text-sm font-medium text-slate-700">{item.latestPaper}</span>
      </div>
    </Link>
  )
}

function SignalCard({ item }: { item: DashboardIntelligenceCard }) {
  return (
    <article className="rounded-2xl border border-slate-100 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="inline-flex rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
          {item.sourceLabel}
        </span>
        <span className="text-xs text-slate-400">{formatRelativeTime(item.timestamp)}</span>
      </div>

      <h3 className="mt-3 text-base font-bold leading-6 text-slate-800">{item.title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">{item.summary}</p>

      {item.reasonChips.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {item.reasonChips.slice(0, 3).map((reason) => (
            <span
              key={`${item.id}-${reason}`}
              className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600"
            >
              {reason}
            </span>
          ))}
        </div>
      ) : null}

      <div className="mt-4 flex items-center justify-between gap-3 border-t border-slate-50 pt-4">
        <span className="text-xs text-slate-500">{item.metricLabel}</span>
        <div className="flex items-center gap-2">
          <ActionLink href={item.researchHref} label={item.researchLabel} />
          <ActionLink href={item.href} label="Source" external={item.isExternal} />
        </div>
      </div>
    </article>
  )
}

function DeadlineCard({ item }: { item: DeadlineRadarItem }) {
  const safeUrl = safeHref(item.url)
  const className = "group flex items-center justify-between gap-3 rounded-2xl border border-slate-100 bg-white p-4 shadow-sm transition-all hover:border-indigo-200 hover:shadow-md"
  const content = (
    <>
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 flex-col items-center justify-center rounded-xl bg-orange-50 text-orange-600 transition-colors group-hover:bg-orange-100">
          <span className="text-[10px] font-bold uppercase">
            {new Date(item.deadline).toLocaleDateString("en-US", { month: "short" })}
          </span>
          <span className="text-lg font-black">
            {new Date(item.deadline).toLocaleDateString("en-US", { day: "2-digit" })}
          </span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-slate-800">{item.name}</h4>
          <p className="text-xs text-slate-500">
            {item.field} · {item.days_left} days left
          </p>
        </div>
      </div>
      {item.days_left <= 14 ? (
        <div className="h-2 w-2 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]" />
      ) : null}
    </>
  )

  if (!safeUrl) {
    return <div className={`${className} cursor-not-allowed opacity-80`}>{content}</div>
  }

  return (
    <a
      href={safeUrl}
      target="_blank"
      rel="noreferrer"
      className={className}
    >
      {content}
    </a>
  )
}

export default async function DashboardPage() {
  noStore()
  const session = await auth()
  const accessToken = session?.accessToken

  const [
    tracksResult,
    readingQueueResult,
    intelligenceResult,
    papersResult,
    latestBriefResult,
    deadlinesResult,
  ] = await Promise.allSettled([
    fetchDashboardTracks(accessToken),
    fetchDashboardReadingQueue(accessToken, 6),
    fetchIntelligenceFeed(accessToken, 6, { sortBy: "delta", sortOrder: "desc" }),
    fetchPapers(accessToken),
    fetchLatestDashboardBrief(),
    fetchDeadlineRadar(accessToken),
  ])

  const tracks = tracksResult.status === "fulfilled" ? tracksResult.value : []
  const readingQueue = readingQueueResult.status === "fulfilled" ? readingQueueResult.value : []
  const intelligenceFeed = intelligenceResult.status === "fulfilled"
    ? intelligenceResult.value
    : { items: [], refreshed_at: null, refresh_scheduled: false, keywords: [], watch_repos: [], subreddits: [] }
  const papers = papersResult.status === "fulfilled" ? papersResult.value : []
  const latestBrief = latestBriefResult.status === "fulfilled" ? latestBriefResult.value : null
  const deadlinesRaw = deadlinesResult.status === "fulfilled" ? deadlinesResult.value : []

  const orderedTracks = [...tracks].sort((left, right) => Number(Boolean(right.is_active)) - Number(Boolean(left.is_active)))
  const activeTrack = orderedTracks[0] || null
  const activeTrackFeed = activeTrack
    ? await fetchDashboardTrackFeed(activeTrack.id, accessToken, 4).catch(() => ({ items: [], total: 0 }))
    : { items: [], total: 0 }
  const recommendationCards: DashboardRecommendationCardData[] = latestBrief?.recommendations.length
    ? latestBrief.recommendations.map((item) => ({
        id: item.id,
        paperRef: normalizePaperRef(item.paperId),
        internalPaperId: parseInternalPaperId(item.paperId),
        title: item.title,
        href: item.href,
        meta: item.meta,
        summary: item.summary,
        tags: item.tags,
        metric: item.metric,
        recommendation: item.recommendation,
        authors: item.authors,
        year: item.year ?? null,
        paperSource: item.paperSource ?? null,
        isSaved: false,
      }))
    : buildRecommendationCards({
        trackFeedItems: activeTrackFeed.items || [],
        activeTrack,
      })

  const paperMap = new Map(papers.map((paper) => [String(paper.id), paper]))
  const queueCards = buildReadingQueueCards({
    recommendations: recommendationCards,
    readingQueue,
    paperMap,
    activeTrack,
    latestBriefTime: latestBrief?.generatedAt || latestBrief?.date || null,
    latestBriefSource: latestBrief?.sourceLabel || null,
  })

  const spotlightSeeds = orderedTracks.slice(0, 3)
  const spotlightItems: TrackSpotlightItem[] = await Promise.all(
    spotlightSeeds.map(async (track) => {
      const feed = await fetchDashboardTrackFeed(track.id, accessToken, 1).catch(() => ({ items: [], total: 0 }))
      return {
        id: track.id,
        name: track.name,
        updateCount: Number(feed.total || 0),
        latestPaper: feed.items[0]?.paper.title || "No recent paper yet",
        href: `/research?track_id=${track.id}`,
      }
    }),
  )

  const intelligenceCards = buildDashboardIntelligenceCards(intelligenceFeed.items)
  const signalCards = intelligenceCards.slice(0, 3)
  const deadlines = [...deadlinesRaw]
    .sort((left, right) => left.days_left - right.days_left)
    .slice(0, 3)

  const greeting = getGreeting()
  const libraryCount = papers.length
  const urgentDeadlines = deadlinesRaw.filter((item) => item.days_left <= 14).length

  return (
    <div className="min-h-screen bg-stone-50/60 pb-12 text-slate-900">
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="space-y-10">
          <header className="space-y-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-600">
                  Dashboard
                </p>
                <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                  {greeting}. Here is today&apos;s research queue.
                </h1>
                <p className="mt-2 text-sm text-slate-600">
                  {queueCards.length} picks ready, {urgentDeadlines} urgent deadlines, {signalCards.length} fresh signals.
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <Link
                  href={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}
                  className="inline-flex min-h-11 items-center gap-2 rounded-full bg-slate-900 px-5 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
                >
                  Open Research
                  <ArrowRight className="size-4" />
                </Link>
                <Link
                  href="/settings"
                  className="inline-flex min-h-11 items-center gap-2 rounded-full border border-slate-200 bg-white px-5 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50"
                >
                  Open Settings
                  <ArrowRight className="size-4" />
                </Link>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3 lg:grid-cols-4">
              <OverviewStat
                label="Library"
                value={libraryCount}
                helper={`${Math.max(0, recommendationCards.length)} ready`}
                icon={BookOpen}
              />
              <OverviewStat
                label="Reading queue"
                value={readingQueue.length}
                helper={`${queueCards.filter((item) => item.priority === "high").length} high priority`}
                alert={queueCards.some((item) => item.priority === "high")}
                icon={Clock3}
              />
              <OverviewStat
                label="Tracks"
                value={tracks.length}
                helper={`${spotlightItems.filter((item) => item.updateCount > 0).length} with updates`}
                icon={Layers}
              />
              <OverviewStat
                label="Signals"
                value={signalCards.length}
                helper={`${urgentDeadlines} deadlines soon`}
                alert={urgentDeadlines > 0}
                icon={BellDot}
              />
            </div>
          </header>

          <section>
            <SectionHeader
              title="Reading Queue"
              actionLabel="Open library"
              actionHref="/papers"
            />
            <DashboardReadingQueuePanel
              initialItems={queueCards}
              activeTrackId={activeTrack?.id ?? null}
            />
          </section>

          <section className="grid gap-8 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
            <div>
              <SectionHeader
                title="Track Spotlight"
                actionLabel="Manage tracks"
                actionHref="/research"
              />
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
                {spotlightItems.length > 0 ? (
                  spotlightItems.map((item) => (
                    <TrackSpotlightCard key={item.id} item={item} />
                  ))
                ) : (
                  <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600 shadow-sm">
                    No tracks yet.
                  </div>
                )}
              </div>
            </div>

            <div>
              <SectionHeader
                title="Signals"
                actionLabel="Open research"
                actionHref="/research"
              />
              <div className="rounded-2xl border border-slate-100 bg-white p-5 shadow-sm">
                {latestBrief?.trendRows.length ? (
                  <div className="mb-5 flex flex-wrap gap-2 border-b border-slate-100 pb-4">
                    {latestBrief.trendRows.slice(0, 3).map((trend) => (
                      <span
                        key={trend.query}
                        className="inline-flex items-center rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
                      >
                        <TrendingUp className="mr-1.5 size-3.5" />
                        {trend.query}
                      </span>
                    ))}
                  </div>
                ) : null}

                <div className="space-y-4">
                  {signalCards.map((item) => (
                    <SignalCard key={item.id} item={item} />
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section>
            <SectionHeader
              title="Deadlines"
              actionLabel="Open research"
              actionHref={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}
            />
            <div className="grid gap-4 lg:grid-cols-3">
              {deadlines.length > 0 ? (
                deadlines.map((item) => (
                  <DeadlineCard key={`${item.name}-${item.deadline}`} item={item} />
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600 shadow-sm lg:col-span-3">
                  No nearby deadlines.
                </div>
              )}
            </div>

            <div className="mt-6 rounded-2xl border border-slate-100 bg-white px-5 py-4 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-3 text-sm text-slate-600">
                  <CalendarDays className="size-4 text-orange-500" />
                  <span>{urgentDeadlines} urgent deadlines</span>
                  <span className="text-slate-300">/</span>
                  <span>{latestBrief?.title || "Latest daily brief"}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Link
                    href="/settings"
                    className="inline-flex items-center gap-1 text-sm font-medium text-indigo-600 transition-colors hover:text-indigo-700"
                  >
                    Manage daily brief
                    <ArrowRight className="size-4" />
                  </Link>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}
