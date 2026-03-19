"use client"

import Link from "next/link"
import { useCallback, useEffect, useMemo, useState, useTransition, type ReactNode } from "react"
import {
  ArrowRight,
  BellDot,
  ExternalLink,
  Layers3,
  type LucideIcon,
  Loader2,
  RefreshCcw,
  Sparkles,
  TrendingUp,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { cn, safeHref, safeInternalHref } from "@/lib/utils"
import {
  buildDashboardIntelligenceCards,
  type DashboardIntelligenceCard,
  summarizeDashboardIntelligence,
} from "@/lib/dashboard-intelligence"
import type {
  IntelligenceFeedResponse,
  ResearchTrackSummary,
} from "@/lib/types"

type SignalsWorkspaceProps = {
  initialFeed: IntelligenceFeedResponse
  initialTracks: ResearchTrackSummary[]
  initialNowMs: number
}

type SignalSort = "delta" | "score" | "freshness" | "published_at"

const SORT_OPTIONS: Array<{ value: SignalSort; label: string }> = [
  { value: "delta", label: "Rising first" },
  { value: "score", label: "Highest score" },
  { value: "freshness", label: "Newest first" },
  { value: "published_at", label: "Published date" },
]

type SignalStatProps = Readonly<{
  label: string
  value: number
  detail: string
  icon: LucideIcon
}>

type WatchListProps = Readonly<{
  title: string
  items: string[]
}>

type SignalListRowProps = Readonly<{
  card: DashboardIntelligenceCard
  active: boolean
  onSelect: () => void
  nowMs: number
}>

type SignalDetailSectionProps = Readonly<{
  title: string
  icon: LucideIcon
  children: ReactNode
}>

type SignalsHeaderProps = Readonly<{
  summary: ReturnType<typeof summarizeDashboardIntelligence>
  refreshedAt: string | null | undefined
  refreshScheduled: boolean | undefined
  nowMs: number
  isPending: boolean
  onRefresh: () => void
}>

type SignalsFiltersPanelProps = Readonly<{
  selectedSource: string
  onSourceChange: (value: string) => void
  sourceOptions: string[]
  selectedTrackId: string
  onTrackChange: (value: string) => void
  tracks: ResearchTrackSummary[]
  sortBy: SignalSort
  onSortChange: (value: SignalSort) => void
  matchedOnly: boolean
  onMatchedOnlyChange: (value: boolean) => void
  watchKeywords: string[]
  watchRepos: string[]
  watchSubreddits: string[]
}>

type SignalQueuePanelProps = Readonly<{
  filteredItemsCount: number
  refreshScheduled: boolean | undefined
  error: string | null
  cards: DashboardIntelligenceCard[]
  selectedSignalId: string
  onSelectSignal: (signalId: string) => void
  nowMs: number
}>

type SignalDetailPanelProps = Readonly<{
  selectedCard: DashboardIntelligenceCard | null
  selectedItem: IntelligenceFeedResponse["items"][number] | null
  safeResearchHref: string | null
  safeSourceHref: string | null
  nowMs: number
}>

function formatCalendarDate(value: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(value)
}

function formatRelativeTime(value: string | null | undefined, nowMs: number): string {
  if (!value) return "Recently"

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return "Recently"

  const diffMs = Math.max(0, nowMs - parsed.getTime())
  const diffMinutes = Math.max(1, Math.floor(diffMs / 60_000))
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`

  return formatCalendarDate(parsed)
}

function SignalStat({
  label,
  value,
  detail,
  icon: Icon,
}: SignalStatProps) {
  return (
    <div className="flex min-h-[64px] items-center gap-3 bg-white/92 px-4 py-2.5">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-1.5 text-slate-600">
        <Icon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            {label}
          </p>
          <span className="text-base font-semibold tabular-nums text-slate-900">{value}</span>
        </div>
        <p className="mt-0.5 line-clamp-1 text-[11px] text-slate-500">{detail}</p>
      </div>
    </div>
  )
}

function WatchList({
  title,
  items,
}: WatchListProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-slate-900">{title}</p>
        <Badge variant="outline" className="rounded-full">
          {items.length}
        </Badge>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {items.length > 0 ? (
          items.slice(0, 10).map((item) => (
            <span
              key={`${title}-${item}`}
              className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600"
            >
              {item}
            </span>
          ))
        ) : (
          <span className="text-sm text-slate-500">No watch items configured yet.</span>
        )}
      </div>
    </div>
  )
}

function SignalListRow({
  card,
  active,
  onSelect,
  nowMs,
}: SignalListRowProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "relative w-full rounded-[14px] border px-3 py-2.5 text-left transition-[background-color,border-color,box-shadow]",
        active
          ? "border-slate-200 bg-slate-50 shadow-[0_1px_2px_rgba(15,23,42,0.04)]"
          : "border-transparent bg-transparent hover:bg-slate-50/85",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "absolute inset-y-2 left-0 w-0.5 rounded-full transition-colors",
          active ? "bg-indigo-500" : "bg-transparent",
        )}
      />
      <div className="flex items-center justify-between gap-3">
        <span className="inline-flex rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-600">
          {card.sourceLabel}
        </span>
        <span className="text-[11px] text-slate-400">{formatRelativeTime(card.timestamp, nowMs)}</span>
      </div>

      <h3 className="mt-1.5 line-clamp-1 text-sm font-semibold leading-6 text-slate-900">
        {card.title}
      </h3>
      <p className="mt-0.5 line-clamp-2 text-xs leading-5 text-slate-600">{card.summary}</p>

      <div className="mt-2 flex flex-wrap gap-1.5">
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-600">
          {card.metricLabel}
        </span>
        {card.matchedTrackNames.slice(0, 2).map((trackName) => (
          <span
            key={`${card.id}-${trackName}`}
            className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700"
          >
            {trackName}
          </span>
        ))}
      </div>
    </button>
  )
}

function SignalDetailSection({
  title,
  icon: Icon,
  children,
}: SignalDetailSectionProps) {
  return (
    <section className="px-5 py-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
        <Icon className="size-4 text-indigo-600" />
        {title}
      </div>
      <div className="mt-3">{children}</div>
    </section>
  )
}

function collectSourceOptions(
  initialItems: IntelligenceFeedResponse["items"],
  currentItems: IntelligenceFeedResponse["items"],
): string[] {
  const values = new Set<string>()
  for (const item of [...initialItems, ...currentItems]) {
    const source = String(item.source || "").trim()
    if (source) values.add(source)
  }
  return Array.from(values)
}

async function fetchSignalsFeed(params: {
  sortBy: SignalSort
  selectedSource: string
  selectedTrackId: string
  refresh?: boolean
}): Promise<IntelligenceFeedResponse> {
  const searchParams = new URLSearchParams({
    limit: "20",
    sort_by: params.sortBy,
    sort_order: "desc",
  })
  if (params.selectedSource !== "all") {
    searchParams.set("source", params.selectedSource)
  }
  if (params.selectedTrackId !== "all") {
    searchParams.set("track_id", params.selectedTrackId)
  }
  if (params.refresh) {
    searchParams.set("refresh", "true")
  }

  const response = await fetch(`/api/intelligence/feed?${searchParams.toString()}`, {
    cache: "no-store",
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Failed to load signals (${response.status})`)
  }
  return (await response.json()) as IntelligenceFeedResponse
}

function resolveSignalSourceHref(card: DashboardIntelligenceCard | null): string | null {
  if (!card?.href) return null
  return card.isExternal ? safeHref(card.href) : safeInternalHref(card.href)
}

function resolveSignalResearchHref(card: DashboardIntelligenceCard | null): string | null {
  return card?.researchHref ? safeInternalHref(card.researchHref) : null
}

function renderSignalSourceAction(
  card: DashboardIntelligenceCard,
  safeSourceHref: string | null,
): ReactNode {
  if (!safeSourceHref) return null
  if (card.isExternal) {
    return (
      <Button asChild size="sm" variant="outline" className="rounded-full">
        <a href={safeSourceHref} target="_blank" rel="noreferrer">
          Open source
          <ExternalLink className="size-4" />
        </a>
      </Button>
    )
  }
  return (
    <Button asChild size="sm" variant="outline" className="rounded-full">
      <Link href={safeSourceHref}>
        Open source
        <ArrowRight className="size-4" />
      </Link>
    </Button>
  )
}

function SignalsHeader({
  summary,
  refreshedAt,
  refreshScheduled,
  nowMs,
  isPending,
  onRefresh,
}: SignalsHeaderProps) {
  return (
    <header className="rounded-[28px] border border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.08),_transparent_30%),linear-gradient(180deg,_rgba(255,255,255,0.98),_rgba(248,250,252,0.98))] p-4.5 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div className="max-w-3xl">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-600">
            Signals
          </p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
            Community radar, track matches, and rising research activity in one place.
          </h1>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Dashboard now keeps only the preview. This workspace holds the full signal queue,
            filter controls, and the research handoff path.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            className="rounded-full"
            onClick={onRefresh}
            disabled={isPending}
          >
            {isPending ? <Loader2 className="size-4 animate-spin" /> : <RefreshCcw className="size-4" />}
            Refresh radar
          </Button>
          <Button asChild className="rounded-full bg-slate-900 hover:bg-slate-800">
            <Link href="/research">
              Open Research
              <ArrowRight className="size-4" />
            </Link>
          </Button>
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200/90 bg-slate-200/80 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]">
        <div className="grid gap-px md:grid-cols-2 xl:grid-cols-4 xl:[&>*:not(:last-child)]:border-r xl:[&>*:not(:last-child)]:border-r-slate-200/80">
          <SignalStat
            label="Signals live"
            value={summary.totalCount}
            detail={refreshScheduled ? "Refresh queued" : "Latest cache ready"}
            icon={BellDot}
          />
          <SignalStat
            label="Track matched"
            value={summary.matchedCount}
            detail={summary.matchedCount > 0 ? "Matched to active tracks" : "No track overlap yet"}
            icon={Layers3}
          />
          <SignalStat
            label="Rising now"
            value={summary.risingCount}
            detail={summary.risingCount > 0 ? "Positive deltas in slice" : "No positive deltas"}
            icon={TrendingUp}
          />
          <SignalStat
            label="Sources"
            value={summary.sourceCount}
            detail={`Updated ${formatRelativeTime(refreshedAt, nowMs)}`}
            icon={Sparkles}
          />
        </div>
      </div>
    </header>
  )
}

function SignalsFiltersPanel({
  selectedSource,
  onSourceChange,
  sourceOptions,
  selectedTrackId,
  onTrackChange,
  tracks,
  sortBy,
  onSortChange,
  matchedOnly,
  onMatchedOnlyChange,
  watchKeywords,
  watchRepos,
  watchSubreddits,
}: SignalsFiltersPanelProps) {
  return (
    <aside className="space-y-4">
      <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center gap-2">
          <BellDot className="size-4 text-indigo-600" />
          <h2 className="text-base font-semibold text-slate-900">Feed controls</h2>
        </div>

        <div className="mt-4 space-y-4">
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
              Source
            </label>
            <Select value={selectedSource} onValueChange={onSourceChange}>
              <SelectTrigger className="bg-slate-50">
                <SelectValue placeholder="All sources" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All sources</SelectItem>
                {sourceOptions.map((source) => (
                  <SelectItem key={source} value={source}>
                    {source}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
              Track
            </label>
            <Select value={selectedTrackId} onValueChange={onTrackChange}>
              <SelectTrigger className="bg-slate-50">
                <SelectValue placeholder="All tracks" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All tracks</SelectItem>
                {tracks.map((track) => (
                  <SelectItem key={track.id} value={String(track.id)}>
                    {track.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
              Sort
            </label>
            <Select value={sortBy} onValueChange={(value) => onSortChange(value as SignalSort)}>
              <SelectTrigger className="bg-slate-50">
                <SelectValue placeholder="Sort signals" />
              </SelectTrigger>
              <SelectContent>
                {SORT_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3">
            <div>
              <p className="text-sm font-medium text-slate-900">Matched only</p>
              <p className="text-xs text-slate-500">Hide unrouted community noise</p>
            </div>
            <Switch checked={matchedOnly} onCheckedChange={onMatchedOnlyChange} />
          </div>
        </div>
      </div>

      <WatchList title="Tracked keywords" items={watchKeywords} />
      <WatchList title="Watched repos" items={watchRepos} />
      <WatchList title="Subreddits" items={watchSubreddits} />
    </aside>
  )
}

function SignalQueuePanel({
  filteredItemsCount,
  refreshScheduled,
  error,
  cards,
  selectedSignalId,
  onSelectSignal,
  nowMs,
}: SignalQueuePanelProps) {
  return (
    <section className="min-h-[640px] rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Signal queue</h2>
          <p className="text-sm text-slate-500">
            {filteredItemsCount} items after the current filters
          </p>
        </div>
        <Badge variant="outline" className="rounded-full">
          {refreshScheduled ? "Refresh queued" : "Live cache"}
        </Badge>
      </div>

      <ScrollArea className="h-[552px]">
        <div className="pb-0.5 pl-2 pr-1 pt-2">
          {error ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {error}
            </div>
          ) : cards.length > 0 ? (
            <div className="space-y-1">
              {cards.map((card) => (
                <SignalListRow
                  key={card.id}
                  card={card}
                  active={card.id === selectedSignalId}
                  onSelect={() => onSelectSignal(card.id)}
                  nowMs={nowMs}
                />
              ))}
            </div>
          ) : (
            <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-5 py-10 text-center">
              <p className="text-base font-semibold text-slate-900">No signals match this filter.</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Try turning off the matched-only view or broadening the selected source.
              </p>
            </div>
          )}
        </div>
      </ScrollArea>
    </section>
  )
}

function SignalDetailPanel({
  selectedCard,
  selectedItem,
  safeResearchHref,
  safeSourceHref,
  nowMs,
}: SignalDetailPanelProps) {
  const sourceAction = selectedCard
    ? renderSignalSourceAction(selectedCard, safeSourceHref)
    : null

  return (
    <section className="min-h-[640px] rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-5 py-3.5">
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
          Selected signal
        </p>
        <h2 className="mt-2 text-lg font-semibold text-slate-900">
          {selectedCard?.title || "Choose a signal"}
        </h2>
        <p className="mt-1.5 text-sm text-slate-500">
          {selectedCard
            ? formatRelativeTime(selectedCard.timestamp, nowMs)
            : "Pick a row from the queue to inspect the handoff."}
        </p>
      </div>

      <ScrollArea className="h-[552px]">
        <div className="p-4">
          {selectedCard ? (
            <div className="divide-y divide-slate-100 overflow-hidden rounded-2xl border border-slate-200 bg-white">
              <section className="px-5 py-4">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge className="rounded-full bg-slate-900 px-2.5 py-0.5 text-[10px] text-white">
                    {selectedCard.sourceLabel}
                  </Badge>
                  <Badge variant="outline" className="rounded-full px-2.5 py-0.5 text-[10px]">
                    {selectedCard.metricLabel}
                  </Badge>
                  {selectedCard.matchedTrackNames.map((trackName) => (
                    <Badge
                      key={`${selectedCard.id}-${trackName}`}
                      variant="secondary"
                      className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-[10px] text-emerald-700"
                    >
                      {trackName}
                    </Badge>
                  ))}
                </div>

                <div className="mt-3 rounded-2xl bg-slate-50/90 px-4 py-3">
                  <p className="text-sm leading-7 text-slate-700">{selectedCard.summary}</p>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {safeResearchHref ? (
                    <Button asChild size="sm" className="rounded-full bg-slate-900 hover:bg-slate-800">
                      <Link href={safeResearchHref}>
                        Open in Research
                        <ArrowRight className="size-4" />
                      </Link>
                    </Button>
                  ) : null}
                  {sourceAction}
                </div>
              </section>

              <SignalDetailSection title="Why this surfaced" icon={TrendingUp}>
                <div className="flex flex-wrap gap-1.5">
                  {selectedCard.reasonChips.length > 0 ? (
                    selectedCard.reasonChips.map((reason) => (
                      <span
                        key={`${selectedCard.id}-${reason}`}
                        className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-600"
                      >
                        {reason}
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-slate-500">
                      No explicit reasons were attached to this signal.
                    </span>
                  )}
                </div>
              </SignalDetailSection>

              <SignalDetailSection title="Research handoff" icon={Layers3}>
                <p className="text-sm leading-6 text-slate-600">
                  {selectedItem?.research_query
                    ? `Prepared query: ${selectedItem.research_query}`
                    : "No prepared research query was attached to this signal."}
                </p>
                {(selectedItem?.matched_tracks || []).length > 0 ? (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {(selectedItem?.matched_tracks || []).map((track) => (
                      <span
                        key={`${selectedCard.id}-${track.track_id}`}
                        className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700"
                      >
                        {track.track_name}
                      </span>
                    ))}
                  </div>
                ) : null}
              </SignalDetailSection>

              {(selectedItem?.keyword_hits || []).length > 0 ||
              (selectedItem?.repo_matches || []).length > 0 ||
              (selectedItem?.author_matches || []).length > 0 ? (
                <SignalDetailSection title="Matched entities" icon={Sparkles}>
                  <div className="flex flex-wrap gap-1.5">
                    {(selectedItem?.keyword_hits || []).map((keyword) => (
                      <span
                        key={`${selectedCard.id}-keyword-${keyword}`}
                        className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-600"
                      >
                        keyword: {keyword}
                      </span>
                    ))}
                    {(selectedItem?.repo_matches || []).map((repo) => (
                      <span
                        key={`${selectedCard.id}-repo-${repo}`}
                        className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-600"
                      >
                        repo: {repo}
                      </span>
                    ))}
                    {(selectedItem?.author_matches || []).map((author) => (
                      <span
                        key={`${selectedCard.id}-author-${author}`}
                        className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-600"
                      >
                        author: {author}
                      </span>
                    ))}
                  </div>
                </SignalDetailSection>
              ) : null}
            </div>
          ) : (
            <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-5 py-12 text-center">
              <p className="text-base font-semibold text-slate-900">No signal selected.</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                As soon as a row appears in the queue, its research handoff and source detail
                will open here.
              </p>
            </div>
          )}
        </div>
      </ScrollArea>
    </section>
  )
}

export default function SignalsWorkspace({
  initialFeed,
  initialTracks,
  initialNowMs,
}: SignalsWorkspaceProps) {
  const [feed, setFeed] = useState<IntelligenceFeedResponse>(initialFeed)
  const [nowMs, setNowMs] = useState(initialNowMs)
  const [selectedSource, setSelectedSource] = useState<string>("all")
  const [selectedTrackId, setSelectedTrackId] = useState<string>("all")
  const [sortBy, setSortBy] = useState<SignalSort>("delta")
  const [matchedOnly, setMatchedOnly] = useState(false)
  const [selectedSignalId, setSelectedSignalId] = useState<string>(initialFeed.items[0]?.id || "")
  const [error, setError] = useState<string | null>(null)
  const [isPending, startTransition] = useTransition()

  const sourceOptions = useMemo(
    () => collectSourceOptions(initialFeed.items, feed.items),
    [feed.items, initialFeed.items],
  )

  const filteredItems = useMemo(() => {
    return matchedOnly
      ? feed.items.filter((item) => (item.matched_tracks || []).length > 0)
      : feed.items
  }, [feed.items, matchedOnly])

  const cards = useMemo(() => buildDashboardIntelligenceCards(filteredItems), [filteredItems])
  const summary = useMemo(() => summarizeDashboardIntelligence(filteredItems), [filteredItems])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNowMs(Date.now())
    }, 60_000)
    return () => window.clearInterval(timer)
  }, [])

  const effectiveSelectedSignalId = useMemo(() => {
    if (cards.length === 0) return ""
    return cards.some((card) => card.id === selectedSignalId) ? selectedSignalId : (cards[0]?.id || "")
  }, [cards, selectedSignalId])

  const selectedCard = useMemo(() => {
    return cards.find((card) => card.id === effectiveSelectedSignalId) || cards[0] || null
  }, [cards, effectiveSelectedSignalId])

  const selectedItem = useMemo(() => {
    if (!selectedCard) return null
    return filteredItems.find((item) => item.id === selectedCard.id) || null
  }, [filteredItems, selectedCard])

  const loadFeed = useCallback(({ refresh = false }: { refresh?: boolean } = {}) => {
    startTransition(() => {
      setError(null)
      fetchSignalsFeed({ sortBy, selectedSource, selectedTrackId, refresh })
        .then((payload) => {
          setFeed(payload)
        })
        .catch((loadError) => {
          setError(loadError instanceof Error ? loadError.message : String(loadError))
        })
    })
  }, [selectedSource, selectedTrackId, sortBy, startTransition])

  useEffect(() => {
    loadFeed()
  }, [loadFeed])

  const safeSourceHref = useMemo(() => resolveSignalSourceHref(selectedCard), [selectedCard])
  const safeResearchHref = useMemo(() => resolveSignalResearchHref(selectedCard), [selectedCard])

  return (
    <main className="mx-auto max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
      <div className="space-y-7">
        <SignalsHeader
          summary={summary}
          refreshedAt={feed.refreshed_at}
          refreshScheduled={feed.refresh_scheduled}
          nowMs={nowMs}
          isPending={isPending}
          onRefresh={() => loadFeed({ refresh: true })}
        />

        <section className="grid gap-6 xl:grid-cols-[300px_minmax(0,0.9fr)_minmax(0,1.05fr)]">
          <SignalsFiltersPanel
            selectedSource={selectedSource}
            onSourceChange={setSelectedSource}
            sourceOptions={sourceOptions}
            selectedTrackId={selectedTrackId}
            onTrackChange={setSelectedTrackId}
            tracks={initialTracks}
            sortBy={sortBy}
            onSortChange={setSortBy}
            matchedOnly={matchedOnly}
            onMatchedOnlyChange={setMatchedOnly}
            watchKeywords={feed.keywords || []}
            watchRepos={feed.watch_repos || []}
            watchSubreddits={feed.subreddits || []}
          />

          <SignalQueuePanel
            filteredItemsCount={filteredItems.length}
            refreshScheduled={feed.refresh_scheduled}
            error={error}
            cards={cards}
            selectedSignalId={selectedCard?.id ?? ""}
            onSelectSignal={setSelectedSignalId}
            nowMs={nowMs}
          />

          <SignalDetailPanel
            selectedCard={selectedCard}
            selectedItem={selectedItem}
            safeResearchHref={safeResearchHref}
            safeSourceHref={safeSourceHref}
            nowMs={nowMs}
          />
        </section>
      </div>
    </main>
  )
}
