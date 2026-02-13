import Link from "next/link"
import { ArrowRight, CalendarDays, Sparkles, Zap } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ActivityFeed } from "@/components/dashboard/ActivityFeed"
import { DashboardCommandCenter } from "@/components/dashboard/DashboardCommandCenter"
import { DeadlineRadar } from "@/components/dashboard/DeadlineRadar"
import { LLMUsageChart } from "@/components/dashboard/LLMUsageChart"
import { PipelineStatus } from "@/components/dashboard/PipelineStatus"
import { ReadingQueue } from "@/components/dashboard/ReadingQueue"
import { ScholarSignalsPanel } from "@/components/dashboard/ScholarSignalsPanel"
import { TrackSpotlight } from "@/components/dashboard/TrackSpotlight"
import {
  fetchDeadlineRadar,
  fetchLLMUsage,
  fetchPipelineTasks,
  fetchScholars,
} from "@/lib/api"
import {
  fetchDashboardActivities,
  fetchDashboardAnchors,
  fetchDashboardReadingQueue,
  fetchDashboardTrackFeed,
  fetchDashboardTracks,
} from "@/lib/dashboard-api"
import type { AnchorPreviewItem, TrackFeedItem } from "@/lib/types"

function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return "0"
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return `${Math.round(value)}`
}

export default async function DashboardPage() {
  const [tracksResult, tasksResult, readingQueueResult, llmUsageResult, deadlineResult, activitiesResult, scholarsResult] =
    await Promise.allSettled([
      fetchDashboardTracks("default"),
      fetchPipelineTasks(),
      fetchDashboardReadingQueue("default", 8),
      fetchLLMUsage(14),
      fetchDeadlineRadar("default"),
      fetchDashboardActivities("default"),
      fetchScholars(),
    ])

  const tracks = tracksResult.status === "fulfilled" ? tracksResult.value : []
  const tasks = tasksResult.status === "fulfilled" ? tasksResult.value : []
  const readingQueue = readingQueueResult.status === "fulfilled" ? readingQueueResult.value : []
  const usageSummary = llmUsageResult.status === "fulfilled"
    ? llmUsageResult.value
    : {
        window_days: 14,
        daily: [],
        provider_models: [],
        totals: { calls: 0, total_tokens: 0, total_cost_usd: 0 },
      }
  const deadlines = deadlineResult.status === "fulfilled" ? deadlineResult.value : []
  const activities = activitiesResult.status === "fulfilled" ? activitiesResult.value : []
  const scholars = scholarsResult.status === "fulfilled" ? scholarsResult.value : []

  const activeTrack = tracks.find((track) => track.is_active) || tracks[0] || null

  let feedItems: TrackFeedItem[] = []
  let feedTotal = 0
  let anchors: AnchorPreviewItem[] = []
  if (activeTrack) {
    const [feedResult, anchorsResult] = await Promise.allSettled([
      fetchDashboardTrackFeed(activeTrack.id, "default", 6),
      fetchDashboardAnchors(activeTrack.id, "default", 4),
    ])
    if (feedResult.status === "fulfilled") {
      feedItems = feedResult.value.items
      feedTotal = feedResult.value.total
    }
    if (anchorsResult.status === "fulfilled") {
      anchors = anchorsResult.value
    }
  }

  const runningPipelines = tasks.filter((task) => !["success", "failed"].includes(task.status)).length
  const deadlineSoon = deadlines.filter((item) => item.days_left <= 30).length
  const todayLabel = new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  }).format(new Date())

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 p-4 pb-10 sm:p-6">
      <Card className="border-border/60 bg-gradient-to-br from-card via-card to-muted/35">
        <CardContent className="space-y-5 p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2">
              <Badge variant="secondary" className="gap-1">
                <CalendarDays className="h-3.5 w-3.5" />
                {todayLabel}
              </Badge>
              <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Research Dashboard</h1>
              <p className="max-w-2xl text-sm text-muted-foreground">
                One place to monitor track feed quality, workflow health, LLM spend, and delivery deadlines.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button asChild>
                <Link href={activeTrack ? `/research?track_id=${activeTrack.id}` : "/research"}>
                  Continue Research
                  <ArrowRight className="ml-1 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/workflows">Run Workflow</Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/papers">Open Library</Link>
              </Button>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Active Tracks</p>
              <p className="mt-1 text-2xl font-semibold">{tracks.length}</p>
              <p className="mt-1 text-xs text-muted-foreground truncate">{activeTrack?.name || "No active track"}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Tracked Scholars</p>
              <p className="mt-1 text-2xl font-semibold">{scholars.length}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {scholars.filter((item) => item.status === "active").length} active now
              </p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Saved Queue</p>
              <p className="mt-1 text-2xl font-semibold">{readingQueue.length}</p>
              <p className="mt-1 text-xs text-muted-foreground">Linked to Papers library</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">LLM Tokens ({usageSummary.window_days}d)</p>
              <p className="mt-1 text-2xl font-semibold">{formatCompactNumber(usageSummary.totals.total_tokens)}</p>
              <p className="mt-1 text-xs text-muted-foreground flex items-center gap-1">
                <Zap className="h-3 w-3" />
                {usageSummary.totals.calls} model calls
              </p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Urgent Deadlines</p>
              <p className="mt-1 text-2xl font-semibold">{deadlineSoon}</p>
              <p className="mt-1 text-xs text-muted-foreground flex items-center gap-1">
                <Sparkles className="h-3 w-3" />
                {deadlines.length} deadlines tracked
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-12">
        <section className="space-y-6 xl:col-span-8">
          <DashboardCommandCenter
            trackCount={tracks.length}
            activeTrackName={activeTrack?.name}
            activeTrackId={activeTrack?.id}
            savedCount={readingQueue.length}
            runningPipelines={runningPipelines}
            deadlineCount={deadlines.length}
            tokenCalls={usageSummary.totals.calls}
          />

          <TrackSpotlight
            track={activeTrack}
            feedItems={feedItems}
            feedTotal={feedTotal}
            anchors={anchors}
          />

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Activity Stream</CardTitle>
            </CardHeader>
            <CardContent>
              <ActivityFeed activities={activities} maxItems={6} showTitle={false} />
            </CardContent>
          </Card>
        </section>

        <section className="space-y-6 xl:col-span-4">
          <PipelineStatus tasks={tasks} />
          <ScholarSignalsPanel scholars={scholars} />
          <ReadingQueue items={readingQueue} />
          <DeadlineRadar items={deadlines} />
          <LLMUsageChart data={usageSummary} />
        </section>
      </div>
    </div>
  )
}
