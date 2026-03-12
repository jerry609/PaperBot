"use client"

import Link from "next/link"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  ArrowUpRightIcon,
  BookOpenIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CompassIcon,
  DownloadIcon,
  FilterIcon,
  Loader2Icon,
  MailIcon,
  PlusIcon,
  PlayIcon,
  SearchIcon,
  SettingsIcon,
  SparklesIcon,
  StarIcon,
  Trash2Icon,
  TrendingUpIcon,
  WorkflowIcon,
  XIcon,
  ZapIcon,
} from "lucide-react"

import JudgeRadarChart from "@/components/research/JudgeRadarChart"
import WorkflowDagView from "@/components/research/WorkflowDagView"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { normalizeSSEMessage, readSSE } from "@/lib/sse"
import { useWorkflowStore } from "@/lib/stores/workflow-store"
import type { DailyResult, WorkflowPhase } from "@/lib/stores/workflow-store"

/* ── Types (local to component) ───────────────────────── */

type DimensionScore = { score?: number; rationale?: string }

type JudgeResult = {
  relevance?: DimensionScore
  novelty?: DimensionScore
  rigor?: DimensionScore
  impact?: DimensionScore
  clarity?: DimensionScore
  overall?: number
  recommendation?: string
  one_line_summary?: string
  judge_model?: string
  judge_cost_tier?: number
}

type SearchItem = {
  title: string
  url?: string
  score?: number
  matched_queries?: string[]
  branches?: string[]
  sources?: string[]
  ai_summary?: string
  relevance?: { score?: number; reason?: string }
  judge?: JudgeResult
}

type RepoRow = {
  title: string
  query?: string
  paper_url?: string
  repo_url: string
  github?: {
    ok?: boolean
    stars?: number
    language?: string
    updated_at?: string
    error?: string
  }
}

type StepStatus = "pending" | "running" | "done" | "error" | "skipped"

/* ── Helpers ──────────────────────────────────────────── */

const DEFAULT_QUERIES = ["In-context compression", "Implicit bias in ICL", "KV cache acceleration"]
const DAILY_STREAM_IDLE_TIMEOUT_MS = 90_000

const REC_COLORS: Record<string, string> = {
  must_read: "bg-green-100 text-green-800 border-green-300",
  worth_reading: "bg-blue-100 text-blue-800 border-blue-300",
  skim: "bg-yellow-100 text-yellow-800 border-yellow-300",
  skip: "bg-slate-100 text-slate-600 border-slate-300",
}

const REC_LABELS: Record<string, string> = {
  must_read: "Must Read",
  worth_reading: "Worth Reading",
  skim: "Skim",
  skip: "Skip",
}

const SOURCE_LABELS: Record<string, string> = {
  papers_cool: "papers.cool",
  arxiv_api: "arXiv API",
  hf_daily: "HF Daily",
}

const BRANCH_LABELS: Record<string, string> = {
  arxiv: "arXiv",
  venue: "Venue",
}

const PHASE_COPY: Record<WorkflowPhase, string> = {
  idle: "Idle",
  searching: "Searching",
  searched: "Candidates ready",
  reporting: "Building brief",
  reported: "Brief ready",
  error: "Blocked",
}

type WorkflowWorkspaceTab = "candidates" | "insights" | "judge" | "report" | "delivery" | "log"

type WorkflowDashboardContext = {
  activeTrackName?: string | null
  activeTrackHref?: string
  readingQueueCount?: number
  urgentDeadlineCount?: number
  signalCount?: number
}

const WORKFLOW_REFERENCE_LINKS: Array<{
  label: string
  description: string
  href: string
  external?: boolean
}> = [
  {
    label: "Open Papers Library",
    description: "Move from the dashboard run into saved papers and reference exports once the shortlist is stable.",
    href: "/papers",
  },
  {
    label: "Tune Providers & Delivery",
    description: "Route models and configure push channels before turning this workflow into a production ritual.",
    href: "/settings",
  },
]

function normalizeWorkflowQueries(values: string[]): string[] {
  return values.map((value) => value.trim()).filter(Boolean)
}

function shouldPersistWorkflowQueries(queries: string[]): boolean {
  const normalizedQueries = normalizeWorkflowQueries(queries)
  if (normalizedQueries.length === 0) return false

  const normalizedDefaults = normalizeWorkflowQueries(DEFAULT_QUERIES)
  if (normalizedQueries.length !== normalizedDefaults.length) return true

  return normalizedQueries.some((value, index) => value !== normalizedDefaults[index])
}

function getNextWorkflowAction(args: {
  hasSearchData: boolean
  hasReportData: boolean
  hasJudgeContent: boolean
  hasLLMContent: boolean
  isLoading: boolean
}) {
  if (args.isLoading) return "A run is in progress."
  if (!args.hasSearchData) return "Run Search."
  if (!args.hasReportData) return "Run DailyPaper."
  if (!args.hasJudgeContent && !args.hasLLMContent) return "Run Analyze."
  return "Review the shortlist and hand off the winners."
}

function formatTimestamp(value?: string | null) {
  if (!value) return "Not run yet"
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

function ScoreBar({ value, max = 5, label }: { value: number; max?: number; label: string }) {
  const pct = Math.round((value / max) * 100)
  const color = value >= 4 ? "bg-green-500" : value >= 3 ? "bg-blue-500" : value >= 2 ? "bg-yellow-500" : "bg-red-400"
  return (
    <div className="flex items-center gap-2 text-xs">
      {label && <span className="w-10 shrink-0 text-muted-foreground">{label}</span>}
      <div className="relative h-1.5 flex-1 rounded-full bg-muted">
        <div className={`absolute inset-y-0 left-0 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-4 text-right font-mono text-muted-foreground">{value}</span>
    </div>
  )
}

function StatCard({
  label,
  value,
  icon,
  compact = false,
}: {
  label: string
  value: string | number
  icon: React.ReactNode
  compact?: boolean
}) {
  return (
    <div
      className={`flex items-center gap-3 rounded-2xl border border-slate-100 bg-slate-50/80 ${
        compact ? "px-3 py-2.5" : "px-4 py-3"
      }`}
    >
      <div className={`flex items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 ${compact ? "size-8" : "size-9"}`}>
        {icon}
      </div>
      <div>
        <div className={`${compact ? "text-xl" : "text-2xl"} font-bold tabular-nums text-slate-800`}>{value}</div>
        <div className="text-xs text-slate-500">{label}</div>
      </div>
    </div>
  )
}

function getStageState(args: {
  isRunning: boolean
  isDone: boolean
  hasError: boolean
}): StepStatus {
  if (args.hasError) return "error"
  if (args.isRunning) return "running"
  if (args.isDone) return "done"
  return "pending"
}

function getStagePresentation(status: StepStatus): {
  label: string
  cardClassName: string
  badgeClassName: string
  dotClassName: string
} {
  switch (status) {
    case "done":
      return {
        label: "Ready",
        cardClassName: "border-emerald-100 bg-emerald-50/70",
        badgeClassName: "border-emerald-200 bg-emerald-50 text-emerald-700",
        dotClassName: "bg-emerald-500",
      }
    case "running":
      return {
        label: "Running",
        cardClassName: "border-indigo-100 bg-indigo-50/70",
        badgeClassName: "border-indigo-200 bg-indigo-50 text-indigo-700",
        dotClassName: "bg-indigo-500 animate-pulse",
      }
    case "error":
      return {
        label: "Blocked",
        cardClassName: "border-rose-100 bg-rose-50/70",
        badgeClassName: "border-rose-200 bg-rose-50 text-rose-700",
        dotClassName: "bg-rose-500",
      }
    case "skipped":
      return {
        label: "Skipped",
        cardClassName: "border-slate-200 bg-slate-50/70",
        badgeClassName: "border-slate-200 bg-white text-slate-500",
        dotClassName: "bg-slate-300",
      }
    default:
      return {
        label: "Pending",
        cardClassName: "border-slate-200 bg-white",
        badgeClassName: "border-slate-200 bg-white text-slate-600",
        dotClassName: "bg-slate-300",
      }
  }
}

function WorkflowStageCard({
  eyebrow,
  title,
  description,
  metric,
  status,
  icon,
}: {
  eyebrow: string
  title: string
  description?: string
  metric: string
  status: StepStatus
  icon: React.ReactNode
}) {
  const presentation = getStagePresentation(status)

  return (
    <div className={`rounded-2xl border p-4 shadow-sm ${presentation.cardClassName}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{eyebrow}</p>
          <div className="mt-2 flex items-center gap-2">
            <span className="flex size-9 items-center justify-center rounded-xl bg-white text-slate-600 shadow-sm">
              {icon}
            </span>
            <div className="min-w-0">
              <h3 className="text-base font-bold text-slate-900">{title}</h3>
            </div>
          </div>
        </div>
        <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-semibold ${presentation.badgeClassName}`}>
          <span className={`size-1.5 rounded-full ${presentation.dotClassName}`} />
          {presentation.label}
        </span>
      </div>
      {description ? <p className="mt-3 text-sm leading-relaxed text-slate-600">{description}</p> : null}
      <p className="mt-4 text-sm font-semibold text-slate-800">{metric}</p>
    </div>
  )
}

function WorkflowChip({
  label,
  active,
  onToggle,
}: {
  label: string
  active: boolean
  onToggle: () => void
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onToggle}
      className={`inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
        active
          ? "border-indigo-200 bg-indigo-50 text-indigo-700"
          : "border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:text-slate-700"
      }`}
    >
      {label}
    </button>
  )
}

function WorkflowTogglePanel({
  title,
  description,
  checked,
  onCheckedChange,
  icon,
  compact = false,
}: {
  title: string
  description?: string
  checked: boolean
  onCheckedChange: (value: boolean) => void
  icon: React.ReactNode
  compact?: boolean
}) {
  if (compact) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
              <span className="rounded-lg bg-slate-50 p-1.5 text-slate-600">{icon}</span>
              {title}
            </div>
            {description ? <p className="text-xs leading-5 text-slate-500">{description}</p> : null}
          </div>
          <Switch checked={checked} onCheckedChange={onCheckedChange} />
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            <span className="rounded-lg bg-white p-1.5 text-slate-600 shadow-sm">{icon}</span>
            {title}
          </div>
          {description ? <p className="text-xs leading-relaxed text-slate-500">{description}</p> : null}
        </div>
        <Switch checked={checked} onCheckedChange={onCheckedChange} />
      </div>
    </div>
  )
}

function WorkflowReferenceCard({
  title,
  description,
  href,
  external = false,
}: {
  title: string
  description?: string
  href: string
  external?: boolean
}) {
  const content = (
    <>
      <div>
        <p className="text-sm font-semibold text-slate-800">{title}</p>
        {description ? <p className="mt-1 text-xs leading-relaxed text-slate-500">{description}</p> : null}
      </div>
      <ArrowUpRightIcon className="size-4 text-slate-400 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-indigo-600" />
    </>
  )

  if (external) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="group flex items-start justify-between gap-3 rounded-2xl border border-slate-200 bg-white p-4 transition-colors hover:border-indigo-200 hover:bg-indigo-50/50"
      >
        {content}
      </a>
    )
  }

  return (
    <Link
      href={href}
      className="group flex items-start justify-between gap-3 rounded-2xl border border-slate-200 bg-white p-4 transition-colors hover:border-indigo-200 hover:bg-indigo-50/50"
    >
      {content}
    </Link>
  )
}

function buildDagStatuses(args: {
  phase: WorkflowPhase
  hasError: boolean
  llmIntent: boolean
  judgeIntent: boolean
  hasSearchData: boolean
  hasReportData: boolean
  hasLLMData: boolean
  hasJudgeData: boolean
  schedulerDone: boolean
}): Record<string, StepStatus> {
  const {
    phase,
    hasError,
    llmIntent,
    judgeIntent,
    hasSearchData,
    hasReportData,
    hasLLMData,
    hasJudgeData,
    schedulerDone,
  } = args

  const statuses: Record<string, StepStatus> = {
    source: hasSearchData ? "done" : "pending",
    normalize: hasSearchData ? "done" : "pending",
    search: hasSearchData ? "done" : "pending",
    rank: hasSearchData ? "done" : "pending",
    llm: "pending",
    judge: "pending",
    report: hasReportData ? "done" : "pending",
    scheduler: schedulerDone ? "done" : "pending",
  }

  if (phase === "searching") {
    statuses.source = "done"
    statuses.normalize = "done"
    statuses.search = "running"
    statuses.rank = "running"
  }

  if (phase === "reporting") {
    statuses.source = "done"
    statuses.normalize = "done"
    statuses.search = "done"
    statuses.rank = "done"
    statuses.report = "running"
  }

  if (hasLLMData) {
    statuses.llm = "done"
  } else if (phase === "reporting" && llmIntent) {
    statuses.llm = "running"
  } else if (hasReportData) {
    statuses.llm = "skipped"
  }

  if (hasJudgeData) {
    statuses.judge = "done"
  } else if (phase === "reporting" && judgeIntent) {
    statuses.judge = "running"
  } else if (hasReportData) {
    statuses.judge = "skipped"
  }

  if (phase === "error" || hasError) {
    if (!hasSearchData) {
      statuses.search = "error"
      statuses.rank = "error"
    }
    if (!hasReportData) {
      statuses.report = "error"
    }
    if (llmIntent && !hasLLMData) statuses.llm = "error"
    if (judgeIntent && !hasJudgeData) statuses.judge = "error"
  }

  return statuses
}

/* ── Stream Progress ─────────────────────────────────── */

type StreamPhase = "idle" | "search" | "build" | "llm" | "insight" | "judge" | "filter" | "save" | "notify" | "done" | "error"

const PHASE_LABELS: Record<StreamPhase, string> = {
  idle: "Idle",
  search: "Searching papers",
  build: "Building report",
  llm: "LLM enrichment",
  insight: "Generating insights",
  judge: "Judge scoring",
  filter: "Filtering papers",
  save: "Saving",
  notify: "Sending notifications",
  done: "Done",
  error: "Error",
}

const PHASE_ORDER: StreamPhase[] = ["search", "build", "llm", "insight", "judge", "filter", "save", "notify", "done"]

function useElapsed(startTime: number | null) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!startTime) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [startTime])
  if (!startTime) return 0
  return Math.max(0, Math.round((now - startTime) / 1000))
}

function StreamProgressCard({
  streamPhase,
  streamLog,
  streamProgress,
  startTime,
}: {
  streamPhase: StreamPhase
  streamLog: string[]
  streamProgress: { done: number; total: number }
  startTime: number | null
}) {
  const elapsed = useElapsed(startTime)
  const currentIdx = PHASE_ORDER.indexOf(streamPhase)
  const pct = streamProgress.total > 0
    ? Math.round((streamProgress.done / streamProgress.total) * 100)
    : currentIdx >= 0
      ? Math.round(((currentIdx + 0.5) / PHASE_ORDER.length) * 100)
      : 0

  return (
    <Card className="rounded-2xl border border-indigo-100 bg-indigo-50/50 shadow-none">
      <CardContent className="space-y-3 py-3">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 font-medium text-indigo-900">
            <Loader2Icon className="size-4 animate-spin" />
            {PHASE_LABELS[streamPhase] || streamPhase}
          </div>
          <div className="flex items-center gap-3 text-xs text-indigo-700">
            {streamProgress.total > 0 && (
              <span>{streamProgress.done}/{streamProgress.total}</span>
            )}
            {elapsed > 0 && <span>{elapsed}s</span>}
          </div>
        </div>
        <Progress value={pct} />
        <div className="flex items-center gap-1.5">
          {PHASE_ORDER.slice(0, -1).map((p) => {
            const idx = PHASE_ORDER.indexOf(p)
            const status = idx < currentIdx ? "done" : idx === currentIdx ? "active" : "pending"
            return (
              <div key={p} className="flex items-center gap-1">
                <div
                  className={`size-2 rounded-full ${
                    status === "done"
                      ? "bg-green-500"
                      : status === "active"
                        ? "bg-indigo-500 animate-pulse"
                        : "bg-slate-200"
                  }`}
                />
                <span className={`text-[10px] ${status === "active" ? "font-medium text-indigo-900" : "text-slate-500"}`}>
                  {PHASE_LABELS[p]}
                </span>
              </div>
            )
          })}
        </div>
        {streamLog.length > 0 && (
          <ScrollArea className="h-32">
            <div className="space-y-0.5 font-mono text-[11px] text-slate-600">
              {streamLog.slice(-20).map((line, idx) => (
                <div key={`sp-${streamLog.length - 20 + idx}`}>{line}</div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}

/* ── Paper Card ───────────────────────────────────────── */

function PaperCard({ item, query, onOpenDetail }: { item: SearchItem; query?: string; onOpenDetail: (item: SearchItem) => void }) {
  const judge = item.judge
  const rec = judge?.recommendation || ""
  const overall = judge?.overall ?? 0

  return (
    <div className="group cursor-pointer rounded-lg border bg-card p-4 transition-all hover:border-primary/40 hover:shadow-md" onClick={() => onOpenDetail(item)}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h4 className="line-clamp-2 text-sm font-semibold leading-snug">
            {item.url ? (
              <a href={item.url} target="_blank" rel="noreferrer" className="hover:text-primary hover:underline" onClick={(e) => e.stopPropagation()}>
                {item.title}
              </a>
            ) : (
              item.title
            )}
          </h4>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {query && <Badge variant="outline" className="text-[10px]">{query}</Badge>}
            {(item.branches || []).map((b) => (<Badge key={b} variant="secondary" className="text-[10px]">{b}</Badge>))}
            {item.score != null && <span className="text-[10px] text-muted-foreground">score {item.score.toFixed(1)}</span>}
          </div>
        </div>
        {judge && overall > 0 && (
          <div className="flex shrink-0 flex-col items-end gap-1">
            <div className="flex size-10 items-center justify-center rounded-full border-2 border-primary/30 bg-primary/5 text-sm font-bold text-primary">{overall.toFixed(1)}</div>
            {rec && <Badge variant="outline" className={`text-[10px] ${REC_COLORS[rec] || ""}`}>{REC_LABELS[rec] || rec}</Badge>}
          </div>
        )}
      </div>
      {judge && overall > 0 && (
        <div className="mt-3 grid gap-1">
          <ScoreBar value={judge.relevance?.score ?? 0} label="Rel" />
          <ScoreBar value={judge.novelty?.score ?? 0} label="Nov" />
          <ScoreBar value={judge.rigor?.score ?? 0} label="Rig" />
          <ScoreBar value={judge.impact?.score ?? 0} label="Imp" />
          <ScoreBar value={judge.clarity?.score ?? 0} label="Clr" />
        </div>
      )}
      {judge?.one_line_summary && <p className="mt-2 text-xs italic text-muted-foreground line-clamp-2">{judge.one_line_summary}</p>}
      <div className="mt-2 flex items-center justify-end text-[10px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">
        Click for details <ChevronRightIcon className="ml-0.5 size-3" />
      </div>
    </div>
  )
}

/* ── Paper Detail Dialog ──────────────────────────────── */

function PaperDetailDialog({ item, open, onClose }: { item: SearchItem | null; open: boolean; onClose: () => void }) {
  if (!item) return null
  const judge = item.judge
  const dims: Array<{ key: string; label: string; dim?: DimensionScore }> = [
    { key: "relevance", label: "Relevance", dim: judge?.relevance },
    { key: "novelty", label: "Novelty", dim: judge?.novelty },
    { key: "rigor", label: "Technical Rigor", dim: judge?.rigor },
    { key: "impact", label: "Impact Potential", dim: judge?.impact },
    { key: "clarity", label: "Clarity", dim: judge?.clarity },
  ]

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-base leading-snug pr-8">{item.title}</DialogTitle>
          <DialogDescription className="sr-only">
            Paper metadata, matched queries, and judge scoring details.
          </DialogDescription>
        </DialogHeader>
        {item.url && <a href={item.url} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline">{item.url}</a>}
        <div className="flex flex-wrap gap-1.5 mt-1">
          {(item.matched_queries || []).map((q) => (<Badge key={q} variant="outline" className="text-[10px]">{q}</Badge>))}
          {(item.branches || []).map((b) => (<Badge key={b} variant="secondary" className="text-[10px]">{b}</Badge>))}
          {item.score != null && <Badge variant="secondary" className="text-[10px]">score {item.score.toFixed(1)}</Badge>}
        </div>
        {judge && (judge.overall ?? 0) > 0 && (
          <>
            <Separator />
            <div className="grid gap-4 md:grid-cols-[200px_1fr]">
              <div>
                <JudgeRadarChart judge={judge} />
                <div className="mt-1 text-center">
                  <span className="text-2xl font-bold text-primary">{judge.overall?.toFixed(2)}</span>
                  <span className="text-sm text-muted-foreground"> / 5.0</span>
                </div>
                {judge.recommendation && <div className="mt-1 text-center"><Badge className={REC_COLORS[judge.recommendation] || ""}>{REC_LABELS[judge.recommendation] || judge.recommendation}</Badge></div>}
                {judge.judge_model && <p className="mt-2 text-center text-[10px] text-muted-foreground">Model: {judge.judge_model}</p>}
              </div>
              <div className="space-y-3">
                {judge.one_line_summary && (
                  <div className="rounded-md bg-muted/50 p-3">
                    <p className="text-sm font-medium">Summary</p>
                    <p className="mt-1 text-sm text-muted-foreground">{judge.one_line_summary}</p>
                  </div>
                )}
                {dims.map(({ key, label, dim }) => (
                  <div key={key} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{label}</span>
                      <span className="font-mono text-muted-foreground">{dim?.score ?? "-"}/5</span>
                    </div>
                    <ScoreBar value={dim?.score ?? 0} label="" />
                    {dim?.rationale && <p className="text-xs text-muted-foreground">{dim.rationale}</p>}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
        {item.ai_summary && (
          <>
            <Separator />
            <div>
              <p className="text-sm font-medium">AI Summary</p>
              <p className="mt-1 text-sm text-muted-foreground">{item.ai_summary}</p>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

/* ── Config Sheet ─────────────────────────────────────── */

function ConfigSheetBody(props: {
  topK: number; setTopK: (v: number) => void
  topN: number; setTopN: (v: number) => void
  showPerBranch: number; setShowPerBranch: (v: number) => void
  saveDaily: boolean; setSaveDaily: (v: boolean) => void
  outputDir: string; setOutputDir: (v: string) => void
  useArxiv: boolean; setUseArxiv: (v: boolean) => void
  useVenue: boolean; setUseVenue: (v: boolean) => void
  usePapersCool: boolean; setUsePapersCool: (v: boolean) => void
  useArxivApi: boolean; setUseArxivApi: (v: boolean) => void
  useHFDaily: boolean; setUseHFDaily: (v: boolean) => void
  enableLLM: boolean; setEnableLLM: (v: boolean) => void
  useSummary: boolean; setUseSummary: (v: boolean) => void
  useTrends: boolean; setUseTrends: (v: boolean) => void
  useInsight: boolean; setUseInsight: (v: boolean) => void
  useRelevance: boolean; setUseRelevance: (v: boolean) => void
  enableJudge: boolean; setEnableJudge: (v: boolean) => void
  judgeRuns: number; setJudgeRuns: (v: number) => void
  judgeMaxItems: number; setJudgeMaxItems: (v: number) => void
  judgeTokenBudget: number; setJudgeTokenBudget: (v: number) => void
  notifyEmail: string; setNotifyEmail: (v: string) => void
  notifyEnabled: boolean; setNotifyEnabled: (v: boolean) => void
  resendEnabled: boolean; setResendEnabled: (v: boolean) => void
}) {
  const {
    topK, setTopK, topN, setTopN,
    showPerBranch, setShowPerBranch, saveDaily, setSaveDaily,
    outputDir, setOutputDir, useArxiv, setUseArxiv, useVenue, setUseVenue,
    usePapersCool, setUsePapersCool, useArxivApi, setUseArxivApi, useHFDaily, setUseHFDaily, enableLLM, setEnableLLM,
    useSummary, setUseSummary, useTrends, setUseTrends,
    useInsight, setUseInsight, useRelevance, setUseRelevance,
    enableJudge, setEnableJudge, judgeRuns, setJudgeRuns,
    judgeMaxItems, setJudgeMaxItems, judgeTokenBudget, setJudgeTokenBudget,
    notifyEmail, setNotifyEmail, notifyEnabled, setNotifyEnabled,
    resendEnabled, setResendEnabled,
  } = props

  return (
    <div className="space-y-5 pr-2">
      <section className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sources &amp; Branches</Label>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-1.5 text-sm"><Checkbox checked={usePapersCool} onCheckedChange={(v) => setUsePapersCool(Boolean(v))} /> papers.cool</label>
            <label className="flex items-center gap-1.5 text-sm"><Checkbox checked={useArxivApi} onCheckedChange={(v) => setUseArxivApi(Boolean(v))} /> arXiv API</label>
            <label className="flex items-center gap-1.5 text-sm"><Checkbox checked={useHFDaily} onCheckedChange={(v) => setUseHFDaily(Boolean(v))} /> HF Daily</label>
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-1.5 text-sm"><Checkbox checked={useArxiv} onCheckedChange={(v) => setUseArxiv(Boolean(v))} /> arxiv</label>
            <label className="flex items-center gap-1.5 text-sm"><Checkbox checked={useVenue} onCheckedChange={(v) => setUseVenue(Boolean(v))} /> venue</label>
          </div>
        </section>

        <section className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Parameters</Label>
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1"><Label className="text-xs">Top K</Label><Input type="number" min={1} value={topK} onChange={(e) => setTopK(Number(e.target.value || 5))} className="h-8 text-sm" /></div>
            <div className="space-y-1"><Label className="text-xs">Show / Branch</Label><Input type="number" min={1} value={showPerBranch} onChange={(e) => setShowPerBranch(Number(e.target.value || 25))} className="h-8 text-sm" /></div>
            <div className="space-y-1"><Label className="text-xs">Daily Top N</Label><Input type="number" min={1} value={topN} onChange={(e) => setTopN(Number(e.target.value || 10))} className="h-8 text-sm" /></div>
          </div>
        </section>

        <section className="space-y-2">
          <label className="flex items-center gap-2 text-sm"><Checkbox checked={saveDaily} onCheckedChange={(v) => setSaveDaily(Boolean(v))} /> Save DailyPaper files</label>
          {saveDaily && <Input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} placeholder="./reports/dailypaper" className="h-8 text-sm" />}
        </section>

        <Separator />

        <section className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium">
            <Checkbox checked={enableLLM} onCheckedChange={(v) => setEnableLLM(Boolean(v))} />
            <SparklesIcon className="size-4" /> LLM Analysis
          </label>
          {enableLLM && (
            <div className="ml-6 grid grid-cols-2 gap-2 text-sm">
              <label className="flex items-center gap-1.5"><Checkbox checked={useSummary} onCheckedChange={(v) => setUseSummary(Boolean(v))} /> Summary</label>
              <label className="flex items-center gap-1.5"><Checkbox checked={useTrends} onCheckedChange={(v) => setUseTrends(Boolean(v))} /> Trends</label>
              <label className="flex items-center gap-1.5"><Checkbox checked={useInsight} onCheckedChange={(v) => setUseInsight(Boolean(v))} /> Insight</label>
              <label className="flex items-center gap-1.5"><Checkbox checked={useRelevance} onCheckedChange={(v) => setUseRelevance(Boolean(v))} /> Relevance</label>
            </div>
          )}
        </section>

        <section className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium">
            <Checkbox checked={enableJudge} onCheckedChange={(v) => setEnableJudge(Boolean(v))} />
            <StarIcon className="size-4" /> LLM Judge
          </label>
          {enableJudge && (
            <div className="ml-6 grid grid-cols-3 gap-2">
              <div className="space-y-1"><Label className="text-xs">Runs</Label><Input type="number" min={1} max={5} value={judgeRuns} onChange={(e) => setJudgeRuns(Number(e.target.value || 1))} className="h-8 text-sm" /></div>
              <div className="space-y-1"><Label className="text-xs">Max Items</Label><Input type="number" min={1} max={200} value={judgeMaxItems} onChange={(e) => setJudgeMaxItems(Number(e.target.value || 20))} className="h-8 text-sm" /></div>
              <div className="space-y-1"><Label className="text-xs">Token Budget</Label><Input type="number" min={0} value={judgeTokenBudget} onChange={(e) => setJudgeTokenBudget(Number(e.target.value || 0))} className="h-8 text-sm" /></div>
            </div>
          )}
        </section>

        <Separator />

        <section className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium">
            <Checkbox checked={notifyEnabled} onCheckedChange={(v) => setNotifyEnabled(Boolean(v))} />
            <MailIcon className="size-4" /> Email Notification
          </label>
          {notifyEnabled && (
            <div className="ml-6 space-y-2">
              <div className="space-y-1">
                <Label className="text-xs">Email Address</Label>
                <Input
                  type="email"
                  value={notifyEmail}
                  onChange={(e) => setNotifyEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="h-8 text-sm"
                />
              </div>
              <p className="text-[10px] text-muted-foreground">
                Requires PAPERBOT_NOTIFY_SMTP_* env vars on the backend. The email address here overrides PAPERBOT_NOTIFY_EMAIL_TO.
              </p>
            </div>
          )}
        </section>

        <section className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium">
            <Checkbox checked={resendEnabled} onCheckedChange={(v) => setResendEnabled(Boolean(v))} />
            <MailIcon className="size-4" /> Newsletter (Resend)
          </label>
          {resendEnabled && (
            <div className="ml-6 space-y-2">
              <p className="text-[10px] text-muted-foreground">
                Send digest to all newsletter subscribers via Resend API. Requires PAPERBOT_RESEND_API_KEY env var.
              </p>
              <NewsletterSubscribeWidget />
            </div>
          )}
        </section>
    </div>
  )
}

/* ── Newsletter Subscribe Widget ─────────────────────── */

function NewsletterSubscribeWidget() {
  const [email, setEmail] = useState("")
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle")
  const [message, setMessage] = useState("")
  const [subCount, setSubCount] = useState<{ active: number; total: number } | null>(null)

  const fetchCount = useCallback(async (): Promise<{ active: number; total: number } | null> => {
    try {
      const res = await fetch("/api/newsletter/subscribers")
      if (!res.ok) return null
      return await res.json()
    } catch { /* ignore */ }
    return null
  }, [])

  useEffect(() => {
    let cancelled = false
    void fetchCount().then((data) => {
      if (!cancelled && data) setSubCount(data)
    })
    return () => {
      cancelled = true
    }
  }, [fetchCount])

  async function handleSubscribe() {
    if (!email.trim()) return
    setStatus("loading"); setMessage("")
    try {
      const res = await fetch("/api/newsletter/subscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim() }),
      })
      const data = await res.json()
      if (res.ok) {
        setStatus("ok"); setMessage(data.message || "Subscribed!"); setEmail("")
        const latest = await fetchCount()
        if (latest) setSubCount(latest)
      } else {
        setStatus("error"); setMessage(data.detail || "Failed to subscribe")
      }
    } catch (err) {
      setStatus("error"); setMessage(String(err))
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5">
        <Input
          type="email"
          value={email}
          onChange={(e) => { setEmail(e.target.value); setStatus("idle") }}
          placeholder="subscriber@example.com"
          className="h-8 text-sm"
          onKeyDown={(e) => e.key === "Enter" && handleSubscribe()}
        />
        <Button size="sm" className="h-8 text-xs" onClick={handleSubscribe} disabled={status === "loading"}>
          {status === "loading" ? <Loader2Icon className="size-3 animate-spin" /> : "Subscribe"}
        </Button>
      </div>
      {message && (
        <p className={`text-[10px] ${status === "ok" ? "text-green-600" : "text-destructive"}`}>{message}</p>
      )}
      {subCount && (
        <p className="text-[10px] text-muted-foreground">{subCount.active} active subscriber{subCount.active !== 1 ? "s" : ""}</p>
      )}
    </div>
  )
}

/* ── Main Dashboard ───────────────────────────────────── */

type TopicWorkflowDashboardProps = {
  initialQueries?: string[]
  dashboardContext?: WorkflowDashboardContext
  compact?: boolean
}

export default function TopicWorkflowDashboard({
  initialQueries,
  dashboardContext,
  compact = false,
}: TopicWorkflowDashboardProps = {}) {
  /* Config state (local — queries only) */
  const [queryItems, setQueryItems] = useState<string[]>([
    ...((initialQueries && initialQueries.length ? initialQueries : DEFAULT_QUERIES) || DEFAULT_QUERIES),
  ])

  /* Persisted state (zustand) */
  const store = useWorkflowStore()
  const { searchResult, dailyResult, phase, analyzeLog, notifyEmail, notifyEnabled, config } = store
  const resendEnabled = store.resendEnabled
  const uc = store.updateConfig

  /* Derived config accessors — read from persisted store */
  const topK = config.topK
  const setTopK = (v: number) => uc({ topK: v })
  const topN = config.topN
  const setTopN = (v: number) => uc({ topN: v })
  const showPerBranch = config.showPerBranch
  const setShowPerBranch = (v: number) => uc({ showPerBranch: v })
  const saveDaily = config.saveDaily
  const setSaveDaily = (v: boolean) => uc({ saveDaily: v })
  const outputDir = config.outputDir
  const setOutputDir = (v: string) => uc({ outputDir: v })
  const useArxiv = config.useArxiv
  const setUseArxiv = (v: boolean) => uc({ useArxiv: v })
  const useVenue = config.useVenue
  const setUseVenue = (v: boolean) => uc({ useVenue: v })
  const usePapersCool = config.usePapersCool
  const setUsePapersCool = (v: boolean) => uc({ usePapersCool: v })
  const useArxivApi = config.useArxivApi
  const setUseArxivApi = (v: boolean) => uc({ useArxivApi: v })
  const useHFDaily = config.useHFDaily
  const setUseHFDaily = (v: boolean) => uc({ useHFDaily: v })
  const enableLLM = config.enableLLM
  const setEnableLLM = (v: boolean) => uc({ enableLLM: v })
  const useSummary = config.useSummary
  const setUseSummary = (v: boolean) => uc({ useSummary: v })
  const useTrends = config.useTrends
  const setUseTrends = (v: boolean) => uc({ useTrends: v })
  const useInsight = config.useInsight
  const setUseInsight = (v: boolean) => uc({ useInsight: v })
  const useRelevance = config.useRelevance
  const setUseRelevance = (v: boolean) => uc({ useRelevance: v })
  const enableJudge = config.enableJudge
  const setEnableJudge = (v: boolean) => uc({ enableJudge: v })
  const judgeRuns = config.judgeRuns
  const setJudgeRuns = (v: number) => uc({ judgeRuns: v })
  const judgeMaxItems = config.judgeMaxItems
  const setJudgeMaxItems = (v: number) => uc({ judgeMaxItems: v })
  const judgeTokenBudget = config.judgeTokenBudget
  const setJudgeTokenBudget = (v: number) => uc({ judgeTokenBudget: v })

  /* Transient loading state (not persisted) */
  const [loadingSearch, setLoadingSearch] = useState(false)
  const [loadingDaily, setLoadingDaily] = useState(false)
  const [loadingAnalyze, setLoadingAnalyze] = useState(false)
  const [analyzeProgress, setAnalyzeProgress] = useState({ done: 0, total: 0 })
  const [loadingRepos, setLoadingRepos] = useState(false)
  const [repoRows, setRepoRows] = useState<RepoRow[]>([])
  const [repoError, setRepoError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  /* Stream progress state */
  const [streamPhase, setStreamPhase] = useState<StreamPhase>("idle")
  const [streamLog, setStreamLog] = useState<string[]>([])
  const [streamProgress, setStreamProgress] = useState({ done: 0, total: 0 })
  const streamStartRef = useRef<number | null>(null)
  const streamAbortRef = useRef<AbortController | null>(null)

  const addStreamLog = useCallback((line: string) => {
    setStreamLog((prev) => [...prev.slice(-50), line])
  }, [])

  /* UI state */
  const [dagOpen, setDagOpen] = useState(false)
  const [compactOptionsOpen, setCompactOptionsOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [selectedPaper, setSelectedPaper] = useState<SearchItem | null>(null)
  const [sortBy, setSortBy] = useState<"score" | "judge">("score")
  const [workspaceTab, setWorkspaceTab] = useState<WorkflowWorkspaceTab>("candidates")

  const queries = useMemo(() => queryItems.map((q) => q.trim()).filter(Boolean), [queryItems])
  const branches = useMemo(() => [useArxiv ? "arxiv" : "", useVenue ? "venue" : ""].filter(Boolean), [useArxiv, useVenue])
  const sources = useMemo(
    () => [
      usePapersCool ? "papers_cool" : "",
      useArxivApi ? "arxiv_api" : "",
      useHFDaily ? "hf_daily" : "",
    ].filter(Boolean),
    [usePapersCool, useArxivApi, useHFDaily],
  )
  const llmFeatures = useMemo(
    () => [useSummary ? "summary" : "", useTrends ? "trends" : "", useInsight ? "insight" : "", useRelevance ? "relevance" : ""].filter(Boolean),
    [useInsight, useRelevance, useSummary, useTrends],
  )

  const hasSearchData = Boolean((searchResult?.items?.length || 0) > 0 || dailyResult?.report)
  const hasReportData = Boolean(dailyResult?.report)
  const hasLLMData = Boolean(
    (dailyResult?.report?.llm_analysis?.daily_insight || "").trim() ||
      (dailyResult?.report?.llm_analysis?.query_trends?.length || 0) > 0,
  )
  const hasJudgeData = Boolean(
    dailyResult?.report?.judge?.enabled ||
      (dailyResult?.report?.queries || []).some((query) =>
        (query.top_items || []).some((item) => (item.judge?.overall || 0) > 0),
      ),
  )
  const schedulerDone = Boolean(dailyResult?.markdown_path)

  const dagStatuses = useMemo(
    () =>
      buildDagStatuses({
        phase,
        hasError: Boolean(error),
        llmIntent: enableLLM,
        judgeIntent: enableJudge,
        hasSearchData,
        hasReportData,
        hasLLMData,
        hasJudgeData,
        schedulerDone,
      }),
    [phase, error, enableLLM, enableJudge, hasSearchData, hasReportData, hasLLMData, hasJudgeData, schedulerDone],
  )

  const paperDataSource = dailyResult?.report?.queries ? "dailypaper" : searchResult?.items ? "search" : null

  const allPapers = useMemo(() => {
    const items: Array<SearchItem & { _query?: string }> = []
    if (dailyResult?.report?.queries) {
      for (const q of dailyResult.report.queries) {
        for (const item of q.top_items || []) {
          items.push({ ...item, _query: q.normalized_query || q.raw_query })
        }
      }
    } else if (searchResult?.items) {
      for (const item of searchResult.items) { items.push(item) }
    }
    const seen = new Set<string>()
    const deduped = items.filter((i) => { if (seen.has(i.title)) return false; seen.add(i.title); return true })
    if (sortBy === "judge") { deduped.sort((a, b) => (b.judge?.overall ?? 0) - (a.judge?.overall ?? 0)) }
    else { deduped.sort((a, b) => (b.score ?? 0) - (a.score ?? 0)) }
    return deduped
  }, [dailyResult, searchResult, sortBy])

  const judgedPapersCount = allPapers.filter((p) => (p.judge?.overall ?? 0) > 0).length
  const hasInsightData = Boolean((dailyResult?.report?.llm_analysis?.daily_insight || "").trim())
  const hasTrendData = (dailyResult?.report?.llm_analysis?.query_trends || []).length > 0
  const hasLLMContent = hasInsightData || hasTrendData
  const hasJudgeContent = hasJudgeData || judgedPapersCount > 0

  const queryHighlightRows = useMemo(() => {
    const rows: Array<{
      query: string
      title: string
      score: number
      recommendation: string
      url: string
    }> = []
    const queriesList = dailyResult?.report?.queries || []
    for (const q of queriesList) {
      const queryName = q.normalized_query || q.raw_query || "-"
      for (const item of (q.top_items || []).slice(0, 5)) {
        rows.push({
          query: queryName,
          title: item.title || "Untitled",
          score: Number(item.score || 0),
          recommendation: item.judge?.recommendation || "-",
          url: item.url || "",
        })
      }
    }
    return rows
  }, [dailyResult])

  const globalTopRows = useMemo(() => {
    return (dailyResult?.report?.global_top || []).slice(0, 10).map((item, index) => ({
      rank: index + 1,
      title: item.title || "Untitled",
      score: Number(item.score || 0),
      queries: (item.matched_queries || []).join(", ") || "-",
      url: item.url || "",
    }))
  }, [dailyResult])

  function updateQuery(index: number, value: string) {
    setQueryItems((prev) => {
      const next = [...prev]
      next[index] = value
      return next
    })
  }

  function removeQuery(index: number) {
    setQueryItems((prev) => {
      if (prev.length <= 1) return prev
      return prev.filter((_, itemIndex) => itemIndex !== index)
    })
  }

  function addQuery() {
    setQueryItems((prev) => [...prev, ""])
  }

  /* Actions */
  async function runTopicSearch() {
    setWorkspaceTab("candidates")
    setLoadingSearch(true); setError(null); store.setPhase("searching")
    store.setDailyResult(null); store.clearAnalyzeLog()
    try {
      const res = await fetch("/api/research/paperscool/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ queries, sources, branches, top_k_per_query: topK, show_per_branch: showPerBranch }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      store.setSearchResult(data)
      store.setPhase("searched")
    } catch (err) { setError(String(err)); store.setPhase("error") } finally { setLoadingSearch(false) }
  }

  async function runDailyPaperStream() {
    setWorkspaceTab("report")
    streamAbortRef.current?.abort()
    const controller = new AbortController()
    streamAbortRef.current = controller
    setLoadingDaily(true); setError(null); setRepoRows([]); setRepoError(null)
    store.setPhase("reporting"); store.clearAnalyzeLog()
    setStreamPhase("search"); setStreamLog([]); setStreamProgress({ done: 0, total: 0 })
    streamStartRef.current = Date.now()

    const requestBody = {
      queries, sources, branches, top_k_per_query: topK, show_per_branch: showPerBranch, top_n: topN,
      title: "DailyPaper Digest", formats: ["both"], save: saveDaily, output_dir: outputDir,
      enable_llm_analysis: enableLLM, llm_features: llmFeatures,
      enable_judge: enableJudge, judge_runs: judgeRuns,
      judge_max_items_per_query: judgeMaxItems, judge_token_budget: judgeTokenBudget,
      notify: notifyEnabled || resendEnabled,
      notify_channels: [...(notifyEnabled ? ["email"] : []), ...(resendEnabled ? ["resend"] : [])],
      notify_email_to: notifyEnabled && notifyEmail.trim() ? [notifyEmail.trim()] : [],
    }

    let streamFailed = false
    let streamIdleTimedOut = false
    let streamIdleTimer: ReturnType<typeof setTimeout> | null = null
    const clearStreamIdleTimer = () => {
      if (streamIdleTimer) {
        clearTimeout(streamIdleTimer)
        streamIdleTimer = null
      }
    }
    const armStreamIdleTimer = () => {
      clearStreamIdleTimer()
      streamIdleTimer = setTimeout(() => {
        streamIdleTimedOut = true
        controller.abort()
      }, DAILY_STREAM_IDLE_TIMEOUT_MS)
    }
    try {
      const res = await fetch("/api/research/paperscool/daily", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream, application/json" },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(await res.text())

      const contentType = res.headers.get("content-type") || ""

      // JSON fallback (fast path — no LLM/Judge)
      if (!contentType.includes("text/event-stream")) {
        const data = await res.json()
        store.setDailyResult(data)
        store.setPhase("reported")
        setStreamPhase("done")
        return
      }

      // SSE streaming path
      if (!res.body) throw new Error("No response body for SSE stream")

      armStreamIdleTimer()
      for await (const rawEvent of readSSE(res.body)) {
        armStreamIdleTimer()
        const event = normalizeSSEMessage(rawEvent, "paperscool_daily")
        if (event.type === "progress") {
          const d = (event.data || {}) as { phase?: string; message?: string; total?: number }
          const p = (d.phase || "search") as StreamPhase
          setStreamPhase(p)
          addStreamLog(`[${p}] ${d.message || "running"}`)
          if (d.total && d.total > 0) {
            setStreamProgress({ done: 0, total: d.total })
          }
          continue
        }

        if (event.type === "search_done") {
          const d = (event.data || {}) as { items_count?: number; unique_items?: number }
          addStreamLog(`search done: ${d.unique_items || 0} unique papers`)
          setStreamPhase("build")
          continue
        }

        if (event.type === "report_built") {
          const d = (event.data || {}) as { report?: DailyResult["report"]; queries_count?: number; global_top_count?: number }
          addStreamLog(`report built: ${d.queries_count || 0} queries, ${d.global_top_count || 0} global top`)
          if (d.report) {
            store.setDailyResult({ report: d.report, markdown: "" })
          }
          continue
        }

        if (event.type === "llm_summary") {
          const d = (event.data || {}) as { title?: string; query?: string; ai_summary?: string; done?: number; total?: number }
          setStreamProgress({ done: d.done || 0, total: d.total || 0 })
          addStreamLog(`summary ${d.done || 0}/${d.total || 0}: ${d.title || "paper"}`)
          if (d.query && d.title && d.ai_summary) {
            store.updateDailyResult((prev) => {
              const nextQueries = (prev.report.queries || []).map((query) => {
                const queryName = query.normalized_query || query.raw_query || ""
                if (queryName !== d.query) return query
                const nextItems = (query.top_items || []).map((item) => {
                  if (item.title === d.title) return { ...item, ai_summary: d.ai_summary }
                  return item
                })
                return { ...query, top_items: nextItems }
              })
              return { ...prev, report: { ...prev.report, queries: nextQueries } }
            })
          }
          continue
        }

        if (event.type === "trend") {
          const d = (event.data || {}) as { query?: string; analysis?: string; done?: number; total?: number }
          addStreamLog(`trend ${d.done || 0}/${d.total || 0}: ${d.query || "query"}`)
          if (d.query && typeof d.analysis === "string") {
            store.updateDailyResult((prev) => {
              const llmAnalysis = prev.report.llm_analysis || { enabled: true, features: [], daily_insight: "", query_trends: [] }
              const features = new Set(llmAnalysis.features || [])
              features.add("trends")
              const trendList = [...(llmAnalysis.query_trends || [])]
              const existingIndex = trendList.findIndex((item) => item.query === d.query)
              if (existingIndex >= 0) {
                trendList[existingIndex] = { query: d.query!, analysis: d.analysis! }
              } else {
                trendList.push({ query: d.query!, analysis: d.analysis! })
              }
              return {
                ...prev,
                report: {
                  ...prev.report,
                  llm_analysis: { ...llmAnalysis, enabled: true, features: Array.from(features), query_trends: trendList },
                },
              }
            })
          }
          continue
        }

        if (event.type === "insight") {
          const d = (event.data || {}) as { analysis?: string }
          addStreamLog("insight generated")
          if (typeof d.analysis === "string") {
            store.updateDailyResult((prev) => {
              const llmAnalysis = prev.report.llm_analysis || { enabled: true, features: [], daily_insight: "", query_trends: [] }
              const features = new Set(llmAnalysis.features || [])
              features.add("insight")
              return {
                ...prev,
                report: {
                  ...prev.report,
                  llm_analysis: { ...llmAnalysis, enabled: true, features: Array.from(features), daily_insight: d.analysis! },
                },
              }
            })
          }
          continue
        }

        if (event.type === "llm_done") {
          const d = (event.data || {}) as { summaries_count?: number; trends_count?: number }
          addStreamLog(`LLM done: ${d.summaries_count || 0} summaries, ${d.trends_count || 0} trends`)
          setStreamPhase("judge")
          continue
        }

        if (event.type === "judge") {
          const d = (event.data || {}) as { query?: string; title?: string; judge?: SearchItem["judge"]; done?: number; total?: number }
          setStreamProgress({ done: d.done || 0, total: d.total || 0 })
          setStreamPhase("judge")
          const rec = d.judge?.recommendation || "?"
          const overall = d.judge?.overall != null ? Number(d.judge.overall).toFixed(2) : "?"
          addStreamLog(`judge ${d.done || 0}/${d.total || 0}: [${rec} ${overall}] ${d.title || "paper"} (${d.query || ""})`)
          // TODO: refactor judge update — current nested map + matched flag is hard
          //  to follow. Use findIndex to locate target query+item, then apply a
          //  single immutable update. See PR #25 review for suggested approach.
          if (d.query && d.title && d.judge) {
            store.updateDailyResult((prev) => {
              const sourceQueries = prev.report.queries || []
              let matched = false
              const nextQueries = sourceQueries.map((query) => {
                const queryName = query.normalized_query || query.raw_query || ""
                if (queryName !== d.query) return query
                const nextItems = (query.top_items || []).map((item) => {
                  if (item.title === d.title) { matched = true; return { ...item, judge: d.judge } }
                  return item
                })
                return { ...query, top_items: nextItems }
              })
              if (!matched) {
                const fallbackQueries = nextQueries.map((query) => {
                  if (matched) return query
                  const nextItems = (query.top_items || []).map((item) => {
                    if (!matched && item.title === d.title) { matched = true; return { ...item, judge: d.judge } }
                    return item
                  })
                  return { ...query, top_items: nextItems }
                })
                return { ...prev, report: { ...prev.report, queries: fallbackQueries } }
              }
              return { ...prev, report: { ...prev.report, queries: nextQueries } }
            })
          }
          continue
        }

        if (event.type === "judge_done") {
          const d = (event.data || {}) as DailyResult["report"]["judge"]
          store.updateDailyResult((prev) => ({
            ...prev,
            report: { ...prev.report, judge: d || prev.report.judge },
          }))
          addStreamLog("judge scoring complete")
          continue
        }

        if (event.type === "filter_done") {
          const d = (event.data || {}) as {
            total_before?: number
            total_after?: number
            removed_count?: number
            log?: Array<{ query?: string; title?: string; recommendation?: string; overall?: number; action?: string }>
          }
          setStreamPhase("filter")
          addStreamLog(`filter: ${d.total_before || 0} papers -> ${d.total_after || 0} kept, ${d.removed_count || 0} removed`)
          if (d.log) {
            for (const entry of d.log) {
              addStreamLog(`  removed [${entry.recommendation || "?"}] ${entry.title || "?"} (${entry.query || ""})`)
            }
          }
          // Update the store with filtered report — the next "result" event will have the final state
          // but we can also re-fetch queries from the filter event if needed
          continue
        }

        if (event.type === "result") {
          const d = (event.data || {}) as {
            report?: DailyResult["report"]
            markdown?: string
            markdown_path?: string | null
            json_path?: string | null
            notify_result?: Record<string, unknown> | null
          }
          if (d.report) {
            store.setDailyResult({
              report: d.report,
              markdown: typeof d.markdown === "string" ? d.markdown : "",
              markdown_path: d.markdown_path,
              json_path: d.json_path,
            })
          }
          setStreamPhase("done")
          addStreamLog("stream complete")
          continue
        }

        if (event.type === "error") {
          const d = (event.data || {}) as { message?: string; detail?: string }
          const msg = event.message || d.message || d.detail || "Unknown stream error"
          addStreamLog(`[error] ${msg}`)
          setError(`DailyPaper failed: ${msg}`)
          streamFailed = true
          setStreamPhase("error")
          store.setPhase("error")
          break
        }
      }
      clearStreamIdleTimer()
      if (!streamFailed) {
        store.setPhase("reported")
      }
    } catch (err) {
      streamFailed = true
      if (streamIdleTimedOut) {
        const timeoutSec = Math.round(DAILY_STREAM_IDLE_TIMEOUT_MS / 1000)
        const message = `DailyPaper stream stalled for ${timeoutSec}s and was aborted.`
        addStreamLog(`[error] ${message}`)
        setError(message)
      } else {
        setError(String(err))
      }
      setStreamPhase("error")
      store.setPhase("error")
    } finally {
      clearStreamIdleTimer()
      setLoadingDaily(false)
      streamStartRef.current = null
      streamAbortRef.current = null
    }
  }

  async function runAnalyzeStream() {
    if (!dailyResult?.report) { setError("Generate DailyPaper first."); return }
    const runJudge = Boolean(enableJudge)
    const runTrends = Boolean(enableLLM && useTrends)
    const runInsight = Boolean(enableLLM && useInsight)
    if (!runJudge && !runTrends && !runInsight) { setError("Enable Judge, LLM trends, or LLM insight before analyzing."); return }

    streamAbortRef.current?.abort()
    const controller = new AbortController()
    streamAbortRef.current = controller
    setWorkspaceTab(runJudge ? "judge" : "insights")
    setLoadingAnalyze(true); setError(null); store.clearAnalyzeLog(); setAnalyzeProgress({ done: 0, total: 0 }); store.setPhase("reporting")
    setStreamPhase("idle"); setStreamLog([]); setStreamProgress({ done: 0, total: 0 })
    streamStartRef.current = Date.now()
    store.addAnalyzeLog(
      `[start] run_judge=${runJudge} run_trends=${runTrends} run_insight=${runInsight} llm_enabled=${enableLLM} judge_enabled=${enableJudge}`,
    )
    if (enableLLM && !useTrends && !useInsight) {
      store.addAnalyzeLog("[hint] Analyze stream currently supports trends and daily insight.")
    }

    let streamFailed = false
    try {
      const res = await fetch("/api/research/paperscool/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          report: dailyResult.report, run_judge: runJudge, run_trends: runTrends, run_insight: runInsight,
          judge_runs: judgeRuns, judge_max_items_per_query: judgeMaxItems,
          judge_token_budget: judgeTokenBudget, trend_max_items_per_query: 3,
        }),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) throw new Error(await res.text())

      for await (const rawEvent of readSSE(res.body)) {
        const event = normalizeSSEMessage(rawEvent, "paperscool_analyze")
        if (event.type === "progress") {
          const d = (event.data || {}) as { phase?: string; message?: string; total?: number }
          const trace = event.envelope.trace_id ? ` trace=${event.envelope.trace_id}` : ""
          store.addAnalyzeLog(`[${d.phase || "step"}] ${d.message || "running"}${trace}`)
          if (d.phase === "judge" && (d.total || 0) > 0) {
            setAnalyzeProgress({ done: 0, total: d.total || 0 })
          }
          continue
        }

        if (event.type === "trend") {
          const d = (event.data || {}) as {
            query?: string
            analysis?: string
            done?: number
            total?: number
          }
          store.addAnalyzeLog(`trend ${d.done || 0}/${d.total || 0}: ${d.query || "query"}`)
          const trendQuery = d.query
          const trendAnalysis = d.analysis
          if (trendQuery && typeof trendAnalysis === "string") {
            store.updateDailyResult((prev) => {
              const llmAnalysis = prev.report.llm_analysis || {
                enabled: true,
                features: [],
                daily_insight: "",
                query_trends: [],
              }
              const features = new Set(llmAnalysis.features || [])
              features.add("trends")
              const trendList = [...(llmAnalysis.query_trends || [])]
              const existingIndex = trendList.findIndex((item) => item.query === trendQuery)
              if (existingIndex >= 0) {
                trendList[existingIndex] = { query: trendQuery, analysis: trendAnalysis }
              } else {
                trendList.push({ query: trendQuery, analysis: trendAnalysis })
              }
              return {
                ...prev,
                report: {
                  ...prev.report,
                  llm_analysis: {
                    ...llmAnalysis,
                    enabled: true,
                    features: Array.from(features),
                    query_trends: trendList,
                  },
                },
              }
            })
          }
          continue
        }

        if (event.type === "insight") {
          const d = (event.data || {}) as {
            analysis?: string
          }
          const insight = d.analysis
          store.addAnalyzeLog("insight generated")
          if (typeof insight === "string") {
            store.updateDailyResult((prev) => {
              const llmAnalysis = prev.report.llm_analysis || {
                enabled: true,
                features: [],
                daily_insight: "",
                query_trends: [],
              }
              const features = new Set(llmAnalysis.features || [])
              features.add("insight")
              return {
                ...prev,
                report: {
                  ...prev.report,
                  llm_analysis: {
                    ...llmAnalysis,
                    enabled: true,
                    features: Array.from(features),
                    daily_insight: insight,
                  },
                },
              }
            })
          }
          continue
        }

        if (event.type === "judge") {
          const d = (event.data || {}) as {
            query?: string
            title?: string
            judge?: SearchItem["judge"]
            done?: number
            total?: number
          }
          setAnalyzeProgress({ done: d.done || 0, total: d.total || 0 })
          store.addAnalyzeLog(`judge ${d.done || 0}/${d.total || 0}: ${d.title || "paper"}`)

          if (d.query && d.title && d.judge) {
            store.updateDailyResult((prev) => {
              const sourceQueries = prev.report.queries || []
              let matched = false
              const nextQueries = sourceQueries.map((query) => {
                const queryName = query.normalized_query || query.raw_query || ""
                if (queryName !== d.query) {
                  return query
                }
                const nextItems = (query.top_items || []).map((item) => {
                  if (item.title === d.title) {
                    matched = true
                    return { ...item, judge: d.judge }
                  }
                  return item
                })
                return { ...query, top_items: nextItems }
              })

              if (!matched) {
                const fallbackQueries = nextQueries.map((query) => {
                  if (matched) {
                    return query
                  }
                  const nextItems = (query.top_items || []).map((item) => {
                    if (!matched && item.title === d.title) {
                      matched = true
                      return { ...item, judge: d.judge }
                    }
                    return item
                  })
                  return { ...query, top_items: nextItems }
                })
                return {
                  ...prev,
                  report: {
                    ...prev.report,
                    queries: fallbackQueries,
                  },
                }
              }

              return {
                ...prev,
                report: {
                  ...prev.report,
                  queries: nextQueries,
                },
              }
            })
          }
          continue
        }

        if (event.type === "judge_done") {
          const d = (event.data || {}) as DailyResult["report"]["judge"]
          store.updateDailyResult((prev) => ({
            ...prev,
            report: {
              ...prev.report,
              judge: d || prev.report.judge,
            },
          }))
          continue
        }

        if (event.type === "result") {
          const d = (event.data || {}) as { report?: DailyResult["report"]; markdown?: string }
          if (d.report) {
            store.updateDailyResult((prev) => ({
              ...prev,
              report: d.report || prev.report,
              markdown: typeof d.markdown === "string" ? d.markdown : prev.markdown,
            }))
          }
          continue
        }

        if (event.type === "error") {
          const d = (event.data || {}) as { message?: string; detail?: string }
          const msg = event.message || d.message || d.detail || "Unknown analyze stream error"
          store.addAnalyzeLog(`[error] ${msg}`)
          setError(`Analyze failed: ${msg}`)
          streamFailed = true
          store.setPhase("error")
          break
        }
      }
      if (!streamFailed) {
        store.setPhase("reported")
      }
    } catch (err) {
      streamFailed = true
      setError(String(err))
      store.setPhase("error")
    } finally {
      setLoadingAnalyze(false)
      streamStartRef.current = null
      streamAbortRef.current = null
    }
  }

  async function runRepoEnrichment() {
    if (!dailyResult?.report) {
      setRepoError("Generate DailyPaper first.")
      return
    }

    setWorkspaceTab("delivery")
    setLoadingRepos(true)
    setRepoError(null)
    try {
      const res = await fetch("/api/research/paperscool/repos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          report: dailyResult.report,
          max_items: 500,
          include_github_api: true,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const payload = await res.json() as { repos?: RepoRow[] }
      setRepoRows(payload.repos || [])
    } catch (err) {
      setRepoError(String(err))
    } finally {
      setLoadingRepos(false)
    }
  }

  const isLoading = loadingSearch || loadingDaily || loadingAnalyze
  const hasWorkspaceOutput = isLoading || hasSearchData || hasReportData || hasJudgeContent || hasLLMContent
  const canSearch = queries.length > 0 && branches.length > 0 && sources.length > 0
  const loadingLabel = loadingSearch
    ? "Searching sources..."
    : "Running judge/trend/insight enrichment..."
  const loadingHint = loadingAnalyze && analyzeProgress.total > 0
    ? `${analyzeProgress.done}/${analyzeProgress.total} judged`
    : loadingSearch
      ? "Multi-query retrieval in progress"
      : "Waiting for LLM events"
  const activeSourceLabels = sources.map((source) => SOURCE_LABELS[source] || source)
  const activeBranchLabels = branches.map((branch) => BRANCH_LABELS[branch] || branch)
  const quickSummaryBadges = [
    `${queries.length} topic${queries.length === 1 ? "" : "s"}`,
    activeSourceLabels.length ? activeSourceLabels.join(" · ") : "No sources",
    activeBranchLabels.length ? activeBranchLabels.join(" · ") : "No branches",
    enableLLM ? "LLM on" : "LLM off",
    enableJudge ? "Judge on" : "Judge off",
  ]
  const searchStageState = getStageState({
    isRunning: loadingSearch,
    isDone: hasSearchData,
    hasError: Boolean(error) && !hasSearchData,
  })
  const dailyStageState = getStageState({
    isRunning: loadingDaily,
    isDone: hasReportData,
    hasError: Boolean(error) && hasSearchData && !hasReportData,
  })
  const analysisStageState = getStageState({
    isRunning: loadingAnalyze,
    isDone: hasJudgeContent || hasLLMContent || schedulerDone,
    hasError: Boolean(error) && hasReportData && !hasJudgeContent && !hasLLMContent,
  })
  const phaseLabel = PHASE_COPY[phase]
  const nextStepLabel = getNextWorkflowAction({
    hasSearchData,
    hasReportData,
    hasJudgeContent,
    hasLLMContent,
    isLoading,
  })
  const deliveryChannels = [notifyEnabled ? "Email" : null, resendEnabled ? "Resend" : null].filter(Boolean) as string[]
  const compactRetrievalSummary = [
    activeSourceLabels.length ? activeSourceLabels.join(" · ") : "No sources",
    activeBranchLabels.length ? activeBranchLabels.join(" + ") : "No branches",
  ]
  const compactAnalysisSummary = [
    enableLLM ? "LLM on" : "LLM off",
    enableJudge ? "Judge on" : "Judge off",
    saveDaily ? "Save on" : "Save off",
  ].join(" · ")
  const compactDeliverySummary = deliveryChannels.length
    ? `${deliveryChannels.join(" + ")} on`
    : "Manual review"
  const showCompactOptions = compact && (compactOptionsOpen || !canSearch || notifyEnabled)
  const controlStateHint = canSearch
    ? compact
      ? "Ready to run Search."
      : "Ready to run."
    : compact
      ? "Pick at least one topic, source, and branch."
      : "Pick at least one topic, source, and branch."
  const emailOverrideField = notifyEnabled ? (
    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-800">
            {compact ? "Email recipient" : "Direct email override"}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {compact
              ? "Override the default recipient for this run."
              : "This address overrides the default backend recipient for the next digest run."}
          </p>
        </div>
        <div className="w-full md:max-w-sm">
          <Input
            type="email"
            value={notifyEmail}
            onChange={(event) => store.setNotifyEmail(event.target.value)}
            placeholder="you@example.com"
            className="h-9 rounded-xl border-slate-200 bg-white"
          />
        </div>
      </div>
    </div>
  ) : null
  const overallProgressValue = Math.round(
    [searchStageState, dailyStageState, analysisStageState].reduce((sum, status) => {
      if (status === "done") return sum + 1
      if (status === "running") return sum + 0.5
      return sum
    }, 0) / 3 * 100,
  )
  const workflowPageHref = shouldPersistWorkflowQueries(queries)
    ? `/workflows?query=${encodeURIComponent(queries.join(","))}`
    : "/workflows"
  const snapshotHighlights = (
    dailyResult?.report?.global_top?.length ? dailyResult.report.global_top : allPapers
  ).slice(0, 3)
  const snapshotInsight = (dailyResult?.report?.llm_analysis?.daily_insight || "").trim()
  const snapshotTrendRows = (dailyResult?.report?.llm_analysis?.query_trends || []).slice(0, 2)

  if (compact) {
    return (
      <div className="space-y-3" id="workflow-console">
        <PaperDetailDialog
          item={selectedPaper}
          open={Boolean(selectedPaper)}
          onClose={() => setSelectedPaper(null)}
        />

        <Card className="rounded-[28px] border border-slate-200/80 bg-white shadow-sm">
          <CardHeader className="pb-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="max-w-3xl">
                <div className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
                  <WorkflowIcon className="size-3.5" />
                  Today&apos;s Research Brief
                </div>
                <h2 className="mt-3 text-2xl font-bold tracking-tight text-slate-900">
                  {dashboardContext?.activeTrackName
                    ? `${dashboardContext.activeTrackName} brief`
                    : "One brief for today"}
                </h2>
              </div>

              <div className="flex flex-wrap gap-2">
                <Badge className="rounded-full bg-indigo-600 px-3 py-1 text-[11px] font-medium text-white">
                  {phaseLabel}
                </Badge>
                <Badge
                  variant="secondary"
                  className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-medium text-slate-600"
                >
                  {queries.length} topics
                </Badge>
                <Badge
                  variant="secondary"
                  className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-medium text-slate-600"
                >
                  {formatTimestamp(store.lastUpdated)}
                </Badge>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3.5">
              <div className="flex flex-wrap gap-2">
                <div className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                  <span className="font-semibold text-slate-900">Focus:</span>{" "}
                  {dashboardContext?.activeTrackName || "Global"} · {dashboardContext?.readingQueueCount ?? 0} queued
                </div>
                <div className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                  <span className="font-semibold text-slate-900">Snapshot:</span>{" "}
                  {hasReportData ? "Daily digest ready" : hasSearchData ? "Candidates ready" : "Waiting"}
                  {" · "}
                  {allPapers.length} candidates / {judgedPapersCount} judged
                </div>
                <div className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                  <span className="font-semibold text-slate-900">Delivery:</span>{" "}
                  {deliveryChannels.length ? deliveryChannels.join(" + ") : "Manual review"}
                  {" · "}
                  {saveDaily ? "persisted" : "preview"}
                </div>
                <div className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                  <span className="font-semibold text-slate-900">Alerts:</span>{" "}
                  {(dashboardContext?.urgentDeadlineCount ?? 0) + (dashboardContext?.signalCount ?? 0)} dashboard items
                </div>
              </div>
            </div>

            {hasWorkspaceOutput ? (
              <div className="rounded-[24px] border border-slate-200 bg-slate-50/60 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div className="max-w-2xl">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
                      Today&apos;s Push
                    </p>
                    <h3 className="mt-2 text-xl font-bold text-slate-900">
                      {dailyResult?.report?.title || "Latest workflow candidate pool"}
                    </h3>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{nextStepLabel}</p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <Button asChild size="sm" className="rounded-full bg-indigo-600 px-4 hover:bg-indigo-700">
                      <Link href={workflowPageHref}>
                        Open Full Workbench
                        <ArrowUpRightIcon className="ml-1.5 size-4" />
                      </Link>
                    </Button>
                    <Button asChild size="sm" variant="outline" className="rounded-full border-slate-200 px-4">
                      <Link href={dashboardContext?.activeTrackHref || "/research"}>
                        Open Research
                        <ArrowUpRightIcon className="ml-1.5 size-4" />
                      </Link>
                    </Button>
                  </div>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Last updated
                    </p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">
                      {formatTimestamp(store.lastUpdated)}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between gap-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      <span>Progress</span>
                      <span>{overallProgressValue}%</span>
                    </div>
                    <Progress value={overallProgressValue} className="mt-3" />
                    <p className="mt-3 text-xs text-slate-500">{phaseLabel}</p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:col-span-2 xl:col-span-1">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Next handoff
                    </p>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      {dashboardContext?.activeTrackName
                        ? `Send promising papers into ${dashboardContext.activeTrackName} and keep the homepage focused on decisions.`
                        : "Move promising papers into Research once the shortlist stabilizes."}
                    </p>
                  </div>
                </div>

                <div className="mt-4 grid gap-3 xl:grid-cols-3">
                  {snapshotHighlights.length > 0 ? (
                    snapshotHighlights.map((item, index) => {
                      const recommendation = item.judge?.recommendation
                        ? REC_LABELS[item.judge.recommendation] || item.judge.recommendation
                        : null
                      const scoreLabel =
                        item.judge?.overall != null
                          ? `Judge ${Number(item.judge.overall).toFixed(1)}`
                          : item.score != null
                            ? `Score ${Number(item.score).toFixed(2)}`
                            : "Candidate"
                      const queryLabel =
                        (item.matched_queries || []).slice(0, 2).join(" · ") || "Latest shortlist"

                      return (
                        <div
                          key={`${item.title}-${index}`}
                          className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex flex-wrap gap-2">
                              <Badge
                                variant="secondary"
                                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-medium text-slate-600"
                              >
                                {queryLabel}
                              </Badge>
                              <Badge
                                variant="secondary"
                                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-medium text-slate-600"
                              >
                                {scoreLabel}
                              </Badge>
                              {recommendation ? (
                                <Badge className="rounded-full bg-indigo-600 px-3 py-1 text-[11px] font-medium text-white">
                                  {recommendation}
                                </Badge>
                              ) : null}
                            </div>
                            <p className="text-xs text-slate-500">#{index + 1}</p>
                          </div>

                          <button
                            type="button"
                            onClick={() => setSelectedPaper(item)}
                            className="mt-3 block text-left"
                          >
                            <p className="text-base font-semibold leading-7 text-slate-900 transition-colors hover:text-indigo-700">
                              {item.title}
                            </p>
                          </button>

                          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                            {(item.sources || []).slice(0, 3).map((source) => (
                              <span
                                key={`${item.title}-${source}`}
                                className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1"
                              >
                                {SOURCE_LABELS[source] || source}
                              </span>
                            ))}
                            {item.url ? (
                              <a
                                href={item.url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-1 font-medium text-indigo-600 hover:text-indigo-700"
                              >
                                Source
                                <ArrowUpRightIcon className="size-3.5" />
                              </a>
                            ) : null}
                          </div>
                        </div>
                      )
                    })
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm leading-6 text-slate-600 xl:col-span-3">
                      No brief snapshot yet. Run Search or DailyPaper from the full workbench.
                    </div>
                  )}
                </div>

                {snapshotInsight || snapshotTrendRows.length > 0 ? (
                  <div className="mt-4 grid gap-3 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,0.95fr)]">
                    {snapshotInsight ? (
                      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
                          Daily Insight
                        </p>
                        <p className="mt-3 text-sm leading-6 text-slate-600">{snapshotInsight}</p>
                      </div>
                    ) : null}

                    {snapshotTrendRows.length > 0 ? (
                      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
                          Topic Trends
                        </p>
                        <div className="mt-3 space-y-3">
                          {snapshotTrendRows.map((trend) => (
                            <div
                              key={trend.query}
                              className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3"
                            >
                              <p className="text-sm font-semibold text-slate-900">{trend.query}</p>
                              <p className="mt-1 text-sm leading-6 text-slate-600">
                                {trend.analysis}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="rounded-[24px] border border-dashed border-slate-300 bg-slate-50/60 p-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="max-w-2xl">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
                      No Snapshot Yet
                    </p>
                    <h3 className="mt-2 text-xl font-bold text-slate-900">
                      No recent workflow run
                    </h3>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <Button asChild size="sm" className="rounded-full bg-indigo-600 px-4 hover:bg-indigo-700">
                      <Link href={workflowPageHref}>
                        Open Full Workbench
                        <ArrowUpRightIcon className="ml-1.5 size-4" />
                      </Link>
                    </Button>
                    <Button asChild size="sm" variant="outline" className="rounded-full border-slate-200 px-4">
                      <Link href={dashboardContext?.activeTrackHref || "/research"}>
                        Open Research
                        <ArrowUpRightIcon className="ml-1.5 size-4" />
                      </Link>
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className={`${compact ? "space-y-3" : "space-y-4"}`}>
      <PaperDetailDialog item={selectedPaper} open={Boolean(selectedPaper)} onClose={() => setSelectedPaper(null)} />

      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <div
          className={`${
            compact
              ? "bg-transparent p-0"
              : "overflow-hidden rounded-[28px] border border-slate-200/80 bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.12),_transparent_34%),linear-gradient(180deg,_#ffffff_0%,_#f8fafc_100%)] p-5 shadow-sm md:p-6"
          }`}
        >
          {!compact ? (
            <>
              <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
                <div className="max-w-3xl">
                  <div className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white/90 px-3 py-1 text-xs font-semibold text-indigo-700 shadow-sm">
                    <WorkflowIcon className="size-3.5" />
                    Research Workflow
                  </div>
                  <h2 className="mt-4 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
                    Run Search, DailyPaper, and Analyze.
                  </h2>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {quickSummaryBadges.map((badge) => (
                      <Badge key={badge} variant="secondary" className="rounded-full border border-slate-200 bg-white/90 px-3 py-1 text-[11px] font-medium text-slate-600 shadow-sm">
                        {badge}
                      </Badge>
                    ))}
                    {dashboardContext?.activeTrackName ? (
                      <Badge className="rounded-full bg-indigo-600 px-3 py-1 text-[11px] font-medium text-white">
                        Focus Track: {dashboardContext.activeTrackName}
                      </Badge>
                    ) : null}
                  </div>
                  {compact ? (
                    <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-slate-500">
                      <span>这里负责多主题检索、DailyPaper 和 Analyze；单篇即时探索继续放在 Research。</span>
                      <Link
                        href={dashboardContext?.activeTrackHref || "/research"}
                        className="inline-flex items-center gap-1 font-medium text-indigo-600 transition-colors hover:text-indigo-700"
                      >
                        Open Research
                        <ArrowUpRightIcon className="size-3.5" />
                      </Link>
                    </div>
                  ) : null}
                </div>

                <div className="grid gap-3 sm:grid-cols-3 xl:w-[420px]">
                  <div className="rounded-2xl border border-white/80 bg-white/90 p-4 shadow-sm">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Current phase</p>
                    <p className="mt-2 text-lg font-bold text-slate-900">{phaseLabel}</p>
                    <p className="mt-1 text-xs text-slate-500">{formatTimestamp(store.lastUpdated)}</p>
                  </div>
                  <div className="rounded-2xl border border-white/80 bg-white/90 p-4 shadow-sm">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Candidates</p>
                    <p className="mt-2 text-lg font-bold text-slate-900">{allPapers.length}</p>
                    <p className="mt-1 text-xs text-slate-500">{paperDataSource === "dailypaper" ? "From DailyPaper" : paperDataSource === "search" ? "From search" : "No data yet"}</p>
                  </div>
                  <div className="rounded-2xl border border-white/80 bg-white/90 p-4 shadow-sm">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Research handoff</p>
                    <p className="mt-2 text-lg font-bold text-slate-900">{dashboardContext?.activeTrackName || "Global"}</p>
                    <p className="mt-1 text-xs text-slate-500">
                      {dashboardContext?.readingQueueCount ?? 0} queued · {(dashboardContext?.urgentDeadlineCount ?? 0) + (dashboardContext?.signalCount ?? 0)} alerts
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 grid gap-3 lg:grid-cols-3">
                <WorkflowStageCard
                  eyebrow="01"
                  title="Search & collect"
                  metric={`${queries.length} topics · ${activeSourceLabels.length || 0} sources`}
                  status={searchStageState}
                  icon={<SearchIcon className="size-4" />}
                />
                <WorkflowStageCard
                  eyebrow="02"
                  title="Build DailyPaper"
                  metric={hasReportData ? `Report date ${dailyResult?.report?.date || "-"}` : `Top N ${topN} · files ${saveDaily ? "saved" : "ephemeral"}`}
                  status={dailyStageState}
                  icon={<BookOpenIcon className="size-4" />}
                />
                <WorkflowStageCard
                  eyebrow="03"
                  title="Analyze & hand off"
                  metric={`${judgedPapersCount} judged · ${deliveryChannels.length || 0} delivery channels`}
                  status={analysisStageState}
                  icon={<CompassIcon className="size-4" />}
                />
              </div>
            </>
          ) : null}

          <div
            className={`${compact ? "mt-0" : "mt-6 grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.85fr)]"}`}
          >
            <div className="rounded-[26px] border border-slate-200/80 bg-white/95 p-5 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-600">
                    Run setup
                  </p>
                  <h3 className="mt-2 text-xl font-bold text-slate-900">
                    Run setup
                  </h3>
                </div>
                <SheetTrigger asChild>
                  <Button size="sm" variant="outline" className="h-9 rounded-full border-slate-200 px-4 text-slate-700">
                    <SettingsIcon className="mr-1.5 size-4" />
                    Advanced settings
                  </Button>
                </SheetTrigger>
              </div>

              <div className={`mt-6 ${compact ? "space-y-4" : "space-y-6"}`}>
                <section className="space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-slate-800">Topics</p>
                    </div>
                    <Button type="button" variant="outline" size="sm" className="h-8 gap-1 rounded-full" onClick={addQuery}>
                      <PlusIcon className="size-3.5" />
                      Add topic
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {queryItems.map((query, index) => (
                      <div key={`${index}-${query}`} className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50/80 px-3 py-2">
                        <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-semibold text-indigo-600">
                          {index + 1}
                        </span>
                        <Input
                          value={query}
                          onChange={(event) => updateQuery(index, event.target.value)}
                          placeholder="Enter a topic, problem, or question"
                          className="h-9 border-none bg-transparent px-0 text-sm shadow-none focus-visible:ring-0"
                        />
                        {queryItems.length > 1 ? (
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="size-8 shrink-0 rounded-full text-slate-400 hover:bg-white hover:text-rose-600"
                            onClick={() => removeQuery(index)}
                          >
                            <XIcon className="size-4" />
                          </Button>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </section>

                {compact ? (
                  <section className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div>
                        <p className="text-sm font-semibold text-slate-800">Run summary</p>
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-9 rounded-full border-slate-200 px-4 text-slate-700"
                        aria-expanded={showCompactOptions}
                        onClick={() => setCompactOptionsOpen((prev) => !prev)}
                      >
                        {showCompactOptions ? <ChevronDownIcon className="mr-1.5 size-4" /> : <ChevronRightIcon className="mr-1.5 size-4" />}
                        {showCompactOptions ? "Hide options" : "Edit options"}
                      </Button>
                    </div>

                    <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="space-y-3">
                        <div className="flex flex-col gap-1 border-b border-slate-100 pb-3 md:flex-row md:items-start md:justify-between md:gap-4">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">检索范围</div>
                          <div className="text-sm font-medium text-slate-700">
                            <div>{compactRetrievalSummary[0]}</div>
                            <div className="mt-1 text-xs text-slate-500">{compactRetrievalSummary[1]}</div>
                          </div>
                        </div>
                        <div className="flex flex-col gap-1 border-b border-slate-100 pb-3 md:flex-row md:items-start md:justify-between md:gap-4">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Analysis</div>
                          <div className="text-sm font-medium text-slate-700">
                            <div>{compactAnalysisSummary}</div>
                          </div>
                        </div>
                        <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between md:gap-4">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Delivery</div>
                          <div className="text-sm font-medium text-slate-700">
                            <div>{compactDeliverySummary}</div>
                            <div className="mt-1 text-xs text-slate-500">
                              {notifyEnabled && notifyEmail.trim() ? notifyEmail.trim() : "Dashboard only"}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {showCompactOptions ? (
                      <div className="mt-4 space-y-4 border-t border-slate-200 pt-4">
                        <section className="space-y-3">
                          <div>
                            <p className="text-sm font-semibold text-slate-800">Retrieval</p>
                          </div>
                          <div className="grid gap-3 lg:grid-cols-2">
                            <div className="rounded-2xl border border-slate-200 bg-white p-3.5">
                              <p className="text-sm font-semibold text-slate-800">Sources</p>
                              <div className="mt-3 flex flex-wrap gap-2">
                                <WorkflowChip label="papers.cool" active={usePapersCool} onToggle={() => setUsePapersCool(!usePapersCool)} />
                                <WorkflowChip label="arXiv API" active={useArxivApi} onToggle={() => setUseArxivApi(!useArxivApi)} />
                                <WorkflowChip label="HF Daily" active={useHFDaily} onToggle={() => setUseHFDaily(!useHFDaily)} />
                              </div>
                            </div>
                            <div className="rounded-2xl border border-slate-200 bg-white p-3.5">
                              <p className="text-sm font-semibold text-slate-800">Branches</p>
                              <div className="mt-3 flex flex-wrap gap-2">
                                <WorkflowChip label="arXiv" active={useArxiv} onToggle={() => setUseArxiv(!useArxiv)} />
                                <WorkflowChip label="Venue" active={useVenue} onToggle={() => setUseVenue(!useVenue)} />
                              </div>
                            </div>
                          </div>
                        </section>

                        <section className="space-y-3">
                          <div>
                            <p className="text-sm font-semibold text-slate-800">Analysis</p>
                          </div>
                          <div className="grid gap-3 md:grid-cols-3">
                            <WorkflowTogglePanel
                              title="LLM analysis"
                              checked={enableLLM}
                              onCheckedChange={setEnableLLM}
                              icon={<SparklesIcon className="size-4" />}
                              compact
                            />
                            <WorkflowTogglePanel
                              title="Judge scoring"
                              checked={enableJudge}
                              onCheckedChange={setEnableJudge}
                              icon={<StarIcon className="size-4" />}
                              compact
                            />
                            <WorkflowTogglePanel
                              title="Save artifacts"
                              checked={saveDaily}
                              onCheckedChange={setSaveDaily}
                              icon={<BookOpenIcon className="size-4" />}
                              compact
                            />
                          </div>
                        </section>

                        <section className="space-y-3">
                          <div>
                            <p className="text-sm font-semibold text-slate-800">Delivery</p>
                          </div>
                          <div className="grid gap-3 md:grid-cols-2">
                            <WorkflowTogglePanel
                              title="Email digest"
                              checked={notifyEnabled}
                              onCheckedChange={store.setNotifyEnabled}
                              icon={<MailIcon className="size-4" />}
                              compact
                            />
                            <WorkflowTogglePanel
                              title="Newsletter push"
                              checked={resendEnabled}
                              onCheckedChange={store.setResendEnabled}
                              icon={<WorkflowIcon className="size-4" />}
                              compact
                            />
                          </div>
                          {emailOverrideField}
                        </section>
                      </div>
                    ) : null}
                  </section>
                ) : (
                  <>
                    <section className="grid gap-4 lg:grid-cols-2">
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <p className="text-sm font-semibold text-slate-800">Sources</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <WorkflowChip label="papers.cool" active={usePapersCool} onToggle={() => setUsePapersCool(!usePapersCool)} />
                          <WorkflowChip label="arXiv API" active={useArxivApi} onToggle={() => setUseArxivApi(!useArxivApi)} />
                          <WorkflowChip label="HF Daily" active={useHFDaily} onToggle={() => setUseHFDaily(!useHFDaily)} />
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <p className="text-sm font-semibold text-slate-800">Branches</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <WorkflowChip label="arXiv" active={useArxiv} onToggle={() => setUseArxiv(!useArxiv)} />
                          <WorkflowChip label="Venue" active={useVenue} onToggle={() => setUseVenue(!useVenue)} />
                        </div>
                      </div>
                    </section>

                    <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                      <WorkflowTogglePanel
                        title="LLM analysis"
                        checked={enableLLM}
                        onCheckedChange={setEnableLLM}
                        icon={<SparklesIcon className="size-4" />}
                      />
                      <WorkflowTogglePanel
                        title="Judge scoring"
                        checked={enableJudge}
                        onCheckedChange={setEnableJudge}
                        icon={<StarIcon className="size-4" />}
                      />
                      <WorkflowTogglePanel
                        title="Save artifacts"
                        checked={saveDaily}
                        onCheckedChange={setSaveDaily}
                        icon={<BookOpenIcon className="size-4" />}
                      />
                      <WorkflowTogglePanel
                        title="Email digest"
                        checked={notifyEnabled}
                        onCheckedChange={store.setNotifyEnabled}
                        icon={<MailIcon className="size-4" />}
                      />
                      <WorkflowTogglePanel
                        title="Newsletter push"
                        checked={resendEnabled}
                        onCheckedChange={store.setResendEnabled}
                        icon={<WorkflowIcon className="size-4" />}
                      />
                    </section>

                    {emailOverrideField}
                  </>
                )}

                <div className="flex flex-col gap-3 border-t border-slate-100 pt-4">
                  <p className="text-sm text-slate-500">{controlStateHint}</p>
                  <div className={`${compact ? "grid gap-2 sm:grid-cols-2 xl:grid-cols-4" : "flex flex-wrap gap-2"}`}>
                    <Button
                      size="sm"
                      className={`${compact ? "h-10 w-full justify-center rounded-xl bg-indigo-600 px-4 hover:bg-indigo-700" : "rounded-full bg-indigo-600 px-4 hover:bg-indigo-700"}`}
                      disabled={isLoading || !canSearch}
                      onClick={runTopicSearch}
                    >
                      {loadingSearch ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : <PlayIcon className="mr-1.5 size-4" />}
                      Search
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      className={`${compact ? "h-10 w-full justify-center rounded-xl px-4" : "rounded-full px-4"}`}
                      disabled={isLoading || !canSearch}
                      onClick={runDailyPaperStream}
                    >
                      {loadingDaily ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : <BookOpenIcon className="mr-1.5 size-4" />}
                      DailyPaper
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className={`${compact ? "h-10 w-full justify-center rounded-xl px-4" : "rounded-full px-4"}`}
                      disabled={isLoading || !dailyResult?.report}
                      onClick={runAnalyzeStream}
                    >
                      {loadingAnalyze ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : <ZapIcon className="mr-1.5 size-4" />}
                      Analyze
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className={`${compact ? "h-10 w-full justify-center rounded-xl border border-slate-200 px-4 text-slate-600 hover:bg-slate-50" : "rounded-full px-3"}`}
                      title="Clear cached data"
                      onClick={() => { store.clearAll(); setError(null) }}
                    >
                      <Trash2Icon className={`${compact ? "mr-1.5 size-4" : "size-4"}`} />
                      {compact ? "Clear" : null}
                    </Button>
                  </div>
                </div>

                {compact ? (
                  <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">Run Status</p>
                        <h4 className="mt-2 text-base font-semibold text-slate-900">Current state</h4>
                        <p className="mt-1 text-sm leading-6 text-slate-600">{nextStepLabel}</p>
                      </div>
                      <Badge variant="secondary" className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-medium text-slate-600">
                        {phaseLabel}
                      </Badge>
                    </div>

                    <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-xl bg-white px-3.5 py-3">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">上次更新</p>
                        <p className="mt-2 text-sm font-semibold text-slate-900">{formatTimestamp(store.lastUpdated)}</p>
                      </div>
                      <div className="rounded-xl bg-white px-3.5 py-3">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">产物模式</p>
                        <p className="mt-2 text-sm font-semibold text-slate-900">{saveDaily ? "持久化产物" : "临时预览"}</p>
                      </div>
                      <div className="rounded-xl bg-white px-3.5 py-3">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Judge 覆盖</p>
                        <p className="mt-2 text-sm font-semibold text-slate-900">{judgedPapersCount} / {allPapers.length || 0} papers</p>
                      </div>
                      <div className="rounded-xl bg-white px-3.5 py-3">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">交付方式</p>
                        <p className="mt-2 text-sm font-semibold text-slate-900">{deliveryChannels.length ? deliveryChannels.join(" + ") : "仅手动复核"}</p>
                      </div>
                    </div>

                    <div className="mt-4 rounded-xl bg-white p-4">
                      <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
                        <span>阶段进度</span>
                        <span>{overallProgressValue}%</span>
                      </div>
                      <Progress value={overallProgressValue} className="mt-2" />
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            {!compact ? (
              <div className="space-y-4">
              <div className="rounded-[26px] border border-slate-200/80 bg-white/95 p-5 shadow-sm">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-600">
                      {compact ? "运行状态" : "Run Snapshot"}
                    </p>
                    <h3 className="mt-2 text-lg font-bold text-slate-900">
                      {compact ? "当前工作流状态" : "Current workflow state"}
                    </h3>
                  </div>
                  <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-medium text-slate-600">
                    {phaseLabel}
                  </Badge>
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {compact ? "上次更新" : "Last updated"}
                    </p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">{formatTimestamp(store.lastUpdated)}</p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {compact ? "产物模式" : "Output mode"}
                    </p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">
                      {saveDaily ? (compact ? "持久化产物" : "Persistent artifacts") : compact ? "临时预览" : "Ephemeral preview"}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {compact ? "Judge 覆盖" : "Judge coverage"}
                    </p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">{judgedPapersCount} / {allPapers.length || 0} papers</p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {compact ? "交付方式" : "Delivery"}
                    </p>
                    <p className="mt-2 text-sm font-semibold text-slate-900">
                      {deliveryChannels.length ? deliveryChannels.join(" + ") : compact ? "仅手动复核" : "Manual review only"}
                    </p>
                  </div>
                </div>
                <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
                    <span>{compact ? "阶段进度" : "Stage progress"}</span>
                    <span>{overallProgressValue}%</span>
                  </div>
                  <Progress value={overallProgressValue} className="mt-2" />
                  <p className="mt-3 text-sm leading-relaxed text-slate-600">{nextStepLabel}</p>
                </div>
              </div>

              <div className="rounded-[26px] border border-slate-200/80 bg-white/95 p-5 shadow-sm">
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-600">Delivery & Handoff</p>
                <h3 className="mt-2 text-lg font-bold text-slate-900">Keep the chain connected</h3>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                    {saveDaily ? `Output: ${outputDir}` : "No artifact path"}
                  </Badge>
                  <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                    {deliveryChannels.length ? `${deliveryChannels.length} delivery channel(s)` : "Delivery disabled"}
                  </Badge>
                  {dailyResult?.markdown_path ? (
                    <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                      MD ready
                    </Badge>
                  ) : null}
                </div>
                <div className="mt-4 space-y-3">
                  <WorkflowReferenceCard
                    title="Open Research Workspace"
                    description={dashboardContext?.activeTrackName
                      ? `Send promising papers into the “${dashboardContext.activeTrackName}” research loop.`
                      : "Hand off reviewed papers to Research for deeper exploration and feed curation."}
                    href={dashboardContext?.activeTrackHref || "/research"}
                  />
                  <WorkflowReferenceCard
                    title="Browse Knowledge Base"
                    description="Use Wiki as the grounding layer for metrics, architectures, and methods before delivery."
                    href="/wiki?q=metric"
                  />
                </div>
              </div>

              <div className="rounded-[26px] border border-slate-200/80 bg-white/95 p-5 shadow-sm">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-600">Workflow Playbook</p>
                  <h3 className="mt-2 text-lg font-bold text-slate-900">Attach docs and wiki to the operator loop</h3>
                  <p className="mt-2 text-sm leading-relaxed text-slate-500">
                    A reasonable chain is: topic selection inside Dashboard, concept grounding in Wiki, evidence review in Research, and then delivery from the same run snapshot.
                  </p>
                  <div className="mt-4 space-y-3">
                    <WorkflowReferenceCard
                      title="Judge Metrics in Wiki"
                      description="Open the concept browser filtered to evaluation and judge-related entries."
                      href="/wiki?q=metric"
                    />
                    <WorkflowReferenceCard
                      title="Architecture Concepts in Wiki"
                      description="Use the knowledge base to refresh related methods before deciding what to save or push."
                      href="/wiki?q=architecture"
                    />
                    {WORKFLOW_REFERENCE_LINKS.map((link) => (
                      <WorkflowReferenceCard
                        key={link.href}
                        title={link.label}
                        description={link.description}
                        href={link.href}
                        external={link.external}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>

        <SheetContent side="right" className="w-[420px] overflow-hidden sm:max-w-[420px]">
          <SheetHeader>
            <SheetTitle>Workflow Configuration</SheetTitle>
            <SheetDescription>Advanced controls for retrieval depth, judge budget, enrichment, and delivery.</SheetDescription>
          </SheetHeader>
          <div className="flex-1 overflow-y-auto px-1 pb-6">
            <ConfigSheetBody {...{
              topK, setTopK, topN, setTopN,
              showPerBranch, setShowPerBranch, saveDaily, setSaveDaily,
              outputDir, setOutputDir, useArxiv, setUseArxiv, useVenue, setUseVenue,
              usePapersCool, setUsePapersCool, useArxivApi, setUseArxivApi, useHFDaily, setUseHFDaily, enableLLM, setEnableLLM,
              useSummary, setUseSummary, useTrends, setUseTrends,
              useInsight, setUseInsight, useRelevance, setUseRelevance,
              enableJudge, setEnableJudge, judgeRuns, setJudgeRuns,
              judgeMaxItems, setJudgeMaxItems, judgeTokenBudget, setJudgeTokenBudget,
              notifyEmail, setNotifyEmail: store.setNotifyEmail,
              notifyEnabled, setNotifyEnabled: store.setNotifyEnabled,
              resendEnabled, setResendEnabled: store.setResendEnabled,
            }} />
          </div>
        </SheetContent>
      </Sheet>

      {error && <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>}

      {/* Stream progress card for DailyPaper SSE */}
      {loadingDaily && streamPhase !== "idle" && (
        <StreamProgressCard
          streamPhase={streamPhase}
          streamLog={streamLog}
          streamProgress={streamProgress}
          startTime={streamStartRef.current}
        />
      )}

      {/* Generic loading card for search / analyze */}
      {isLoading && !loadingDaily && !loadingAnalyze && (
        <Card className="border-blue-200 bg-blue-50/40">
          <CardContent className="space-y-2 py-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2 font-medium text-blue-900">
                <Loader2Icon className="size-4 animate-spin" />
                {loadingLabel}
              </div>
              <span className="text-xs text-blue-700">{loadingHint}</span>
            </div>
            <Progress value={loadingAnalyze && analyzeProgress.total > 0 ? (analyzeProgress.done / analyzeProgress.total) * 100 : 35} />
            <p className="text-xs text-blue-800/80">任务执行中，结果会逐步填充，不会空白等待。</p>
          </CardContent>
        </Card>
      )}

      {(loadingAnalyze || analyzeLog.length > 0) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Analyze Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-24">
              <div className="space-y-0.5 font-mono text-xs text-muted-foreground">
                {analyzeLog.slice(-12).map((line, idx) => (<div key={`global-log-${idx}`}>{line}</div>))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {compact && !hasWorkspaceOutput ? (
        <Card className="rounded-[26px] border border-slate-200/80 bg-white shadow-sm">
          <CardHeader className="pb-3">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <CardTitle className="text-base text-slate-900">结果区</CardTitle>
                <p className="mt-1 text-sm text-slate-500">先完成检索，再逐步生成报告和分析结果。</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                  {queries.length} 个主题
                </Badge>
                <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                  {activeSourceLabels.length} 个来源
                </Badge>
                <Badge variant="secondary" className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
                  {activeBranchLabels.length} 个分支
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">01 Search</p>
                <p className="mt-2 text-sm font-semibold text-slate-900">先拉起候选池</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">基于当前主题、来源和分支完成首轮检索。</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">02 DailyPaper</p>
                <p className="mt-2 text-sm font-semibold text-slate-900">整理成日报</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">把候选变成可读报告和可保存产物。</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">03 Analyze</p>
                <p className="mt-2 text-sm font-semibold text-slate-900">补齐 Judge 与洞察</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">在结果值得保留时再运行更重的分析步骤。</p>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 text-sm leading-6 text-slate-600">
              {nextStepLabel}
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Stats Row */}
          <div className={`grid grid-cols-2 ${compact ? "gap-2 xl:grid-cols-4" : "gap-3 md:grid-cols-4"}`}>
            <StatCard
              compact={compact}
              label={compact ? "主题数" : "Queries"}
              value={queries.length}
              icon={<FilterIcon className="size-5" />}
            />
            <StatCard
              compact={compact}
              label={compact ? "候选数" : "Papers Found"}
              value={searchResult?.summary?.unique_items ?? dailyResult?.report?.stats?.unique_items ?? 0}
              icon={<BookOpenIcon className="size-5" />}
            />
            <StatCard
              compact={compact}
              label={compact ? "Judge 数" : "Judged"}
              value={judgedPapersCount}
              icon={<StarIcon className="size-5" />}
            />
            <StatCard
              compact={compact}
              label={compact ? "阶段" : "Phase"}
              value={phaseLabel}
              icon={<TrendingUpIcon className="size-5" />}
            />
          </div>

          {/* DAG (collapsible) */}
          <Card>
            <CardHeader className="cursor-pointer py-3" onClick={() => setDagOpen(!dagOpen)}>
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">{compact ? "流程图" : "Workflow DAG"}</CardTitle>
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    {Object.entries(dagStatuses).map(([key, status]) => (
                      <div key={key} className={`size-2 rounded-full ${status === "done" ? "bg-green-500" : status === "running" ? "bg-blue-500 animate-pulse" : status === "error" ? "bg-red-500" : status === "skipped" ? "bg-slate-300" : "bg-slate-200"}`} title={`${key}: ${status}`} />
                    ))}
                  </div>
                  {dagOpen ? <ChevronDownIcon className="size-4" /> : <ChevronRightIcon className="size-4" />}
                </div>
              </div>
            </CardHeader>
            {dagOpen && (
              <CardContent className="pt-0">
                <WorkflowDagView statuses={dagStatuses} queriesCount={queries.length} hitCount={searchResult?.summary?.total_query_hits ?? dailyResult?.report?.stats?.total_query_hits ?? 0} uniqueCount={searchResult?.summary?.unique_items ?? dailyResult?.report?.stats?.unique_items ?? 0} llmEnabled={enableLLM || hasLLMData} judgeEnabled={enableJudge || hasJudgeData} />
              </CardContent>
            )}
          </Card>

          {/* Result Tabs */}
          <Tabs value={workspaceTab} onValueChange={(value) => setWorkspaceTab(value as WorkflowWorkspaceTab)} className="w-full">
            <TabsList className={`${compact ? "flex h-auto w-full flex-wrap justify-start gap-2 rounded-2xl bg-slate-100 p-2" : "grid w-full grid-cols-3 rounded-2xl bg-slate-100 p-1 lg:grid-cols-6"}`}>
              <TabsTrigger value="candidates" className="gap-1.5 rounded-xl text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"><BookOpenIcon className="size-3.5" /> {compact ? "候选" : "Candidates"}</TabsTrigger>
              <TabsTrigger value="insights" className="gap-1.5 rounded-xl text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"><SparklesIcon className="size-3.5" /> {compact ? "洞察" : "Insights"}</TabsTrigger>
              <TabsTrigger value="judge" className="gap-1.5 rounded-xl text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"><StarIcon className="size-3.5" /> {compact ? "Judge" : "Judge"}</TabsTrigger>
              <TabsTrigger value="report" className="gap-1.5 rounded-xl text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"><BookOpenIcon className="size-3.5" /> {compact ? "报告" : "Report"}</TabsTrigger>
              <TabsTrigger value="delivery" className="gap-1.5 rounded-xl text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"><MailIcon className="size-3.5" /> {compact ? "交付" : "Delivery"}</TabsTrigger>
              <TabsTrigger value="log" className="gap-1.5 rounded-xl text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"><WorkflowIcon className="size-3.5" /> {compact ? "日志" : "Logs"}</TabsTrigger>
            </TabsList>

        {/* Papers */}
        <TabsContent value="candidates" className="mt-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <p className="text-sm text-muted-foreground">{allPapers.length} {compact ? "篇结果" : "papers"}</p>
              {paperDataSource && (
                <Badge variant="outline" className="text-[10px]">
                  {paperDataSource === "dailypaper" ? "DailyPaper" : "Search"}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Label className="text-xs">{compact ? "排序" : "Sort:"}</Label>
              <select className="h-7 rounded-md border bg-background px-2 text-xs" value={sortBy} onChange={(e) => setSortBy(e.target.value as "score" | "judge")}>
                <option value="score">{compact ? "检索分" : "Search Score"}</option>
                <option value="judge">{compact ? "Judge 分" : "Judge Score"}</option>
              </select>
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {allPapers.length > 0 ? (
              allPapers.map((item, idx) => (
                <PaperCard key={`${item.title}-${idx}`} item={item} query={(item as SearchItem & { _query?: string })._query} onOpenDetail={(p) => setSelectedPaper(p)} />
              ))
            ) : isLoading ? (
              Array.from({ length: 4 }).map((_, idx) => (
                <div key={`paper-skeleton-${idx}`} className="rounded-lg border p-4">
                  <div className="h-4 w-4/5 animate-pulse rounded bg-muted" />
                  <div className="mt-2 h-3 w-2/5 animate-pulse rounded bg-muted" />
                  <div className="mt-4 space-y-2">
                    <div className="h-2 w-full animate-pulse rounded bg-muted" />
                    <div className="h-2 w-11/12 animate-pulse rounded bg-muted" />
                    <div className="h-2 w-10/12 animate-pulse rounded bg-muted" />
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-2 rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
                {compact
                  ? "先运行 Search 建立候选池，再用 DailyPaper 生成报告，最后用 Analyze 补齐 Judge 和洞察。"
                  : "Run Search to find papers, then DailyPaper to rank and compose a report, then Analyze to run Judge/Trends."}
              </div>
            )}
          </div>
        </TabsContent>

        {/* Insights */}
        <TabsContent value="insights" className="mt-4 space-y-4">
          {hasInsightData ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Insight</CardTitle></CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert text-sm">
                  <Markdown remarkPlugins={[remarkGfm]}>{dailyResult?.report?.llm_analysis?.daily_insight || ""}</Markdown>
                </div>
              </CardContent>
            </Card>
          ) : loadingAnalyze ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Insight (Generating...)</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="h-3 w-full animate-pulse rounded bg-muted" />
                <div className="h-3 w-11/12 animate-pulse rounded bg-muted" />
                <div className="h-3 w-9/12 animate-pulse rounded bg-muted" />
              </CardContent>
            </Card>
          ) : null}

          {hasTrendData ? (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold">Query Trend Analysis</h3>
              {(dailyResult?.report?.llm_analysis?.query_trends || []).map((trend, idx) => (
                <Card key={`${trend.query}-${idx}`}>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">{trend.query}</CardTitle></CardHeader>
                  <CardContent>
                    {(trend.analysis || "").trim() ? (
                      <div className="prose prose-sm max-w-none dark:prose-invert text-sm">
                        <Markdown remarkPlugins={[remarkGfm]}>{trend.analysis}</Markdown>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No trend content returned by model for this query.</p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : loadingAnalyze ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Query Trend Analysis (Generating...)</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="h-3 w-1/3 animate-pulse rounded bg-muted" />
                <div className="h-3 w-full animate-pulse rounded bg-muted" />
                <div className="h-3 w-10/12 animate-pulse rounded bg-muted" />
              </CardContent>
            </Card>
          ) : null}

          {!hasLLMContent && !loadingAnalyze && (
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              {dailyResult?.report?.llm_analysis?.enabled
                ? "LLM enrichment ran but returned empty content. Check reasoning model route/API key and analyze log."
                : "Run DailyPaper with LLM Analysis enabled, or run Analyze."}
            </div>
          )}
        </TabsContent>

        {/* Judge */}
        <TabsContent value="judge" className="mt-4 space-y-4">
          {dailyResult?.report?.judge?.enabled && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {Object.entries(dailyResult.report.judge.recommendation_count || {}).map(([name, count]) => (
                <div key={name} className={`rounded-lg border px-3 py-2 text-center ${REC_COLORS[name] || ""}`}>
                  <div className="text-lg font-bold">{count}</div>
                  <div className="text-xs">{REC_LABELS[name] || name}</div>
                </div>
              ))}
            </div>
          )}
          {dailyResult?.report?.judge?.budget && (
            <div className="rounded-lg border bg-muted/30 px-4 py-3 text-sm">
              <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                <span>Candidates: {dailyResult.report.judge.budget.candidate_items ?? 0}</span>
                <span>Judged: {dailyResult.report.judge.budget.judged_items ?? 0}</span>
                <span>Tokens: {dailyResult.report.judge.budget.estimated_tokens ?? 0}/{(dailyResult.report.judge.budget.token_budget ?? 0) > 0 ? dailyResult.report.judge.budget.token_budget : "unlimited"}</span>
                {(dailyResult.report.judge.budget.skipped_due_budget ?? 0) > 0 && <span className="text-yellow-600">Skipped (budget): {dailyResult.report.judge.budget.skipped_due_budget}</span>}
              </div>
            </div>
          )}
          <div className="grid gap-3 md:grid-cols-2">
            {allPapers.filter((p) => (p.judge?.overall ?? 0) > 0).sort((a, b) => (b.judge?.overall ?? 0) - (a.judge?.overall ?? 0)).map((item, idx) => (
              <PaperCard key={`judge-${item.title}-${idx}`} item={item} query={(item as SearchItem & { _query?: string })._query} onOpenDetail={(p) => setSelectedPaper(p)} />
            ))}
            {!hasJudgeContent && (loadingAnalyze ? (
              Array.from({ length: 2 }).map((_, idx) => (
                <div key={`judge-skeleton-${idx}`} className="rounded-lg border p-4">
                  <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
                  <div className="mt-2 h-3 w-1/3 animate-pulse rounded bg-muted" />
                  <div className="mt-4 h-36 animate-pulse rounded bg-muted" />
                </div>
              ))
            ) : (
              <div className="col-span-2 rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
                {dailyResult?.report?.judge?.enabled
                  ? "Judge ran but no score was attached. Check candidate count/token budget and analyze log."
                  : "Run DailyPaper with Judge enabled, or run Analyze to see judge results."}
              </div>
            ))}
          </div>
          {analyzeLog.length > 0 && (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Analyze Stream Log</CardTitle></CardHeader>
              <CardContent>
                <ScrollArea className="h-32">
                  <div className="space-y-0.5 font-mono text-xs text-muted-foreground">
                    {analyzeLog.map((line, idx) => (<div key={`log-${idx}`}>{line}</div>))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Report (Structured) */}
        <TabsContent value="report" className="mt-4 space-y-3">
          {dailyResult?.report ? (
            <>
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">DailyPaper Report</CardTitle>
                    <div className="flex items-center gap-2">
                      {dailyResult.markdown && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 gap-1 text-xs"
                          onClick={() => {
                            const blob = new Blob([dailyResult.markdown || ""], { type: "text/markdown" })
                            const url = URL.createObjectURL(blob)
                            const a = document.createElement("a")
                            a.href = url
                            a.download = `dailypaper-${dailyResult.report.date || "report"}.md`
                            a.click()
                            URL.revokeObjectURL(url)
                          }}
                        >
                          <DownloadIcon className="size-3.5" /> Download .md
                        </Button>
                      )}
                      <div className="flex gap-2 text-xs text-muted-foreground">
                        {dailyResult.markdown_path && <span>MD: {dailyResult.markdown_path}</span>}
                        {dailyResult.json_path && <span>JSON: {dailyResult.json_path}</span>}
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-2 text-xs md:grid-cols-4">
                    <div className="rounded-md border bg-muted/20 px-3 py-2">Date: {dailyResult.report.date}</div>
                    <div className="rounded-md border bg-muted/20 px-3 py-2">Source: {dailyResult.report.source || "papers.cool"}</div>
                    <div className="rounded-md border bg-muted/20 px-3 py-2">Unique: {dailyResult.report.stats.unique_items}</div>
                    <div className="rounded-md border bg-muted/20 px-3 py-2">Hits: {dailyResult.report.stats.total_query_hits}</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Query Highlights Table</CardTitle></CardHeader>
                <CardContent>
                  {queryHighlightRows.length > 0 ? (
                    <ScrollArea className="h-72 rounded-md border">
                      <table className="w-full text-left text-xs">
                        <thead className="sticky top-0 bg-muted/70 backdrop-blur">
                          <tr>
                            <th className="border-b px-3 py-2 font-medium">Query</th>
                            <th className="border-b px-3 py-2 font-medium">Title</th>
                            <th className="border-b px-3 py-2 font-medium">Score</th>
                            <th className="border-b px-3 py-2 font-medium">Judge</th>
                          </tr>
                        </thead>
                        <tbody>
                          {queryHighlightRows.map((row, idx) => (
                            <tr key={`${row.query}-${row.title}-${idx}`} className="odd:bg-muted/20">
                              <td className="border-b px-3 py-2 text-muted-foreground">{row.query}</td>
                              <td className="border-b px-3 py-2">
                                {row.url ? <a href={row.url} target="_blank" rel="noreferrer" className="hover:underline text-primary">{row.title}</a> : row.title}
                              </td>
                              <td className="border-b px-3 py-2 font-mono">{row.score.toFixed(4)}</td>
                              <td className="border-b px-3 py-2">
                                <Badge variant="outline" className={REC_COLORS[row.recommendation] || ""}>{REC_LABELS[row.recommendation] || row.recommendation}</Badge>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </ScrollArea>
                  ) : (
                    <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">No query highlights yet.</div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Global Top Table</CardTitle></CardHeader>
                <CardContent>
                  {globalTopRows.length > 0 ? (
                    <ScrollArea className="h-64 rounded-md border">
                      <table className="w-full text-left text-xs">
                        <thead className="sticky top-0 bg-muted/70 backdrop-blur">
                          <tr>
                            <th className="border-b px-3 py-2 font-medium">#</th>
                            <th className="border-b px-3 py-2 font-medium">Title</th>
                            <th className="border-b px-3 py-2 font-medium">Score</th>
                            <th className="border-b px-3 py-2 font-medium">Matched Queries</th>
                          </tr>
                        </thead>
                        <tbody>
                          {globalTopRows.map((row) => (
                            <tr key={`${row.rank}-${row.title}`} className="odd:bg-muted/20">
                              <td className="border-b px-3 py-2 font-mono">{row.rank}</td>
                              <td className="border-b px-3 py-2">
                                {row.url ? <a href={row.url} target="_blank" rel="noreferrer" className="hover:underline text-primary">{row.title}</a> : row.title}
                              </td>
                              <td className="border-b px-3 py-2 font-mono">{row.score.toFixed(4)}</td>
                              <td className="border-b px-3 py-2 text-muted-foreground">{row.queries}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </ScrollArea>
                  ) : (
                    <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">No global top papers yet.</div>
                  )}
                </CardContent>
              </Card>

            </>
          ) : isLoading ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Building structured report...</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="h-8 w-full animate-pulse rounded bg-muted" />
                <div className="h-44 w-full animate-pulse rounded bg-muted" />
              </CardContent>
            </Card>
          ) : (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">Generate a DailyPaper to see the rendered report.</div>
          )}
        </TabsContent>

        <TabsContent value="delivery" className="mt-4 space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Delivery Readiness</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Artifacts</p>
                  <p className="mt-2 font-medium">{saveDaily ? outputDir : "Not persisted"}</p>
                </div>
                <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Channels</p>
                  <p className="mt-2 font-medium">{deliveryChannels.length ? deliveryChannels.join(" + ") : "Manual review"}</p>
                </div>
                <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Research Handoff</p>
                  <p className="mt-2 font-medium">{dashboardContext?.activeTrackName || "Global queue"}</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {dailyResult?.markdown_path ? <span>MD: {dailyResult.markdown_path}</span> : null}
                {dailyResult?.json_path ? <span>JSON: {dailyResult.json_path}</span> : null}
                {notifyEnabled && notifyEmail.trim() ? <span>Email override: {notifyEmail.trim()}</span> : null}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="outline" asChild>
                  <Link href={dashboardContext?.activeTrackHref || "/research"}>
                    Open Research
                    <ArrowUpRightIcon className="ml-1.5 size-4" />
                  </Link>
                </Button>
                <Button size="sm" variant="outline" asChild>
                  <Link href="/wiki?q=metric">
                    Open Wiki
                    <ArrowUpRightIcon className="ml-1.5 size-4" />
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-sm">Repository Enrichment</CardTitle>
                <Button size="sm" variant="outline" disabled={loadingRepos || !dailyResult?.report} onClick={runRepoEnrichment}>
                  {loadingRepos ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : null}
                  {loadingRepos ? "Enriching..." : "Find Repos"}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {repoError && <div className="mb-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{repoError}</div>}
              {repoRows.length > 0 ? (
                <ScrollArea className="h-64 rounded-md border">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-muted/70 backdrop-blur">
                      <tr>
                        <th className="border-b px-3 py-2 font-medium">Title</th>
                        <th className="border-b px-3 py-2 font-medium">Repository</th>
                        <th className="border-b px-3 py-2 font-medium">Stars</th>
                        <th className="border-b px-3 py-2 font-medium">Language</th>
                      </tr>
                    </thead>
                    <tbody>
                      {repoRows.map((row, idx) => (
                        <tr key={`delivery-${row.repo_url}-${idx}`} className="odd:bg-muted/20">
                          <td className="border-b px-3 py-2">
                            {row.paper_url ? <a href={row.paper_url} target="_blank" rel="noreferrer" className="hover:underline text-primary">{row.title}</a> : row.title}
                          </td>
                          <td className="border-b px-3 py-2">
                            <a href={row.repo_url} target="_blank" rel="noreferrer" className="hover:underline text-primary">{row.repo_url}</a>
                          </td>
                          <td className="border-b px-3 py-2 font-mono">{row.github?.stars ?? "-"}</td>
                          <td className="border-b px-3 py-2 text-muted-foreground">{row.github?.language || "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollArea>
              ) : (
                <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">Click &quot;Find Repos&quot; to enrich papers with code repositories.</div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="log" className="mt-4 space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Analyze Stream Log</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-48 rounded-md border">
                <div className="space-y-0.5 p-3 font-mono text-xs text-muted-foreground">
                  {analyzeLog.length > 0 ? analyzeLog.map((line, index) => <div key={`analysis-log-${index}`}>{line}</div>) : (
                    <div>No analyze log yet. Run Analyze to stream judge and insight events.</div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">DailyPaper Stream Log</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-48 rounded-md border">
                <div className="space-y-0.5 p-3 font-mono text-xs text-muted-foreground">
                  {streamLog.length > 0 ? streamLog.map((line, index) => <div key={`daily-log-${index}`}>{line}</div>) : (
                    <div>No DailyPaper stream log yet. Generate a report to inspect the stream phases.</div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
        </>
      )}
    </div>
  )
}
