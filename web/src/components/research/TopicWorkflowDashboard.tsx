"use client"

import { useMemo, useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  BookOpenIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  FilterIcon,
  Loader2Icon,
  PlayIcon,
  SettingsIcon,
  SparklesIcon,
  StarIcon,
  Trash2Icon,
  TrendingUpIcon,
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
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { readSSE } from "@/lib/sse"
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

type StepStatus = "pending" | "running" | "done" | "error" | "skipped"

/* ── Helpers ──────────────────────────────────────────── */

const DEFAULT_QUERIES = ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]

function parseLines(text: string): string[] {
  return text.split("\n").map((l) => l.trim()).filter(Boolean)
}

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

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card px-4 py-3">
      <div className="flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">{icon}</div>
      <div>
        <div className="text-2xl font-bold tabular-nums">{value}</div>
        <div className="text-xs text-muted-foreground">{label}</div>
      </div>
    </div>
  )
}

function buildDagStatuses(args: {
  phase: WorkflowPhase
  hasError: boolean
  llmEnabled: boolean
  judgeEnabled: boolean
  reportReady: boolean
}): Record<string, StepStatus> {
  const { phase, hasError, llmEnabled, judgeEnabled, reportReady } = args
  const base: Record<string, StepStatus> = {
    source: "pending",
    normalize: "pending",
    search: "pending",
    rank: "pending",
    llm: llmEnabled ? "pending" : "skipped",
    judge: judgeEnabled ? "pending" : "skipped",
    report: "pending",
    scheduler: "pending",
  }
  if (phase === "searching") return { ...base, source: "done", normalize: "done", search: "running", rank: "running" }
  if (phase === "searched") return { ...base, source: "done", normalize: "done", search: "done", rank: "done" }
  if (phase === "reporting")
    return { ...base, source: "done", normalize: "done", search: "done", rank: "done", llm: llmEnabled ? "running" : "skipped", judge: judgeEnabled ? "running" : "skipped", report: "running" }
  if (phase === "reported")
    return { ...base, source: "done", normalize: "done", search: "done", rank: "done", llm: llmEnabled ? "done" : "skipped", judge: judgeEnabled ? "done" : "skipped", report: reportReady ? "done" : "pending", scheduler: reportReady ? "done" : "pending" }
  if (phase === "error" || hasError)
    return { ...base, source: "done", normalize: "done", search: "error", rank: "error", llm: llmEnabled ? "error" : "skipped", judge: judgeEnabled ? "error" : "skipped", report: "error", scheduler: "pending" }
  return base
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
  queriesText: string; setQueriesText: (v: string) => void
  topK: number; setTopK: (v: number) => void
  topN: number; setTopN: (v: number) => void
  showPerBranch: number; setShowPerBranch: (v: number) => void
  saveDaily: boolean; setSaveDaily: (v: boolean) => void
  outputDir: string; setOutputDir: (v: string) => void
  useArxiv: boolean; setUseArxiv: (v: boolean) => void
  useVenue: boolean; setUseVenue: (v: boolean) => void
  usePapersCool: boolean; setUsePapersCool: (v: boolean) => void
  enableLLM: boolean; setEnableLLM: (v: boolean) => void
  useSummary: boolean; setUseSummary: (v: boolean) => void
  useTrends: boolean; setUseTrends: (v: boolean) => void
  useInsight: boolean; setUseInsight: (v: boolean) => void
  useRelevance: boolean; setUseRelevance: (v: boolean) => void
  enableJudge: boolean; setEnableJudge: (v: boolean) => void
  judgeRuns: number; setJudgeRuns: (v: number) => void
  judgeMaxItems: number; setJudgeMaxItems: (v: number) => void
  judgeTokenBudget: number; setJudgeTokenBudget: (v: number) => void
}) {
  const {
    queriesText, setQueriesText, topK, setTopK, topN, setTopN,
    showPerBranch, setShowPerBranch, saveDaily, setSaveDaily,
    outputDir, setOutputDir, useArxiv, setUseArxiv, useVenue, setUseVenue,
    usePapersCool, setUsePapersCool, enableLLM, setEnableLLM,
    useSummary, setUseSummary, useTrends, setUseTrends,
    useInsight, setUseInsight, useRelevance, setUseRelevance,
    enableJudge, setEnableJudge, judgeRuns, setJudgeRuns,
    judgeMaxItems, setJudgeMaxItems, judgeTokenBudget, setJudgeTokenBudget,
  } = props

  return (
    <div className="space-y-5 pr-2">
      <section className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Topics</Label>
          <Textarea value={queriesText} onChange={(e) => setQueriesText(e.target.value)} rows={5} className="text-sm" placeholder="One topic per line..." />
        </section>

        <section className="space-y-2">
          <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sources &amp; Branches</Label>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-1.5 text-sm"><Checkbox checked={usePapersCool} onCheckedChange={(v) => setUsePapersCool(Boolean(v))} /> papers.cool</label>
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
              <div className="space-y-1"><Label className="text-xs">Max Items</Label><Input type="number" min={1} max={20} value={judgeMaxItems} onChange={(e) => setJudgeMaxItems(Number(e.target.value || 5))} className="h-8 text-sm" /></div>
              <div className="space-y-1"><Label className="text-xs">Token Budget</Label><Input type="number" min={0} value={judgeTokenBudget} onChange={(e) => setJudgeTokenBudget(Number(e.target.value || 0))} className="h-8 text-sm" /></div>
            </div>
          )}
        </section>
    </div>
  )
}

/* ── Main Dashboard ───────────────────────────────────── */

export default function TopicWorkflowDashboard() {
  /* Config state (local) */
  const [queriesText, setQueriesText] = useState(DEFAULT_QUERIES.join("\n"))
  const [topK, setTopK] = useState(5)
  const [topN, setTopN] = useState(10)
  const [showPerBranch, setShowPerBranch] = useState(25)
  const [saveDaily, setSaveDaily] = useState(false)
  const [outputDir, setOutputDir] = useState("./reports/dailypaper")
  const [useArxiv, setUseArxiv] = useState(true)
  const [useVenue, setUseVenue] = useState(true)
  const [usePapersCool, setUsePapersCool] = useState(true)
  const [enableLLM, setEnableLLM] = useState(false)
  const [useSummary, setUseSummary] = useState(true)
  const [useTrends, setUseTrends] = useState(true)
  const [useInsight, setUseInsight] = useState(true)
  const [useRelevance, setUseRelevance] = useState(false)
  const [enableJudge, setEnableJudge] = useState(false)
  const [judgeRuns, setJudgeRuns] = useState(1)
  const [judgeMaxItems, setJudgeMaxItems] = useState(5)
  const [judgeTokenBudget, setJudgeTokenBudget] = useState(0)

  /* Persisted state (zustand) */
  const store = useWorkflowStore()
  const { searchResult, dailyResult, phase, analyzeLog } = store

  /* Transient loading state (not persisted) */
  const [loadingSearch, setLoadingSearch] = useState(false)
  const [loadingDaily, setLoadingDaily] = useState(false)
  const [loadingAnalyze, setLoadingAnalyze] = useState(false)
  const [analyzeProgress, setAnalyzeProgress] = useState({ done: 0, total: 0 })
  const [error, setError] = useState<string | null>(null)

  /* UI state */
  const [dagOpen, setDagOpen] = useState(false)
  const [selectedPaper, setSelectedPaper] = useState<SearchItem | null>(null)
  const [sortBy, setSortBy] = useState<"score" | "judge">("score")

  const queries = useMemo(() => parseLines(queriesText), [queriesText])
  const branches = useMemo(() => [useArxiv ? "arxiv" : "", useVenue ? "venue" : ""].filter(Boolean), [useArxiv, useVenue])
  const sources = useMemo(() => [usePapersCool ? "papers_cool" : ""].filter(Boolean), [usePapersCool])
  const llmFeatures = useMemo(
    () => [useSummary ? "summary" : "", useTrends ? "trends" : "", useInsight ? "insight" : "", useRelevance ? "relevance" : ""].filter(Boolean),
    [useInsight, useRelevance, useSummary, useTrends],
  )

  const dagStatuses = useMemo(
    () => buildDagStatuses({ phase, hasError: Boolean(error), llmEnabled: enableLLM, judgeEnabled: enableJudge, reportReady: Boolean(dailyResult?.report) }),
    [phase, error, enableLLM, enableJudge, dailyResult],
  )

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

  /* Actions */
  async function runTopicSearch() {
    setLoadingSearch(true); setError(null); store.setPhase("searching")
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

  async function runDailyPaper() {
    setLoadingDaily(true); setError(null); store.setPhase("reporting")
    try {
      const res = await fetch("/api/research/paperscool/daily", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queries, sources, branches, top_k_per_query: topK, show_per_branch: showPerBranch, top_n: topN,
          title: "DailyPaper Digest", formats: ["both"], save: saveDaily, output_dir: outputDir,
          enable_llm_analysis: enableLLM, llm_features: llmFeatures,
          enable_judge: enableJudge, judge_runs: judgeRuns,
          judge_max_items_per_query: judgeMaxItems, judge_token_budget: judgeTokenBudget,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      store.setDailyResult(await res.json())
      store.setPhase("reported")
    } catch (err) { setError(String(err)); store.setPhase("error") } finally { setLoadingDaily(false) }
  }

  async function runAnalyzeStream() {
    if (!dailyResult?.report) { setError("Generate DailyPaper first."); return }
    const runJudge = Boolean(enableJudge)
    const runTrends = Boolean(enableLLM && useTrends)
    if (!runJudge && !runTrends) { setError("Enable Judge or LLM trends before analyzing."); return }

    setLoadingAnalyze(true); setError(null); store.clearAnalyzeLog(); setAnalyzeProgress({ done: 0, total: 0 }); store.setPhase("reporting")

    try {
      const res = await fetch("/api/research/paperscool/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          report: dailyResult.report, run_judge: runJudge, run_trends: runTrends,
          judge_runs: judgeRuns, judge_max_items_per_query: judgeMaxItems,
          judge_token_budget: judgeTokenBudget, trend_max_items_per_query: 3,
        }),
      })
      if (!res.ok || !res.body) throw new Error(await res.text())

      for await (const event of readSSE(res.body)) {
        if (event.type === "progress") {
          const d = (event.data || {}) as { phase?: string; message?: string }
          store.addAnalyzeLog(`[${d.phase || "step"}] ${d.message || "running"}`)
        } else if (event.type === "trend") {
          const d = (event.data || {}) as { query?: string; done?: number; total?: number }
          store.addAnalyzeLog(`trend ${d.done}/${d.total}: ${d.query}`)
        } else if (event.type === "judge") {
          const d = (event.data || {}) as { title?: string; done?: number; total?: number }
          setAnalyzeProgress({ done: d.done || 0, total: d.total || 0 })
          store.addAnalyzeLog(`judge ${d.done}/${d.total}: ${d.title}`)
        } else if (event.type === "result") {
          const d = (event.data || {}) as { report?: DailyResult["report"]; markdown?: string }
          if (d.report) {
            store.updateDailyResult((prev) => ({
              ...prev,
              report: d.report || prev.report,
              markdown: typeof d.markdown === "string" ? d.markdown : prev.markdown,
            }))
          }
        }
      }
      store.setPhase("reported")
    } catch (err) { setError(String(err)); store.setPhase("error") } finally { setLoadingAnalyze(false) }
  }

  const isLoading = loadingSearch || loadingDaily || loadingAnalyze
  const canSearch = queries.length > 0 && branches.length > 0 && sources.length > 0
  const loadingLabel = loadingSearch
    ? "Searching papers.cool sources..."
    : loadingDaily
      ? "Generating DailyPaper report and enrichment..."
      : "Running judge/trend enrichment..."
  const loadingHint = loadingAnalyze && analyzeProgress.total > 0
    ? `${analyzeProgress.done}/${analyzeProgress.total} judged`
    : loadingDaily
      ? "Fetching, ranking, and composing report"
      : loadingSearch
        ? "Multi-query retrieval in progress"
        : "Waiting for LLM events"

  return (
    <div className="space-y-4">
      <PaperDetailDialog item={selectedPaper} open={Boolean(selectedPaper)} onClose={() => setSelectedPaper(null)} />

      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h2 className="text-xl font-bold">Topic Workflow</h2>
          <p className="text-sm text-muted-foreground">
            Search, analyze, and judge research papers
            {store.lastUpdated && <span className="ml-2 text-[10px]">(last: {new Date(store.lastUpdated).toLocaleString()})</span>}
          </p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-2">
          <Button size="sm" disabled={isLoading || !canSearch} onClick={runTopicSearch}>
            {loadingSearch ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : <PlayIcon className="mr-1.5 size-4" />} Search
          </Button>
          <Button size="sm" variant="secondary" disabled={isLoading || !canSearch} onClick={runDailyPaper}>
            {loadingDaily ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : <BookOpenIcon className="mr-1.5 size-4" />} DailyPaper
          </Button>
          <Button size="sm" variant="outline" disabled={isLoading || !dailyResult?.report} onClick={runAnalyzeStream}>
            {loadingAnalyze ? <Loader2Icon className="mr-1.5 size-4 animate-spin" /> : <ZapIcon className="mr-1.5 size-4" />} Analyze
          </Button>
          <Separator orientation="vertical" className="mx-1 h-6" />
          <Button size="sm" variant="ghost" title="Clear cached data" onClick={() => { store.clearAll(); setError(null) }}>
            <Trash2Icon className="size-4" />
          </Button>
          <Sheet>
            <SheetTrigger asChild>
              <Button size="sm" variant="ghost"><SettingsIcon className="size-4" /></Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[400px] sm:max-w-[400px] overflow-hidden">
              <SheetHeader>
                <SheetTitle>Workflow Configuration</SheetTitle>
                <SheetDescription>Topics, sources, LLM and Judge settings</SheetDescription>
              </SheetHeader>
              <div className="flex-1 overflow-y-auto px-1 pb-6">
                <ConfigSheetBody {...{
                queriesText, setQueriesText, topK, setTopK, topN, setTopN,
                showPerBranch, setShowPerBranch, saveDaily, setSaveDaily,
                outputDir, setOutputDir, useArxiv, setUseArxiv, useVenue, setUseVenue,
                usePapersCool, setUsePapersCool, enableLLM, setEnableLLM,
                useSummary, setUseSummary, useTrends, setUseTrends,
                useInsight, setUseInsight, useRelevance, setUseRelevance,
                enableJudge, setEnableJudge, judgeRuns, setJudgeRuns,
                judgeMaxItems, setJudgeMaxItems, judgeTokenBudget, setJudgeTokenBudget,
              }} />
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>

      {error && <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>}

      {isLoading && (
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

      {/* Stats Row */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Queries" value={queries.length} icon={<FilterIcon className="size-5" />} />
        <StatCard label="Papers Found" value={searchResult?.summary?.unique_items ?? dailyResult?.report?.stats?.unique_items ?? 0} icon={<BookOpenIcon className="size-5" />} />
        <StatCard label="Judged" value={judgedPapersCount} icon={<StarIcon className="size-5" />} />
        <StatCard label="Phase" value={phase} icon={<TrendingUpIcon className="size-5" />} />
      </div>

      {/* DAG (collapsible) */}
      <Card>
        <CardHeader className="cursor-pointer py-3" onClick={() => setDagOpen(!dagOpen)}>
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Workflow DAG</CardTitle>
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
            <WorkflowDagView statuses={dagStatuses} queriesCount={queries.length} hitCount={searchResult?.summary?.total_query_hits ?? 0} uniqueCount={searchResult?.summary?.unique_items ?? 0} llmEnabled={enableLLM} judgeEnabled={enableJudge} />
          </CardContent>
        )}
      </Card>

      {/* Result Tabs */}
      <Tabs defaultValue="papers" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="papers" className="gap-1.5"><BookOpenIcon className="size-3.5" /> Papers</TabsTrigger>
          <TabsTrigger value="insights" className="gap-1.5"><SparklesIcon className="size-3.5" /> Insights</TabsTrigger>
          <TabsTrigger value="judge" className="gap-1.5"><StarIcon className="size-3.5" /> Judge</TabsTrigger>
          <TabsTrigger value="markdown" className="gap-1.5"><BookOpenIcon className="size-3.5" /> Report</TabsTrigger>
        </TabsList>

        {/* Papers */}
        <TabsContent value="papers" className="mt-4 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">{allPapers.length} papers</p>
            <div className="flex items-center gap-2">
              <Label className="text-xs">Sort:</Label>
              <select className="h-7 rounded-md border bg-background px-2 text-xs" value={sortBy} onChange={(e) => setSortBy(e.target.value as "score" | "judge")}>
                <option value="score">Search Score</option>
                <option value="judge">Judge Score</option>
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
                Run a search or generate a DailyPaper to see papers here.
              </div>
            )}
          </div>
        </TabsContent>

        {/* Insights */}
        <TabsContent value="insights" className="mt-4 space-y-4">
          {dailyResult?.report?.llm_analysis?.daily_insight ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Insight</CardTitle></CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert text-sm">
                  <Markdown remarkPlugins={[remarkGfm]}>{dailyResult.report.llm_analysis.daily_insight}</Markdown>
                </div>
              </CardContent>
            </Card>
          ) : isLoading ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Insight (Generating...)</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="h-3 w-full animate-pulse rounded bg-muted" />
                <div className="h-3 w-11/12 animate-pulse rounded bg-muted" />
                <div className="h-3 w-9/12 animate-pulse rounded bg-muted" />
              </CardContent>
            </Card>
          ) : (
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">Enable LLM Analysis and run DailyPaper to see insights.</div>
          )}
          {(dailyResult?.report?.llm_analysis?.query_trends || []).length > 0 ? (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold">Query Trend Analysis</h3>
              {dailyResult!.report.llm_analysis!.query_trends!.map((trend, idx) => (
                <Card key={`${trend.query}-${idx}`}>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">{trend.query}</CardTitle></CardHeader>
                  <CardContent>
                    <div className="prose prose-sm max-w-none dark:prose-invert text-sm">
                      <Markdown remarkPlugins={[remarkGfm]}>{trend.analysis}</Markdown>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : isLoading && enableLLM && useTrends ? (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Query Trend Analysis (Generating...)</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="h-3 w-1/3 animate-pulse rounded bg-muted" />
                <div className="h-3 w-full animate-pulse rounded bg-muted" />
                <div className="h-3 w-10/12 animate-pulse rounded bg-muted" />
              </CardContent>
            </Card>
          ) : null}
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
            {judgedPapersCount === 0 && (isLoading && enableJudge ? (
              Array.from({ length: 2 }).map((_, idx) => (
                <div key={`judge-skeleton-${idx}`} className="rounded-lg border p-4">
                  <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
                  <div className="mt-2 h-3 w-1/3 animate-pulse rounded bg-muted" />
                  <div className="mt-4 h-36 animate-pulse rounded bg-muted" />
                </div>
              ))
            ) : (
              <div className="col-span-2 rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">Enable LLM Judge and run Analyze to see judge results.</div>
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
        <TabsContent value="markdown" className="mt-4 space-y-3">
          {dailyResult?.report ? (
            <>
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">DailyPaper Report</CardTitle>
                    <div className="flex gap-2 text-xs text-muted-foreground">
                      {dailyResult.markdown_path && <span>MD: {dailyResult.markdown_path}</span>}
                      {dailyResult.json_path && <span>JSON: {dailyResult.json_path}</span>}
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
      </Tabs>
    </div>
  )
}
