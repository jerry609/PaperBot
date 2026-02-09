"use client"

import { useMemo, useState } from "react"

import WorkflowDagView from "@/components/research/WorkflowDagView"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"

type JudgeResult = {
  overall?: number
  recommendation?: string
  one_line_summary?: string
}

type SearchItem = {
  title: string
  url?: string
  score?: number
  matched_queries?: string[]
  branches?: string[]
  sources?: string[]
  ai_summary?: string
  relevance?: {
    score?: number
    reason?: string
  }
  judge?: JudgeResult
}

type SearchResult = {
  source: string
  fetched_at: string
  sources: string[]
  items: SearchItem[]
  summary: {
    unique_items: number
    total_query_hits: number
    source_breakdown?: Record<string, number>
  }
}

type LLMAnalysis = {
  enabled?: boolean
  features?: string[]
  daily_insight?: string
  query_trends?: Array<{ query: string; analysis: string }>
}

type JudgeSummary = {
  enabled?: boolean
  max_items_per_query?: number
  n_runs?: number
  recommendation_count?: Record<string, number>
}

type DailyResult = {
  report: {
    title: string
    date: string
    stats: {
      unique_items: number
      total_query_hits: number
      query_count: number
    }
    queries?: Array<{ normalized_query?: string; raw_query?: string; top_items?: SearchItem[] }>
    global_top: SearchItem[]
    llm_analysis?: LLMAnalysis
    judge?: JudgeSummary
  }
  markdown: string
  markdown_path?: string | null
  json_path?: string | null
}

type WorkflowPhase = "idle" | "searching" | "searched" | "reporting" | "reported" | "error"

type StepStatus = "pending" | "running" | "done" | "error" | "skipped"

const DEFAULT_QUERIES = ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]

function parseLines(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
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

  if (phase === "searching") {
    return {
      ...base,
      source: "done",
      normalize: "done",
      search: "running",
      rank: "running",
    }
  }

  if (phase === "searched") {
    return {
      ...base,
      source: "done",
      normalize: "done",
      search: "done",
      rank: "done",
    }
  }

  if (phase === "reporting") {
    return {
      ...base,
      source: "done",
      normalize: "done",
      search: "done",
      rank: "done",
      llm: llmEnabled ? "running" : "skipped",
      judge: judgeEnabled ? "running" : "skipped",
      report: "running",
    }
  }

  if (phase === "reported") {
    return {
      ...base,
      source: "done",
      normalize: "done",
      search: "done",
      rank: "done",
      llm: llmEnabled ? "done" : "skipped",
      judge: judgeEnabled ? "done" : "skipped",
      report: reportReady ? "done" : "pending",
      scheduler: reportReady ? "done" : "pending",
    }
  }

  if (phase === "error" || hasError) {
    return {
      ...base,
      source: "done",
      normalize: "done",
      search: "error",
      rank: "error",
      llm: llmEnabled ? "error" : "skipped",
      judge: judgeEnabled ? "error" : "skipped",
      report: "error",
      scheduler: "pending",
    }
  }

  return base
}

export default function TopicWorkflowDashboard() {
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

  const [phase, setPhase] = useState<WorkflowPhase>("idle")
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null)
  const [dailyResult, setDailyResult] = useState<DailyResult | null>(null)
  const [loadingSearch, setLoadingSearch] = useState(false)
  const [loadingDaily, setLoadingDaily] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const queries = useMemo(() => parseLines(queriesText), [queriesText])
  const branches = useMemo(
    () => [useArxiv ? "arxiv" : "", useVenue ? "venue" : ""].filter(Boolean),
    [useArxiv, useVenue],
  )
  const sources = useMemo(() => [usePapersCool ? "papers_cool" : ""].filter(Boolean), [usePapersCool])
  const llmFeatures = useMemo(
    () => [
      useSummary ? "summary" : "",
      useTrends ? "trends" : "",
      useInsight ? "insight" : "",
      useRelevance ? "relevance" : "",
    ].filter(Boolean),
    [useInsight, useRelevance, useSummary, useTrends],
  )

  const dagStatuses = useMemo(
    () =>
      buildDagStatuses({
        phase,
        hasError: Boolean(error),
        llmEnabled: enableLLM,
        judgeEnabled: enableJudge,
        reportReady: Boolean(dailyResult?.report),
      }),
    [phase, error, enableLLM, enableJudge, dailyResult],
  )

  async function runTopicSearch() {
    setLoadingSearch(true)
    setError(null)
    setPhase("searching")
    try {
      const res = await fetch("/api/research/paperscool/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queries,
          sources,
          branches,
          top_k_per_query: topK,
          show_per_branch: showPerBranch,
        }),
      })
      if (!res.ok) {
        throw new Error(await res.text())
      }
      const data = (await res.json()) as SearchResult
      setSearchResult(data)
      setPhase("searched")
    } catch (err) {
      setError(String(err))
      setPhase("error")
    } finally {
      setLoadingSearch(false)
    }
  }

  async function runDailyPaper() {
    setLoadingDaily(true)
    setError(null)
    setPhase("reporting")
    try {
      const res = await fetch("/api/research/paperscool/daily", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queries,
          sources,
          branches,
          top_k_per_query: topK,
          show_per_branch: showPerBranch,
          top_n: topN,
          title: "DailyPaper Digest",
          formats: ["both"],
          save: saveDaily,
          output_dir: outputDir,
          enable_llm_analysis: enableLLM,
          llm_features: llmFeatures,
          enable_judge: enableJudge,
          judge_runs: judgeRuns,
          judge_max_items_per_query: judgeMaxItems,
        }),
      })
      if (!res.ok) {
        throw new Error(await res.text())
      }
      const data = (await res.json()) as DailyResult
      setDailyResult(data)
      setPhase("reported")
    } catch (err) {
      setError(String(err))
      setPhase("error")
    } finally {
      setLoadingDaily(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Canvas (XYFlow Read-Only DAG)</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">Source</Badge>
            <Badge variant="secondary">Normalize</Badge>
            <Badge variant="secondary">Search</Badge>
            <Badge variant="secondary">Dedupe/Rank</Badge>
            <Badge variant={enableLLM ? "default" : "secondary"}>LLM Analysis</Badge>
            <Badge variant={enableJudge ? "default" : "secondary"}>LLM Judge</Badge>
            <Badge variant="secondary">DailyPaper</Badge>
            <Badge variant="secondary">Scheduler/Feed</Badge>
          </div>
          <WorkflowDagView
            statuses={dagStatuses}
            queriesCount={queries.length}
            hitCount={searchResult?.summary?.total_query_hits ?? 0}
            uniqueCount={searchResult?.summary?.unique_items ?? 0}
            llmEnabled={enableLLM}
            judgeEnabled={enableJudge}
          />
          <p className="text-sm text-muted-foreground">
            这是预定义拓扑的只读 DAG，可视化状态流，不是 n8n/coze 式自由拖拽。
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Workflow Config</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Topics (one per line)</Label>
              <Textarea value={queriesText} onChange={(e) => setQueriesText(e.target.value)} rows={6} />
            </div>

            <div className="space-y-2">
              <Label>Sources</Label>
              <div className="flex items-center gap-2">
                <Checkbox checked={usePapersCool} onCheckedChange={(v) => setUsePapersCool(Boolean(v))} />
                <span className="text-sm">papers_cool</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Branches</Label>
              <div className="flex items-center gap-6">
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox checked={useArxiv} onCheckedChange={(v) => setUseArxiv(Boolean(v))} />
                  arxiv
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox checked={useVenue} onCheckedChange={(v) => setUseVenue(Boolean(v))} />
                  venue
                </label>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-1">
                <Label>Top K</Label>
                <Input type="number" min={1} value={topK} onChange={(e) => setTopK(Number(e.target.value || 5))} />
              </div>
              <div className="space-y-1">
                <Label>Show</Label>
                <Input
                  type="number"
                  min={1}
                  value={showPerBranch}
                  onChange={(e) => setShowPerBranch(Number(e.target.value || 25))}
                />
              </div>
              <div className="space-y-1">
                <Label>Daily Top N</Label>
                <Input type="number" min={1} value={topN} onChange={(e) => setTopN(Number(e.target.value || 10))} />
              </div>
            </div>

            <div className="space-y-2 rounded border p-3">
              <label className="flex items-center gap-2 text-sm">
                <Checkbox checked={saveDaily} onCheckedChange={(v) => setSaveDaily(Boolean(v))} />
                Save DailyPaper files
              </label>
              <Input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} placeholder="./reports/dailypaper" />
            </div>

            <div className="space-y-2 rounded border p-3">
              <label className="flex items-center gap-2 text-sm">
                <Checkbox checked={enableLLM} onCheckedChange={(v) => setEnableLLM(Boolean(v))} />
                Enable LLM Analysis
              </label>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <label className="flex items-center gap-2">
                  <Checkbox checked={useSummary} onCheckedChange={(v) => setUseSummary(Boolean(v))} /> summary
                </label>
                <label className="flex items-center gap-2">
                  <Checkbox checked={useTrends} onCheckedChange={(v) => setUseTrends(Boolean(v))} /> trends
                </label>
                <label className="flex items-center gap-2">
                  <Checkbox checked={useInsight} onCheckedChange={(v) => setUseInsight(Boolean(v))} /> insight
                </label>
                <label className="flex items-center gap-2">
                  <Checkbox checked={useRelevance} onCheckedChange={(v) => setUseRelevance(Boolean(v))} /> relevance
                </label>
              </div>
            </div>

            <div className="space-y-2 rounded border p-3">
              <label className="flex items-center gap-2 text-sm">
                <Checkbox checked={enableJudge} onCheckedChange={(v) => setEnableJudge(Boolean(v))} />
                Enable LLM Judge
              </label>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>Judge Runs</Label>
                  <Input type="number" min={1} max={5} value={judgeRuns} onChange={(e) => setJudgeRuns(Number(e.target.value || 1))} />
                </div>
                <div className="space-y-1">
                  <Label>Judge Max Items</Label>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={judgeMaxItems}
                    onChange={(e) => setJudgeMaxItems(Number(e.target.value || 5))}
                  />
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button disabled={loadingSearch || queries.length === 0 || branches.length === 0 || sources.length === 0} onClick={runTopicSearch}>
                {loadingSearch ? "Running Search..." : "Run Topic Search"}
              </Button>
              <Button
                variant="secondary"
                disabled={loadingDaily || queries.length === 0 || branches.length === 0 || sources.length === 0}
                onClick={runDailyPaper}
              >
                {loadingDaily ? "Generating DailyPaper..." : "Generate DailyPaper"}
              </Button>
            </div>

            {error ? <p className="text-sm text-red-500">{error}</p> : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Execution Output</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="search" className="w-full">
              <TabsList>
                <TabsTrigger value="search">Search Results</TabsTrigger>
                <TabsTrigger value="daily">DailyPaper</TabsTrigger>
                <TabsTrigger value="llm">LLM Analysis</TabsTrigger>
                <TabsTrigger value="judge">Judge</TabsTrigger>
              </TabsList>
              <TabsContent value="search" className="space-y-2 text-sm pt-3">
                <div>Source: {searchResult?.source ?? "-"}</div>
                <div>Fetched At: {searchResult?.fetched_at ?? "-"}</div>
                <div>Unique Items: {searchResult?.summary?.unique_items ?? 0}</div>
                <div>Total Query Hits: {searchResult?.summary?.total_query_hits ?? 0}</div>
                <div className="flex flex-wrap gap-2 pt-2">
                  {Object.entries(searchResult?.summary?.source_breakdown || {}).map(([name, count]) => (
                    <Badge key={name} variant="outline">
                      {name}: {count}
                    </Badge>
                  ))}
                </div>
                <ScrollArea className="h-72 rounded border p-3">
                  <div className="space-y-2">
                    {(searchResult?.items || []).slice(0, 15).map((item, idx) => (
                      <div key={`${item.title}-${idx}`} className="rounded border p-2">
                        <div className="font-medium">
                          {item.url ? (
                            <a className="underline" href={item.url} target="_blank" rel="noreferrer">
                              {item.title}
                            </a>
                          ) : (
                            item.title
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          score={item.score} | branches={(item.branches || []).join(", ")} | sources={(item.sources || []).join(", ")}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="daily" className="space-y-2 text-sm pt-3">
                <div>Title: {dailyResult?.report?.title ?? "-"}</div>
                <div>Date: {dailyResult?.report?.date ?? "-"}</div>
                <div>Unique Items: {dailyResult?.report?.stats?.unique_items ?? 0}</div>
                <div>Query Count: {dailyResult?.report?.stats?.query_count ?? 0}</div>
                <div>Markdown File: {dailyResult?.markdown_path || "(not saved)"}</div>
                <div>JSON File: {dailyResult?.json_path || "(not saved)"}</div>
                <ScrollArea className="h-64 rounded border p-3">
                  <pre className="whitespace-pre-wrap text-xs">{dailyResult?.markdown || "Run Generate DailyPaper to preview markdown."}</pre>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="llm" className="space-y-3 text-sm pt-3">
                <div>Enabled: {dailyResult?.report?.llm_analysis?.enabled ? "Yes" : "No"}</div>
                <div>Features: {(dailyResult?.report?.llm_analysis?.features || []).join(", ") || "-"}</div>
                <div className="rounded border p-3 text-sm">
                  <div className="font-medium">Daily Insight</div>
                  <p className="mt-1 text-muted-foreground">{dailyResult?.report?.llm_analysis?.daily_insight || "-"}</p>
                </div>
                <ScrollArea className="h-48 rounded border p-3">
                  <div className="space-y-2">
                    {(dailyResult?.report?.llm_analysis?.query_trends || []).map((trend, idx) => (
                      <div key={`${trend.query}-${idx}`} className="rounded border p-2">
                        <div className="font-medium">{trend.query}</div>
                        <div className="text-xs text-muted-foreground whitespace-pre-wrap">{trend.analysis}</div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="judge" className="space-y-3 text-sm pt-3">
                <div>Enabled: {dailyResult?.report?.judge?.enabled ? "Yes" : "No"}</div>
                <div>Runs: {dailyResult?.report?.judge?.n_runs ?? 0}</div>
                <div>Max Items / Query: {dailyResult?.report?.judge?.max_items_per_query ?? 0}</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(dailyResult?.report?.judge?.recommendation_count || {}).map(([name, count]) => (
                    <Badge key={name} variant="outline">
                      {name}: {count}
                    </Badge>
                  ))}
                </div>
                <ScrollArea className="h-52 rounded border p-3">
                  <div className="space-y-2">
                    {(dailyResult?.report?.queries || []).flatMap((query) =>
                      (query.top_items || []).slice(0, 2).map((item, idx) => (
                        <div key={`${query.normalized_query}-${item.title}-${idx}`} className="rounded border p-2">
                          <div className="font-medium">{item.title}</div>
                          <div className="text-xs text-muted-foreground">
                            {query.normalized_query || query.raw_query} | overall={item.judge?.overall} | rec={item.judge?.recommendation}
                          </div>
                          <div className="text-xs text-muted-foreground">{item.judge?.one_line_summary || "-"}</div>
                        </div>
                      )),
                    )}
                  </div>
                </ScrollArea>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
