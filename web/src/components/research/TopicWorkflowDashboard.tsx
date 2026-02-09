"use client"

import { useMemo, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"

type SearchItem = {
  title: string
  url?: string
  score?: number
  matched_queries?: string[]
  branches?: string[]
  sources?: string[]
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

type DailyResult = {
  report: {
    title: string
    date: string
    stats: {
      unique_items: number
      total_query_hits: number
      query_count: number
    }
    global_top: SearchItem[]
  }
  markdown: string
  markdown_path?: string | null
  json_path?: string | null
}

const DEFAULT_QUERIES = ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]

function parseLines(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
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

  async function runTopicSearch() {
    setLoadingSearch(true)
    setError(null)
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
    } catch (err) {
      setError(String(err))
    } finally {
      setLoadingSearch(false)
    }
  }

  async function runDailyPaper() {
    setLoadingDaily(true)
    setError(null)
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
        }),
      })
      if (!res.ok) {
        throw new Error(await res.text())
      }
      const data = (await res.json()) as DailyResult
      setDailyResult(data)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoadingDaily(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Canvas (MVP)</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">Source</Badge>
            <Badge variant="secondary">Search</Badge>
            <Badge variant="secondary">Rank</Badge>
            <Badge variant="secondary">DailyPaper</Badge>
            <Badge variant="secondary">Scheduler/Feed</Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            当前先做参数化流程面板（比 n8n/coze 全自由拖拽更轻量，后续可升级为节点拖拽）。
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
            <CardTitle>Search Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
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
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>DailyPaper Preview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>Title: {dailyResult?.report?.title ?? "-"}</div>
          <div>Date: {dailyResult?.report?.date ?? "-"}</div>
          <div>Unique Items: {dailyResult?.report?.stats?.unique_items ?? 0}</div>
          <div>Query Count: {dailyResult?.report?.stats?.query_count ?? 0}</div>
          <div>Markdown File: {dailyResult?.markdown_path || "(not saved)"}</div>
          <div>JSON File: {dailyResult?.json_path || "(not saved)"}</div>
          <ScrollArea className="h-64 rounded border p-3">
            <pre className="whitespace-pre-wrap text-xs">{dailyResult?.markdown || "Run Generate DailyPaper to preview markdown."}</pre>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
