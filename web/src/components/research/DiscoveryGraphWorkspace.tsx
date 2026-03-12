"use client"

import { useEffect, useMemo, useState } from "react"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"

type DiscoveryPaper = {
  id?: number
  title?: string
  abstract?: string
  authors?: string[]
  year?: number
  venue?: string
  citation_count?: number
  url?: string
  doi?: string
  arxiv_id?: string
  semantic_scholar_id?: string
  openalex_id?: string
  source?: string
}

type DiscoveryItem = {
  candidate_key?: string
  paper: DiscoveryPaper
  edge_types: string[]
  score: number
  why_this_paper: string[]
}

type DiscoveryResponse = {
  seed: { seed_type?: string; seed_id?: string; title?: string; name?: string }
  nodes: Array<{ id: string; label: string; type: string; year?: number; score?: number }>
  edges: Array<{ source: string; target: string; type: string; weight?: number }>
  items: DiscoveryItem[]
  stats?: Record<string, unknown>
}

type SavePaperPayload = {
  paper_id: string
  title: string
  abstract?: string
  authors?: string[]
  year?: number
  venue?: string
  citation_count?: number
  url?: string
  source?: string
}

type SeedType = "doi" | "arxiv" | "openalex" | "semantic_scholar" | "author"

const EDGE_COLORS: Record<string, string> = {
  related: "#3b82f6",
  cited: "#10b981",
  citing: "#f59e0b",
  coauthor: "#8b5cf6",
}

function toSavePayload(item: DiscoveryItem): SavePaperPayload | null {
  const paper = item.paper || {}
  const title = String(paper.title || "").trim()
  if (!title) return null
  const paperId =
    String(paper.semantic_scholar_id || "").trim() ||
    String(paper.openalex_id || "").trim() ||
    String(paper.arxiv_id || "").trim() ||
    String(paper.doi || "").trim() ||
    `title:${title}`
  return {
    paper_id: paperId,
    title,
    abstract: paper.abstract || "",
    authors: paper.authors || [],
    year: paper.year,
    venue: paper.venue,
    citation_count: paper.citation_count || 0,
    url: paper.url || "",
    source: paper.source || "semantic_scholar",
  }
}

function toValidYear(value: string): number | undefined {
  const text = value.trim()
  if (!text) return undefined
  const num = Number(text)
  if (!Number.isInteger(num) || num < 1900 || num > 2100) return undefined
  return num
}

interface DiscoveryGraphWorkspaceProps {
  trackId: number | null
  onSavePaper: (paper: SavePaperPayload) => Promise<void>
  initialSeedType?: SeedType
  initialSeedId?: string
  className?: string
}

export default function DiscoveryGraphWorkspace({
    trackId,
  onSavePaper,
  initialSeedType = "doi",
  initialSeedId = "",
  className,
}: DiscoveryGraphWorkspaceProps) {
  const [seedType, setSeedType] = useState<SeedType>(initialSeedType)
  const [seedId, setSeedId] = useState(initialSeedId)
  const [limit, setLimit] = useState("30")
  const [requestYearFrom, setRequestYearFrom] = useState("")
  const [requestYearTo, setRequestYearTo] = useState("")
  const [edgeFilter, setEdgeFilter] = useState<Record<string, boolean>>({
    related: true,
    cited: true,
    citing: true,
    coauthor: true,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [payload, setPayload] = useState<DiscoveryResponse | null>(null)
  const [savingKey, setSavingKey] = useState<string | null>(null)

  const [timelineFrom, setTimelineFrom] = useState<number | null>(null)
  const [timelineTo, setTimelineTo] = useState<number | null>(null)

  useEffect(() => {
    setSeedType(initialSeedType)
  }, [initialSeedType])

  useEffect(() => {
    setSeedId(initialSeedId)
  }, [initialSeedId])

  const timelineYearCounts = useMemo(() => {
    const rows = new Map<number, number>()
    for (const item of payload?.items || []) {
      const year = Number(item.paper?.year)
      if (Number.isInteger(year) && year >= 1900 && year <= 2100) {
        rows.set(year, (rows.get(year) || 0) + 1)
      }
    }
    return Array.from(rows.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([year, count]) => ({ year, count }))
  }, [payload])

  useEffect(() => {
    if (timelineYearCounts.length === 0) {
      setTimelineFrom(null)
      setTimelineTo(null)
      return
    }
    setTimelineFrom(timelineYearCounts[0].year)
    setTimelineTo(timelineYearCounts[timelineYearCounts.length - 1].year)
  }, [timelineYearCounts])

  const maxTimelineCount = useMemo(() => {
    return Math.max(1, ...timelineYearCounts.map((row) => row.count))
  }, [timelineYearCounts])

  const filteredItems = useMemo(() => {
    return (payload?.items || []).filter((item) => {
      if (!(item.edge_types || []).some((edge) => edgeFilter[edge])) {
        return false
      }

      const year = Number(item.paper?.year)
      if (timelineFrom !== null && timelineTo !== null) {
        if (!Number.isInteger(year)) return false
        if (year < timelineFrom || year > timelineTo) return false
      }
      return true
    })
  }, [payload, edgeFilter, timelineFrom, timelineTo])

  const filteredNodeIds = useMemo(() => {
    const rows = new Set<string>()
    for (const item of filteredItems) {
      const key = String(item.candidate_key || "").trim()
      if (key) rows.add(key)
    }
    return rows
  }, [filteredItems])

  const displayEdges = useMemo(() => {
    const edges = payload?.edges || []
    return edges.filter(
      (edge) => edgeFilter[edge.type] && (filteredNodeIds.size === 0 || filteredNodeIds.has(edge.target)),
    )
  }, [payload, edgeFilter, filteredNodeIds])

  const graphPoints = useMemo(() => {
    const nodes = payload?.nodes || []
    const seed = nodes.find((node) => node.type === "seed")
    const papers = nodes.filter((node) => node.type === "paper" && filteredNodeIds.has(node.id))
    const center = { x: 220, y: 150 }
    const radius = 110
    const map: Record<string, { x: number; y: number; label: string; type: string }> = {}
    if (seed) map[seed.id] = { x: center.x, y: center.y, label: seed.label, type: "seed" }
    papers.forEach((node, index) => {
      const angle = (index / Math.max(papers.length, 1)) * Math.PI * 2
      map[node.id] = {
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
        label: node.label,
        type: "paper",
      }
    })
    return map
  }, [payload, filteredNodeIds])

  async function runDiscovery() {
    if (!seedId.trim()) return
    setLoading(true)
    setError(null)
    try {
      const parsedLimit = Number(limit) || 30
      const res = await fetch("/api/research/discovery/seed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_id: trackId ?? undefined,
          seed_type: seedType,
          seed_id: seedId.trim(),
          limit: Math.max(1, Math.min(parsedLimit, 200)),
          include_related: true,
          include_cited: true,
          include_citing: true,
          include_coauthor: true,
          personalized: true,
          year_from: toValidYear(requestYearFrom),
          year_to: toValidYear(requestYearTo),
        }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }
      setPayload((await res.json()) as DiscoveryResponse)
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setError(detail)
      setPayload(null)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave(item: DiscoveryItem) {
    const savePayload = toSavePayload(item)
    if (!savePayload) return
    setSavingKey(savePayload.paper_id)
    try {
      await onSavePaper(savePayload)
    } finally {
      setSavingKey(null)
    }
  }

  return (
    <Card className={cn("mt-6 border-border/80", className)}>
      <CardHeader>
        <CardTitle>Discovery Graph</CardTitle>
        <CardDescription>
          Seed paper/author expansion with graph, explainable ranking, and timeline slicing.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-6">
          <select
            className="h-9 min-w-0 rounded-md border bg-background px-2 text-sm"
            value={seedType}
            onChange={(event) => setSeedType(event.target.value as typeof seedType)}
            disabled={loading}
          >
            <option value="doi">DOI</option>
            <option value="arxiv">arXiv</option>
            <option value="openalex">OpenAlex</option>
            <option value="semantic_scholar">Semantic Scholar</option>
            <option value="author">Author ID</option>
          </select>
          <Input
            value={seedId}
            onChange={(event) => setSeedId(event.target.value)}
            placeholder="Seed identifier"
            disabled={loading}
            className="min-w-0 xl:col-span-2"
          />
          <Input
            value={limit}
            onChange={(event) => setLimit(event.target.value)}
            placeholder="30"
            disabled={loading}
            className="min-w-0"
          />
          <Input
            value={requestYearFrom}
            onChange={(event) => setRequestYearFrom(event.target.value)}
            placeholder="Year from"
            disabled={loading}
            className="min-w-0"
          />
          <Input
            value={requestYearTo}
            onChange={(event) => setRequestYearTo(event.target.value)}
            placeholder="Year to"
            disabled={loading}
            className="min-w-0"
          />
          <Button
            onClick={runDiscovery}
            disabled={loading || !seedId.trim()}
            className="w-full xl:w-auto"
          >
            {loading ? "Discovering..." : "Discover"}
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {Object.keys(edgeFilter).map((edge) => (
            <label key={edge} className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={edgeFilter[edge]}
                onCheckedChange={(checked) =>
                  setEdgeFilter((prev) => ({ ...prev, [edge]: Boolean(checked) }))
                }
              />
              <span className="capitalize">{edge}</span>
            </label>
          ))}
          {payload?.stats ? (
            <>
              <Badge variant="outline">Candidates: {String(payload.stats.candidate_count || 0)}</Badge>
              <Badge variant="outline">Visible: {filteredItems.length}</Badge>
            </>
          ) : null}
        </div>

        {timelineYearCounts.length > 0 && timelineFrom !== null && timelineTo !== null ? (
          <div className="space-y-3 rounded-md border bg-muted/20 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-medium">
                Timeline window: {timelineFrom} - {timelineTo}
              </p>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setTimelineFrom(timelineYearCounts[0].year)
                  setTimelineTo(timelineYearCounts[timelineYearCounts.length - 1].year)
                }}
              >
                Reset Timeline
              </Button>
            </div>

            <div className="grid gap-2 md:grid-cols-2">
              <label className="space-y-1 text-xs text-muted-foreground">
                From year
                <Input
                  type="range"
                  min={timelineYearCounts[0].year}
                  max={timelineYearCounts[timelineYearCounts.length - 1].year}
                  value={timelineFrom}
                  onChange={(event) => {
                    const nextFrom = Number(event.target.value)
                    setTimelineFrom(nextFrom)
                    setTimelineTo((prev) => (prev !== null ? Math.max(prev, nextFrom) : nextFrom))
                  }}
                />
              </label>
              <label className="space-y-1 text-xs text-muted-foreground">
                To year
                <Input
                  type="range"
                  min={timelineYearCounts[0].year}
                  max={timelineYearCounts[timelineYearCounts.length - 1].year}
                  value={timelineTo}
                  onChange={(event) => {
                    const nextTo = Number(event.target.value)
                    setTimelineTo(nextTo)
                    setTimelineFrom((prev) => (prev !== null ? Math.min(prev, nextTo) : nextTo))
                  }}
                />
              </label>
            </div>

            <div className="grid grid-cols-6 gap-1 md:grid-cols-10">
              {timelineYearCounts.map((row) => (
                <button
                  key={row.year}
                  type="button"
                  className="group rounded-sm border bg-background p-1 text-[10px] hover:bg-muted"
                  onClick={() => {
                    setTimelineFrom(row.year)
                    setTimelineTo(row.year)
                  }}
                  title={`${row.year}: ${row.count} papers`}
                >
                  <div
                    className="mx-auto w-full bg-primary/70 transition-opacity group-hover:opacity-90"
                    style={{ height: `${Math.max(8, (row.count / maxTimelineCount) * 36)}px` }}
                  />
                  <p className="mt-1 truncate text-[10px] text-muted-foreground">{row.year}</p>
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {!!payload && (
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-md border bg-muted/30 p-2">
              <svg viewBox="0 0 440 300" className="h-[300px] w-full">
                {displayEdges.map((edge, idx) => {
                  const from = graphPoints[edge.source]
                  const to = graphPoints[edge.target]
                  if (!from || !to) return null
                  return (
                    <line
                      key={`${edge.source}-${edge.target}-${idx}`}
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      stroke={EDGE_COLORS[edge.type] || "#94a3b8"}
                      strokeWidth={Math.max(1, Number(edge.weight || 1))}
                      opacity={0.8}
                    />
                  )
                })}
                {Object.entries(graphPoints).map(([id, node]) => (
                  <g key={id}>
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.type === "seed" ? 12 : 8}
                      fill={node.type === "seed" ? "#111827" : "#2563eb"}
                    />
                    <text
                      x={node.x}
                      y={node.y + (node.type === "seed" ? -16 : -12)}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#334155"
                    >
                      {node.label.slice(0, 20)}
                    </text>
                  </g>
                ))}
              </svg>
            </div>

            <div className="space-y-2">
              {filteredItems.slice(0, 12).map((item, idx) => {
                const paper = item.paper || {}
                const savePayload = toSavePayload(item)
                const saveKey = savePayload?.paper_id || `${idx}`
                return (
                  <div key={`${saveKey}-${idx}`} className="rounded-md border p-3">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="font-medium text-sm">{paper.title || "Untitled"}</p>
                        <p className="text-xs text-muted-foreground">
                          {(paper.authors || []).slice(0, 3).join(", ") || "Unknown authors"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {[paper.year, paper.venue].filter(Boolean).join(" · ") || "-"}
                        </p>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {(item.edge_types || []).map((edge) => (
                            <Badge key={edge} variant="secondary">
                              {edge}
                            </Badge>
                          ))}
                          <Badge variant="outline">score {item.score.toFixed(2)}</Badge>
                        </div>
                        {(item.why_this_paper || []).length > 0 ? (
                          <p className="mt-1 text-xs text-muted-foreground">
                            {item.why_this_paper.join(" · ")}
                          </p>
                        ) : null}
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleSave(item)}
                        disabled={!savePayload || savingKey === saveKey}
                      >
                        {savingKey === saveKey ? "Saving..." : "Save"}
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
