"use client"

import Link from "next/link"
import { useCallback, useEffect, useMemo, useState } from "react"
import { ArrowRight, BookOpen, FlaskConical, Plus, Search, Trash2, Users, Workflow } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import type { Scholar } from "@/lib/types"

interface ScholarsWatchlistProps {
  scholars: Scholar[]
}

type StatusFilter = "all" | "active" | "idle"

function toResearchLink(scholar: Scholar): string {
  const keyword = scholar.keywords?.[0] || scholar.name
  return `/research?query=${encodeURIComponent(keyword)}&scholar=${encodeURIComponent(scholar.name)}`
}

function splitTags(raw: string): string[] {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
}

export function ScholarsWatchlist({ scholars }: ScholarsWatchlistProps) {
  const [items, setItems] = useState<Scholar[]>(scholars)
  const [query, setQuery] = useState("")
  const [status, setStatus] = useState<StatusFilter>("all")

  const [createOpen, setCreateOpen] = useState(false)
  const [createName, setCreateName] = useState("")
  const [createSemanticId, setCreateSemanticId] = useState("")
  const [createAffiliation, setCreateAffiliation] = useState("")
  const [createKeywords, setCreateKeywords] = useState("")
  const [createLoading, setCreateLoading] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)

  const [rowBusy, setRowBusy] = useState<Record<string, boolean>>({})

  useEffect(() => {
    setItems(scholars)
  }, [scholars])

  const refreshScholars = useCallback(async () => {
    const res = await fetch("/api/research/scholars?limit=200", { cache: "no-store" })
    if (!res.ok) {
      throw new Error(`Failed to refresh scholars: ${res.status}`)
    }
    const payload = (await res.json()) as { items?: Scholar[] }
    setItems(payload.items || [])
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return items
      .filter((item) => {
        if (status !== "all" && item.status !== status) return false
        if (!q) return true
        const bag = [item.name, item.affiliation, ...(item.keywords || [])].join(" ").toLowerCase()
        return bag.includes(q)
      })
      .sort((a, b) => {
        if (a.status !== b.status) return a.status === "active" ? -1 : 1
        return (b.h_index || 0) - (a.h_index || 0)
      })
  }, [items, query, status])

  const activeCount = items.filter((item) => item.status === "active").length

  async function handleCreateScholar() {
    setCreateLoading(true)
    setCreateError(null)
    try {
      const res = await fetch("/api/research/scholars", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: createName,
          semantic_scholar_id: createSemanticId,
          affiliations: createAffiliation ? [createAffiliation.trim()] : [],
          keywords: splitTags(createKeywords),
          research_fields: [],
        }),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `HTTP ${res.status}`)
      }
      await refreshScholars()
      setCreateOpen(false)
      setCreateName("")
      setCreateSemanticId("")
      setCreateAffiliation("")
      setCreateKeywords("")
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : String(error))
    } finally {
      setCreateLoading(false)
    }
  }

  async function handleDeleteScholar(scholar: Scholar) {
    const ok = window.confirm(`Remove ${scholar.name} from watchlist?`)
    if (!ok) return

    setRowBusy((prev) => ({ ...prev, [scholar.id]: true }))
    try {
      const res = await fetch(`/api/research/scholars/${encodeURIComponent(scholar.id)}`, {
        method: "DELETE",
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `HTTP ${res.status}`)
      }
      await refreshScholars()
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error))
    } finally {
      setRowBusy((prev) => ({ ...prev, [scholar.id]: false }))
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 pb-10 sm:p-6">
      <Card className="border-border/60 bg-gradient-to-br from-card via-card to-muted/30">
        <CardContent className="p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Scholar Intelligence</h1>
              <p className="max-w-2xl text-sm text-muted-foreground">
                Watch momentum shifts, track high-signal authors, and jump straight into Research Feed decisions.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => setCreateOpen(true)}>
                <Plus className="mr-1 h-4 w-4" />
                Add Scholar
              </Button>
              <Button asChild variant="outline">
                <Link href="/research">Open Research</Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/workflows">Run Tracking Job</Link>
              </Button>
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Tracked Scholars</p>
              <p className="mt-1 text-2xl font-semibold">{items.length}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Active Signals</p>
              <p className="mt-1 text-2xl font-semibold">{activeCount}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Avg H-Index</p>
              <p className="mt-1 text-2xl font-semibold">
                {items.length
                  ? Math.round(items.reduce((acc, row) => acc + Number(row.h_index || 0), 0) / items.length)
                  : 0}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Watchlist</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <div className="relative min-w-[220px] flex-1">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="pl-9"
                placeholder="Search scholar, affiliation, keyword"
              />
            </div>
            <Button
              variant={status === "all" ? "default" : "outline"}
              size="sm"
              onClick={() => setStatus("all")}
            >
              All
            </Button>
            <Button
              variant={status === "active" ? "default" : "outline"}
              size="sm"
              onClick={() => setStatus("active")}
            >
              Active
            </Button>
            <Button
              variant={status === "idle" ? "default" : "outline"}
              size="sm"
              onClick={() => setStatus("idle")}
            >
              Idle
            </Button>
          </div>

          {!filtered.length ? (
            <div className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">
              No scholars matched current filters.
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {filtered.map((scholar) => (
                <Card key={scholar.id} className="border-border/60">
                  <CardContent className="space-y-3 p-4">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="line-clamp-1 text-sm font-semibold">{scholar.name}</p>
                        <p className="line-clamp-1 text-xs text-muted-foreground">{scholar.affiliation}</p>
                      </div>
                      <Badge variant={scholar.status === "active" ? "default" : "secondary"} className="capitalize">
                        {scholar.status}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="rounded-md bg-muted/40 p-2">
                        <p className="text-xs text-muted-foreground">H-Index</p>
                        <p className="text-sm font-semibold">{scholar.h_index || 0}</p>
                      </div>
                      <div className="rounded-md bg-muted/40 p-2">
                        <p className="text-xs text-muted-foreground">Papers</p>
                        <p className="text-sm font-semibold">{scholar.papers_tracked || 0}</p>
                      </div>
                      <div className="rounded-md bg-muted/40 p-2">
                        <p className="text-xs text-muted-foreground">Cached</p>
                        <p className="text-sm font-semibold">{scholar.cached_papers || 0}</p>
                      </div>
                    </div>

                    <p className="text-xs text-muted-foreground">{scholar.recent_activity}</p>

                    {(scholar.keywords || []).length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {(scholar.keywords || []).slice(0, 3).map((keyword) => (
                          <Badge key={`${scholar.id}-${keyword}`} variant="outline" className="text-[11px]">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    ) : null}

                    <div className="flex flex-wrap gap-1.5 pt-1">
                      <Button asChild size="sm" className="h-8">
                        <Link href={`/scholars/${encodeURIComponent(scholar.id)}`}>
                          <Users className="mr-1 h-3.5 w-3.5" />
                          Signals
                        </Link>
                      </Button>
                      <Button asChild size="sm" variant="outline" className="h-8">
                        <Link href={toResearchLink(scholar)}>
                          <FlaskConical className="mr-1 h-3.5 w-3.5" />
                          Research
                        </Link>
                      </Button>
                      <Button asChild size="sm" variant="ghost" className="h-8">
                        <Link href="/workflows">
                          <Workflow className="mr-1 h-3.5 w-3.5" />
                          Workflow
                        </Link>
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-8 text-destructive hover:text-destructive"
                        disabled={!!rowBusy[scholar.id]}
                        onClick={() => handleDeleteScholar(scholar)}
                      >
                        <Trash2 className="mr-1 h-3.5 w-3.5" />
                        Remove
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          <div className="flex items-center justify-between rounded-xl border bg-muted/20 p-3">
            <p className="text-xs text-muted-foreground">
              Watchlist is persisted in subscription config and powers scholar tracking workflows.
            </p>
            <Button asChild size="sm" variant="ghost">
              <Link href="/settings">
                Open Settings
                <ArrowRight className="ml-1 h-3.5 w-3.5" />
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-3 sm:grid-cols-2">
        <Card>
          <CardContent className="flex items-center justify-between p-4">
            <div>
              <p className="text-sm font-medium">Research Feed Linkage</p>
              <p className="text-xs text-muted-foreground">Jump from author watchlist to track-scoped search context.</p>
            </div>
            <Button asChild size="sm" variant="outline">
              <Link href="/research">
                <BookOpen className="mr-1 h-3.5 w-3.5" />
                Open Research
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center justify-between p-4">
            <div>
              <p className="text-sm font-medium">Automation Linkage</p>
              <p className="text-xs text-muted-foreground">Run periodic scholar tracking and evidence pipelines.</p>
            </div>
            <Button asChild size="sm" variant="outline">
              <Link href="/workflows">
                <Workflow className="mr-1 h-3.5 w-3.5" />
                Open Workflows
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Scholar</DialogTitle>
            <DialogDescription>
              Add a scholar to watchlist. This updates scholar subscriptions used by tracking workflows.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3">
            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">Name</p>
              <Input value={createName} onChange={(event) => setCreateName(event.target.value)} placeholder="Dawn Song" />
            </div>
            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">Semantic Scholar ID</p>
              <Input value={createSemanticId} onChange={(event) => setCreateSemanticId(event.target.value)} placeholder="1741101" />
            </div>
            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">Affiliation (optional)</p>
              <Input value={createAffiliation} onChange={(event) => setCreateAffiliation(event.target.value)} placeholder="UC Berkeley" />
            </div>
            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">Keywords (comma separated)</p>
              <Input value={createKeywords} onChange={(event) => setCreateKeywords(event.target.value)} placeholder="AI Security, LLM Safety" />
            </div>
            {createError ? <p className="text-xs text-destructive">{createError}</p> : null}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)} disabled={createLoading}>Cancel</Button>
            <Button
              onClick={handleCreateScholar}
              disabled={createLoading || !createName.trim() || !createSemanticId.trim()}
            >
              {createLoading ? "Saving..." : "Save Scholar"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
