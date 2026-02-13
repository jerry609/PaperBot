"use client"

import Link from "next/link"
import { useMemo, useState } from "react"
import { ArrowRight, BookOpen, FlaskConical, Search, Users, Workflow } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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

export function ScholarsWatchlist({ scholars }: ScholarsWatchlistProps) {
  const [query, setQuery] = useState("")
  const [status, setStatus] = useState<StatusFilter>("all")

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return scholars
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
  }, [query, scholars, status])

  const activeCount = scholars.filter((item) => item.status === "active").length

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
              <p className="mt-1 text-2xl font-semibold">{scholars.length}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Active Signals</p>
              <p className="mt-1 text-2xl font-semibold">{activeCount}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Avg H-Index</p>
              <p className="mt-1 text-2xl font-semibold">
                {scholars.length
                  ? Math.round(scholars.reduce((acc, row) => acc + Number(row.h_index || 0), 0) / scholars.length)
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
            <Button variant={status === "all" ? "default" : "outline"} size="sm" onClick={() => setStatus("all")}>All</Button>
            <Button variant={status === "active" ? "default" : "outline"} size="sm" onClick={() => setStatus("active")}>Active</Button>
            <Button variant={status === "idle" ? "default" : "outline"} size="sm" onClick={() => setStatus("idle")}>Idle</Button>
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
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          <div className="flex items-center justify-between rounded-xl border bg-muted/20 p-3">
            <p className="text-xs text-muted-foreground">
              Need new scholars? Edit `config/scholar_subscriptions.yaml` or extend settings UI next.
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
    </div>
  )
}
