"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useCallback, useEffect, useMemo, useState } from "react"
import {
  ArrowRight,
  BellOff,
  BellRing,
  BookOpen,
  Check,
  CircleAlert,
  FlaskConical,
  Plus,
  Search,
  Trash2,
  Users,
  Workflow,
} from "lucide-react"

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
import type { ResearchTrackSummary, Scholar } from "@/lib/types"

interface ScholarsWatchlistProps {
  scholars: Scholar[]
}

type StatusFilter = "all" | "active" | "idle"
type TrackMatchFilter = "all" | "matched" | "unmatched"

type ScholarInsight = {
  scholar: Scholar
  matchedTracks: ResearchTrackSummary[]
  momentumLabel: "rising" | "stable" | "cooling"
  momentumScore: number
  newSinceLastSeen: number
  riskLabel: "low" | "medium" | "high"
}

type ScholarApiRow = {
  id?: string
  scholar_id?: string
  semantic_scholar_id?: string | null
  name?: string
  affiliation?: string
  keywords?: string[]
  h_index?: number
  paper_count?: number
  recent_activity?: string
  status?: "active" | "idle"
  cached_papers?: number
  last_updated?: string | null
  muted?: boolean
  last_seen_at?: string | null
  last_seen_cached_papers?: number
  digest_enabled?: boolean
  digest_frequency?: "daily" | "weekly" | "monthly" | string
  alert_enabled?: boolean
  alert_keywords?: string[]
}

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

function normalizeWords(values: string[] | undefined): Set<string> {
  return new Set((values || []).map((item) => item.trim().toLowerCase()).filter(Boolean))
}

function deriveMomentum(scholar: Scholar): { label: "rising" | "stable" | "cooling"; score: number } {
  const activity = (scholar.recent_activity || "").toLowerCase()
  if (activity.includes("today") || activity.includes("1 day")) {
    return { label: "rising", score: 3 }
  }
  if (scholar.status === "active") {
    return { label: "stable", score: 2 }
  }
  return { label: "cooling", score: 1 }
}

function deriveRisk(scholar: Scholar): "low" | "medium" | "high" {
  const keywordCount = (scholar.keywords || []).length
  const cached = scholar.cached_papers || 0

  if (scholar.status === "idle" && (cached <= 2 || keywordCount <= 1)) return "high"
  if (scholar.status === "idle" || keywordCount <= 1) return "medium"
  return "low"
}

function mapScholarRow(row: ScholarApiRow): Scholar {
  return {
    id: String(row.id || row.semantic_scholar_id || row.scholar_id || row.name || "unknown"),
    semantic_scholar_id: row.semantic_scholar_id || undefined,
    name: String(row.name || "Unknown"),
    affiliation: String(row.affiliation || "Unknown affiliation"),
    h_index: Number(row.h_index || 0),
    papers_tracked: Number(row.paper_count || 0),
    recent_activity: String(row.recent_activity || "No tracking runs yet"),
    status: row.status === "active" ? "active" : "idle",
    keywords: row.keywords || [],
    cached_papers: Number(row.cached_papers || 0),
    last_updated: row.last_updated || null,
    muted: !!row.muted,
    last_seen_at: row.last_seen_at || null,
    last_seen_cached_papers: Number(row.last_seen_cached_papers || 0),
    digest_enabled: !!row.digest_enabled,
    digest_frequency:
      row.digest_frequency === "daily" || row.digest_frequency === "monthly"
        ? row.digest_frequency
        : "weekly",
    alert_enabled: !!row.alert_enabled,
    alert_keywords: Array.isArray(row.alert_keywords)
      ? row.alert_keywords.map((keyword) => String(keyword).trim()).filter(Boolean)
      : [],
  }
}

export function ScholarsWatchlist({ scholars }: ScholarsWatchlistProps) {
  const router = useRouter()

  const [items, setItems] = useState<Scholar[]>(scholars)
  const [tracks, setTracks] = useState<ResearchTrackSummary[]>([])
  const [query, setQuery] = useState("")
  const [institutionQuery, setInstitutionQuery] = useState("")
  const [status, setStatus] = useState<StatusFilter>("all")
  const [trackMatchFilter, setTrackMatchFilter] = useState<TrackMatchFilter>("all")
  const [showMuted, setShowMuted] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

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

  useEffect(() => {
    const loadTracks = async () => {
      try {
        const res = await fetch("/api/research/tracks", { cache: "no-store" })
        if (!res.ok) return
        const payload = (await res.json()) as { tracks?: ResearchTrackSummary[] }
        setTracks(payload.tracks || [])
      } catch {
        setTracks([])
      }
    }
    loadTracks().catch(() => {})
  }, [])

  const refreshScholars = useCallback(async () => {
    const res = await fetch("/api/research/scholars?limit=200", { cache: "no-store" })
    if (!res.ok) {
      throw new Error(`Failed to refresh scholars: ${res.status}`)
    }
    const payload = (await res.json()) as { items?: ScholarApiRow[] }
    setItems((payload.items || []).map(mapScholarRow))
  }, [])

  const insights = useMemo<ScholarInsight[]>(() => {
    return items.map((scholar) => {
      const scholarKeywords = normalizeWords(scholar.keywords)
      const matchedTracks = tracks.filter((track) => {
        const trackKeywords = normalizeWords(track.keywords)
        for (const keyword of scholarKeywords) {
          if (trackKeywords.has(keyword)) return true
        }
        return false
      })

      const momentum = deriveMomentum(scholar)
      const baseline = Number(scholar.last_seen_cached_papers || 0)
      const newSinceLastSeen = Math.max(0, (scholar.cached_papers || 0) - baseline)
      const riskLabel = deriveRisk(scholar)

      return {
        scholar,
        matchedTracks,
        momentumLabel: momentum.label,
        momentumScore: momentum.score,
        newSinceLastSeen,
        riskLabel,
      }
    })
  }, [items, tracks])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const institution = institutionQuery.trim().toLowerCase()

    return insights
      .filter((item) => {
        const scholar = item.scholar
        if (status !== "all" && scholar.status !== status) return false
        if (!showMuted && scholar.muted) return false
        if (trackMatchFilter === "matched" && item.matchedTracks.length === 0) return false
        if (trackMatchFilter === "unmatched" && item.matchedTracks.length > 0) return false

        if (institution && !scholar.affiliation.toLowerCase().includes(institution)) return false

        if (!q) return true
        const bag = [scholar.name, scholar.affiliation, ...(scholar.keywords || [])]
          .join(" ")
          .toLowerCase()
        return bag.includes(q)
      })
      .sort((a, b) => {
        if (a.momentumScore !== b.momentumScore) return b.momentumScore - a.momentumScore
        if (a.newSinceLastSeen !== b.newSinceLastSeen) return b.newSinceLastSeen - a.newSinceLastSeen
        return (b.scholar.h_index || 0) - (a.scholar.h_index || 0)
      })
  }, [insights, query, institutionQuery, status, trackMatchFilter, showMuted])

  const activeCount = items.filter((item) => item.status === "active").length
  const mutedCount = items.filter((item) => item.muted).length

  const selectedInsights = useMemo(() => {
    return insights.filter((row) => selectedIds.has(row.scholar.id))
  }, [insights, selectedIds])

  const selectedForTrack = selectedInsights.length === 1 ? selectedInsights[0].scholar : null

  function buildUniqueTrackName(baseName: string): string {
    const existingNames = new Set(tracks.map((track) => String(track.name || "").trim().toLowerCase()))
    if (!existingNames.has(baseName.toLowerCase())) {
      return baseName
    }

    let suffix = 2
    while (existingNames.has(`${baseName} (${suffix})`.toLowerCase())) {
      suffix += 1
    }
    return `${baseName} (${suffix})`
  }

  function toggleSelected(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  async function patchScholarState(
    scholar: Scholar,
    patch: {
      muted?: boolean
      last_seen_cached_papers?: number
      last_seen_at?: string
      digest_enabled?: boolean
      digest_frequency?: "daily" | "weekly" | "monthly"
      alert_enabled?: boolean
      alert_keywords?: string[]
    },
  ) {
    const res = await fetch(`/api/research/scholars/${encodeURIComponent(scholar.id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => "")
      throw new Error(text || `HTTP ${res.status}`)
    }
  }

  async function markSeen(scholar: Scholar) {
    await patchScholarState(scholar, {
      last_seen_cached_papers: Number(scholar.cached_papers || 0),
      last_seen_at: new Date().toISOString(),
    })
    await refreshScholars()
  }

  async function toggleMuteScholar(scholar: Scholar) {
    await patchScholarState(scholar, { muted: !scholar.muted })
    await refreshScholars()
  }

  async function toggleDigestScholar(scholar: Scholar) {
    const nextEnabled = !scholar.digest_enabled
    await patchScholarState(scholar, {
      digest_enabled: nextEnabled,
      digest_frequency: scholar.digest_frequency || "weekly",
    })
    await refreshScholars()
  }

  async function toggleAlertScholar(scholar: Scholar) {
    const nextEnabled = !scholar.alert_enabled
    await patchScholarState(scholar, {
      alert_enabled: nextEnabled,
      alert_keywords:
        nextEnabled && (!scholar.alert_keywords || scholar.alert_keywords.length === 0)
          ? scholar.keywords || []
          : scholar.alert_keywords || [],
    })
    await refreshScholars()
  }

  async function muteSelected() {
    if (selectedIds.size === 0) return
    const targets = items.filter((item) => selectedIds.has(item.id) && !item.muted)
    if (!targets.length) {
      setSelectedIds(new Set())
      return
    }
    try {
      await Promise.all(targets.map((item) => patchScholarState(item, { muted: true })))
      setSelectedIds(new Set())
      await refreshScholars()
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error))
    }
  }


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

  async function handleCreateTrackFromScholar(scholar: Scholar) {
    setRowBusy((prev) => ({ ...prev, [scholar.id]: true }))
    try {
      const existingMatch = insights.find((row) => row.scholar.id === scholar.id)?.matchedTracks[0]
      const queryValue = scholar.keywords?.[0] || scholar.name
      if (existingMatch?.id) {
        await markSeen(scholar)
        router.push(
          `/research?track_id=${encodeURIComponent(String(existingMatch.id))}&query=${encodeURIComponent(queryValue)}`,
        )
        return
      }

      const trackName = buildUniqueTrackName(`${scholar.name} Watch`)
      const fallbackKeywords = [scholar.name]
      const keywords = (scholar.keywords || []).length > 0 ? scholar.keywords || [] : fallbackKeywords

      const res = await fetch("/api/research/tracks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trackName,
          description: `Auto-generated from scholar ${scholar.name}`,
          keywords,
          activate: true,
        }),
      })

      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `HTTP ${res.status}`)
      }

      const payload = (await res.json()) as { track?: { id?: number } }
      const trackId = payload.track?.id

      await markSeen(scholar)
      if (trackId) {
        router.push(
          `/research?track_id=${encodeURIComponent(String(trackId))}&query=${encodeURIComponent(queryValue)}`,
        )
      } else {
        router.push(toResearchLink(scholar))
      }
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error))
    } finally {
      setRowBusy((prev) => ({ ...prev, [scholar.id]: false }))
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
              <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Researcher Intelligence Layer</h1>
              <p className="max-w-2xl text-sm text-muted-foreground">
                Decide who to follow, detect direction shifts, and trigger immediate research actions in one console.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => setCreateOpen(true)}>
                <Plus className="mr-1 h-4 w-4" />
                Add to Watchlist
              </Button>
              <Button variant="outline" onClick={() => muteSelected().catch(() => {})} disabled={selectedIds.size === 0}>
                <BellOff className="mr-1 h-4 w-4" />
                Mute
              </Button>
              <Button
                variant="outline"
                onClick={() => selectedForTrack && handleCreateTrackFromScholar(selectedForTrack)}
                disabled={!selectedForTrack}
              >
                <FlaskConical className="mr-1 h-4 w-4" />
                Create Track from Scholar
              </Button>
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Tracked Scholars</p>
              <p className="mt-1 text-2xl font-semibold">{items.length}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Active Signals</p>
              <p className="mt-1 text-2xl font-semibold">{activeCount}</p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">New Since Last Seen</p>
              <p className="mt-1 text-2xl font-semibold">
                {insights.reduce((acc, row) => acc + row.newSinceLastSeen, 0)}
              </p>
            </div>
            <div className="rounded-xl border bg-background/70 p-4">
              <p className="text-xs text-muted-foreground">Muted</p>
              <p className="mt-1 text-2xl font-semibold">{mutedCount}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Watchlist Console</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-5">
            <div className="relative xl:col-span-2">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="pl-9"
                placeholder="Search scholar, domain, or keyword"
              />
            </div>
            <Input
              value={institutionQuery}
              onChange={(event) => setInstitutionQuery(event.target.value)}
              placeholder="Filter by institution"
            />
            <Button
              variant={status === "active" ? "default" : "outline"}
              size="sm"
              onClick={() => setStatus(status === "active" ? "all" : "active")}
            >
              Activity: {status === "active" ? "Active" : "All"}
            </Button>
            <Button
              variant={showMuted ? "default" : "outline"}
              size="sm"
              onClick={() => setShowMuted((prev) => !prev)}
            >
              {showMuted ? "Showing muted" : "Hide muted"}
            </Button>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              variant={trackMatchFilter === "all" ? "default" : "outline"}
              size="sm"
              onClick={() => setTrackMatchFilter("all")}
            >
              Track Match: All
            </Button>
            <Button
              variant={trackMatchFilter === "matched" ? "default" : "outline"}
              size="sm"
              onClick={() => setTrackMatchFilter("matched")}
            >
              Matched only
            </Button>
            <Button
              variant={trackMatchFilter === "unmatched" ? "default" : "outline"}
              size="sm"
              onClick={() => setTrackMatchFilter("unmatched")}
            >
              Unmatched only
            </Button>
          </div>

          {!filtered.length ? (
            <div className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">
              No scholars matched current filters.
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {filtered.map((row) => {
                const scholar = row.scholar
                const isSelected = selectedIds.has(scholar.id)
                const isMuted = !!scholar.muted
                const digestEnabled = !!scholar.digest_enabled
                const alertEnabled = !!scholar.alert_enabled

                return (
                  <Card key={scholar.id} className={isSelected ? "border-primary" : "border-border/60"}>
                    <CardContent className="space-y-3 p-4">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <p className="line-clamp-1 text-sm font-semibold">{scholar.name}</p>
                          <p className="line-clamp-1 text-xs text-muted-foreground">{scholar.affiliation}</p>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <Button
                            size="sm"
                            variant={isSelected ? "default" : "ghost"}
                            className="h-7 px-2"
                            onClick={() => toggleSelected(scholar.id)}
                          >
                            {isSelected ? <Check className="h-3.5 w-3.5" /> : "Select"}
                          </Button>
                          <Badge variant={scholar.status === "active" ? "default" : "secondary"} className="capitalize">
                            {scholar.status}
                          </Badge>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-center">
                        <div className="rounded-md bg-muted/40 p-2">
                          <p className="text-[11px] text-muted-foreground">Momentum</p>
                          <p className="text-sm font-semibold capitalize">{row.momentumLabel}</p>
                        </div>
                        <div className="rounded-md bg-muted/40 p-2">
                          <p className="text-[11px] text-muted-foreground">Track Match</p>
                          <p className="text-sm font-semibold">{row.matchedTracks.length}</p>
                        </div>
                        <div className="rounded-md bg-muted/40 p-2">
                          <p className="text-[11px] text-muted-foreground">New Since Last Seen</p>
                          <p className="text-sm font-semibold">{row.newSinceLastSeen}</p>
                        </div>
                        <div className="rounded-md bg-muted/40 p-2">
                          <p className="text-[11px] text-muted-foreground">Risk/Noise</p>
                          <p className="text-sm font-semibold uppercase">{row.riskLabel}</p>
                        </div>
                      </div>

                      {row.matchedTracks.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {row.matchedTracks.slice(0, 2).map((track) => (
                            <Badge key={`${scholar.id}-${track.id}`} variant="outline" className="text-[11px]">
                              {track.name}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <CircleAlert className="h-3.5 w-3.5" />
                          No matched track yet.
                        </div>
                      )}

                      <div className="flex flex-wrap gap-1.5">
                        <Badge variant={digestEnabled ? "default" : "outline"} className="text-[11px] capitalize">
                          Digest {digestEnabled ? scholar.digest_frequency || "weekly" : "off"}
                        </Badge>
                        <Badge variant={alertEnabled ? "default" : "outline"} className="text-[11px]">
                          Alert {alertEnabled ? `${(scholar.alert_keywords || []).length} keywords` : "off"}
                        </Badge>
                      </div>

                      <p className="text-xs text-muted-foreground">{scholar.recent_activity}</p>

                      <div className="flex flex-wrap gap-1.5 pt-1">
                        <Button
                          size="sm"
                          className="h-8"
                          onClick={() => {
                            markSeen(scholar).catch(() => {})
                            router.push(`/scholars/${encodeURIComponent(scholar.id)}`)
                          }}
                        >
                          <Users className="mr-1 h-3.5 w-3.5" />
                          Signals
                        </Button>
                        <Button asChild size="sm" variant="outline" className="h-8">
                          <Link href={toResearchLink(scholar)}>
                            <FlaskConical className="mr-1 h-3.5 w-3.5" />
                            Research
                          </Link>
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-8"
                          disabled={!!rowBusy[scholar.id]}
                          onClick={() => handleCreateTrackFromScholar(scholar)}
                        >
                          <Workflow className="mr-1 h-3.5 w-3.5" />
                          Create Track
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8"
                          onClick={() => markSeen(scholar).catch(() => {})}
                        >
                          Mark Seen
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8"
                          onClick={() => toggleMuteScholar(scholar).catch(() => {})}
                        >
                          <BellOff className="mr-1 h-3.5 w-3.5" />
                          {isMuted ? "Unmute" : "Mute"}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8"
                          onClick={() => toggleDigestScholar(scholar).catch(() => {})}
                        >
                          <BellRing className="mr-1 h-3.5 w-3.5" />
                          {digestEnabled ? "Digest Off" : "Digest On"}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8"
                          onClick={() => toggleAlertScholar(scholar).catch(() => {})}
                        >
                          <CircleAlert className="mr-1 h-3.5 w-3.5" />
                          {alertEnabled ? "Alert Off" : "Alert On"}
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
                )
              })}
            </div>
          )}

          <div className="flex items-center justify-between rounded-xl border bg-muted/20 p-3">
            <p className="text-xs text-muted-foreground">
              Scholar watchlist drives Research feed prioritization and daily digest automation.
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
              <p className="text-xs text-muted-foreground">
                Route scholar signals into track-scoped search and ranking.
              </p>
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
              <p className="text-sm font-medium">Digest Linkage</p>
              <p className="text-xs text-muted-foreground">
                Manage periodic digest delivery and signal refresh from settings.
              </p>
            </div>
            <Button asChild size="sm" variant="outline">
              <Link href="/settings">
                <BellRing className="mr-1 h-3.5 w-3.5" />
                Manage Delivery
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
              Add a scholar to watchlist. This updates scholar subscriptions used by tracking and digest delivery.
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
