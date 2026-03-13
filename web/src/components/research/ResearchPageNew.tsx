"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"

import {
  currentFeedbackFromRequestAction,
  normalizePaperFeedbackAction,
  type PaperFeedbackAction,
  type PaperFeedbackRequestAction,
} from "@/lib/paper-feedback"
import { cn } from "@/lib/utils"
import { fetchJson, getErrorMessage } from "@/lib/fetch"
import { ArrowRight, BookOpen } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"

import { SearchBox } from "./SearchBox"
import { TrackPills } from "./TrackPills"
import { SearchResults } from "./SearchResults"
import { MemoryTab } from "./MemoryTab"
import { SavedTab } from "./SavedTab"
import { CreateTrackModal } from "./CreateTrackModal"
import { EditTrackModal } from "./EditTrackModal"
import { ManageTracksModal } from "./ManageTracksModal"
import type { Track } from "./TrackSelector"
import type { Paper } from "./PaperCard"
import type { ResearchTrackContextResponse } from "@/lib/types"

type ContextPack = {
  context_run_id?: number | null
  routing: {
    track_id: number | null
    stage?: string
    exploration_ratio?: number
    diversity_strength?: number
  }
  paper_recommendations?: Paper[]
  paper_recommendation_reasons?: Record<string, string[]>
}

const RESULT_LIMIT_OPTIONS = [10, 25, 50] as const

function getGreeting(): string {
  const hour = new Date().getHours()
  if (hour < 12) return "Good morning"
  if (hour < 18) return "Good afternoon"
  return "Good evening"
}

export default function ResearchPageNew() {
  const searchParams = useSearchParams()

  // User state

  // Track state
  const [tracks, setTracks] = useState<Track[]>([])
  const [activeTrackId, setActiveTrackId] = useState<number | null>(null)
  const [trackContext, setTrackContext] = useState<ResearchTrackContextResponse | null>(null)
  const [trackContextLoading, setTrackContextLoading] = useState(false)

  // All available sources
  const ALL_SOURCES = ["semantic_scholar", "arxiv", "openalex", "papers_cool", "hf_daily"]

  // Search state
  const [query, setQuery] = useState("")
  const [hasSearched, setHasSearched] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [contextPack, setContextPack] = useState<ContextPack | null>(null)
  const [searchSources, setSearchSources] = useState<string[]>(ALL_SOURCES)
  const [paperLimit, setPaperLimit] = useState<(typeof RESULT_LIMIT_OPTIONS)[number]>(25)
  const [yearFrom, setYearFrom] = useState("")
  const [yearTo, setYearTo] = useState("")

  // Memory drawer state
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [workspaceTab, setWorkspaceTab] = useState<"saved" | "memory">("saved")

  // Anchor mode state (used for search filtering)
  const [anchorPersonalized, setAnchorPersonalized] = useState(true)

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  const [trackToEdit, setTrackToEdit] = useState<Track | null>(null)
  const [manageModalOpen, setManageModalOpen] = useState(false)
  const [confirmClearOpen, setConfirmClearOpen] = useState(false)
  const [trackToClear, setTrackToClear] = useState<number | null>(null)

  // Derived state
  const activeTrack = useMemo(
    () => {
      if (trackContext?.track && trackContext.track.id === activeTrackId) {
        return trackContext.track as Track
      }
      return tracks.find((t) => t.id === activeTrackId) || null
    },
    [trackContext, tracks, activeTrackId]
  )

  const papers = contextPack?.paper_recommendations || []
  const reasons = contextPack?.paper_recommendation_reasons || {}
  const routeTrackId = Number(searchParams.get("track_id") || 0)
  const routeQuery = searchParams.get("query")?.trim() || ""

  // Load tracks on mount
  useEffect(() => {
    refreshTracks().catch((e) => setError(getErrorMessage(e)))
  }, [])

  useEffect(() => {
    if (!routeTrackId || !Number.isFinite(routeTrackId)) return
    if (!tracks.some((track) => track.id === routeTrackId)) return
    if (activeTrackId === routeTrackId) return
    activateTrack(routeTrackId).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeTrackId, tracks, activeTrackId])

  useEffect(() => {
    if (!routeQuery || hasSearched) return
    if (query === routeQuery) return
    setQuery(routeQuery)
  }, [routeQuery, query, hasSearched])

  async function refreshTrackContext(trackId: number): Promise<void> {
    const data = await fetchJson<ResearchTrackContextResponse>(
      `/api/research/tracks/${trackId}/context`
    )
    setTrackContext(data)
  }

  useEffect(() => {
    let cancelled = false

    async function loadTrackContext(trackId: number) {
      setTrackContextLoading(true)
      try {
        const data = await fetchJson<ResearchTrackContextResponse>(
          `/api/research/tracks/${trackId}/context`
        )
        if (!cancelled) {
          setTrackContext(data)
        }
      } catch (e) {
        if (!cancelled) {
          setTrackContext(null)
          setError(getErrorMessage(e))
        }
      } finally {
        if (!cancelled) {
          setTrackContextLoading(false)
        }
      }
    }

    if (!activeTrackId) {
      setTrackContext(null)
      setTrackContextLoading(false)
      return () => {
        cancelled = true
      }
    }

    loadTrackContext(activeTrackId).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [activeTrackId])

  async function refreshTracks(): Promise<number | null> {
    const data = await fetchJson<{ tracks: Track[] }>(
      `/api/research/tracks`
    )
    setTracks(data.tracks || [])
    const active = data.tracks.find((t) => t.is_active)
    const activeId = active?.id ?? null
    setActiveTrackId(activeId)
    return activeId
  }

  async function activateTrack(trackId: number) {
    setLoading(true)
    try {
      await fetchJson(
        `/api/research/tracks/${trackId}/activate`,
        {
          method: "POST",
          body: "{}",
          headers: { "Content-Type": "application/json" },
        }
      )
      await refreshTracks()
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleSearch() {
    if (!query.trim()) return

    setIsSearching(true)
    setHasSearched(true)
    setError(null)

    try {
      const parseYear = (value: string): number | undefined => {
        const trimmed = value.trim()
        if (!trimmed) return undefined
        const n = Number(trimmed)
        if (!Number.isInteger(n)) return undefined
        if (n < 1900 || n > 2100) return undefined
        return n
      }
      const parsedYearFrom = parseYear(yearFrom)
      const parsedYearTo = parseYear(yearTo)

      const body = {
        query,
        track_id: activeTrackId ?? undefined,
        paper_limit: paperLimit,
        memory_limit: 8,
        sources: searchSources,
        offline: false,
        include_cross_track: false,
        stage: "auto",
        personalized: anchorPersonalized,
        year_from: parsedYearFrom,
        year_to: parsedYearTo,
      }

      const data = await fetchJson<{ context_pack: ContextPack }>(
        `/api/research/context`,
        {
          method: "POST",
          body: JSON.stringify(body),
          headers: { "Content-Type": "application/json" },
        }
      )

      setContextPack(data.context_pack)
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setIsSearching(false)
    }
  }

  function toggleSearchSource(source: string) {
    setSearchSources((prev) => {
      const exists = prev.includes(source)
      if (exists) {
        const next = prev.filter((x) => x !== source)
        // Keep at least one source selected
        return next.length ? next : prev
      }
      return [...prev, source]
    })
  }

  // Auto-refresh search when sources or personalization mode change (after initial search)
  useEffect(() => {
    if (!hasSearched || !query.trim() || isSearching) return

    // Debounce to avoid rapid re-fetching
    const timer = setTimeout(() => {
      handleSearch()
    }, 300)

    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchSources, anchorPersonalized, paperLimit, yearFrom, yearTo])

  // If the page is opened with a query parameter, run it once automatically.
  useEffect(() => {
    if (!routeQuery || hasSearched || isSearching || loading) return
    if (query.trim() !== routeQuery) return
    if (routeTrackId && Number.isFinite(routeTrackId) && activeTrackId !== routeTrackId) return
    handleSearch().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeQuery, query, hasSearched, isSearching, loading, routeTrackId, activeTrackId])

  async function handleCreateTrack(data: {
    name: string
    description: string
    keywords: string[]
  }): Promise<boolean> {
    const name = data.name.trim()
    const duplicate = tracks.some((track) => track.name.trim() === name)
    if (duplicate) {
      setCreateError(`Track "${name}" already exists.`)
      return false
    }
    setLoading(true)
    setError(null)
    setCreateError(null)
    try {
      await fetchJson(`/api/research/tracks`, {
        method: "POST",
        body: JSON.stringify({
          name,
          description: data.description,
          keywords: data.keywords,
          activate: true,
        }),
        headers: { "Content-Type": "application/json" },
      })
      setCreateModalOpen(false)
      setCreateError(null)
      await refreshTracks()
      return true
    } catch (e) {
      const message = getErrorMessage(e)
      if (message.startsWith("409")) {
        setCreateError(`Track "${name}" already exists.`)
      } else {
        setCreateError(message)
      }
      return false
    } finally {
      setLoading(false)
    }
  }

  function handleEditTrack(track: Track) {
    setTrackToEdit(track)
    setEditError(null)
    setEditModalOpen(true)
  }

  async function handleUpdateTrack(
    trackId: number,
    data: { name: string; description: string; keywords: string[] }
  ): Promise<boolean> {
    const name = data.name.trim()
    const duplicate = tracks.some(
      (track) => track.id !== trackId && track.name.trim() === name
    )
    if (duplicate) {
      setEditError(`Track "${name}" already exists.`)
      return false
    }
    setLoading(true)
    setError(null)
    setEditError(null)
    try {
      await fetchJson(`/api/research/tracks/${trackId}`, {
        method: "PATCH",
        body: JSON.stringify({
          name,
          description: data.description,
          keywords: data.keywords,
        }),
        headers: { "Content-Type": "application/json" },
      })
      await refreshTracks()
      if (trackId === activeTrackId) {
        await refreshTrackContext(trackId)
      }
      return true
    } catch (e) {
      const message = getErrorMessage(e)
      if (message.startsWith("409")) {
        setEditError(`Track "${name}" already exists.`)
      } else {
        setEditError(message)
      }
      return false
    } finally {
      setLoading(false)
    }
  }

  async function handleClearTrackMemory(trackId: number) {
    setTrackToClear(trackId)
    setConfirmClearOpen(true)
  }

  async function confirmClearMemory() {
    if (!trackToClear) return

    setLoading(true)
    setError(null)
    try {
      await fetchJson(
        `/api/research/tracks/${trackToClear}/memory/clear?confirm=true`,
        {
          method: "POST",
          body: "{}",
          headers: { "Content-Type": "application/json" },
        }
      )
      if (trackToClear === activeTrackId) {
        await refreshTrackContext(trackToClear)
      }
      setConfirmClearOpen(false)
      setTrackToClear(null)
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleFeedback(
    paperId: string,
    action: PaperFeedbackRequestAction,
    rank?: number,
    paper?: Paper
  ): Promise<PaperFeedbackAction | null | undefined> {
    setError(null)
    try {
      const body: Record<string, unknown> = {
        track_id: activeTrackId,
        paper_id: paperId,
        action,
        weight: 0.0,
        context_run_id: contextPack?.context_run_id ?? null,
        context_rank: typeof rank === "number" ? rank : undefined,
        metadata: {
          retrieval_sources: Array.isArray(paper?.retrieval_sources)
            ? paper?.retrieval_sources
            : [],
          retrieval_score:
            typeof paper?.retrieval_score === "number" ? paper.retrieval_score : undefined,
          anchor_mode: anchorPersonalized ? "personalized" : "global",
        },
      }

      if (action === "save" && paper) {
        body.paper_title = paper.title
        body.paper_abstract = paper.abstract || ""
        body.paper_authors = paper.authors || []
        body.paper_year = paper.year
        body.paper_venue = paper.venue
        body.paper_citation_count = paper.citation_count
        body.paper_url = paper.url
        body.paper_source = paper.source || "semantic_scholar"
      }

      const payload = await fetchJson<{ current_action?: string | null }>(
        `/api/research/papers/feedback`,
        {
          method: "POST",
          body: JSON.stringify(body),
          headers: { "Content-Type": "application/json" },
        }
      )

      return (
        normalizePaperFeedbackAction(payload.current_action) ??
        currentFeedbackFromRequestAction(action)
      )
    } catch (e) {
      setError(getErrorMessage(e))
      return undefined
    }
  }

  const trackToClearName = tracks.find((t) => t.id === trackToClear)?.name || "this track"
  const discoveryHref = useMemo(() => {
    const params = new URLSearchParams()
    if (query.trim()) params.set("query", query.trim())
    if (activeTrackId) params.set("track_id", String(activeTrackId))
    const qs = params.toString()
    return qs ? `/research/discovery?${qs}` : "/research/discovery"
  }, [query, activeTrackId])

  const communityRadarHref = useMemo(() => {
    const params = new URLSearchParams()
    if (activeTrackId) params.set("radar_track", String(activeTrackId))

    const keywordSeed = query.trim() || activeTrack?.keywords?.[0] || ""
    if (keywordSeed) {
      params.set("radar_keyword", keywordSeed)
    }

    const qs = params.toString()
    return qs ? `/dashboard?${qs}` : "/dashboard"
  }, [query, activeTrack, activeTrackId])

  const workflowHref = useMemo(() => {
    const params = new URLSearchParams()
    if (query.trim()) {
      params.set("query", query.trim())
    }
    const qs = params.toString()
    return qs ? `/workflows?${qs}` : "/workflows"
  }, [query])

  return (
    <div
      className={cn(
        "min-h-[calc(100vh-4rem)] bg-gradient-to-b from-background via-background to-muted/20 transition-all duration-500 ease-out",
        !hasSearched && "flex flex-col items-center justify-center",
        hasSearched && "py-6 sm:py-8"
      )}
    >
      {/* Confirm Clear Memory Dialog */}
      <Dialog open={confirmClearOpen} onOpenChange={setConfirmClearOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear Track Memory?</DialogTitle>
            <DialogDescription>
              This will delete all memories for &quot;{trackToClearName}&quot;. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setConfirmClearOpen(false)
                setTrackToClear(null)
              }}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmClearMemory}
              disabled={loading}
            >
              Clear Memory
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Track Modal */}
      <CreateTrackModal
        open={createModalOpen}
        onOpenChange={(open) => {
          setCreateModalOpen(open)
          if (!open) setCreateError(null)
        }}
        onCreateTrack={handleCreateTrack}
        isLoading={loading}
        error={createError}
        onClearError={() => setCreateError(null)}
      />

      {/* Edit Track Modal */}
      <EditTrackModal
        open={editModalOpen}
        onOpenChange={(open) => {
          setEditModalOpen(open)
          if (!open) {
            setTrackToEdit(null)
            setEditError(null)
          }
        }}
        track={trackToEdit}
        onUpdateTrack={handleUpdateTrack}
        isLoading={loading}
        error={editError}
        onClearError={() => setEditError(null)}
      />

      {/* Manage Tracks Modal */}
      <ManageTracksModal
        open={manageModalOpen}
        onOpenChange={setManageModalOpen}
        tracks={tracks}
        activeTrackId={activeTrackId}
        onEditTrack={handleEditTrack}
        onClearTrackMemory={handleClearTrackMemory}
        isLoading={loading}
      />

      {/* Main Content */}
      <div
        className={cn(
          "w-full px-4 sm:px-6 lg:px-8 transition-all duration-500 ease-out mx-auto",
          hasSearched ? "max-w-[1400px]" : "max-w-4xl"
        )}
      >
        {/* Greeting - only show before search */}
        {!hasSearched && (
          <div className="text-center mb-8 sm:mb-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight mb-2 sm:mb-3">
              <BookOpen className="inline-block mr-2 sm:mr-3 h-8 w-8 sm:h-10 sm:w-10 align-middle" />
              {getGreeting()}
            </h1>
            <p className="text-lg sm:text-xl text-muted-foreground">
              What papers are you looking for?
            </p>
          </div>
        )}

        {/* Search Box */}
        <div
          className={cn(
            "transition-all duration-500 ease-out",
            hasSearched ? "mb-6" : "mb-10"
          )}
        >
          <SearchBox
            query={query}
            onQueryChange={setQuery}
            onSearch={handleSearch}
            tracks={tracks}
            activeTrack={activeTrack}
            onSelectTrack={activateTrack}
            onNewTrack={() => setCreateModalOpen(true)}
            onManageTracks={() => setManageModalOpen(true)}
            isSearching={isSearching}
            disabled={loading}
            anchorMode={anchorPersonalized ? "personalized" : "global"}
            onAnchorModeChange={(mode) => setAnchorPersonalized(mode === "personalized")}
            onOpenLibrary={() => {
              setWorkspaceTab("saved")
              setMemoryOpen(true)
            }}
            onOpenMemory={() => {
              setWorkspaceTab("memory")
              setMemoryOpen(true)
            }}
            yearFrom={yearFrom}
            yearTo={yearTo}
            onYearFromChange={setYearFrom}
            onYearToChange={setYearTo}
          />
        </div>

        {/* Track context panel removed per request */}

        {/* Track Pills - only show before search */}
        {!hasSearched && tracks.length > 0 && (
          <div className="mb-8 flex justify-center animate-in fade-in slide-in-from-bottom-2 duration-500 delay-150">
            <TrackPills
              tracks={tracks}
              activeTrackId={activeTrackId}
              onSelectTrack={activateTrack}
              onNewTrack={() => setCreateModalOpen(true)}
              disabled={loading}
            />
          </div>
        )}

        {/* Error Display */}
        {error && (
          <Card className="border-destructive/40 mb-6 max-w-3xl mx-auto">
            <CardHeader className="pb-2">
              <CardTitle className="text-destructive text-base">Error</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="whitespace-pre-wrap text-sm">{error}</pre>
            </CardContent>
          </Card>
        )}

        {/* Search Results - only shown after search */}
        {hasSearched && (
          <div className="space-y-4">
            <Card className="border-border/70 bg-card/80 backdrop-blur-sm">
              <CardContent className="space-y-3 p-3 sm:p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline">
                      Track: {activeTrack?.name || "Global"}
                    </Badge>
                    {trackContextLoading ? (
                      <Badge variant="outline">Track snapshot: loading</Badge>
                    ) : trackContext ? (
                      <Badge variant="outline">
                        Pending memory: {trackContext.memory.pending_items}
                      </Badge>
                    ) : null}
                    <Badge variant="outline">
                      Mode: {anchorPersonalized ? "Personalized" : "Global"}
                    </Badge>
                    <Badge variant="outline">Sources: {searchSources.length}</Badge>
                    <Badge variant="secondary">Results: {papers.length}</Badge>
                    <div className="flex flex-wrap items-center gap-1">
                      <span className="text-xs text-muted-foreground">Cap</span>
                      {RESULT_LIMIT_OPTIONS.map((limit) => (
                        <Button
                          key={limit}
                          type="button"
                          size="sm"
                          variant={paperLimit === limit ? "default" : "outline"}
                          className="h-7 px-2 text-xs"
                          onClick={() => setPaperLimit(limit)}
                          disabled={isSearching}
                        >
                          {limit}
                        </Button>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button asChild size="sm" variant="outline" className="gap-1.5">
                      <Link href={workflowHref}>
                        Open Workflows
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    </Button>
                    <Button asChild size="sm" variant="outline" className="gap-1.5">
                      <Link href={communityRadarHref}>
                        Open Community Radar
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    </Button>
                    <Button asChild size="sm" className="gap-1.5">
                      <Link href={discoveryHref}>
                        Open Discovery Workspace
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                </div>
                <div className="rounded-xl border border-border/60 bg-background/80 px-3 py-2.5 text-sm">
                  <p className="font-medium text-foreground">
                    Research is for fast context search. Workflow is for batch topics, DailyPaper, and Judge.
                  </p>
                  <p className="mt-1 text-muted-foreground">
                    这里更适合单问题探索和即时反馈；需要把多个主题整理成 digest、评分和交付链路时，直接切到 Workflows。
                  </p>
                </div>
              </CardContent>
            </Card>

            <SearchResults
              papers={papers}
              reasons={reasons}
              isSearching={isSearching}
              hasSearched={hasSearched}
              selectedSources={searchSources}
              onToggleSource={toggleSearchSource}
              onFeedbackAction={(paperId, action, rank, paper) =>
                handleFeedback(paperId, action, rank, paper)
              }
            />
          </div>
        )}

        {/* Memory Sheet Drawer */}
        <Sheet open={memoryOpen} onOpenChange={setMemoryOpen}>
          <SheetContent className="w-full border-l border-border/70 p-0 sm:max-w-2xl">
            <div className="flex h-full flex-col bg-gradient-to-b from-slate-50 via-white to-white">
              <SheetHeader className="border-b border-border/70 bg-white/90 px-5 py-4 backdrop-blur">
                <div className="flex flex-wrap items-center gap-2">
                  <SheetTitle>Workspace Library</SheetTitle>
                  <Badge variant="secondary">
                    {activeTrack?.name ? `Track: ${activeTrack.name}` : "Global scope"}
                  </Badge>
                </div>
                <SheetDescription>
                  Review saved papers, trigger export handoffs, and inspect track memory without leaving the
                  research workspace.
                </SheetDescription>
                <div className="flex flex-wrap gap-2 pt-2">
                  <Badge variant="outline">{activeTrack?.name || "Global scope"}</Badge>
                  <Badge variant="outline">
                    Mode: {anchorPersonalized ? "Personalized" : "Global"}
                  </Badge>
                  <Badge variant="outline">Query: {query.trim() || "No active query"}</Badge>
                </div>
              </SheetHeader>
              <div className="flex min-h-0 flex-1 flex-col px-5 pb-5 pt-4">
                <Tabs
                  value={workspaceTab}
                  onValueChange={(value) => {
                    if (value === "saved" || value === "memory") {
                      setWorkspaceTab(value)
                    }
                  }}
                  className="flex min-h-0 flex-1 flex-col"
                >
                  <TabsList className="grid h-auto w-full grid-cols-2 rounded-2xl bg-slate-100 p-1">
                    <TabsTrigger value="saved" className="rounded-xl">
                      Saved
                    </TabsTrigger>
                    <TabsTrigger value="memory" className="rounded-xl">
                      Memory
                    </TabsTrigger>
                  </TabsList>
                  <TabsContent value="saved" className="mt-4 min-h-0 flex-1 overflow-y-auto pr-1">
                    <SavedTab
                      trackId={activeTrackId}
                      trackName={activeTrack?.name ?? null}
                    />
                  </TabsContent>
                  <TabsContent value="memory" className="mt-4 min-h-0 flex-1 overflow-y-auto pr-1">
                    <MemoryTab trackId={activeTrackId} />
                  </TabsContent>
                </Tabs>
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </div>
  )
}
