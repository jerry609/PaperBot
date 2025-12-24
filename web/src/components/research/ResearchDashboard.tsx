"use client"

import { useEffect, useMemo, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

type Track = {
  id: number
  name: string
  description?: string
  keywords?: string[]
  venues?: string[]
  methods?: string[]
  is_active?: boolean
}

type MemoryItem = {
  id: number
  kind: string
  content: string
  status?: string
  scope_type?: string
  scope_id?: string | null
  score?: number
  tags?: string[]
}

type Paper = {
  paper_id: string
  title: string
  year?: number
  venue?: string
  citation_count?: number
  authors?: string[]
  url?: string
}

type ContextPack = {
  context_run_id?: number | null
  routing: {
    track_id: number | null
    stage?: string
    exploration_ratio?: number
    diversity_strength?: number
    suggestion?: {
      track_id: number
      track_name?: string
      score: number
      margin: number
      features?: Record<string, number>
      top_memory_hits?: MemoryItem[]
    } | null
  }
  paper_recommendations?: Paper[]
  paper_recommendation_reasons?: Record<string, string[]>
  progress_state?: { tasks?: { title: string; status: string; priority: number }[] }
}

type EvalSummary = {
  window_days: number
  runs: number
  impressions: number
  unique_recommended_papers: number
  repeat_rate: number
  feedback_on_recommended: Record<string, number>
  feedback_coverage: number
  linked_feedback_rows: number
}

type ConfirmAction =
  | { type: "bulk_moderate"; status: "approved" | "rejected"; itemIds: number[] }
  | { type: "bulk_move"; itemIds: number[]; targetTrackId: number }
  | { type: "clear_track_memory"; trackId: number }

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(`${res.status} ${res.statusText} ${text}`.trim())
  }
  return res.json() as Promise<T>
}

function clampNumber(value: number, min: number, max: number, fallback: number) {
  if (!Number.isFinite(value)) return fallback
  return Math.min(max, Math.max(min, value))
}

export default function ResearchDashboard() {
  const [userId, setUserId] = useState("default")
  const [tracks, setTracks] = useState<Track[]>([])
  const [activeTrackId, setActiveTrackId] = useState<number | null>(null)
  const [query, setQuery] = useState("")
  const [stage, setStage] = useState<"auto" | "survey" | "writing" | "rebuttal">("auto")
  const [explorationRatio, setExplorationRatio] = useState<number | "">("")
  const [diversityStrength, setDiversityStrength] = useState<number | "">("")
  const [contextPack, setContextPack] = useState<ContextPack | null>(null)
  const [inbox, setInbox] = useState<MemoryItem[]>([])
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null)
  const [evalDays, setEvalDays] = useState(30)
  const [suggestText, setSuggestText] = useState("")
  const [selectedInboxIds, setSelectedInboxIds] = useState<Set<number>>(new Set())
  const [moveTargetTrackId, setMoveTargetTrackId] = useState<number | "">("")
  const [createOpen, setCreateOpen] = useState(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [confirmAction, setConfirmAction] = useState<ConfirmAction | null>(null)
  const [newTrackName, setNewTrackName] = useState("")
  const [newTrackDescription, setNewTrackDescription] = useState("")
  const [newTrackKeywords, setNewTrackKeywords] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selectedCount = selectedInboxIds.size
  const allSelected = inbox.length > 0 && selectedCount === inbox.length

  const activeTrack = useMemo(() => tracks.find((t) => t.id === activeTrackId) || null, [tracks, activeTrackId])

  const confirmUi = useMemo(() => {
    if (!confirmAction) return null

    if (confirmAction.type === "bulk_moderate") {
      const verb = confirmAction.status === "approved" ? "Approve" : "Reject"
      return {
        title: `${verb} ${confirmAction.itemIds.length} memory item(s)?`,
        description:
          confirmAction.status === "approved"
            ? "Approved items become eligible for context injection and routing signals."
            : "Rejected items will remain stored but not used for context injection.",
        confirmLabel: verb,
        destructive: confirmAction.status === "rejected",
      }
    }

    if (confirmAction.type === "bulk_move") {
      const targetName = tracks.find((t) => t.id === confirmAction.targetTrackId)?.name || "target track"
      return {
        title: `Move ${confirmAction.itemIds.length} item(s) to “${targetName}”?`,
        description: "This changes the memory scope to the selected track.",
        confirmLabel: "Move",
        destructive: false,
      }
    }

    const trackName = tracks.find((t) => t.id === confirmAction.trackId)?.name || "active track"
    return {
      title: `Clear all memories for “${trackName}”?`,
      description:
        "This performs a soft delete of all memory items scoped to the active track. You can re-ingest or re-create memories later.",
      confirmLabel: "Clear",
      destructive: true,
    }
  }, [confirmAction, tracks])

  function openConfirm(action: ConfirmAction) {
    setConfirmAction(action)
    setConfirmOpen(true)
  }

  function toggleSelected(id: number, nextChecked: boolean) {
    setSelectedInboxIds((prev) => {
      const next = new Set(prev)
      if (nextChecked) next.add(id)
      else next.delete(id)
      return next
    })
  }

  function toggleSelectAll() {
    if (allSelected) {
      setSelectedInboxIds(new Set())
      return
    }
    setSelectedInboxIds(new Set(inbox.map((m) => m.id)))
  }

  async function refreshTracks(): Promise<number | null> {
    const data = await fetchJson<{ tracks: Track[] }>(`/api/research/tracks?user_id=${encodeURIComponent(userId)}`)
    setTracks(data.tracks || [])
    const active = data.tracks.find((t) => t.is_active)
    const activeId = active?.id ?? null
    setActiveTrackId(activeId)
    setMoveTargetTrackId("")
    return activeId
  }

  async function refreshInbox(trackId?: number | null) {
    const tid = trackId ?? activeTrackId
    const qs = new URLSearchParams({ user_id: userId })
    if (tid) qs.set("track_id", String(tid))
    const data = await fetchJson<{ items: MemoryItem[] }>(`/api/research/memory/inbox?${qs.toString()}`)
    setInbox(data.items || [])
    setSelectedInboxIds(new Set())
  }

  async function refreshEval() {
    const qs = new URLSearchParams({ user_id: userId, days: String(evalDays) })
    if (activeTrackId) qs.set("track_id", String(activeTrackId))
    const data = await fetchJson<{ summary: EvalSummary }>(`/api/research/evals/summary?${qs.toString()}`)
    setEvalSummary(data.summary)
  }

  useEffect(() => {
    setError(null)
    refreshTracks().catch((e) => setError(String(e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (activeTrackId != null) {
      refreshInbox(activeTrackId).catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTrackId, userId])

  async function activateTrack(trackId: number) {
    await fetchJson(`/api/research/tracks/${trackId}/activate?user_id=${encodeURIComponent(userId)}`, {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json" },
    })
    const activeId = await refreshTracks()
    await refreshInbox(activeId ?? trackId)
  }

  async function buildContext(activateSuggestion?: boolean) {
    setLoading(true)
    setError(null)
    try {
      const suggestion = contextPack?.routing?.suggestion
      const activateTrackId = activateSuggestion ? suggestion?.track_id : null
      const body: Record<string, unknown> = {
        user_id: userId,
        query,
        paper_limit: 8,
        memory_limit: 8,
        offline: false,
        include_cross_track: false,
        activate_track_id: activateTrackId,
        stage,
      }
      if (explorationRatio !== "") {
        body.exploration_ratio = clampNumber(explorationRatio, 0, 0.5, 0.15)
      }
      if (diversityStrength !== "") {
        body.diversity_strength = clampNumber(diversityStrength, 0, 2, 0.55)
      }
      const data = await fetchJson<{ context_pack: ContextPack }>(`/api/research/context`, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      })
      setContextPack(data.context_pack)
      if (activateTrackId) {
        await refreshTracks()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function suggestMemories() {
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/api/research/memory/suggest`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          text: suggestText,
          scope_type: "track",
          scope_id: activeTrackId ? String(activeTrackId) : undefined,
          use_llm: false,
          redact: true,
        }),
        headers: { "Content-Type": "application/json" },
      })
      setSuggestText("")
      await refreshInbox(activeTrackId)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function bulkModerate(status: "approved" | "rejected", itemIds?: number[]) {
    const ids = itemIds ?? Array.from(selectedInboxIds)
    if (ids.length === 0) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/api/research/memory/bulk_moderate`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          item_ids: ids,
          status,
        }),
        headers: { "Content-Type": "application/json" },
      })
      await refreshInbox(activeTrackId)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function bulkMoveToTrack(targetTrackId: number, itemIds?: number[]) {
    const ids = itemIds ?? Array.from(selectedInboxIds)
    if (ids.length === 0) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/api/research/memory/bulk_move`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          item_ids: ids,
          scope_type: "track",
          scope_id: String(targetTrackId),
        }),
        headers: { "Content-Type": "application/json" },
      })
      await refreshInbox(activeTrackId)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function createTrack() {
    const name = newTrackName.trim()
    if (!name) return
    setLoading(true)
    setError(null)
    try {
      const keywords = newTrackKeywords
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
      await fetchJson(`/api/research/tracks`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          name,
          description: newTrackDescription.trim(),
          keywords,
          activate: true,
        }),
        headers: { "Content-Type": "application/json" },
      })
      setCreateOpen(false)
      setNewTrackName("")
      setNewTrackDescription("")
      setNewTrackKeywords("")
      const activeId = await refreshTracks()
      await refreshInbox(activeId)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function sendFeedback(paperId: string, action: string, rank?: number) {
    setLoading(true)
    setError(null)
    try {
      const contextRunId = contextPack?.context_run_id ?? null
      await fetchJson(`/api/research/papers/feedback`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          track_id: activeTrackId,
          paper_id: paperId,
          action,
          weight: 0.0,
          context_run_id: contextRunId,
          context_rank: typeof rank === "number" ? rank : undefined,
          metadata: {},
        }),
        headers: { "Content-Type": "application/json" },
      })
      await buildContext(false)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function clearTrackMemory(trackId: number) {
    setLoading(true)
    setError(null)
    try {
      await fetchJson(
        `/api/research/tracks/${trackId}/memory/clear?user_id=${encodeURIComponent(userId)}&confirm=true`,
        { method: "POST", body: "{}", headers: { "Content-Type": "application/json" } },
      )
      await refreshInbox(trackId)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function runConfirmAction() {
    const action = confirmAction
    if (!action) return

    if (action.type === "bulk_moderate") {
      await bulkModerate(action.status, action.itemIds)
      return
    }

    if (action.type === "bulk_move") {
      await bulkMoveToTrack(action.targetTrackId, action.itemIds)
      return
    }

    await clearTrackMemory(action.trackId)
  }

  return (
    <div className="space-y-6">
      <Dialog
        open={confirmOpen}
        onOpenChange={(open) => {
          setConfirmOpen(open)
          if (!open) setConfirmAction(null)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{confirmUi?.title || "Confirm"}</DialogTitle>
            <DialogDescription>{confirmUi?.description}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="secondary"
              onClick={() => {
                setConfirmOpen(false)
                setConfirmAction(null)
              }}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button
              variant={confirmUi?.destructive ? "destructive" : "default"}
              onClick={async () => {
                setConfirmOpen(false)
                try {
                  await runConfirmAction()
                } finally {
                  setConfirmAction(null)
                }
              }}
              disabled={loading}
            >
              {confirmUi?.confirmLabel || "Confirm"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Research</h2>
          <p className="text-muted-foreground">Tracks, memory inbox, and personalized paper recommendations</p>
        </div>
        <div className="flex items-center gap-3">
          <Label className="text-sm text-muted-foreground">user_id</Label>
          <Input value={userId} onChange={(e) => setUserId(e.target.value)} className="w-[200px]" />
          <Button variant="secondary" onClick={() => refreshTracks()} disabled={loading}>
            Refresh
          </Button>
        </div>
      </div>

      {error ? (
        <Card className="border-destructive/40">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="whitespace-pre-wrap text-sm">{error}</pre>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Tracks</CardTitle>
              <Dialog open={createOpen} onOpenChange={setCreateOpen}>
                <DialogTrigger asChild>
                  <Button size="sm" variant="outline">
                    New
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Create Track</DialogTitle>
                    <DialogDescription>
                      Track is the isolation boundary for memories, progress, and recommendations.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-3">
                    <div className="space-y-1">
                      <Label>Name</Label>
                      <Input value={newTrackName} onChange={(e) => setNewTrackName(e.target.value)} />
                    </div>
                    <div className="space-y-1">
                      <Label>Description</Label>
                      <Textarea
                        value={newTrackDescription}
                        onChange={(e) => setNewTrackDescription(e.target.value)}
                        rows={3}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>Keywords (comma separated)</Label>
                      <Input value={newTrackKeywords} onChange={(e) => setNewTrackKeywords(e.target.value)} />
                    </div>
                  </div>
                  <DialogFooter>
                    <Button variant="secondary" onClick={() => setCreateOpen(false)} disabled={loading}>
                      Cancel
                    </Button>
                    <Button onClick={() => createTrack()} disabled={loading || !newTrackName.trim()}>
                      Create & Activate
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {tracks.length === 0 ? (
              <p className="text-sm text-muted-foreground">No tracks yet. Create one via API.</p>
            ) : null}
            <div className="space-y-2">
              {tracks.map((t) => (
                <div key={t.id} className="flex items-start justify-between gap-3 rounded-md border p-3">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">{t.name}</div>
                      {t.is_active ? <Badge>Active</Badge> : null}
                    </div>
                    {t.description ? (
                      <div className="text-xs text-muted-foreground line-clamp-2">{t.description}</div>
                    ) : null}
                    {(t.keywords?.length || 0) > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {t.keywords?.slice(0, 6).map((k) => (
                          <Badge key={k} variant="outline" className="text-[10px]">
                            {k}
                          </Badge>
                        ))}
                      </div>
                    ) : null}
                  </div>
                  <Button
                    size="sm"
                    variant={t.is_active ? "secondary" : "default"}
                    onClick={() => activateTrack(t.id)}
                    disabled={loading}
                  >
                    {t.is_active ? "Re-activate" : "Activate"}
                  </Button>
                </div>
              ))}
            </div>
            <Button
              variant="outline"
              onClick={() => activeTrackId && openConfirm({ type: "clear_track_memory", trackId: activeTrackId })}
              disabled={loading || !activeTrackId}
            >
              Clear Track Memory
            </Button>
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Context Builder</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <Label>Query</Label>
                <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. reranking for RAG" />
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="space-y-1">
                  <Label>Stage</Label>
                  <select
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={stage}
                    onChange={(e) => setStage(e.target.value as typeof stage)}
                    disabled={loading}
                  >
                    <option value="auto">Auto (infer from progress)</option>
                    <option value="survey">Survey</option>
                    <option value="writing">Writing</option>
                    <option value="rebuttal">Rebuttal</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <Label>Exploration ratio (optional, 0–0.5)</Label>
                  <Input
                    type="number"
                    value={explorationRatio}
                    min={0}
                    max={0.5}
                    step={0.05}
                    placeholder="(auto)"
                    onChange={(e) => setExplorationRatio(e.target.value === "" ? "" : Number(e.target.value))}
                  />
                </div>
                <div className="space-y-1">
                  <Label>Diversity strength (optional, 0–2)</Label>
                  <Input
                    type="number"
                    value={diversityStrength}
                    min={0}
                    max={2}
                    step={0.05}
                    placeholder="(auto)"
                    onChange={(e) => setDiversityStrength(e.target.value === "" ? "" : Number(e.target.value))}
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={() => buildContext(false)} disabled={loading || !query.trim()}>
                  Build Context
                </Button>
              </div>

              {contextPack?.routing?.suggestion ? (
                <div className="rounded-md border p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm">
                      Suggested track:{" "}
                      <span className="font-medium">{contextPack.routing.suggestion.track_name || "Track"}</span>{" "}
                      <span className="text-muted-foreground">
                        (score {contextPack.routing.suggestion.score.toFixed(2)}, margin{" "}
                        {contextPack.routing.suggestion.margin.toFixed(2)})
                      </span>
                    </div>
                    <Button size="sm" onClick={() => buildContext(true)} disabled={loading}>
                      Switch & Rebuild
                    </Button>
                  </div>
                </div>
              ) : null}

              {contextPack?.progress_state?.tasks?.length ? (
                <div className="text-sm text-muted-foreground">
                  Top tasks:{" "}
                  {contextPack.progress_state.tasks
                    .slice(0, 3)
                    .map((t) => t.title)
                    .join(" · ")}
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Tabs defaultValue="papers">
              <TabsList>
                <TabsTrigger value="papers">Recommendations</TabsTrigger>
                <TabsTrigger value="inbox">Memory Inbox</TabsTrigger>
                <TabsTrigger value="evals">Evals</TabsTrigger>
              </TabsList>

            <TabsContent value="papers">
              <Card>
                <CardHeader>
                  <CardTitle>Paper Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  {!contextPack?.paper_recommendations?.length ? (
                    <p className="text-sm text-muted-foreground">
                      Build context to fetch recommendations. (Needs backend running + network)
                    </p>
                  ) : (
                    <ScrollArea className="h-[420px] pr-4">
                      <div className="space-y-3">
                        {contextPack.paper_recommendations.map((p, idx) => (
                          <div key={p.paper_id} className="rounded-md border p-3 space-y-2">
                            <div className="flex items-start justify-between gap-3">
                              <div className="space-y-1">
                                <div className="font-medium leading-snug">{p.title}</div>
                                <div className="text-xs text-muted-foreground">
                                  {(p.venue || "Unknown venue") + (p.year ? ` · ${p.year}` : "")}
                                  {" · "}
                                  {p.citation_count ?? 0} citations
                                </div>
                                {contextPack.paper_recommendation_reasons?.[p.paper_id]?.length ? (
                                  <div className="flex flex-wrap gap-1">
                                    {contextPack.paper_recommendation_reasons[p.paper_id].map((r) => (
                                      <Badge key={r} variant="outline" className="text-[10px]">
                                        {r}
                                      </Badge>
                                    ))}
                                  </div>
                                ) : null}
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => sendFeedback(p.paper_id, "like", idx)}
                                  disabled={loading}
                                >
                                  Like
                                </Button>
                                <Button size="sm" variant="outline" onClick={() => sendFeedback(p.paper_id, "save", idx)} disabled={loading}>
                                  Save
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => sendFeedback(p.paper_id, "dislike", idx)}
                                  disabled={loading}
                                >
                                  Dislike
                                </Button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="inbox">
              <Card>
                <CardHeader>
                  <CardTitle>Memory Inbox</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Suggest memories (creates pending items)</Label>
                    <Textarea value={suggestText} onChange={(e) => setSuggestText(e.target.value)} rows={3} />
                    <Button onClick={() => suggestMemories()} disabled={loading || !suggestText.trim() || !activeTrackId}>
                      Extract to Inbox
                    </Button>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                      Track: <span className="font-medium">{activeTrack?.name || "None"}</span>
                      {" · "}
                      Pending: {inbox.length}
                      {" · "}
                      Selected: {selectedCount}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="secondary"
                        onClick={() => openConfirm({ type: "bulk_moderate", status: "approved", itemIds: Array.from(selectedInboxIds) })}
                        disabled={loading || selectedCount === 0}
                      >
                        Approve Selected
                      </Button>
                      <Button
                        variant="destructive"
                        onClick={() => openConfirm({ type: "bulk_moderate", status: "rejected", itemIds: Array.from(selectedInboxIds) })}
                        disabled={loading || selectedCount === 0}
                      >
                        Reject Selected
                      </Button>
                      <Button variant="outline" onClick={() => toggleSelectAll()} disabled={loading || inbox.length === 0}>
                        {allSelected ? "Clear Selection" : "Select All"}
                      </Button>
                      <Button variant="outline" onClick={() => refreshInbox(activeTrackId)} disabled={loading || !activeTrackId}>
                        Refresh Inbox
                      </Button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm text-muted-foreground">Move selected to another track:</div>
                    <div className="flex items-center gap-2">
                      <select
                        className="h-9 rounded-md border bg-background px-3 text-sm"
                        value={moveTargetTrackId}
                        onChange={(e) => setMoveTargetTrackId(e.target.value ? Number(e.target.value) : "")}
                        disabled={loading}
                      >
                        <option value="">Select track…</option>
                        {tracks
                          .filter((t) => t.id !== activeTrackId)
                          .map((t) => (
                            <option key={t.id} value={t.id}>
                              {t.name}
                            </option>
                          ))}
                      </select>
                      <Button
                        variant="outline"
                        onClick={() =>
                          openConfirm({
                            type: "bulk_move",
                            itemIds: Array.from(selectedInboxIds),
                            targetTrackId: Number(moveTargetTrackId),
                          })
                        }
                        disabled={loading || selectedCount === 0 || moveTargetTrackId === ""}
                      >
                        Move
                      </Button>
                    </div>
                  </div>

                  {!inbox.length ? (
                    <p className="text-sm text-muted-foreground">No pending items.</p>
                  ) : (
                    <ScrollArea className="h-[320px] pr-4">
                      <div className="space-y-2">
                        {inbox.map((m) => {
                          const checked = selectedInboxIds.has(m.id)
                          return (
                            <div key={m.id} className="flex items-start justify-between gap-3 rounded-md border p-3">
                              <div className="flex items-start gap-3">
                                <Checkbox
                                  checked={checked}
                                  onCheckedChange={(next) => toggleSelected(m.id, next === true)}
                                  disabled={loading}
                                  aria-label={`Select memory item ${m.id}`}
                                />
                                <div className="space-y-1">
                                  <button
                                    type="button"
                                    className="text-left font-medium underline-offset-4 hover:underline"
                                    onClick={() => toggleSelected(m.id, !checked)}
                                    disabled={loading}
                                  >
                                    {m.kind}
                                  </button>
                                  <div className="text-sm">{m.content}</div>
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => {
                                    bulkModerate("approved", [m.id])
                                  }}
                                  disabled={loading}
                                >
                                  Approve
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => {
                                    bulkModerate("rejected", [m.id])
                                  }}
                                  disabled={loading}
                                >
                                  Reject
                                </Button>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="evals">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between gap-3">
                    <CardTitle>Eval Summary</CardTitle>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        className="w-[120px]"
                        min={1}
                        max={365}
                        value={evalDays}
                        onChange={(e) => setEvalDays(Number(e.target.value))}
                        disabled={loading}
                      />
                      <Button
                        variant="outline"
                        onClick={() => refreshEval().catch((e) => setError(String(e)))}
                        disabled={loading}
                      >
                        Refresh
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-sm text-muted-foreground">
                    Track: <span className="font-medium">{activeTrack?.name || "None"}</span>
                    {" · "}
                    Window: {evalDays} days
                  </div>

                  {!evalSummary ? (
                    <Button variant="secondary" onClick={() => refreshEval().catch((e) => setError(String(e)))} disabled={loading}>
                      Load Summary
                    </Button>
                  ) : (
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                      <div className="rounded-md border p-3 space-y-1">
                        <div className="text-sm font-medium">Volume</div>
                        <div className="text-sm text-muted-foreground">
                          Runs: {evalSummary.runs} · Impressions: {evalSummary.impressions}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Unique papers: {evalSummary.unique_recommended_papers} · Repeat rate:{" "}
                          {(evalSummary.repeat_rate * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="rounded-md border p-3 space-y-1">
                        <div className="text-sm font-medium">Feedback</div>
                        <div className="text-sm text-muted-foreground">
                          Coverage: {(evalSummary.feedback_coverage * 100).toFixed(1)}% · Linked rows:{" "}
                          {evalSummary.linked_feedback_rows}
                        </div>
                        <div className="flex flex-wrap gap-1 pt-1">
                          {Object.entries(evalSummary.feedback_on_recommended || {}).map(([k, v]) => (
                            <Badge key={k} variant="outline" className="text-[10px]">
                              {k}:{v}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
