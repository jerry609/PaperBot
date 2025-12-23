"use client"

import { useEffect, useMemo, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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
  routing: {
    track_id: number | null
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

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(`${res.status} ${res.statusText} ${text}`.trim())
  }
  return res.json() as Promise<T>
}

export default function ResearchDashboard() {
  const [userId, setUserId] = useState("default")
  const [tracks, setTracks] = useState<Track[]>([])
  const [activeTrackId, setActiveTrackId] = useState<number | null>(null)
  const [query, setQuery] = useState("")
  const [contextPack, setContextPack] = useState<ContextPack | null>(null)
  const [inbox, setInbox] = useState<MemoryItem[]>([])
  const [suggestText, setSuggestText] = useState("")
  const [selectedInboxIds, setSelectedInboxIds] = useState<Set<number>>(new Set())
  const [moveTargetTrackId, setMoveTargetTrackId] = useState<number | "">("")
  const [createOpen, setCreateOpen] = useState(false)
  const [newTrackName, setNewTrackName] = useState("")
  const [newTrackDescription, setNewTrackDescription] = useState("")
  const [newTrackKeywords, setNewTrackKeywords] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selectedCount = selectedInboxIds.size

  const activeTrack = useMemo(() => tracks.find((t) => t.id === activeTrackId) || null, [tracks, activeTrackId])

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
      const body = {
        user_id: userId,
        query,
        paper_limit: 8,
        memory_limit: 8,
        offline: false,
        include_cross_track: false,
        activate_track_id: activateTrackId,
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

  async function bulkModerate(status: "approved" | "rejected") {
    if (selectedInboxIds.size === 0) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/api/research/memory/bulk_moderate`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          item_ids: Array.from(selectedInboxIds),
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

  async function bulkMoveToTrack() {
    if (selectedInboxIds.size === 0) return
    if (moveTargetTrackId === "" || !moveTargetTrackId) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/api/research/memory/bulk_move`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          item_ids: Array.from(selectedInboxIds),
          scope_type: "track",
          scope_id: String(moveTargetTrackId),
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

  async function sendFeedback(paperId: string, action: string) {
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`/api/research/papers/feedback`, {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          track_id: activeTrackId,
          paper_id: paperId,
          action,
          weight: 0.0,
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

  async function clearTrackMemory() {
    if (!activeTrackId) return
    const ok = window.confirm("Clear all memories in this track? (soft delete)")
    if (!ok) return
    setLoading(true)
    setError(null)
    try {
      await fetchJson(
        `/api/research/tracks/${activeTrackId}/memory/clear?user_id=${encodeURIComponent(userId)}&confirm=true`,
        { method: "POST", body: "{}", headers: { "Content-Type": "application/json" } },
      )
      await refreshInbox(activeTrackId)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
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
            <Button variant="outline" onClick={() => clearTrackMemory()} disabled={loading || !activeTrackId}>
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
                        {contextPack.paper_recommendations.map((p) => (
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
                                <Button size="sm" variant="secondary" onClick={() => sendFeedback(p.paper_id, "like")} disabled={loading}>
                                  Like
                                </Button>
                                <Button size="sm" variant="outline" onClick={() => sendFeedback(p.paper_id, "save")} disabled={loading}>
                                  Save
                                </Button>
                                <Button size="sm" variant="destructive" onClick={() => sendFeedback(p.paper_id, "dislike")} disabled={loading}>
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
                      <Button variant="secondary" onClick={() => bulkModerate("approved")} disabled={loading || selectedCount === 0}>
                        Approve Selected
                      </Button>
                      <Button variant="destructive" onClick={() => bulkModerate("rejected")} disabled={loading || selectedCount === 0}>
                        Reject Selected
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
                        onClick={() => bulkMoveToTrack()}
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
                              <div className="space-y-1">
                                <button
                                  type="button"
                                  className="text-left font-medium underline-offset-4 hover:underline"
                                  onClick={() => {
                                    const next = new Set(selectedInboxIds)
                                    if (next.has(m.id)) next.delete(m.id)
                                    else next.add(m.id)
                                    setSelectedInboxIds(next)
                                  }}
                                >
                                  {checked ? "✓ " : ""}
                                  {m.kind}
                                </button>
                                <div className="text-sm">{m.content}</div>
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => {
                                    setSelectedInboxIds(new Set([m.id]))
                                    bulkModerate("approved")
                                  }}
                                  disabled={loading}
                                >
                                  Approve
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => {
                                    setSelectedInboxIds(new Set([m.id]))
                                    bulkModerate("rejected")
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
          </Tabs>
        </div>
      </div>
    </div>
  )
}
