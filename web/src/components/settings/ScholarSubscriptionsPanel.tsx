"use client"

import { useEffect, useMemo, useState } from "react"
import {
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  UserRound,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"

type ScholarRow = {
  id: string
  semantic_scholar_id?: string | null
  name: string
  affiliation: string
  keywords?: string[]
  status: "active" | "idle"
  recent_activity: string
  cached_papers?: number
}

type ScholarCandidate = {
  author_id: string
  name: string
  affiliation?: string
  affiliations?: string[]
  paper_count?: number
  citation_count?: number
  h_index?: number
}

type ScholarFormState = {
  id?: string
  name: string
  semantic_scholar_id: string
  affiliation: string
  keywords: string
}

const EMPTY_FORM: ScholarFormState = {
  name: "",
  semantic_scholar_id: "",
  affiliation: "",
  keywords: "",
}

function splitKeywords(raw: string): string[] {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
}

export function ScholarSubscriptionsPanel() {
  const [items, setItems] = useState<ScholarRow[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [removing, setRemoving] = useState<string | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [form, setForm] = useState<ScholarFormState>(EMPTY_FORM)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const [searchQuery, setSearchQuery] = useState("")
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<ScholarCandidate[]>([])
  const [searchError, setSearchError] = useState<string | null>(null)

  const editing = useMemo(() => !!form.id, [form.id])
  const knownSemanticIds = useMemo(
    () => new Set(items.map((item) => String(item.semantic_scholar_id || item.id || "").toLowerCase())),
    [items],
  )

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/research/scholars?limit=200", { cache: "no-store" })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const payload = (await res.json()) as { items?: ScholarRow[] }
      setItems(payload.items || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load().catch(() => {})
  }, [])

  function openAdd(prefill?: Partial<ScholarFormState>) {
    setForm({ ...EMPTY_FORM, ...prefill })
    setDialogOpen(true)
    setError(null)
    setMessage(null)
  }

  function openEdit(item: ScholarRow) {
    setForm({
      id: item.id,
      name: item.name,
      semantic_scholar_id: String(item.semantic_scholar_id || item.id),
      affiliation: item.affiliation || "",
      keywords: (item.keywords || []).join(", "),
    })
    setDialogOpen(true)
    setError(null)
    setMessage(null)
  }

  async function searchCandidates() {
    const q = searchQuery.trim()
    if (!q) return

    setSearching(true)
    setSearchError(null)
    try {
      const qs = new URLSearchParams({ query: q, limit: "8" })
      const res = await fetch(`/api/research/scholars/search?${qs.toString()}`, { cache: "no-store" })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `${res.status} ${res.statusText}`)
      }
      const payload = (await res.json()) as { items?: ScholarCandidate[] }
      setSearchResults(payload.items || [])
    } catch (e) {
      setSearchError(e instanceof Error ? e.message : String(e))
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }

  async function saveScholar() {
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      if (!form.name.trim()) throw new Error("Name is required")
      if (!form.semantic_scholar_id.trim()) throw new Error("Semantic Scholar ID is required")

      const payload = {
        name: form.name.trim(),
        semantic_scholar_id: form.semantic_scholar_id.trim(),
        affiliations: form.affiliation.trim() ? [form.affiliation.trim()] : [],
        keywords: splitKeywords(form.keywords),
        research_fields: [],
      }

      const url = editing
        ? `/api/research/scholars/${encodeURIComponent(String(form.id))}`
        : "/api/research/scholars"
      const method = editing ? "PATCH" : "POST"

      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `${res.status} ${res.statusText}`)
      }

      await load()
      setDialogOpen(false)
      setForm({ ...EMPTY_FORM })
      setMessage(editing ? "Scholar updated." : "Scholar added.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  async function removeScholar(item: ScholarRow) {
    if (!confirm(`Remove ${item.name} from subscriptions?`)) return

    setRemoving(item.id)
    setError(null)
    setMessage(null)
    try {
      const res = await fetch(`/api/research/scholars/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `${res.status} ${res.statusText}`)
      }
      await load()
      setMessage("Scholar removed.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRemoving(null)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold">Scholar Subscriptions</h3>
          <p className="text-sm text-muted-foreground">
            Manage watchlist used by scholar tracking workflows
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => load().catch(() => {})}>
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Refresh
          </Button>
          <Button size="sm" onClick={() => openAdd()}>
            <Plus className="mr-1.5 h-3.5 w-3.5" />
            Add Scholar
          </Button>
        </div>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {message && <p className="text-sm text-green-600">{message}</p>}

      <Card>
        <CardContent className="space-y-3 p-4">
          <p className="text-sm font-medium">Import from Semantic Scholar</p>
          <div className="flex flex-wrap gap-2">
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search author name"
              className="min-w-[220px] flex-1"
            />
            <Button size="sm" onClick={searchCandidates} disabled={searching || !searchQuery.trim()}>
              {searching ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Search className="mr-1.5 h-3.5 w-3.5" />}
              Search
            </Button>
          </div>
          {searchError ? <p className="text-xs text-destructive">{searchError}</p> : null}

          {!!searchResults.length && (
            <div className="space-y-2">
              {searchResults.map((candidate) => {
                const alreadyAdded = knownSemanticIds.has(String(candidate.author_id || "").toLowerCase())
                return (
                  <div key={candidate.author_id} className="flex items-center justify-between gap-3 rounded-lg border p-2.5">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{candidate.name}</p>
                      <p className="truncate text-xs text-muted-foreground">
                        {candidate.affiliation || "Unknown affiliation"} · {candidate.author_id}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        papers {candidate.paper_count || 0} · citations {candidate.citation_count || 0} · h-index {candidate.h_index || 0}
                      </p>
                    </div>
                    <Button
                      size="sm"
                      variant={alreadyAdded ? "secondary" : "outline"}
                      disabled={alreadyAdded}
                      onClick={() =>
                        openAdd({
                          name: candidate.name,
                          semantic_scholar_id: candidate.author_id,
                          affiliation: candidate.affiliation || "",
                        })
                      }
                    >
                      {alreadyAdded ? "Added" : "Use"}
                    </Button>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading scholars...</p>
      ) : !items.length ? (
        <p className="text-sm text-muted-foreground">No scholars configured yet.</p>
      ) : (
        <div className="space-y-2">
          {items.map((item) => (
            <Card key={item.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <UserRound className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="text-sm font-medium">{item.name}</span>
                      <Badge variant={item.status === "active" ? "default" : "secondary"} className="capitalize">
                        {item.status}
                      </Badge>
                    </div>
                    <p className="mt-1 truncate text-xs text-muted-foreground">
                      {item.affiliation || "Unknown affiliation"} · {item.semantic_scholar_id || item.id}
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {(item.keywords || []).slice(0, 3).map((keyword) => (
                        <Badge key={`${item.id}-${keyword}`} variant="outline" className="text-[11px]">
                          {keyword}
                        </Badge>
                      ))}
                      <span className="text-xs text-muted-foreground">{item.recent_activity}</span>
                    </div>
                  </div>

                  <div className="flex shrink-0 gap-1">
                    <Button size="sm" variant="ghost" onClick={() => openEdit(item)}>
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      onClick={() => removeScholar(item)}
                      disabled={removing === item.id}
                    >
                      {removing === item.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editing ? "Edit Scholar" : "Add Scholar"}</DialogTitle>
            <DialogDescription>
              Configure scholar metadata used for watchlist and tracking jobs.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-3">
            <div className="space-y-1">
              <label className="text-sm font-medium">Name</label>
              <Input
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="Dawn Song"
              />
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium">Semantic Scholar ID</label>
              <Input
                value={form.semantic_scholar_id}
                onChange={(e) => setForm((prev) => ({ ...prev, semantic_scholar_id: e.target.value }))}
                placeholder="1741101"
              />
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium">Affiliation</label>
              <Input
                value={form.affiliation}
                onChange={(e) => setForm((prev) => ({ ...prev, affiliation: e.target.value }))}
                placeholder="UC Berkeley"
              />
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium">Keywords (comma separated)</label>
              <Input
                value={form.keywords}
                onChange={(e) => setForm((prev) => ({ ...prev, keywords: e.target.value }))}
                placeholder="AI Security, LLM Safety"
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={saveScholar} disabled={saving}>
              {saving ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> : null}
              {editing ? "Update" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
