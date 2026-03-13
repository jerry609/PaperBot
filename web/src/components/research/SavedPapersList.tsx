"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { Check, ChevronDown, Copy, Download, FileText, Filter, Loader2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  currentFeedbackFromRequestAction,
  toggleSaveFeedbackAction,
  type PaperFeedbackRequestAction,
} from "@/lib/paper-feedback"


type SavedPaperSort = "saved_at" | "judge_score" | "published_at"

type ReadingStatus = "unread" | "reading" | "read" | "archived"

type SavedPaperItem = {
  paper: {
    id: number
    title: string
    authors?: string[]
    primary_source?: string
    source?: string
    venue?: string
    url?: string
    external_url?: string
    published_at?: string | null
    publication_date?: string | null
    citation_count?: number
  }
  saved_at?: string | null
  track_id?: number | null
  action?: string
  reading_status?: {
    status?: string
    updated_at?: string | null
  } | null
  latest_judge?: {
    overall?: number | null
    recommendation?: string | null
  } | null
}

type SavedPapersResponse = {
  items?: SavedPaperItem[]
  papers?: SavedPaperItem[]
  total?: number
  limit?: number
  offset?: number
}

type UpdatingAction = "toggleRead" | "unsave"

type Track = {
  id: number
  name: string
  is_active?: boolean
}

type BibtexImportResponse = {
  parsed?: number
  imported?: number
  created?: number
  updated?: number
  skipped?: number
  errors?: string[]
}

type ZoteroPullResponse = {
  imported?: number
  created?: number
  updated?: number
  skipped?: number
  errors?: string[]
}

type ZoteroPushResponse = {
  to_push?: number
  pushed?: number
  skipped?: number
  dry_run?: boolean
  errors?: string[]
}

type CollectionSummary = {
  id: number
  name: string
  description?: string
  item_count?: number
}

type CollectionItem = {
  id: number
  paper_id: number
  note?: string
  tags?: string[]
  paper?: {
    title?: string
    authors?: string[]
  }
}

const PAGE_SIZE = 20
const SORT_OPTIONS: Array<{ value: SavedPaperSort; label: string }> = [
  { value: "saved_at", label: "Saved Time" },
  { value: "judge_score", label: "Judge Score" },
  { value: "published_at", label: "Published Time" },
]

function formatDate(value?: string | null): string {
  if (!value) return "-"
  const dt = new Date(value)
  if (Number.isNaN(dt.getTime())) return "-"
  return dt.toLocaleString()
}

function formatJudge(value?: number | null): string {
  if (typeof value !== "number") return "-"
  return `${value.toFixed(2)} / 5.0`
}

function normalizeStatus(value?: string | null): ReadingStatus {
  if (value === "reading" || value === "read" || value === "archived") return value
  return "unread"
}

type SavedPaperListItemProps = {
  item: SavedPaperItem
  status: ReadingStatus
  selected: boolean
  togglingRead: boolean
  unsaving: boolean
  onToggleSelect: (paperId: number) => void
  onToggleReadStatus: (paperId: number, currentStatus: ReadingStatus) => void
  onUnsave: (paperId: number, externalId: string | null) => void
}

function SavedPaperListItem({
  item,
  status,
  selected,
  togglingRead,
  unsaving,
  onToggleSelect,
  onToggleReadStatus,
  onUnsave,
}: SavedPaperListItemProps) {
  const { paper, latest_judge } = item
  const rowUpdating = togglingRead || unsaving
  const authorsText = (paper.authors || []).slice(0, 4).join(", ") || "Unknown authors"
  const source = paper.primary_source || paper.source || "unknown"
  const savedDate = formatDate(item.saved_at)
  const publishedDate = formatDate(paper.publication_date || paper.published_at)
  const judgeScore = formatJudge(latest_judge?.overall)
  const judgeRecommendation = latest_judge?.recommendation

  const handleUnsave = () => {
    onUnsave(paper.id, paper.external_url || paper.url || null)
  }

  return (
    <div className="flex gap-3 border-b border-border px-3 py-3 last:border-b-0 hover:bg-muted">
      <div className="pt-1">
        <Checkbox
          checked={selected}
          onCheckedChange={() => onToggleSelect(paper.id)}
          aria-label={`Select ${paper.title}`}
        />
      </div>
      <div className="flex-1 space-y-1.5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div
              className="line-clamp-2 text-sm md:text-base font-semibold leading-snug"
              title={paper.title}
            >
              {paper.title}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <Button asChild size="sm" variant="outline">
              <Link href={`/papers/${paper.id}`}>Detail</Link>
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={rowUpdating}
              onClick={() => onToggleReadStatus(paper.id, status)}
            >
              {togglingRead ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : status === "read" ? (
                "Reading"
              ) : (
                "Mark Read"
              )}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={rowUpdating}
              onClick={handleUnsave}
            >
              {unsaving ? <Loader2 className="h-3 w-3 animate-spin" /> : "Unsave"}
            </Button>
          </div>
        </div>

        <div className="text-xs text-muted-foreground">{authorsText}</div>
        {paper.venue ? (
          <div className="text-xs text-muted-foreground">{paper.venue}</div>
        ) : null}

        <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] md:text-xs text-muted-foreground">
          <Badge variant="outline" className="text-[11px] md:text-xs">
            {source}
          </Badge>
          {savedDate !== "-" ? <span>Saved · {savedDate}</span> : null}
          {publishedDate !== "-" ? <span>Published · {publishedDate}</span> : null}
          {judgeScore !== "-" ? (
            <Badge variant="secondary" className="text-[11px] md:text-xs">
              Judge {judgeScore}
            </Badge>
          ) : null}
          <Badge variant="outline" className="text-[11px] md:text-xs">
            {status}
          </Badge>
        </div>

        {judgeRecommendation ? (
          <div className="text-[11px] md:text-xs text-muted-foreground">{judgeRecommendation}</div>
        ) : null}
      </div>
    </div>
  )
}

export default function SavedPapersList() {
  const [items, setItems] = useState<SavedPaperItem[]>([])
  const [sortBy, setSortBy] = useState<SavedPaperSort>("saved_at")
  const [page, setPage] = useState<number>(1)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [updatingAction, setUpdatingAction] = useState<{ paperId: number; action: UpdatingAction } | null>(null)

  // Related Work state
  const [rwOpen, setRwOpen] = useState(false)
  const [rwTopic, setRwTopic] = useState("")
  const [rwLoading, setRwLoading] = useState(false)
  const [rwMarkdown, setRwMarkdown] = useState<string | null>(null)
  const [rwCopied, setRwCopied] = useState(false)

  // BibTeX import state
  const [importOpen, setImportOpen] = useState(false)
  const [importLoading, setImportLoading] = useState(false)
  const [importTrackName, setImportTrackName] = useState("")
  const [importBibtex, setImportBibtex] = useState("")
  const [importResult, setImportResult] = useState<string | null>(null)

  // Zotero sync state
  const [zoteroOpen, setZoteroOpen] = useState(false)
  const [zoteroLoading, setZoteroLoading] = useState(false)
  const [zoteroMode, setZoteroMode] = useState<"pull" | "push">("pull")
  const [zoteroLibraryType, setZoteroLibraryType] = useState<"user" | "group">("user")
  const [zoteroLibraryId, setZoteroLibraryId] = useState("")
  const [zoteroApiKey, setZoteroApiKey] = useState("")
  const [zoteroDryRun, setZoteroDryRun] = useState(true)
  const [zoteroTrackName, setZoteroTrackName] = useState("")
  const [zoteroResult, setZoteroResult] = useState<string | null>(null)

  // Collections state
  const [collectionsOpen, setCollectionsOpen] = useState(false)
  const [collectionsLoading, setCollectionsLoading] = useState(false)
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [collectionItems, setCollectionItems] = useState<CollectionItem[]>([])
  const [selectedCollectionId, setSelectedCollectionId] = useState<number | null>(null)
  const [newCollectionName, setNewCollectionName] = useState("")
  const [newCollectionDesc, setNewCollectionDesc] = useState("")
  const [collectionsMessage, setCollectionsMessage] = useState<string | null>(null)
  const [editingCollectionPaperId, setEditingCollectionPaperId] = useState<number | null>(null)
  const [editingCollectionNote, setEditingCollectionNote] = useState("")
  const [editingCollectionTags, setEditingCollectionTags] = useState("")

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

  // Track filter state
  const [tracks, setTracks] = useState<Track[]>([])
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null)
  const activeTrackId = selectedTrackId

  // Fetch tracks on mount
  useEffect(() => {
    fetch("/api/research/tracks?user_id=default", { cache: "no-store" })
      .then((res) => res.json())
      .then((data) => setTracks(data.tracks || []))
      .catch(() => setTracks([]))
  }, [])

  const loadSavedPapers = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({
        sort_by: sortBy,
        limit: "500",
        user_id: "default",
      })
      if (selectedTrackId) {
        qs.set("track_id", String(selectedTrackId))
      }
      const res = await fetch(`/api/research/papers/saved?${qs.toString()}`, { cache: "no-store" })
      if (!res.ok) {
        const errorText = await res.text()
        // Avoid showing raw HTML in error messages
        if (errorText.startsWith("<!DOCTYPE") || errorText.startsWith("<html")) {
          throw new Error(`Server error: ${res.status} ${res.statusText}`)
        }
        throw new Error(errorText)
      }
      const payload = (await res.json()) as SavedPapersResponse
      setItems(payload.items || payload.papers || [])
      setPage(1)
      setSelectedIds(new Set()) // Clear selection on reload
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setError(detail)
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [sortBy, selectedTrackId])

  useEffect(() => {
    loadSavedPapers().catch(() => {})
  }, [loadSavedPapers])

  const totalPages = useMemo(() => {
    return Math.max(1, Math.ceil(items.length / PAGE_SIZE))
  }, [items.length])

  const pagedItems = useMemo(() => {
    const safePage = Math.min(page, totalPages)
    const start = (safePage - 1) * PAGE_SIZE
    return items.slice(start, start + PAGE_SIZE)
  }, [items, page, totalPages])

  const hasSelection = selectedIds.size > 0
  const selectedCollection = useMemo(
    () => collections.find((item) => item.id === selectedCollectionId) || null,
    [collections, selectedCollectionId],
  )

  const toggleSelect = useCallback((paperId: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(paperId)) {
        next.delete(paperId)
      } else {
        next.add(paperId)
      }
      return next
    })
  }, [])

  const toggleSelectAll = useCallback(() => {
    if (selectedIds.size === pagedItems.length && pagedItems.length > 0) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(pagedItems.map((item) => item.paper.id)))
    }
  }, [selectedIds.size, pagedItems])

  const isAllSelected = pagedItems.length > 0 && selectedIds.size === pagedItems.length

  const unsavePaper = useCallback(
    async (paperId: number, externalId: string | null) => {
      const requestAction: PaperFeedbackRequestAction = toggleSaveFeedbackAction(true)
      setUpdatingAction({ paperId, action: "unsave" })
      setError(null)
      try {
        const payload = {
          user_id: "default",
          track_id: activeTrackId,
          paper_id: externalId || String(paperId),
          action: requestAction,
          weight: 0.0,
          context_run_id: null,
          context_rank: undefined,
          metadata: {},
        }

        const res = await fetch(`/api/research/papers/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })

        if (!res.ok) {
          const errorText = await res.text()
          if (errorText.startsWith("<!DOCTYPE") || errorText.startsWith("<html")) {
            throw new Error(`Server error: ${res.status} ${res.statusText}`)
          }
          throw new Error(errorText)
        }
        const currentAction = currentFeedbackFromRequestAction(requestAction)

        if (currentAction !== "save") {
          setItems((prev) => prev.filter((row) => row.paper.id !== paperId))
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err)
        setError(detail)
      } finally {
        setUpdatingAction(null)
      }
    },
    [activeTrackId],
  )

  const toggleReadStatus = useCallback(
    async (paperId: number, currentStatus: ReadingStatus) => {
      setUpdatingAction({ paperId, action: "toggleRead" })
      setError(null)
      const newStatus = currentStatus === "read" ? "reading" : "read"
      try {
        // Persist reading status so refreshes keep the latest state
        const res = await fetch(`/api/research/papers/${paperId}/status`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "default",
            status: newStatus,
          }),
        })

        if (!res.ok) {
          const errorText = await res.text()
          if (errorText.startsWith("<!DOCTYPE") || errorText.startsWith("<html")) {
            throw new Error(`Server error: ${res.status} ${res.statusText}`)
          }
          throw new Error(errorText)
        }

        setItems((prev) =>
          prev.map((row) => {
            if (row.paper.id !== paperId) return row
            return {
              ...row,
              reading_status: {
                ...row.reading_status,
                status: newStatus,
                updated_at: new Date().toISOString(),
              },
            }
          }),
        )
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err)
        setError(detail)
      } finally {
        setUpdatingAction(null)
      }
    },
    [],
  )

  const handleExport = useCallback(async (format: "bibtex" | "ris" | "markdown" | "csl_json") => {
    const qs = new URLSearchParams({ format, user_id: "default" })
    // Add selected paper IDs
    selectedIds.forEach((id) => qs.append("paper_id", String(id)))
    try {
      const res = await fetch(`/api/papers/export?${qs.toString()}`, { cache: "no-store" })
      if (!res.ok) throw new Error(`${res.status}`)
      const blob = await res.blob()
      const extMap: Record<string, string> = { bibtex: "bib", ris: "ris", markdown: "md", csl_json: "csl.json" }
      const ext = extMap[format] || "txt"
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `papers.${ext}`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      setError("Export failed")
    }
  }, [selectedIds])

  const handleGenerateRelatedWork = useCallback(async () => {
    if (!rwTopic.trim()) return
    setRwLoading(true)
    setRwMarkdown(null)
    try {
      const res = await fetch("/api/research/papers/related-work", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "default",
          topic: rwTopic.trim(),
        }),
      })
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json()
      setRwMarkdown(data.markdown || "No output generated.")
    } catch {
      setRwMarkdown("Failed to generate related work. Please try again.")
    } finally {
      setRwLoading(false)
    }
  }, [rwTopic])

  const handleCopyRw = useCallback(async () => {
    if (!rwMarkdown) return
    await navigator.clipboard.writeText(rwMarkdown)
    setRwCopied(true)
    setTimeout(() => setRwCopied(false), 2000)
  }, [rwMarkdown])

  const handleBibtexImport = useCallback(async () => {
    if (!importBibtex.trim()) return
    setImportLoading(true)
    setImportResult(null)
    setError(null)
    try {
      const res = await fetch("/api/research/papers/import/bibtex", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "default",
          content: importBibtex,
          track_id: selectedTrackId ?? undefined,
          track_name: selectedTrackId ? undefined : (importTrackName.trim() || undefined),
        }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const payload = (await res.json()) as BibtexImportResponse
      const parsed = payload.parsed ?? 0
      const imported = payload.imported ?? 0
      const skipped = payload.skipped ?? 0
      setImportResult(`Imported ${imported}/${parsed}, skipped ${skipped}.`)
      await loadSavedPapers()
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setImportResult(`Import failed: ${detail}`)
    } finally {
      setImportLoading(false)
    }
  }, [importBibtex, importTrackName, loadSavedPapers, selectedTrackId])

  const handleZoteroSync = useCallback(async () => {
    if (!zoteroLibraryId.trim() || !zoteroApiKey.trim()) return
    setZoteroLoading(true)
    setZoteroResult(null)
    setError(null)
    try {
      const endpoint =
        zoteroMode === "pull"
          ? "/api/research/integrations/zotero/pull"
          : "/api/research/integrations/zotero/push"
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "default",
          track_id: selectedTrackId ?? undefined,
          track_name: selectedTrackId ? undefined : (zoteroTrackName.trim() || undefined),
          library_type: zoteroLibraryType,
          library_id: zoteroLibraryId.trim(),
          api_key: zoteroApiKey.trim(),
          max_items: 200,
          dry_run: zoteroMode === "push" ? zoteroDryRun : undefined,
        }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }

      if (zoteroMode === "pull") {
        const payload = (await res.json()) as ZoteroPullResponse
        setZoteroResult(
          `Pulled ${payload.imported ?? 0} (created ${payload.created ?? 0}, updated ${payload.updated ?? 0}).`,
        )
        await loadSavedPapers()
      } else {
        const payload = (await res.json()) as ZoteroPushResponse
        setZoteroResult(
          `Push ${payload.dry_run ? "preview" : "done"}: ${payload.pushed ?? 0}/${payload.to_push ?? 0}.`,
        )
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setZoteroResult(`Zotero sync failed: ${detail}`)
    } finally {
      setZoteroLoading(false)
    }
  }, [
    loadSavedPapers,
    selectedTrackId,
    zoteroApiKey,
    zoteroDryRun,
    zoteroLibraryId,
    zoteroLibraryType,
    zoteroMode,
    zoteroTrackName,
  ])

  const loadCollections = useCallback(async () => {
    setCollectionsLoading(true)
    setCollectionsMessage(null)
    try {
      const qs = new URLSearchParams({ user_id: "default", limit: "200" })
      if (selectedTrackId) qs.set("track_id", String(selectedTrackId))
      const res = await fetch(`/api/research/collections?${qs.toString()}`, { cache: "no-store" })
      if (!res.ok) throw new Error(`${res.status}`)
      const payload = await res.json()
      const rows = (payload.items || []) as CollectionSummary[]
      setCollections(rows)
      const nextSelectedId =
        rows.find((row) => row.id === selectedCollectionId)?.id ?? rows[0]?.id ?? null
      setSelectedCollectionId(nextSelectedId)
      if (!nextSelectedId) {
        setCollectionItems([])
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setCollectionsMessage(`Failed to load collections: ${detail}`)
    } finally {
      setCollectionsLoading(false)
    }
  }, [selectedCollectionId, selectedTrackId])

  const loadCollectionItems = useCallback(async (collectionId: number) => {
    setCollectionsLoading(true)
    setCollectionsMessage(null)
    try {
      const qs = new URLSearchParams({ user_id: "default", limit: "500" })
      const res = await fetch(`/api/research/collections/${collectionId}/items?${qs.toString()}`, {
        cache: "no-store",
      })
      if (!res.ok) throw new Error(`${res.status}`)
      const payload = await res.json()
      setCollectionItems((payload.items || []) as CollectionItem[])
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setCollectionsMessage(`Failed to load collection items: ${detail}`)
      setCollectionItems([])
    } finally {
      setCollectionsLoading(false)
    }
  }, [])

  const createCollection = useCallback(async () => {
    if (!newCollectionName.trim()) return
    setCollectionsLoading(true)
    setCollectionsMessage(null)
    try {
      const res = await fetch("/api/research/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "default",
          name: newCollectionName.trim(),
          description: newCollectionDesc.trim(),
          track_id: selectedTrackId ?? undefined,
        }),
      })
      if (!res.ok) throw new Error(`${res.status}`)
      setNewCollectionName("")
      setNewCollectionDesc("")
      await loadCollections()
      setCollectionsMessage("Collection created.")
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setCollectionsMessage(`Create failed: ${detail}`)
    } finally {
      setCollectionsLoading(false)
    }
  }, [loadCollections, newCollectionDesc, newCollectionName, selectedTrackId])

  const addSelectedToCollection = useCallback(async () => {
    if (!selectedCollectionId || selectedIds.size === 0) return
    setCollectionsLoading(true)
    setCollectionsMessage(null)
    try {
      for (const paperId of selectedIds) {
        await fetch(`/api/research/collections/${selectedCollectionId}/items`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "default",
            paper_id: String(paperId),
            note: "",
            tags: [],
          }),
        })
      }
      await loadCollectionItems(selectedCollectionId)
      await loadCollections()
      setCollectionsMessage(`Added ${selectedIds.size} papers to collection.`)
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setCollectionsMessage(`Add failed: ${detail}`)
    } finally {
      setCollectionsLoading(false)
    }
  }, [loadCollectionItems, loadCollections, selectedCollectionId, selectedIds])

  const saveCollectionItemMeta = useCallback(
    async (item: CollectionItem, note: string, tagsRaw: string) => {
      if (!selectedCollectionId) return
      setCollectionsLoading(true)
      setCollectionsMessage(null)
      try {
        const tags = tagsRaw.split(",").map((tag) => tag.trim()).filter(Boolean)
        const res = await fetch(
          `/api/research/collections/${selectedCollectionId}/items/${item.paper_id}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              user_id: "default",
              note,
              tags,
            }),
          },
        )
        if (!res.ok) throw new Error(`${res.status}`)
        await loadCollectionItems(selectedCollectionId)
        setCollectionsMessage("Note/tags updated.")
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err)
        setCollectionsMessage(`Update failed: ${detail}`)
      } finally {
        setCollectionsLoading(false)
      }
    },
    [loadCollectionItems, selectedCollectionId],
  )

  const startEditingCollectionItem = useCallback((item: CollectionItem) => {
    setEditingCollectionPaperId(item.paper_id)
    setEditingCollectionNote(item.note || "")
    setEditingCollectionTags((item.tags || []).join(", "))
  }, [])

  const cancelEditingCollectionItem = useCallback(() => {
    setEditingCollectionPaperId(null)
    setEditingCollectionNote("")
    setEditingCollectionTags("")
  }, [])

  const saveEditingCollectionItem = useCallback(async () => {
    if (editingCollectionPaperId === null) return
    const item = collectionItems.find((row) => row.paper_id === editingCollectionPaperId)
    if (!item) return
    await saveCollectionItemMeta(item, editingCollectionNote, editingCollectionTags)
    cancelEditingCollectionItem()
  }, [
    cancelEditingCollectionItem,
    collectionItems,
    editingCollectionNote,
    editingCollectionPaperId,
    editingCollectionTags,
    saveCollectionItemMeta,
  ])

  const removeCollectionItem = useCallback(
    async (paperId: number) => {
      if (!selectedCollectionId) return
      setCollectionsLoading(true)
      setCollectionsMessage(null)
      try {
        const qs = new URLSearchParams({ user_id: "default" })
        const res = await fetch(
          `/api/research/collections/${selectedCollectionId}/items/${paperId}?${qs.toString()}`,
          { method: "DELETE" },
        )
        if (!res.ok) throw new Error(`${res.status}`)
        await loadCollectionItems(selectedCollectionId)
        await loadCollections()
        if (editingCollectionPaperId === paperId) {
          cancelEditingCollectionItem()
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err)
        setCollectionsMessage(`Remove failed: ${detail}`)
      } finally {
        setCollectionsLoading(false)
      }
    },
    [
      cancelEditingCollectionItem,
      editingCollectionPaperId,
      loadCollectionItems,
      loadCollections,
      selectedCollectionId,
    ],
  )

  useEffect(() => {
    if (!collectionsOpen || !selectedCollectionId) return
    loadCollectionItems(selectedCollectionId).catch(() => {})
  }, [collectionsOpen, selectedCollectionId, loadCollectionItems])

  useEffect(() => {
    if (!collectionsOpen) {
      cancelEditingCollectionItem()
    }
  }, [cancelEditingCollectionItem, collectionsOpen])

  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle>Saved Papers</CardTitle>
            <CardDescription>
              View saved items, sort by score/time, and update reading status.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground" htmlFor="saved-sort">
              Sort
            </label>
            <select
              id="saved-sort"
              className="h-9 rounded-md border bg-background px-2 text-sm"
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SavedPaperSort)}
            >
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" disabled={loading}>
                  <Filter className="mr-1 h-4 w-4" />
                  {selectedTrackId
                    ? tracks.find((t) => t.id === selectedTrackId)?.name || "Track"
                    : "All Tracks"}
                  <ChevronDown className="ml-1 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  onClick={() => setSelectedTrackId(null)}
                  className="flex items-center gap-2"
                >
                  {selectedTrackId === null ? <Check className="h-4 w-4" /> : <span className="w-4" />}
                  All Tracks
                </DropdownMenuItem>
                {tracks.length > 0 && <DropdownMenuSeparator />}
                {tracks.map((track) => (
                  <DropdownMenuItem
                    key={track.id}
                    onClick={() => setSelectedTrackId(track.id)}
                    className="flex items-center gap-2"
                  >
                    {selectedTrackId === track.id ? <Check className="h-4 w-4" /> : <span className="w-4" />}
                    <span className="truncate">{track.name}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              variant="outline"
              size="sm"
              disabled={loading || items.length === 0}
              onClick={() => { setRwOpen(true); setRwMarkdown(null); setRwTopic(""); }}
            >
              <FileText className="mr-1 h-4 w-4" />
              Related Work
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={loading}
              onClick={() => {
                setCollectionsOpen(true)
                loadCollections().catch(() => {})
              }}
            >
              Collections
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={loading}
              onClick={() => { setImportOpen(true); setImportResult(null) }}
            >
              Import BibTeX
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={loading}
              onClick={() => { setZoteroOpen(true); setZoteroResult(null) }}
            >
              Zotero Sync
            </Button>
            {hasSelection && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" disabled={loading}>
                    <Download className="mr-1 h-4 w-4" />
                    Export ({selectedIds.size})
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleExport("bibtex")}>BibTeX</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExport("ris")}>RIS</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExport("markdown")}>Markdown</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExport("csl_json")}>Zotero (CSL-JSON)</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        </div>
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading saved papers...
          </div>
        ) : items.length === 0 ? (
          <div className="py-8 text-sm text-muted-foreground">No saved papers yet.</div>
        ) : (
          <>
            <div className="rounded-md border bg-card">
              <div className="flex items-center justify-between border-b border-border px-3 py-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={isAllSelected}
                    onCheckedChange={toggleSelectAll}
                    aria-label="Select all"
                  />
                  <span>Select all</span>
                </div>
                <div>
                  <span>{items.length} papers</span>
                </div>
              </div>
              <div>
                {pagedItems.map((item) => {
                  const paper = item.paper
                  const status = normalizeStatus(item.reading_status?.status)
                  const togglingRead =
                    updatingAction?.paperId === paper.id && updatingAction?.action === "toggleRead"
                  const unsaving =
                    updatingAction?.paperId === paper.id && updatingAction?.action === "unsave"
                  return (
                    <SavedPaperListItem
                      key={paper.id}
                      item={item}
                      status={status}
                      selected={selectedIds.has(paper.id)}
                      togglingRead={togglingRead}
                      unsaving={unsaving}
                      onToggleSelect={toggleSelect}
                      onToggleReadStatus={toggleReadStatus}
                      onUnsave={unsavePaper}
                    />
                  )
                })}
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between text-sm text-muted-foreground">
              <span>
                Showing {(Math.min(page, totalPages) - 1) * PAGE_SIZE + 1} -{" "}
                {Math.min(Math.min(page, totalPages) * PAGE_SIZE, items.length)} of {items.length}
              </span>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page <= 1}
                  onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                >
                  Prev
                </Button>
                <span>
                  Page {Math.min(page, totalPages)} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page >= totalPages}
                  onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </CardContent>

      <Dialog open={collectionsOpen} onOpenChange={setCollectionsOpen}>
        <DialogContent className="h-[90vh] w-[min(96vw,1280px)] max-w-none overflow-hidden p-0 sm:max-w-none">
          <div className="flex h-full flex-col">
            <DialogHeader className="border-b px-6 py-4">
              <DialogTitle>Collections Workspace</DialogTitle>
              <DialogDescription>
                Group saved papers, attach note/tags, and reuse collection-scoped context.
              </DialogDescription>
            </DialogHeader>
            <div className="grid min-h-0 flex-1 gap-4 px-6 py-4 lg:grid-cols-[320px_1fr]">
              <div className="min-h-0 min-w-0 space-y-4">
                <div className="space-y-3 rounded-md border bg-muted/20 p-3">
                  <p className="text-sm font-medium">Create Collection</p>
                  <Input
                    placeholder="Collection name"
                    value={newCollectionName}
                    onChange={(event) => setNewCollectionName(event.target.value)}
                    disabled={collectionsLoading}
                  />
                  <Textarea
                    placeholder="Description (optional)"
                    value={newCollectionDesc}
                    onChange={(event) => setNewCollectionDesc(event.target.value)}
                    disabled={collectionsLoading}
                    className="min-h-[80px]"
                  />
                  <Button
                    size="sm"
                    onClick={createCollection}
                    disabled={collectionsLoading || !newCollectionName.trim()}
                  >
                    Create Collection
                  </Button>
                </div>

                <div className="flex min-h-0 flex-col space-y-2 rounded-md border p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium">Collections</p>
                    <Badge variant="outline">{collections.length}</Badge>
                  </div>
                  <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
                    {collections.length === 0 ? (
                      <p className="text-xs text-muted-foreground">
                        No collections yet. Create one to start grouping papers.
                      </p>
                    ) : (
                      collections.map((collection) => (
                        <button
                          key={collection.id}
                          className={`min-w-0 w-full rounded-md border px-3 py-2 text-left transition-colors ${
                            selectedCollectionId === collection.id
                              ? "border-primary bg-muted"
                              : "hover:bg-muted/40"
                          }`}
                          onClick={() => {
                            setSelectedCollectionId(collection.id)
                            cancelEditingCollectionItem()
                            loadCollectionItems(collection.id).catch(() => {})
                          }}
                          type="button"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <p className="line-clamp-2 break-words text-sm font-medium">
                                {collection.name}
                              </p>
                              {collection.description ? (
                                <p className="mt-0.5 line-clamp-2 break-words text-xs text-muted-foreground">
                                  {collection.description}
                                </p>
                              ) : null}
                            </div>
                            <Badge variant="secondary">{collection.item_count || 0}</Badge>
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </div>

              <div className="flex min-h-0 min-w-0 flex-col space-y-3 rounded-md border p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="break-words text-sm font-medium">
                      {selectedCollection?.name || "Select a collection"}
                    </p>
                    {selectedCollection?.description ? (
                      <p className="break-words text-xs text-muted-foreground">
                        {selectedCollection.description}
                      </p>
                    ) : null}
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">
                      {selectedCollection?.item_count ?? collectionItems.length} papers
                    </Badge>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!selectedCollectionId || selectedIds.size === 0 || collectionsLoading}
                      onClick={addSelectedToCollection}
                    >
                      Add Selected ({selectedIds.size})
                    </Button>
                  </div>
                </div>

                {collectionsMessage ? (
                  <p className="rounded-md border bg-muted/20 px-2 py-1 text-sm text-muted-foreground">
                    {collectionsMessage}
                  </p>
                ) : null}

                {!selectedCollectionId ? (
                  <p className="text-sm text-muted-foreground">Choose a collection from the left.</p>
                ) : collectionItems.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No items in this collection yet.</p>
                ) : (
                  <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
                    {collectionItems.map((item) => {
                      const isEditing = editingCollectionPaperId === item.paper_id
                      return (
                        <div key={item.id} className="min-w-0 rounded-md border p-3">
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="min-w-0">
                              <p className="break-words text-sm font-medium">
                                {item.paper?.title || `Paper #${item.paper_id}`}
                              </p>
                              {(item.paper?.authors || []).length > 0 ? (
                                <p className="break-words text-xs text-muted-foreground">
                                  {(item.paper?.authors || []).slice(0, 4).join(", ")}
                                </p>
                              ) : null}
                            </div>
                            <div className="flex items-center gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => startEditingCollectionItem(item)}
                                disabled={collectionsLoading}
                              >
                                Edit note/tags
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => removeCollectionItem(item.paper_id).catch(() => {})}
                                disabled={collectionsLoading}
                              >
                                Remove
                              </Button>
                            </div>
                          </div>

                          {isEditing ? (
                            <div className="mt-3 space-y-2 rounded-md border bg-muted/20 p-2">
                              <Textarea
                                placeholder="Add a note for this paper..."
                                value={editingCollectionNote}
                                onChange={(event) => setEditingCollectionNote(event.target.value)}
                                disabled={collectionsLoading}
                                className="min-h-[80px]"
                              />
                              <Input
                                placeholder="tags: theory, baseline, survey"
                                value={editingCollectionTags}
                                onChange={(event) => setEditingCollectionTags(event.target.value)}
                                disabled={collectionsLoading}
                              />
                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  onClick={() => saveEditingCollectionItem().catch(() => {})}
                                  disabled={collectionsLoading}
                                >
                                  Save
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={cancelEditingCollectionItem}
                                  disabled={collectionsLoading}
                                >
                                  Cancel
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <div className="mt-2 space-y-1">
                              <div className="flex flex-wrap gap-1">
                                {(item.tags || []).length > 0 ? (
                                  (item.tags || []).map((tag) => (
                                    <Badge key={tag} variant="secondary">
                                      {tag}
                                    </Badge>
                                  ))
                                ) : (
                                  <span className="text-xs text-muted-foreground">No tags</span>
                                )}
                              </div>
                              <p className="break-words text-xs text-muted-foreground">
                                {item.note?.trim() ? item.note : "No note yet."}
                              </p>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={importOpen} onOpenChange={setImportOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Import BibTeX</DialogTitle>
            <DialogDescription>
              Paste BibTeX entries to import and save to current track.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <Input
              placeholder="Track name (optional when no track filter)"
              value={importTrackName}
              onChange={(event) => setImportTrackName(event.target.value)}
              disabled={importLoading || selectedTrackId !== null}
            />
            <Textarea
              placeholder="@article{key, title={...}, author={...}}"
              value={importBibtex}
              onChange={(event) => setImportBibtex(event.target.value)}
              disabled={importLoading}
              className="min-h-[220px] font-mono text-xs"
            />
            {importResult ? <p className="text-sm text-muted-foreground">{importResult}</p> : null}
          </div>
          <DialogFooter>
            <Button onClick={handleBibtexImport} disabled={importLoading || !importBibtex.trim()}>
              {importLoading ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : null}
              Import
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={zoteroOpen} onOpenChange={setZoteroOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Zotero Sync</DialogTitle>
            <DialogDescription>
              Pull from Zotero into PaperBot or push saved papers to Zotero.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <select
                className="h-9 rounded-md border bg-background px-2 text-sm"
                value={zoteroMode}
                onChange={(event) => setZoteroMode(event.target.value as "pull" | "push")}
                disabled={zoteroLoading}
              >
                <option value="pull">Pull from Zotero</option>
                <option value="push">Push to Zotero</option>
              </select>
              <select
                className="h-9 rounded-md border bg-background px-2 text-sm"
                value={zoteroLibraryType}
                onChange={(event) => setZoteroLibraryType(event.target.value as "user" | "group")}
                disabled={zoteroLoading}
              >
                <option value="user">User Library</option>
                <option value="group">Group Library</option>
              </select>
            </div>
            <Input
              placeholder="Library ID"
              value={zoteroLibraryId}
              onChange={(event) => setZoteroLibraryId(event.target.value)}
              disabled={zoteroLoading}
            />
            <Input
              type="password"
              placeholder="Zotero API Key"
              value={zoteroApiKey}
              onChange={(event) => setZoteroApiKey(event.target.value)}
              disabled={zoteroLoading}
            />
            <Input
              placeholder="Track name (optional when no track filter)"
              value={zoteroTrackName}
              onChange={(event) => setZoteroTrackName(event.target.value)}
              disabled={zoteroLoading || selectedTrackId !== null}
            />
            {zoteroMode === "push" ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Checkbox
                  checked={zoteroDryRun}
                  onCheckedChange={(checked) => setZoteroDryRun(Boolean(checked))}
                  disabled={zoteroLoading}
                />
                Dry run (preview only)
              </div>
            ) : null}
            {zoteroResult ? <p className="text-sm text-muted-foreground">{zoteroResult}</p> : null}
          </div>
          <DialogFooter>
            <Button
              onClick={handleZoteroSync}
              disabled={zoteroLoading || !zoteroLibraryId.trim() || !zoteroApiKey.trim()}
            >
              {zoteroLoading ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : null}
              {zoteroMode === "pull" ? "Start Pull" : "Start Push"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Related Work Dialog */}
      <Dialog open={rwOpen} onOpenChange={setRwOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Generate Related Work</DialogTitle>
            <DialogDescription>
              Generate a Related Work section draft from your saved papers.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="flex gap-2">
              <Input
                placeholder="Enter research topic..."
                value={rwTopic}
                onChange={(e) => setRwTopic(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleGenerateRelatedWork() }}
                disabled={rwLoading}
              />
              <Button onClick={handleGenerateRelatedWork} disabled={rwLoading || !rwTopic.trim()}>
                {rwLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Generate"}
              </Button>
            </div>
            {rwMarkdown && (
              <div className="rounded-md border bg-muted/50 p-4">
                <pre className="whitespace-pre-wrap text-sm font-mono">{rwMarkdown}</pre>
              </div>
            )}
          </div>
          {rwMarkdown && (
            <DialogFooter>
              <Button variant="outline" size="sm" onClick={handleCopyRw}>
                <Copy className="mr-1 h-3.5 w-3.5" />
                {rwCopied ? "Copied" : "Copy"}
              </Button>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>
    </Card>
  )
}
