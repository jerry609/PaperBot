"use client"

import { useEffect, useMemo, useState } from "react"
import { Copy, Download, FileText, Loader2, RefreshCw } from "lucide-react"

import { Button } from "@/components/ui/button"
import { getErrorMessage } from "@/lib/fetch"
import { Card, CardContent } from "@/components/ui/card"
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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"

import { PaperCard, type Paper } from "./PaperCard"

type SavedItem = {
  paper: {
    id?: number
    title?: string
    abstract?: string
    authors?: string[]
    year?: number
    venue?: string
    citation_count?: number
    url?: string
  }
  latest_judge?: Paper["latest_judge"]
  saved_at?: string | null
}

type SavedResponse = {
  user_id: string
  items: SavedItem[]
}

interface SavedTabProps {
  userId: string
  trackId: number | null
}

function toPaper(item: SavedItem): Paper {
  return {
    paper_id: String(item.paper.id || ""),
    title: item.paper.title || "Untitled",
    abstract: item.paper.abstract || "",
    authors: item.paper.authors || [],
    year: item.paper.year,
    venue: item.paper.venue,
    citation_count: item.paper.citation_count || 0,
    url: item.paper.url,
    latest_judge: item.latest_judge,
    is_saved: true,
  }
}

export function SavedTab({ userId, trackId }: SavedTabProps) {
  const [items, setItems] = useState<SavedItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rwOpen, setRwOpen] = useState(false)
  const [rwTopic, setRwTopic] = useState("")
  const [rwLoading, setRwLoading] = useState(false)
  const [rwMarkdown, setRwMarkdown] = useState<string | null>(null)
  const [rwCopied, setRwCopied] = useState(false)

  const papers = useMemo(() => items.map(toPaper), [items])

  const handleExport = async (format: "bibtex" | "ris" | "markdown" | "csl_json") => {
    const qs = new URLSearchParams({ format, user_id: userId })
    if (trackId) qs.set("track_id", String(trackId))
    try {
      const res = await fetch(`/api/papers/export?${qs.toString()}`)
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
  }

  const handleGenerateRelatedWork = async () => {
    if (!rwTopic.trim()) return
    setRwLoading(true)
    setRwMarkdown(null)
    try {
      const res = await fetch("/api/research/papers/related-work", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          track_id: trackId,
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
  }

  const handleCopyRw = async () => {
    if (!rwMarkdown) return
    await navigator.clipboard.writeText(rwMarkdown)
    setRwCopied(true)
    setTimeout(() => setRwCopied(false), 2000)
  }

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({
        user_id: userId,
        sort_by: "saved_at",
        limit: "100",
      })
      if (trackId) {
        qs.set("track_id", String(trackId))
      }
      const res = await fetch(`/api/research/papers/saved?${qs.toString()}`)
      if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
      }
      const payload = (await res.json()) as SavedResponse
      setItems(payload.items || [])
    } catch (e) {
      setError(getErrorMessage(e))
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, trackId])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Saved papers scoped to current track.</p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={loading || !papers.length}
            onClick={() => { setRwOpen(true); setRwMarkdown(null); setRwTopic(""); }}
          >
            <FileText className="h-3.5 w-3.5" /> Related Work
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" disabled={loading || !papers.length}>
                <Download className="h-3.5 w-3.5" /> Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => handleExport("bibtex")}>BibTeX</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleExport("ris")}>RIS</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleExport("markdown")}>Markdown</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleExport("csl_json")}>Zotero (CSL-JSON)</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="outline" size="sm" onClick={() => load().catch(() => {})} disabled={loading}>
            {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {loading && !papers.length ? (
        <div className="py-8 text-sm text-muted-foreground">Loading saved papers...</div>
      ) : !papers.length ? (
        <div className="py-8 text-sm text-muted-foreground">No saved papers in this track.</div>
      ) : (
        <div className="space-y-3">
          {papers.map((paper, idx) => (
            <PaperCard key={`${paper.paper_id}-${idx}`} paper={paper} rank={idx} />
          ))}
        </div>
      )}

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
                <Copy className="h-3.5 w-3.5" />
                {rwCopied ? "Copied" : "Copy"}
              </Button>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}