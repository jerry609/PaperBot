"use client"

import { useEffect, useMemo, useState } from "react"
import { Copy, Download, ExternalLink, FileText, FolderSync, Loader2, RefreshCw } from "lucide-react"

import { Button } from "@/components/ui/button"
import { getErrorMessage } from "@/lib/fetch"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
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
import { buildObsidianExportCommand, describeObsidianScope } from "@/lib/obsidian"

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
  trackId: number | null
  trackName?: string | null
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

export function SavedTab({ trackId, trackName = null }: SavedTabProps) {
  const [items, setItems] = useState<SavedItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rwOpen, setRwOpen] = useState(false)
  const [rwTopic, setRwTopic] = useState("")
  const [rwLoading, setRwLoading] = useState(false)
  const [rwMarkdown, setRwMarkdown] = useState<string | null>(null)
  const [rwCopied, setRwCopied] = useState(false)
  const [obsidianOpen, setObsidianOpen] = useState(false)
  const [obsidianVaultPath, setObsidianVaultPath] = useState("~/Obsidian/MyVault")
  const [obsidianRootDir, setObsidianRootDir] = useState("PaperBot")
  const [obsidianCopied, setObsidianCopied] = useState(false)
  const [obsidianCopyError, setObsidianCopyError] = useState<string | null>(null)

  const papers = useMemo(() => items.map(toPaper), [items])
  const obsidianCommand = useMemo(
    () =>
      buildObsidianExportCommand({
        vaultPath: obsidianVaultPath,
        rootDir: obsidianRootDir,
        trackId,
      }),
    [obsidianRootDir, obsidianVaultPath, trackId]
  )
  const obsidianScope = useMemo(
    () => describeObsidianScope(trackId, trackName),
    [trackId, trackName]
  )

  const handleExport = async (format: "bibtex" | "ris" | "markdown" | "csl_json") => {
    const qs = new URLSearchParams({ format })
    if (trackId != null) qs.set("track_id", String(trackId))
    try {
      const res = await fetch(`/api/papers/export`)
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

  const handleCopyObsidianCommand = async () => {
    try {
      await navigator.clipboard.writeText(obsidianCommand)
      setObsidianCopyError(null)
      setObsidianCopied(true)
      setTimeout(() => setObsidianCopied(false), 2000)
    } catch {
      setObsidianCopied(false)
      setObsidianCopyError("Clipboard access failed. Copy the command manually.")
      setTimeout(() => setObsidianCopyError(null), 3000)
    }
  }

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({
        sort_by: "saved_at",
        limit: "100",
      })
      if (trackId != null) {
        qs.set("track_id", String(trackId))
      }
      const res = await fetch(`/api/research/papers/saved`)
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
  }, [trackId])

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-border/70 bg-background/90 p-4 shadow-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{obsidianScope}</Badge>
              <Badge variant="outline">{papers.length} saved</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              {trackId != null
                ? "Saved papers scoped to the current track."
                : "Saved papers across your library."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={loading || !papers.length}
              onClick={() => setObsidianOpen(true)}
            >
              <FolderSync className="h-3.5 w-3.5" /> Obsidian
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={loading || !papers.length}
              onClick={() => {
                setRwOpen(true)
                setRwMarkdown(null)
                setRwTopic("")
              }}
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
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {loading && !papers.length ? (
        <div className="py-8 text-sm text-muted-foreground">Loading saved papers...</div>
      ) : !papers.length ? (
        <div className="rounded-2xl border border-dashed bg-muted/20 py-10 text-center text-sm text-muted-foreground">
          {trackId != null ? "No saved papers in this track yet." : "No saved papers in your library yet."}
        </div>
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

      <Dialog open={obsidianOpen} onOpenChange={setObsidianOpen}>
        <DialogContent className="max-h-[85vh] max-w-2xl overflow-hidden border-border/70 bg-white/95 p-0 shadow-2xl backdrop-blur">
          <div className="flex h-full max-h-[85vh] flex-col">
            <DialogHeader className="border-b border-border/70 px-6 pb-4 pt-6 pr-14">
              <DialogTitle>Export to Obsidian</DialogTitle>
              <DialogDescription>
                This dashboard does not write vault files directly. Copy the local CLI command below and run
                it on the machine that owns your Obsidian vault.
              </DialogDescription>
            </DialogHeader>

            <div className="flex-1 space-y-4 overflow-y-auto px-6 py-5">
              <Card className="border-border/70 bg-slate-50">
                <CardContent className="space-y-3 py-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary">{obsidianScope}</Badge>
                    <Badge variant="outline">{papers.length} saved paper{papers.length === 1 ? "" : "s"}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Markdown fallback stays available from the normal export menu if you only need a flat bundle.
                  </p>
                </CardContent>
              </Card>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-2 text-sm">
                  <span className="font-medium">Vault path</span>
                  <Input
                    aria-label="Vault path"
                    value={obsidianVaultPath}
                    onChange={(e) => setObsidianVaultPath(e.target.value)}
                    placeholder="~/Obsidian/MyVault"
                  />
                </label>
                <label className="space-y-2 text-sm">
                  <span className="font-medium">Root directory</span>
                  <Input
                    aria-label="Root directory"
                    value={obsidianRootDir}
                    onChange={(e) => setObsidianRootDir(e.target.value)}
                    placeholder="PaperBot"
                  />
                </label>
              </div>

              <div className="rounded-2xl border bg-slate-950 p-4 text-sm text-slate-50 shadow-inner">
                <pre className="whitespace-pre-wrap break-all font-mono">{obsidianCommand}</pre>
              </div>
              {obsidianCopyError ? (
                <p className="text-xs text-red-600">{obsidianCopyError}</p>
              ) : null}

              <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  Direct dashboard-triggered vault writes stay out of scope until we have a trusted local bridge or
                  allowlisted directory model. See issue #346.
                </span>
              </div>
            </div>

            <DialogFooter className="border-t border-border/70 px-6 py-4 sm:justify-between">
              <Button variant="outline" size="sm" onClick={() => handleExport("markdown")}>
                <Download className="h-3.5 w-3.5" />
                Download Markdown Fallback
              </Button>
              <Button size="sm" onClick={handleCopyObsidianCommand}>
                <Copy className="h-3.5 w-3.5" />
                {obsidianCopied ? "Copied" : "Copy CLI Command"}
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
