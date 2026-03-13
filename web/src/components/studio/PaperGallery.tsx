"use client"

import { useState, useMemo, useEffect, type CSSProperties } from "react"
import { useStudioStore, type StudioPaper } from "@/lib/store/studio-store"
import { NewPaperModal } from "./NewPaperModal"
import { deleteProjectFiles } from "@/lib/runbook/deleteProjectFiles"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { cn } from "@/lib/utils"
import { Plus, Search, Trash2, FlaskConical, Loader2 } from "lucide-react"

const STOP_WORDS = new Set([
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "from", "by", "at", "is", "are",
    "this", "that", "these", "those", "into", "through", "using", "use",
])

function formatRelativeTime(dateStr: string): string {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (minutes < 1) return "now"
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString()
}

function PixelCornerIcon({
    className,
    style,
    size = 108,
    frame = 0,
}: {
    className?: string
    style?: CSSProperties
    size?: number
    frame?: number
}) {
    return (
        <div
            className={cn(
                "shrink-0 bg-no-repeat",
                className,
            )}
            style={{
                width: size,
                height: size,
                backgroundImage: "url('/icons/llama-sprite.png')",
                backgroundSize: `${size * 6}px ${size}px`,
                backgroundPosition: `${-frame * size}px 0`,
                imageRendering: "pixelated",
                ...style,
            }}
            aria-hidden="true"
        />
    )
}

function buildDisplayWords(title: string): string[] {
    const cleaned = (title || "")
        .replace(/[^A-Za-z0-9\s-]/g, " ")
        .split(/\s+/)
        .map((word) => word.trim())
        .filter(Boolean)
        .filter((word) => !STOP_WORDS.has(word.toLowerCase()))
    if (cleaned.length === 0) return ["untitled"]
    const words = cleaned.slice(0, 2).map((word) => word.toLowerCase())
    return words
}

function summarizeAbstract(abstract: string): string {
    const text = (abstract || "").trim()
    if (!text) return "No abstract yet."

    const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map((part) => part.trim())
        .filter(Boolean)

    if (sentences.length === 0) return text.slice(0, 180)
    const summary = sentences.slice(0, 2).join(" ")
    if (summary.length <= 260) return summary
    return `${summary.slice(0, 257).trimEnd()}...`
}

function inferAuthor(paper: StudioPaper): string {
    const authors = Array.isArray(paper.authors) ? paper.authors : []
    const normalized = authors
        .map((name) => String(name || "").trim())
        .filter(Boolean)
    if (normalized.length > 0) return normalized[0]
    return "Author unavailable"
}

function inferTags(paper: StudioPaper): { shown: string[]; extraCount: number } {
    const directAreas = Array.isArray(paper.researchAreas)
        ? paper.researchAreas
            .map((area) => String(area || "").trim().toLowerCase())
            .filter(Boolean)
        : []

    if (directAreas.length > 0) {
        const uniq = Array.from(new Set(directAreas))
        return { shown: uniq.slice(0, 3), extraCount: Math.max(0, uniq.length - 3) }
    }

    const source = `${paper.title} ${paper.abstract}`.toLowerCase()
    const tokens = source
        .replace(/[^a-z0-9\s-]/g, " ")
        .split(/\s+/)
        .map((token) => token.trim())
        .filter((token) => token.length >= 4 && !STOP_WORDS.has(token))
    const unique: string[] = []
    for (const token of tokens) {
        if (!unique.includes(token)) unique.push(token)
    }
    const shown = unique.slice(0, 3)
    const extraCount = Math.max(0, unique.length - shown.length)
    return { shown, extraCount }
}

export function PaperGallery() {
    const { papers, selectPaper, deletePaper, updatePaper } = useStudioStore()
    const [query, setQuery] = useState("")
    const [activeFilter, setActiveFilter] = useState<"all" | "draft" | "running" | "completed">("all")
    const [newPaperOpen, setNewPaperOpen] = useState(false)
    const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
    const [paperToDelete, setPaperToDelete] = useState<{
        id: string
        title: string
        outputDir?: string
    } | null>(null)
    const [deleting, setDeleting] = useState(false)
    const [deleteError, setDeleteError] = useState<string | null>(null)
    const [gridColumns, setGridColumns] = useState(4)
    const [spriteFrame, setSpriteFrame] = useState(0)

    useEffect(() => {
        // Drive all paper icons with one shared frame clock for stable, synchronized animation.
        const interval = window.setInterval(() => {
            setSpriteFrame((prev) => (prev + 1) % 6)
        }, 140)
        return () => window.clearInterval(interval)
    }, [])

    useEffect(() => {
        const updateColumns = () => {
            const width = window.innerWidth
            if (width >= 1280) {
                setGridColumns(4)
                return
            }
            if (width >= 768) {
                setGridColumns(2)
                return
            }
            setGridColumns(1)
        }

        updateColumns()
        window.addEventListener("resize", updateColumns)
        return () => window.removeEventListener("resize", updateColumns)
    }, [])

    useEffect(() => {
        const missingMetadata = papers.filter(
            (paper) =>
                (!Array.isArray(paper.authors) || paper.authors.length === 0) ||
                (!Array.isArray(paper.researchAreas) || paper.researchAreas.length === 0),
        )
        if (missingMetadata.length === 0) return

        let cancelled = false

        const hydratePaperMetadata = async () => {
            try {
                const response = await fetch("/api/papers/library")
                if (!response.ok) return
                const payload = (await response.json()) as {
                    papers?: Array<{ paper?: Record<string, unknown> }>
                }
                const libraryRows = Array.isArray(payload.papers) ? payload.papers : []
                const byTitle = new Map<
                    string,
                    { authors: string[]; researchAreas: string[] }
                >()

                for (const row of libraryRows) {
                    const raw = row.paper ?? {}
                    const title = String(raw.title || "").trim().toLowerCase()
                    if (!title) continue
                    const authors = Array.isArray(raw.authors)
                        ? raw.authors
                            .map((author) => String(author || "").trim())
                            .filter(Boolean)
                        : []
                    const areasSource =
                        raw.research_areas ||
                        raw.researchAreas ||
                        raw.keywords ||
                        raw.fields ||
                        []
                    const researchAreas = Array.isArray(areasSource)
                        ? areasSource
                            .map((area) => String(area || "").trim())
                            .filter(Boolean)
                        : []
                    byTitle.set(title, { authors, researchAreas })
                }

                for (const paper of missingMetadata) {
                    if (cancelled) break
                    const key = paper.title.trim().toLowerCase()
                    const match = byTitle.get(key)
                    if (!match) continue

                    const updates: Partial<StudioPaper> = {}
                    if (
                        (!Array.isArray(paper.authors) || paper.authors.length === 0) &&
                        match.authors.length > 0
                    ) {
                        updates.authors = match.authors
                    }
                    if (
                        (!Array.isArray(paper.researchAreas) || paper.researchAreas.length === 0) &&
                        match.researchAreas.length > 0
                    ) {
                        updates.researchAreas = match.researchAreas
                    }
                    if (Object.keys(updates).length > 0) {
                        updatePaper(paper.id, updates)
                    }
                }
            } catch {
                // Keep the gallery render stable if metadata hydration fails.
            }
        }

        void hydratePaperMetadata()
        return () => {
            cancelled = true
        }
    }, [papers, updatePaper])

    const filteredPapers = useMemo(() => {
        const q = query.trim().toLowerCase()
        if (!q) return papers
        return papers.filter(
            (p) =>
                p.title.toLowerCase().includes(q) ||
                p.abstract.toLowerCase().includes(q),
        )
    }, [papers, query])

    const statusFilteredPapers = useMemo(() => {
        if (activeFilter === "all") return filteredPapers
        return filteredPapers.filter((paper) => paper.status === activeFilter)
    }, [activeFilter, filteredPapers])

    const sortedPapers = useMemo(() => {
        return [...statusFilteredPapers].sort(
            (a, b) =>
                new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
        )
    }, [statusFilteredPapers])

    const handleDeleteClick = (
        e: React.MouseEvent,
        paper: (typeof papers)[0],
    ) => {
        e.stopPropagation()
        setPaperToDelete({
            id: paper.id,
            title: paper.title,
            outputDir: paper.outputDir,
        })
        setDeleteError(null)
        setDeleteConfirmOpen(true)
    }

    const handleConfirmDelete = async () => {
        if (!paperToDelete) return
        setDeleting(true)
        setDeleteError(null)
        let deleted = false
        try {
            if (paperToDelete.outputDir) {
                try {
                    await deleteProjectFiles(paperToDelete.outputDir)
                } catch (e) {
                    console.error("Failed to delete project files:", e)
                    setDeleteError(
                        "Failed to delete project files. Please try again.",
                    )
                    return
                }
            }
            deletePaper(paperToDelete.id)
            deleted = true
        } finally {
            setDeleting(false)
            if (deleted) {
                setDeleteConfirmOpen(false)
                setPaperToDelete(null)
            }
        }
    }

    return (
        <div className="flex h-full flex-col bg-background">
            <style jsx>{`
                @keyframes paperRowSync {
                    0%,
                    100% {
                        transform: translateY(0);
                    }
                    50% {
                        transform: translateY(-2px);
                    }
                }
            `}</style>
            {/* Header */}
            <div className="border-b border-zinc-300 bg-background px-6 py-5 shrink-0">
                <div className="mx-auto w-full max-w-[1360px] flex items-center justify-between">
                    <div>
                        <h1 className="text-xl font-semibold">DeepCode Studio</h1>
                        <p className="mt-0.5 text-sm text-zinc-500">
                            Paper-to-code reproduction workspace
                        </p>
                    </div>
                    <Button onClick={() => setNewPaperOpen(true)} size="sm">
                        <Plus className="h-4 w-4 mr-1.5" />
                        New Paper
                    </Button>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-auto">
                <div className="mx-auto w-full max-w-[1360px] px-6 py-6">
                    {/* Search */}
                    <div className="relative mb-5 max-w-[520px]">
                        <Search className="absolute left-3 top-3 h-4 w-4 text-zinc-400" />
                        <Input
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search papers..."
                            className="h-11 rounded-none border-zinc-300 bg-white pl-9"
                        />
                    </div>

                    {/* Filter strip to mirror catalog-like browsing */}
                    <div className="mb-5 flex flex-wrap items-center gap-1 border-b border-zinc-300 pb-2 text-sm">
                        <button
                            onClick={() => setActiveFilter("all")}
                            className={cn(
                                "rounded px-2 py-1",
                                activeFilter === "all" ? "font-medium text-zinc-900" : "text-zinc-600 hover:text-zinc-900",
                            )}
                        >
                            All Papers ({filteredPapers.length})
                        </button>
                        <span className="px-1 text-zinc-300">|</span>
                        <button
                            onClick={() => setActiveFilter("draft")}
                            className={cn(
                                "rounded px-2 py-1",
                                activeFilter === "draft" ? "font-medium text-zinc-900" : "text-zinc-600 hover:text-zinc-900",
                            )}
                        >
                            Drafts ({filteredPapers.filter((paper) => paper.status === "draft").length})
                        </button>
                        <button
                            onClick={() => setActiveFilter("running")}
                            className={cn(
                                "rounded px-2 py-1",
                                activeFilter === "running" ? "font-medium text-zinc-900" : "text-zinc-600 hover:text-zinc-900",
                            )}
                        >
                            Running ({filteredPapers.filter((paper) => paper.status === "running").length})
                        </button>
                        <button
                            onClick={() => setActiveFilter("completed")}
                            className={cn(
                                "rounded px-2 py-1",
                                activeFilter === "completed" ? "font-medium text-zinc-900" : "text-zinc-600 hover:text-zinc-900",
                            )}
                        >
                            Completed ({filteredPapers.filter((paper) => paper.status === "completed").length})
                        </button>
                    </div>

                    {sortedPapers.length === 0 ? (
                        /* Empty state */
                        <div className="flex flex-col items-center justify-center border border-zinc-300 bg-white py-24 text-center">
                            <div className="rounded-full bg-muted p-4 mb-4">
                                <FlaskConical className="h-8 w-8 text-muted-foreground" />
                            </div>
                            <h2 className="text-lg font-medium mb-1">
                                {query ? "No papers found" : "Add your first paper"}
                            </h2>
                            <p className="text-sm text-muted-foreground mb-4 max-w-sm">
                                {query
                                    ? "Try a different search term."
                                    : "Start by adding a paper to reproduce its code. Paste a title and abstract to get started."}
                            </p>
                            {!query && (
                                <Button
                                    onClick={() => setNewPaperOpen(true)}
                                    size="sm"
                                >
                                    <Plus className="h-4 w-4 mr-1.5" />
                                    Add Paper
                                </Button>
                            )}
                        </div>
                    ) : (
                        /* Tiled catalog grid (Image-2 style) */
                        <div className="grid grid-cols-1 gap-0 md:grid-cols-2 xl:grid-cols-4">
                            {sortedPapers.map((paper, index) => {
                                const rowIndex = Math.floor(index / gridColumns)
                                const colIndex = index % gridColumns
                                const isFirstRow = rowIndex === 0
                                const isFirstCol = colIndex === 0
                                return (
                                    <article
                                        key={paper.id}
                                        onClick={() => selectPaper(paper.id)}
                                        onKeyDown={(event) => {
                                            if (event.key === "Enter" || event.key === " ") {
                                                event.preventDefault()
                                                selectPaper(paper.id)
                                            }
                                        }}
                                        role="button"
                                        tabIndex={0}
                                        className={cn(
                                            "group relative flex min-h-[300px] cursor-pointer flex-col border-r border-b border-zinc-300 bg-background p-6 text-left transition-colors hover:bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500 focus-visible:ring-offset-2",
                                            isFirstRow && "border-t",
                                            isFirstCol && "border-l",
                                        )}
                                    >
                                        {/* Delete button */}
                                        <button
                                            onClick={(e) =>
                                                handleDeleteClick(e, paper)
                                            }
                                            className="absolute right-3 top-3 z-10 rounded-md p-1.5 opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
                                            title="Delete paper"
                                        >
                                            <Trash2 className="h-3.5 w-3.5" />
                                        </button>

                                        {/* Header row: icon + abbreviation */}
                                        <div className="mb-4 flex items-center gap-4">
                                            <PixelCornerIcon
                                                frame={spriteFrame}
                                                style={{
                                                    animation: "paperRowSync 2.2s ease-in-out infinite",
                                                    animationDelay: `${rowIndex * 140}ms`,
                                                }}
                                            />
                                            <div className="leading-[0.95]">
                                                {buildDisplayWords(paper.title).map((word) => (
                                                    <p
                                                        key={`${paper.id}-${word}`}
                                                        className="text-[18px] font-bold tracking-[-0.01em] text-zinc-900"
                                                    >
                                                        {word}
                                                    </p>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Main content */}
                                        <div className="flex-1 space-y-2">
                                            <p className="line-clamp-4 text-[17px] leading-[1.4] text-zinc-500">
                                                {summarizeAbstract(paper.abstract)}
                                            </p>
                                            <p className="line-clamp-1 text-[13px] leading-5 text-zinc-500">
                                                {inferAuthor(paper)}
                                            </p>
                                            <p className="line-clamp-2 text-[13px] leading-5 text-zinc-500">
                                                {(() => {
                                                    const tags = inferTags(paper)
                                                    if (tags.shown.length === 0) return "#research #paper"
                                                    const prefix = tags.shown.map((tag) => `#${tag}`).join(" ")
                                                    return tags.extraCount > 0 ? `${prefix} +${tags.extraCount} more` : prefix
                                                })()}
                                            </p>
                                        </div>

                                        <p className="mt-4 text-[12px] text-zinc-500">
                                            Updated {formatRelativeTime(paper.updatedAt)}
                                        </p>
                                    </article>
                                )
                            })}
                        </div>
                    )}
                </div>
            </div>

            <NewPaperModal open={newPaperOpen} onOpenChange={setNewPaperOpen} />

            {/* Delete Confirmation Dialog */}
            <AlertDialog
                open={deleteConfirmOpen}
                onOpenChange={(open) => {
                    setDeleteConfirmOpen(open)
                    if (!open) {
                        setPaperToDelete(null)
                        setDeleteError(null)
                    }
                }}
            >
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Paper</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete &quot;
                            {paperToDelete?.title}&quot;?
                            {paperToDelete?.outputDir && (
                                <span className="mt-2 block text-destructive">
                                    This will also delete all generated code
                                    files.
                                </span>
                            )}
                            {deleteError && (
                                <span className="mt-2 block text-sm text-destructive">
                                    {deleteError}
                                </span>
                            )}
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={deleting}>
                            Cancel
                        </AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleConfirmDelete}
                            disabled={deleting}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            {deleting ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Deleting...
                                </>
                            ) : (
                                "Delete"
                            )}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    )
}
