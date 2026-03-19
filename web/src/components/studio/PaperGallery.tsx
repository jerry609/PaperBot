"use client"

import { useState, useMemo, useEffect, type CSSProperties } from "react"
import { useRouter } from "next/navigation"
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
import { Plus, Search, Trash2, FlaskConical, Loader2, FolderOpen, MessageSquare, ArrowRight } from "lucide-react"

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

function formatPaperStatus(status: StudioPaper["status"]): string {
    if (status === "running") return "Running"
    if (status === "completed") return "Completed"
    return "Draft"
}

function paperStatusClassName(status: StudioPaper["status"]): string {
    if (status === "running") return "border-amber-200 bg-amber-50 text-amber-700"
    if (status === "completed") return "border-emerald-200 bg-emerald-50 text-emerald-700"
    return "border-slate-200 bg-[#f7f8f4] text-slate-600"
}

function buildPaperStageBadges(paper: StudioPaper): Array<{
    label: string
    className: string
}> {
    const badges = [
        paper.outputDir
            ? {
                label: "workspace ready",
                className: "border-emerald-200 bg-emerald-50 text-emerald-700",
            }
            : {
                label: "review workspace",
                className: "border-slate-200 bg-[#f7f8f4] text-slate-600",
            },
        paper.contextPackId
            ? {
                label: "skills ready",
                className: "border-emerald-200 bg-emerald-50 text-emerald-700",
            }
            : {
                label: "skills pending",
                className: "border-slate-200 bg-[#f7f8f4] text-slate-600",
            },
    ]

    if ((paper.taskIds?.length ?? 0) > 0 || paper.boardSessionId) {
        badges.push({
            label: "thread started",
            className: "border-sky-200 bg-sky-50 text-sky-700",
        })
    }

    return badges
}

function hasMonitorEntry(paper: StudioPaper): boolean {
    return Boolean(paper.boardSessionId || (paper.taskIds?.length ?? 0) > 0)
}

export function PaperGallery() {
    const router = useRouter()
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
    const [spriteFrame, setSpriteFrame] = useState(0)

    useEffect(() => {
        // Drive all paper icons with one shared frame clock for stable, synchronized animation.
        const interval = window.setInterval(() => {
            setSpriteFrame((prev) => (prev + 1) % 6)
        }, 140)
        return () => window.clearInterval(interval)
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
    const totalPaperCount = papers.length
    const draftCount = papers.filter((paper) => paper.status === "draft").length
    const runningCount = papers.filter((paper) => paper.status === "running").length
    const completedCount = papers.filter((paper) => paper.status === "completed").length
    const featuredPaper =
        !query.trim() && activeFilter === "all" && sortedPapers.length > 1 ? sortedPapers[0] : null
    const remainingPapers = featuredPaper ? sortedPapers.slice(1) : sortedPapers
    const latestMonitorPaper = useMemo(
        () =>
            sortedPapers.find((paper) => hasMonitorEntry(paper)) ?? null,
        [sortedPapers],
    )

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
                <div className="mx-auto flex w-full max-w-[1360px] flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div className="max-w-[720px] space-y-3">
                        <span className="inline-flex rounded-full border border-zinc-300 bg-white px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
                            Step 1 of 3
                        </span>
                        <div>
                            <h1 className="text-[28px] font-semibold tracking-[-0.03em] text-zinc-950">
                                Choose a paper to start Studio
                            </h1>
                            <p className="mt-1.5 text-sm leading-6 text-zinc-500">
                                Pick an existing paper or add a new one. Workspace review and Claude Code chat happen next,
                                and Monitor stays secondary until you need raw execution detail.
                            </p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            <span className="rounded-full border border-zinc-300 bg-white px-2.5 py-1 text-[11px] text-zinc-600">
                                {totalPaperCount} papers
                            </span>
                            <span className="rounded-full border border-zinc-300 bg-white px-2.5 py-1 text-[11px] text-zinc-600">
                                {draftCount} draft
                            </span>
                            <span className="rounded-full border border-zinc-300 bg-white px-2.5 py-1 text-[11px] text-zinc-600">
                                {runningCount} running
                            </span>
                            <span className="rounded-full border border-zinc-300 bg-white px-2.5 py-1 text-[11px] text-zinc-600">
                                {completedCount} completed
                            </span>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {latestMonitorPaper ? (
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="h-10 rounded-full border-zinc-300 bg-white px-4 text-zinc-700"
                                onClick={() =>
                                    router.push(`/studio/agent-board?paperId=${encodeURIComponent(latestMonitorPaper.id)}`)
                                }
                            >
                                Monitor latest
                            </Button>
                        ) : null}
                        <Button onClick={() => setNewPaperOpen(true)} size="sm" className="h-10 rounded-full px-4">
                            <Plus className="mr-1.5 h-4 w-4" />
                            Add Paper
                        </Button>
                    </div>
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
                            className="h-11 rounded-full border-zinc-300 bg-white pl-9"
                        />
                    </div>

                    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                        <div className="flex flex-wrap gap-2">
                            {[
                                {
                                    key: "all" as const,
                                    label: `All Papers (${filteredPapers.length})`,
                                },
                                {
                                    key: "draft" as const,
                                    label: `Drafts (${filteredPapers.filter((paper) => paper.status === "draft").length})`,
                                },
                                {
                                    key: "running" as const,
                                    label: `Running (${filteredPapers.filter((paper) => paper.status === "running").length})`,
                                },
                                {
                                    key: "completed" as const,
                                    label: `Completed (${filteredPapers.filter((paper) => paper.status === "completed").length})`,
                                },
                            ].map((filter) => (
                                <button
                                    key={filter.key}
                                    onClick={() => setActiveFilter(filter.key)}
                                    className={cn(
                                        "rounded-full border px-3 py-1.5 text-[12px] transition-colors",
                                        activeFilter === filter.key
                                            ? "border-zinc-900 bg-zinc-900 text-white"
                                            : "border-zinc-300 bg-white text-zinc-600 hover:border-zinc-400 hover:text-zinc-900",
                                    )}
                                >
                                    {filter.label}
                                </button>
                            ))}
                        </div>

                        <p className="text-[12px] text-zinc-500">
                            {sortedPapers.length} paper{sortedPapers.length === 1 ? "" : "s"} ready for the next Studio step
                        </p>
                    </div>

                    {sortedPapers.length === 0 ? (
                        /* Empty state */
                        <div className="rounded-[30px] border border-zinc-300 bg-white px-6 py-8 shadow-[0_18px_40px_rgba(15,23,42,0.04)]">
                            <div className="mx-auto max-w-[920px] text-center">
                                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-3xl border border-zinc-300 bg-[#f7f8f4]">
                                    <FlaskConical className="h-7 w-7 text-zinc-500" />
                                </div>
                                <h2 className="mt-5 text-[26px] font-semibold tracking-[-0.03em] text-zinc-950">
                                    {query ? "No papers match this search" : "Add your first paper"}
                                </h2>
                                <p className="mx-auto mt-2 max-w-[38rem] text-sm leading-6 text-zinc-500">
                                    {query
                                        ? "Try a different title, keyword, or abstract phrase."
                                        : "Start with a paper title and abstract. Studio will guide the next steps: workspace review first, then Claude Code chat, then Monitor only when execution detail matters."}
                                </p>

                                {!query ? (
                                    <>
                                        <div className="mt-6 grid gap-3 text-left md:grid-cols-3">
                                            <div className="rounded-[22px] border border-zinc-300 bg-[#fafaf7] px-4 py-4">
                                                <div className="flex items-center gap-2">
                                                    <FlaskConical className="h-4 w-4 text-zinc-500" />
                                                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
                                                        1. Add paper
                                                    </p>
                                                </div>
                                                <p className="mt-2 text-sm font-medium text-zinc-900">
                                                    Paste the title and abstract for the paper you want to reproduce.
                                                </p>
                                            </div>
                                            <div className="rounded-[22px] border border-zinc-300 bg-[#fafaf7] px-4 py-4">
                                                <div className="flex items-center gap-2">
                                                    <FolderOpen className="h-4 w-4 text-zinc-500" />
                                                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
                                                        2. Review workspace
                                                    </p>
                                                </div>
                                                <p className="mt-2 text-sm font-medium text-zinc-900">
                                                    Confirm where generated code should live before the Studio session starts.
                                                </p>
                                            </div>
                                            <div className="rounded-[22px] border border-zinc-300 bg-[#fafaf7] px-4 py-4">
                                                <div className="flex items-center gap-2">
                                                    <MessageSquare className="h-4 w-4 text-zinc-500" />
                                                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
                                                        3. Start chat
                                                    </p>
                                                </div>
                                                <p className="mt-2 text-sm font-medium text-zinc-900">
                                                    Open a focused Claude Code thread and keep Monitor for compressed execution detail only.
                                                </p>
                                            </div>
                                        </div>

                                        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
                                            <Button
                                                onClick={() => setNewPaperOpen(true)}
                                                size="sm"
                                                className="h-10 rounded-full px-4"
                                            >
                                                <Plus className="mr-1.5 h-4 w-4" />
                                                Add Paper
                                            </Button>
                                            <span className="rounded-full border border-zinc-300 bg-[#f7f8f4] px-3 py-1 text-[11px] text-zinc-600">
                                                Review before runtime
                                            </span>
                                            <span className="rounded-full border border-zinc-300 bg-[#f7f8f4] px-3 py-1 text-[11px] text-zinc-600">
                                                Chat first, Monitor second
                                            </span>
                                        </div>
                                    </>
                                ) : null}
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {featuredPaper ? (
                                <article
                                    onClick={() => selectPaper(featuredPaper.id)}
                                    onKeyDown={(event) => {
                                        if (event.key === "Enter" || event.key === " ") {
                                            event.preventDefault()
                                            selectPaper(featuredPaper.id)
                                        }
                                    }}
                                    role="button"
                                    tabIndex={0}
                                    className="group rounded-[30px] border border-zinc-300 bg-white px-5 py-5 text-left shadow-[0_18px_40px_rgba(15,23,42,0.04)] transition-[border-color,box-shadow] hover:border-zinc-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500 focus-visible:ring-offset-2"
                                >
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                        <div>
                                            <span className="inline-flex rounded-full border border-zinc-300 bg-[#f7f8f4] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
                                                Resume latest
                                            </span>
                                            <h2 className="mt-3 text-[26px] font-semibold tracking-[-0.03em] text-zinc-950">
                                                {featuredPaper.title}
                                            </h2>
                                            <p className="mt-2 max-w-[48rem] text-sm leading-6 text-zinc-500">
                                                {summarizeAbstract(featuredPaper.abstract)}
                                            </p>
                                        </div>
                                        <span className={cn(
                                            "rounded-full border px-2.5 py-1 text-[11px] font-medium",
                                            paperStatusClassName(featuredPaper.status),
                                        )}>
                                            {formatPaperStatus(featuredPaper.status)}
                                        </span>
                                    </div>

                                    <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
                                        <div className="flex gap-4">
                                            <PixelCornerIcon
                                                frame={spriteFrame}
                                                size={84}
                                                style={{
                                                    animation: "paperRowSync 2.2s ease-in-out infinite",
                                                }}
                                            />
                                            <div className="min-w-0 space-y-3">
                                                <p className="text-sm text-zinc-600">
                                                    {inferAuthor(featuredPaper)}
                                                </p>
                                                <div className="flex flex-wrap gap-2">
                                                    {buildPaperStageBadges(featuredPaper).map((badge) => (
                                                        <span
                                                            key={badge.label}
                                                            className={cn(
                                                                "rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]",
                                                                badge.className,
                                                            )}
                                                        >
                                                            {badge.label}
                                                        </span>
                                                    ))}
                                                </div>
                                                <p className="text-[13px] leading-6 text-zinc-500">
                                                    {(() => {
                                                        const tags = inferTags(featuredPaper)
                                                        if (tags.shown.length === 0) return "#research #paper"
                                                        const prefix = tags.shown.map((tag) => `#${tag}`).join(" ")
                                                        return tags.extraCount > 0 ? `${prefix} +${tags.extraCount} more` : prefix
                                                    })()}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="flex h-full flex-col rounded-[24px] border border-zinc-300 bg-[linear-gradient(180deg,#fafaf7_0%,#f4f5ef_100%)] px-4 py-4">
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
                                                        Next Studio move
                                                    </p>
                                                    <p className="mt-1 text-[18px] font-semibold tracking-[-0.02em] text-zinc-950">
                                                        Continue in Studio
                                                    </p>
                                                </div>
                                                <span className="shrink-0 rounded-full border border-zinc-300 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-zinc-500">
                                                    Workspace first
                                                </span>
                                            </div>
                                            <p className="mt-2 text-sm leading-6 text-zinc-700">
                                                Review workspace, then continue in Claude Code chat.
                                            </p>
                                            <div className="mt-3 grid gap-2">
                                                <div className="rounded-[16px] border border-zinc-300/80 bg-white/80 px-3 py-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-300 bg-[#f7f8f4] text-[10px] font-semibold text-zinc-600">
                                                            1
                                                        </span>
                                                        <FolderOpen className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                                                        <p className="text-[11px] font-medium text-zinc-900">Review workspace</p>
                                                    </div>
                                                    <p className="mt-1 pl-7 text-[10px] leading-5 text-zinc-500">
                                                        Confirm where generated code should live before the next run.
                                                    </p>
                                                </div>
                                                <div className="rounded-[16px] border border-zinc-300/80 bg-white/80 px-3 py-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-300 bg-[#f7f8f4] text-[10px] font-semibold text-zinc-600">
                                                            2
                                                        </span>
                                                        <MessageSquare className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                                                        <p className="text-[11px] font-medium text-zinc-900">Continue in chat</p>
                                                    </div>
                                                    <p className="mt-1 pl-7 text-[10px] leading-5 text-zinc-500">
                                                        Keep Monitor secondary unless you need worker or tool detail.
                                                    </p>
                                                </div>
                                            </div>
                                            <div className="mt-auto border-t border-zinc-300 pt-3">
                                                <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] text-zinc-500">
                                                    <span>Updated {formatRelativeTime(featuredPaper.updatedAt)}</span>
                                                    {hasMonitorEntry(featuredPaper) ? (
                                                        <span className="rounded-full border border-zinc-300 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-zinc-500">
                                                            Monitor ready
                                                        </span>
                                                    ) : null}
                                                </div>
                                                <Button
                                                    type="button"
                                                    className="mt-3 h-10 w-full rounded-full px-4"
                                                    onClick={(event) => {
                                                        event.stopPropagation()
                                                        selectPaper(featuredPaper.id)
                                                    }}
                                                >
                                                    Continue in Studio
                                                    <ArrowRight className="ml-1.5 h-4 w-4" />
                                                </Button>
                                                <div className="mt-2 flex flex-wrap items-center gap-2">
                                                    {hasMonitorEntry(featuredPaper) ? (
                                                        <Button
                                                            type="button"
                                                            variant="outline"
                                                            className="h-9 rounded-full border-zinc-300 bg-white px-4 text-zinc-700"
                                                            onClick={(event) => {
                                                                event.stopPropagation()
                                                                router.push(`/studio/agent-board?paperId=${encodeURIComponent(featuredPaper.id)}`)
                                                            }}
                                                        >
                                                            Monitor
                                                        </Button>
                                                    ) : null}
                                                    <button
                                                        type="button"
                                                        onClick={(event) => handleDeleteClick(event, featuredPaper)}
                                                        className="ml-auto inline-flex h-9 w-9 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-white hover:text-rose-600"
                                                        title="Delete paper"
                                                        aria-label="Delete paper"
                                                    >
                                                        <Trash2 className="h-3.5 w-3.5" />
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </article>
                            ) : null}

                            {remainingPapers.length > 0 ? (
                                <div className="space-y-3">
                                    {featuredPaper ? (
                                        <div className="flex items-center justify-between gap-3 px-1">
                                            <h3 className="text-[13px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
                                                More papers
                                            </h3>
                                            <p className="text-[12px] text-zinc-500">
                                                Choose one and continue straight into Studio
                                            </p>
                                        </div>
                                    ) : null}

                                    {remainingPapers.map((paper, index) => (
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
                                            className="group rounded-[26px] border border-zinc-300 bg-white px-4 py-4 text-left transition-[border-color,box-shadow] hover:border-zinc-400 hover:shadow-[0_12px_28px_rgba(15,23,42,0.04)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500 focus-visible:ring-offset-2"
                                        >
                                            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                                                <div className="flex min-w-0 gap-4">
                                                    <PixelCornerIcon
                                                        frame={spriteFrame}
                                                        size={68}
                                                        style={{
                                                            animation: "paperRowSync 2.2s ease-in-out infinite",
                                                            animationDelay: `${index * 120}ms`,
                                                        }}
                                                    />
                                                    <div className="min-w-0">
                                                        <div className="flex flex-wrap items-center gap-2">
                                                            <h3 className="text-[18px] font-semibold tracking-[-0.02em] text-zinc-950">
                                                                {paper.title}
                                                            </h3>
                                                            <span className={cn(
                                                                "rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em]",
                                                                paperStatusClassName(paper.status),
                                                            )}>
                                                                {formatPaperStatus(paper.status)}
                                                            </span>
                                                        </div>
                                                        <p className="mt-2 line-clamp-2 text-sm leading-6 text-zinc-500">
                                                            {summarizeAbstract(paper.abstract)}
                                                        </p>
                                                        <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px] text-zinc-500">
                                                            <span>{inferAuthor(paper)}</span>
                                                            <span className="text-zinc-300">/</span>
                                                            <span>Updated {formatRelativeTime(paper.updatedAt)}</span>
                                                        </div>
                                                        <div className="mt-3 flex flex-wrap gap-2">
                                                            {buildPaperStageBadges(paper).map((badge) => (
                                                                <span
                                                                    key={badge.label}
                                                                    className={cn(
                                                                        "rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]",
                                                                        badge.className,
                                                                    )}
                                                                >
                                                                    {badge.label}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="flex min-w-[220px] shrink-0 flex-col gap-2">
                                                    <div className="flex items-center justify-between gap-2 text-[11px] text-zinc-500">
                                                        <span>Next Studio move</span>
                                                        <span className="rounded-full border border-zinc-300 bg-[#f7f8f4] px-2 py-0.5 text-[9px] uppercase tracking-[0.12em] text-zinc-500">
                                                            {paper.outputDir ? "Chat ready" : "Workspace first"}
                                                        </span>
                                                    </div>
                                                    <Button
                                                        type="button"
                                                        className="h-9 rounded-full px-4"
                                                        onClick={(event) => {
                                                            event.stopPropagation()
                                                            selectPaper(paper.id)
                                                        }}
                                                    >
                                                        Continue in Studio
                                                        <ArrowRight className="ml-1.5 h-4 w-4" />
                                                    </Button>
                                                    <div className="flex items-center gap-2">
                                                        {hasMonitorEntry(paper) ? (
                                                            <Button
                                                                type="button"
                                                                variant="outline"
                                                                className="h-9 rounded-full border-zinc-300 bg-white px-4 text-zinc-700"
                                                                onClick={(event) => {
                                                                    event.stopPropagation()
                                                                    router.push(`/studio/agent-board?paperId=${encodeURIComponent(paper.id)}`)
                                                                }}
                                                            >
                                                                Monitor
                                                            </Button>
                                                        ) : null}
                                                        <button
                                                            type="button"
                                                            onClick={(event) => handleDeleteClick(event, paper)}
                                                            className="ml-auto inline-flex h-9 w-9 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-white hover:text-rose-600"
                                                            title="Delete paper"
                                                            aria-label="Delete paper"
                                                        >
                                                            <Trash2 className="h-3.5 w-3.5" />
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        </article>
                                    ))}
                                </div>
                            ) : null}
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
