"use client"

import { useState, useMemo } from "react"
import { useStudioStore, type StudioPaperStatus } from "@/lib/store/studio-store"
import { NewPaperModal } from "./NewPaperModal"
import { PaperIcon } from "./PaperIcon"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
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
import { Plus, Search, Loader2, Trash2, FlaskConical } from "lucide-react"

const statusConfig: Record<StudioPaperStatus, { label: string; className: string }> = {
    draft: { label: "Draft", className: "bg-zinc-500/90 text-white" },
    generating: { label: "Code", className: "bg-blue-500 text-white" },
    ready: { label: "Ready", className: "bg-emerald-500 text-white" },
    running: { label: "Run", className: "bg-violet-500 text-white" },
    completed: { label: "Done", className: "bg-emerald-500 text-white" },
    error: { label: "Error", className: "bg-red-500 text-white" },
}

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

export function PaperGallery() {
    const { papers, selectPaper, deletePaper } = useStudioStore()
    const [query, setQuery] = useState("")
    const [newPaperOpen, setNewPaperOpen] = useState(false)
    const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
    const [paperToDelete, setPaperToDelete] = useState<{
        id: string
        title: string
        outputDir?: string
    } | null>(null)
    const [deleting, setDeleting] = useState(false)
    const [deleteError, setDeleteError] = useState<string | null>(null)

    const filteredPapers = useMemo(() => {
        const q = query.trim().toLowerCase()
        if (!q) return papers
        return papers.filter(
            (p) =>
                p.title.toLowerCase().includes(q) ||
                p.abstract.toLowerCase().includes(q),
        )
    }, [papers, query])

    const sortedPapers = useMemo(() => {
        return [...filteredPapers].sort(
            (a, b) =>
                new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
        )
    }, [filteredPapers])

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
                    const response = await fetch("/api/runbook/delete", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            project_dir: paperToDelete.outputDir,
                        }),
                    })
                    if (!response.ok) {
                        throw new Error(
                            `Delete failed with status ${response.status}`,
                        )
                    }
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
        <div className="flex h-full flex-col">
            {/* Header */}
            <div className="border-b bg-background px-6 py-5 shrink-0">
                <div className="mx-auto max-w-6xl flex items-center justify-between">
                    <div>
                        <h1 className="text-xl font-semibold">DeepCode Studio</h1>
                        <p className="text-sm text-muted-foreground mt-0.5">
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
                <div className="mx-auto max-w-6xl px-6 py-6">
                    {/* Search */}
                    <div className="relative mb-6 max-w-md">
                        <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search papers..."
                            className="pl-9 h-9"
                        />
                    </div>

                    {sortedPapers.length === 0 ? (
                        /* Empty state */
                        <div className="flex flex-col items-center justify-center py-24 text-center">
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
                        /* Responsive grid */
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                            {sortedPapers.map((paper) => {
                                const config = statusConfig[paper.status]
                                const isLoading =
                                    paper.status === "generating" ||
                                    paper.status === "running"

                                return (
                                    <Card
                                        key={paper.id}
                                        onClick={() => selectPaper(paper.id)}
                                        className="group relative cursor-pointer gap-0 py-0 transition-all hover:shadow-md hover:border-primary/30"
                                    >
                                        {/* Delete button */}
                                        <button
                                            onClick={(e) =>
                                                handleDeleteClick(e, paper)
                                            }
                                            className="absolute right-2 top-2 z-10 rounded-md p-1.5 opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
                                            title="Delete paper"
                                        >
                                            <Trash2 className="h-3.5 w-3.5" />
                                        </button>

                                        {/* Icon */}
                                        <div className="flex items-center gap-3 px-4 pt-4">
                                            <PaperIcon
                                                paperId={paper.id}
                                                title={paper.title}
                                                size={44}
                                            />
                                        </div>

                                        {/* Title */}
                                        <div className="px-4 pt-3 pb-3">
                                            <p className="text-sm font-medium leading-snug">
                                                {paper.title || "Untitled"}
                                            </p>
                                        </div>

                                        {/* Footer: status + time */}
                                        <div className="flex items-center gap-1.5 border-t px-4 py-2.5">
                                            <span
                                                className={cn(
                                                    "inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium",
                                                    config.className,
                                                )}
                                            >
                                                {isLoading && (
                                                    <Loader2 className="mr-1 h-2.5 w-2.5 animate-spin" />
                                                )}
                                                {config.label}
                                            </span>
                                            <span className="text-[10px] text-muted-foreground/60">
                                                ·
                                            </span>
                                            <span className="text-[11px] text-muted-foreground/60">
                                                {formatRelativeTime(
                                                    paper.updatedAt,
                                                )}
                                            </span>
                                        </div>
                                    </Card>
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
