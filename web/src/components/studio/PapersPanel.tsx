"use client"

import { useState, useMemo, useEffect } from "react"
import { useStudioStore, StudioPaperStatus } from "@/lib/store/studio-store"
import { NewPaperModal } from "./NewPaperModal"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
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
import { deleteProjectFiles } from "@/lib/runbook/deleteProjectFiles"
import {
    Plus,
    Search,
    Loader2,
    Trash2,
} from "lucide-react"

// Vibrant colored badges like CodePilot
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

    if (minutes < 1) return 'now'
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString()
}

function getDateGroup(dateStr: string): string {
    const date = new Date(dateStr)
    const now = new Date()
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    const yesterday = new Date(today.getTime() - 86400000)
    const weekAgo = new Date(today.getTime() - 7 * 86400000)

    if (date >= today) return "TODAY"
    if (date >= yesterday) return "YESTERDAY"
    if (date >= weekAgo) return "THIS WEEK"
    return "OLDER"
}

export function PapersPanel() {
    const { papers, selectedPaperId, selectPaper, deletePaper, loadPapers } = useStudioStore()
    const [query, setQuery] = useState("")
    const [newPaperOpen, setNewPaperOpen] = useState(false)
    const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
    const [paperToDelete, setPaperToDelete] = useState<{ id: string; title: string; outputDir?: string } | null>(null)
    const [deleting, setDeleting] = useState(false)

    useEffect(() => {
        loadPapers()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const filteredPapers = useMemo(() => {
        const q = query.trim().toLowerCase()
        if (!q) return papers
        return papers.filter(p =>
            p.title.toLowerCase().includes(q) ||
            p.abstract.toLowerCase().includes(q)
        )
    }, [papers, query])

    const sortedPapers = useMemo(() => {
        return [...filteredPapers].sort((a, b) =>
            new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
        )
    }, [filteredPapers])

    // Group papers by date
    const groupedPapers = useMemo(() => {
        const groups: Record<string, typeof sortedPapers> = {}
        for (const paper of sortedPapers) {
            const group = getDateGroup(paper.updatedAt)
            if (!groups[group]) groups[group] = []
            groups[group].push(paper)
        }
        return groups
    }, [sortedPapers])

    const groupOrder = ["TODAY", "YESTERDAY", "THIS WEEK", "OLDER"]

    const handleDeleteClick = (e: React.MouseEvent, paper: typeof papers[0]) => {
        e.stopPropagation()
        setPaperToDelete({ id: paper.id, title: paper.title, outputDir: paper.outputDir })
        setDeleteConfirmOpen(true)
    }

    const handleConfirmDelete = async () => {
        if (!paperToDelete) return

        setDeleting(true)
        try {
            // Delete generated files if outputDir exists
            if (paperToDelete.outputDir) {
                try {
                    await deleteProjectFiles(paperToDelete.outputDir)
                } catch (e) {
                    console.error('Failed to delete project files:', e)
                }
            }

            // Delete paper from store
            deletePaper(paperToDelete.id)
        } finally {
            setDeleting(false)
            setDeleteConfirmOpen(false)
            setPaperToDelete(null)
        }
    }

    return (
        <div className="h-full w-full min-w-0 flex flex-col bg-muted/30 dark:bg-zinc-900/50 overflow-hidden">
            {/* Header */}
            <div className="px-4 h-12 flex items-center justify-between shrink-0">
                <span className="text-sm font-semibold">Papers</span>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setNewPaperOpen(true)}
                >
                    <Plus className="h-4 w-4" />
                </Button>
            </div>

            {/* Search */}
            <div className="px-3 pb-3">
                <div className="relative">
                    <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search papers..."
                        className="pl-9 h-9 bg-muted/50 dark:bg-zinc-800/50 border-0 rounded-lg focus-visible:ring-1"
                    />
                </div>
            </div>

            {/* Paper list */}
            <ScrollArea className="flex-1">
                <div className="px-2 pb-4">
                    {sortedPapers.length === 0 ? (
                        <div className="flex flex-col items-center justify-center text-muted-foreground py-12 px-4">
                            <p className="text-sm">
                                {query ? "No papers found" : "No papers yet"}
                            </p>
                            {!query && (
                                <Button
                                    variant="link"
                                    size="sm"
                                    className="mt-2"
                                    onClick={() => setNewPaperOpen(true)}
                                >
                                    <Plus className="h-4 w-4 mr-1" />
                                    Add paper
                                </Button>
                            )}
                        </div>
                    ) : (
                        groupOrder.map(group => {
                            const papersInGroup = groupedPapers[group]
                            if (!papersInGroup?.length) return null
                            return (
                                <div key={group}>
                                    <div className="px-2 py-2 text-[11px] font-medium text-muted-foreground/70 tracking-wide">
                                        {group}
                                    </div>
                                    {papersInGroup.map(paper => {
                                        const config = statusConfig[paper.status]
                                        const isSelected = selectedPaperId === paper.id
                                        const isLoading = paper.status === 'generating' || paper.status === 'running'

                                        return (
                                            <div
                                                key={paper.id}
                                                onClick={() => selectPaper(paper.id)}
                                                className={cn(
                                                    "group px-3 py-3 rounded-lg cursor-pointer transition-all mb-0.5 relative",
                                                    isSelected
                                                        ? "bg-muted/80 dark:bg-zinc-800/80"
                                                        : "hover:bg-muted/40 dark:hover:bg-zinc-800/40"
                                                )}
                                            >
                                                {/* Delete button - appears on hover */}
                                                <button
                                                    onClick={(e) => handleDeleteClick(e, paper)}
                                                    className="absolute right-2 top-2 p-1.5 rounded-md opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
                                                    title="Delete paper"
                                                >
                                                    <Trash2 className="h-3.5 w-3.5" />
                                                </button>

                                                <p className="text-sm font-medium leading-snug line-clamp-2 pr-6">
                                                    {paper.title || "Untitled"}
                                                </p>
                                                <div className="flex items-center gap-1.5 mt-1.5">
                                                    <span className={cn(
                                                        "text-[10px] font-medium px-1.5 py-0.5 rounded inline-flex items-center",
                                                        config.className
                                                    )}>
                                                        {isLoading && <Loader2 className="h-2.5 w-2.5 mr-1 animate-spin" />}
                                                        {config.label}
                                                    </span>
                                                    <span className="text-[10px] text-muted-foreground/60">·</span>
                                                    <span className="text-[11px] text-muted-foreground/60">
                                                        {formatRelativeTime(paper.updatedAt)}
                                                    </span>
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            )
                        })
                    )}
                </div>
            </ScrollArea>

            <NewPaperModal open={newPaperOpen} onOpenChange={setNewPaperOpen} />

            {/* Delete Confirmation Dialog */}
            <AlertDialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Paper</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete &quot;{paperToDelete?.title}&quot;?
                            {paperToDelete?.outputDir && (
                                <span className="block mt-2 text-destructive">
                                    This will also delete all generated code files.
                                </span>
                            )}
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleConfirmDelete}
                            disabled={deleting}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            {deleting ? (
                                <>
                                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
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
