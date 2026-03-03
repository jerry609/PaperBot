"use client"

import { useState, useEffect } from "react"
import { useStudioStore } from "@/lib/store/studio-store"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChevronDown, ChevronRight, Loader2, FileText, Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface LibraryPaper {
    id: string
    title: string
    abstract?: string
    authors?: string[]
    venue?: string
}

interface NewPaperModalProps {
    open: boolean
    onOpenChange: (open: boolean) => void
}

export function NewPaperModal({ open, onOpenChange }: NewPaperModalProps) {
    const { addPaper, papers } = useStudioStore()

    // Manual entry state
    const [title, setTitle] = useState("")
    const [abstract, setAbstract] = useState("")
    const [methodSection, setMethodSection] = useState("")
    const [showAdvanced, setShowAdvanced] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Library state
    const [libraryPapers, setLibraryPapers] = useState<LibraryPaper[]>([])
    const [loadingLibrary, setLoadingLibrary] = useState(false)
    const [selectedPaperIds, setSelectedPaperIds] = useState<Set<string>>(new Set())
    const [searchQuery, setSearchQuery] = useState("")

    const canSubmit = title.trim().length > 0 && abstract.trim().length > 0
    const canImport = selectedPaperIds.size > 0

    // Fetch library papers when modal opens
    useEffect(() => {
        if (open) {
            fetchLibraryPapers()
        }
    }, [open])

    const fetchLibraryPapers = async () => {
        setLoadingLibrary(true)
        try {
            const res = await fetch('/api/papers/library')
            if (res.ok) {
                const data = await res.json()
                const papers = (data.papers || []).map((item: { paper: Record<string, unknown> }) => ({
                    id: String(item.paper.id || item.paper.paper_id),
                    title: String(item.paper.title || "Untitled"),
                    abstract: String(item.paper.abstract || ""),
                    authors: Array.isArray(item.paper.authors) ? item.paper.authors : [],
                    venue: String(item.paper.venue || ""),
                }))
                setLibraryPapers(papers)
            }
        } catch (e) {
            console.error('Failed to fetch library papers:', e)
        } finally {
            setLoadingLibrary(false)
        }
    }

    const togglePaperSelection = (paperId: string) => {
        setSelectedPaperIds(prev => {
            const next = new Set(prev)
            if (next.has(paperId)) {
                next.delete(paperId)
            } else {
                next.add(paperId)
            }
            return next
        })
    }

    const handleManualSubmit = () => {
        if (!canSubmit) {
            setError("Title and Abstract are required.")
            return
        }

        const normalizedTitle = title.trim().toLowerCase()
        const duplicate = papers.find(p => p.title.trim().toLowerCase() === normalizedTitle)
        if (duplicate) {
            setError("A paper with this title already exists in the studio.")
            return
        }

        addPaper({
            title: title.trim(),
            abstract: abstract.trim(),
            methodSection: methodSection.trim() || undefined,
        })

        resetAndClose()
    }

    const handleImportSelected = () => {
        const existingTitles = new Set(papers.map(p => p.title.trim().toLowerCase()))

        let importedCount = 0
        const skippedTitles: string[] = []

        for (const paperId of selectedPaperIds) {
            const paper = libraryPapers.find(p => p.id === paperId)
            if (!paper) continue

            const normalizedTitle = paper.title.trim().toLowerCase()

            // Skip if already in studio (by title)
            if (existingTitles.has(normalizedTitle)) {
                skippedTitles.push(paper.title)
                continue
            }

            addPaper({
                title: paper.title,
                abstract: paper.abstract || "",
            })
            existingTitles.add(normalizedTitle)
            importedCount++
        }

        if (importedCount === 0 && skippedTitles.length > 0) {
            setError(
                skippedTitles.length === 1
                    ? "This paper already exists in the studio."
                    : "The selected papers already exist in the studio."
            )
            return
        }

        resetAndClose()
    }

    const resetAndClose = () => {
        setTitle("")
        setAbstract("")
        setMethodSection("")
        setShowAdvanced(false)
        setError(null)
        setSelectedPaperIds(new Set())
        setSearchQuery("")
        onOpenChange(false)
    }

    const filteredLibraryPapers = libraryPapers.filter(paper => {
        if (!searchQuery.trim()) return true
        const q = searchQuery.toLowerCase()
        return paper.title.toLowerCase().includes(q) ||
            (paper.abstract?.toLowerCase().includes(q)) ||
            (paper.authors?.some(a => a.toLowerCase().includes(q)))
    })

    // Check which papers are already in studio
    const studioPaperTitles = new Set(papers.map(p => p.title.trim().toLowerCase()))

    return (
        <Dialog open={open} onOpenChange={resetAndClose}>
            <DialogContent className="sm:max-w-lg">
                <DialogHeader>
                    <DialogTitle>Add Paper</DialogTitle>
                    <DialogDescription>
                        Import from your library or enter manually.
                    </DialogDescription>
                </DialogHeader>

                <Tabs defaultValue="library" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="library">From Library</TabsTrigger>
                        <TabsTrigger value="manual">Manual Entry</TabsTrigger>
                    </TabsList>

                    <TabsContent value="library" className="mt-4">
                        <div className="space-y-3">
                            {error && (
                                <div className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
                                    {error}
                                </div>
                            )}
                            <Input
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Search papers..."
                                className="h-9"
                            />

                            <ScrollArea className="h-[280px] border rounded-md">
                                {loadingLibrary ? (
                                    <div className="flex items-center justify-center h-full py-12">
                                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                                    </div>
                                ) : filteredLibraryPapers.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center h-full py-12 text-muted-foreground">
                                        <FileText className="h-8 w-8 mb-2 opacity-30" />
                                        <p className="text-sm">
                                            {searchQuery ? "No papers found" : "No papers in library"}
                                        </p>
                                    </div>
                                ) : (
                                    <div className="p-2 space-y-1">
                                        {filteredLibraryPapers.map(paper => {
                                            const isSelected = selectedPaperIds.has(paper.id)
                                            const isInStudio = studioPaperTitles.has(
                                                paper.title.trim().toLowerCase(),
                                            )

                                            return (
                                                <button
                                                    key={paper.id}
                                                    onClick={() => !isInStudio && togglePaperSelection(paper.id)}
                                                    disabled={isInStudio}
                                                    className={cn(
                                                        "w-full text-left px-3 py-2.5 rounded-md transition-colors",
                                                        isInStudio
                                                            ? "opacity-50 cursor-not-allowed bg-muted/30"
                                                            : isSelected
                                                                ? "bg-primary/10 border border-primary/30"
                                                                : "hover:bg-muted/50"
                                                    )}
                                                >
                                                    <div className="flex items-start gap-2">
                                                        <div className={cn(
                                                            "w-4 h-4 mt-0.5 rounded border shrink-0 flex items-center justify-center",
                                                            isSelected ? "bg-primary border-primary" : "border-muted-foreground/30"
                                                        )}>
                                                            {isSelected && <Check className="h-3 w-3 text-primary-foreground" />}
                                                        </div>
                                                        <div className="flex-1 min-w-0">
                                                            <p className="text-sm font-medium leading-tight line-clamp-2">
                                                                {paper.title}
                                                            </p>
                                                            {paper.authors && paper.authors.length > 0 && (
                                                                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                                                                    {paper.authors.slice(0, 3).join(", ")}
                                                                    {paper.authors.length > 3 && " et al."}
                                                                </p>
                                                            )}
                                                            {isInStudio && (
                                                                <span className="text-[10px] text-muted-foreground">
                                                                    Already in studio
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                </button>
                                            )
                                        })}
                                    </div>
                                )}
                            </ScrollArea>

                            {selectedPaperIds.size > 0 && (
                                <p className="text-xs text-muted-foreground">
                                    {selectedPaperIds.size} paper{selectedPaperIds.size > 1 ? "s" : ""} selected
                                </p>
                            )}
                        </div>

                        <DialogFooter className="mt-4">
                            <Button variant="outline" onClick={resetAndClose}>
                                Cancel
                            </Button>
                            <Button onClick={handleImportSelected} disabled={!canImport}>
                                Import Selected
                            </Button>
                        </DialogFooter>
                    </TabsContent>

                    <TabsContent value="manual" className="mt-4">
                        <div className="space-y-4">
                            {error && (
                                <div className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
                                    {error}
                                </div>
                            )}

                            <div className="space-y-2">
                                <Label htmlFor="paper-title" className="text-sm">
                                    Title <span className="text-destructive">*</span>
                                </Label>
                                <Input
                                    id="paper-title"
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                    placeholder="Enter paper title"
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="paper-abstract" className="text-sm">
                                    Abstract <span className="text-destructive">*</span>
                                </Label>
                                <Textarea
                                    id="paper-abstract"
                                    value={abstract}
                                    onChange={(e) => setAbstract(e.target.value)}
                                    placeholder="Paste the paper abstract"
                                    className="min-h-[100px]"
                                />
                            </div>

                            <div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-7 px-2 text-xs -ml-2"
                                    onClick={() => setShowAdvanced(!showAdvanced)}
                                >
                                    {showAdvanced ? (
                                        <ChevronDown className="h-3.5 w-3.5 mr-1" />
                                    ) : (
                                        <ChevronRight className="h-3.5 w-3.5 mr-1" />
                                    )}
                                    Advanced Options
                                </Button>
                            </div>

                            {showAdvanced && (
                                <div className="space-y-2">
                                    <Label htmlFor="paper-method" className="text-sm">
                                        Method Section <span className="text-muted-foreground text-xs">(optional)</span>
                                    </Label>
                                    <Textarea
                                        id="paper-method"
                                        value={methodSection}
                                        onChange={(e) => setMethodSection(e.target.value)}
                                        placeholder="Optionally paste the methodology section"
                                        className="min-h-[80px]"
                                    />
                                </div>
                            )}
                        </div>

                        <DialogFooter className="mt-4">
                            <Button variant="outline" onClick={resetAndClose}>
                                Cancel
                            </Button>
                            <Button onClick={handleManualSubmit} disabled={!canSubmit}>
                                Add Paper
                            </Button>
                        </DialogFooter>
                    </TabsContent>
                </Tabs>
            </DialogContent>
        </Dialog>
    )
}
