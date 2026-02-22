"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Folder, FolderOpen, Loader2, Circle, CheckCircle2 } from "lucide-react"
import { StudioPaper } from "@/lib/store/studio-store"
import { cn } from "@/lib/utils"

interface WorkspaceSetupDialogProps {
    paper: StudioPaper
    open: boolean
    onConfirm: (directory: string) => void
    onCancel: () => void
}

export function WorkspaceSetupDialog({
    paper,
    open,
    onConfirm,
    onCancel,
}: WorkspaceSetupDialogProps) {
    const [loading, setLoading] = useState(true)
    const [currentDir, setCurrentDir] = useState("")
    const [choice, setChoice] = useState<"current" | "custom">("current")
    const [customDir, setCustomDir] = useState("")
    const [error, setError] = useState<string | null>(null)

    // Fetch current working directory on mount
    useEffect(() => {
        if (open) {
            setLoading(true)
            setError(null)
            fetch("/api/studio/cwd")
                .then((res) => res.json())
                .then((data) => {
                    const dir = data.cwd || data.home || "/tmp"
                    setCurrentDir(dir)
                    // Suggest a paper-specific subdirectory
                    const safeName = paper.title
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, "-")
                        .slice(0, 40)
                    setCustomDir(`${dir}/${safeName}`)
                })
                .catch(() => {
                    setCurrentDir("/tmp")
                    setError("Could not detect current directory")
                })
                .finally(() => {
                    setLoading(false)
                })
        }
    }, [open, paper.title])

    const handleConfirm = () => {
        const directory = choice === "current" ? currentDir : customDir.trim()
        if (!directory) {
            setError("Please enter a valid directory path")
            return
        }
        onConfirm(directory)
    }

    const selectedDir = choice === "current" ? currentDir : customDir

    return (
        <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <FolderOpen className="h-5 w-5 text-blue-500" />
                        Set Up Workspace
                    </DialogTitle>
                    <DialogDescription>
                        Choose where to save generated code for &quot;{paper.title.slice(0, 50)}
                        {paper.title.length > 50 ? "..." : ""}&quot;
                    </DialogDescription>
                </DialogHeader>

                {loading ? (
                    <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                    </div>
                ) : (
                    <div className="space-y-4 py-4">
                        {error && (
                            <div className="text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 px-3 py-2 rounded-md">
                                {error}
                            </div>
                        )}

                        <div className="space-y-3">
                            {/* Option 1: Current directory */}
                            <button
                                type="button"
                                onClick={() => setChoice("current")}
                                className={cn(
                                    "w-full flex items-start gap-3 p-3 rounded-lg border text-left transition-colors",
                                    choice === "current"
                                        ? "border-primary bg-primary/5"
                                        : "border-border hover:bg-muted/50"
                                )}
                            >
                                {choice === "current" ? (
                                    <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                                ) : (
                                    <Circle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                                )}
                                <div className="flex-1 min-w-0">
                                    <Label className="font-medium cursor-pointer">
                                        Use current directory
                                    </Label>
                                    <div className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
                                        <Folder className="h-4 w-4 shrink-0" />
                                        <code className="bg-muted px-2 py-0.5 rounded text-xs break-all">
                                            {currentDir}
                                        </code>
                                    </div>
                                </div>
                            </button>

                            {/* Option 2: Custom directory */}
                            <button
                                type="button"
                                onClick={() => setChoice("custom")}
                                className={cn(
                                    "w-full flex items-start gap-3 p-3 rounded-lg border text-left transition-colors",
                                    choice === "custom"
                                        ? "border-primary bg-primary/5"
                                        : "border-border hover:bg-muted/50"
                                )}
                            >
                                {choice === "custom" ? (
                                    <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                                ) : (
                                    <Circle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                                )}
                                <div className="flex-1 min-w-0 space-y-2">
                                    <Label className="font-medium cursor-pointer">
                                        Choose a different directory
                                    </Label>
                                    <Input
                                        value={customDir}
                                        onChange={(e) => {
                                            setCustomDir(e.target.value)
                                            setChoice("custom")
                                        }}
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            setChoice("custom")
                                        }}
                                        placeholder="/path/to/project"
                                        className="font-mono text-sm"
                                    />
                                    <p className="text-xs text-muted-foreground">
                                        Tip: Use terminal <code className="bg-muted px-1 rounded">pwd</code> to get current path
                                    </p>
                                </div>
                            </button>
                        </div>

                        <div className="pt-2 border-t">
                            <p className="text-xs text-muted-foreground">
                                Generated code will be saved to:{" "}
                                <code className="bg-muted px-1 rounded">{selectedDir || "..."}</code>
                            </p>
                        </div>
                    </div>
                )}

                <DialogFooter>
                    <Button variant="outline" onClick={onCancel}>
                        Cancel
                    </Button>
                    <Button onClick={handleConfirm} disabled={loading || !selectedDir}>
                        Start Session
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}
