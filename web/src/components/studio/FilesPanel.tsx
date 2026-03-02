"use client"

import { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useStudioStore } from "@/lib/store/studio-store"
import { useProjectContext } from "@/lib/store/project-context"
import { cn } from "@/lib/utils"
import { FileText, Folder, FolderOpen, RefreshCw, Search, ChevronRight, ChevronDown, FolderInput } from "lucide-react"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"

type FileIndexResponse = {
    project_dir: string
    files: string[]
    directories?: string[]
    truncated?: boolean
}

type PrepareProjectDirResponse = {
    project_dir: string
    created?: boolean
}

async function readErrorDetail(res: Response): Promise<string> {
    const cloned = res.clone()
    try {
        const data = await cloned.json() as { detail?: string }
        if (typeof data?.detail === "string" && data.detail.trim()) {
            return data.detail
        }
    } catch {
        // Fall back to plain text below.
    }
    try {
        const text = await res.text()
        if (text.trim()) return text
    } catch {
        // ignore
    }
    return `Request failed (${res.status})`
}

function languageForPath(path: string): string {
    const lower = path.toLowerCase()
    if (lower.endsWith(".py")) return "python"
    if (lower.endsWith(".ts") || lower.endsWith(".tsx")) return "typescript"
    if (lower.endsWith(".js") || lower.endsWith(".jsx")) return "javascript"
    if (lower.endsWith(".json")) return "json"
    if (lower.endsWith(".yaml") || lower.endsWith(".yml")) return "yaml"
    if (lower.endsWith(".md")) return "markdown"
    if (lower.endsWith(".toml")) return "toml"
    if (lower.endsWith(".txt")) return "plaintext"
    if (lower.endsWith(".sh")) return "shell"
    return "plaintext"
}

interface FileTreeNode {
    name: string
    path: string
    isDirectory: boolean
    children: FileTreeNode[]
}

function buildFileTree(files: string[]): FileTreeNode[] {
    const root: FileTreeNode[] = []
    const nodeMap = new Map<string, FileTreeNode>()

    for (const filePath of files) {
        const parts = filePath.split('/')
        let currentPath = ''
        let currentLevel = root

        for (let i = 0; i < parts.length; i++) {
            const part = parts[i]
            const isLast = i === parts.length - 1
            currentPath = currentPath ? `${currentPath}/${part}` : part

            let node = nodeMap.get(currentPath)
            if (!node) {
                node = {
                    name: part,
                    path: currentPath,
                    isDirectory: !isLast,
                    children: [],
                }
                nodeMap.set(currentPath, node)
                currentLevel.push(node)
            }
            currentLevel = node.children
        }
    }

    const sortNodes = (nodes: FileTreeNode[]) => {
        nodes.sort((a, b) => {
            if (a.isDirectory !== b.isDirectory) return a.isDirectory ? -1 : 1
            return a.name.localeCompare(b.name)
        })
        nodes.forEach(n => sortNodes(n.children))
    }
    sortNodes(root)
    return root
}

interface FileTreeItemProps {
    node: FileTreeNode
    depth: number
    activeFile: string | null
    expandedDirs: Set<string>
    onToggleDir: (path: string) => void
    onSelectFile: (path: string) => void
}

function FileTreeItem({ node, depth, activeFile, expandedDirs, onToggleDir, onSelectFile }: FileTreeItemProps) {
    const isExpanded = expandedDirs.has(node.path)
    const isActive = activeFile === node.path

    if (node.isDirectory) {
        return (
            <div>
                <button
                    onClick={() => onToggleDir(node.path)}
                    className="w-full flex items-center gap-1.5 px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors rounded"
                    style={{ paddingLeft: `${depth * 16 + 8}px` }}
                >
                    {isExpanded ? (
                        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    ) : (
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    )}
                    {isExpanded ? (
                        <FolderOpen className="h-4 w-4 text-blue-500 shrink-0" />
                    ) : (
                        <Folder className="h-4 w-4 text-blue-500 shrink-0" />
                    )}
                    <span className="truncate">{node.name}</span>
                </button>
                {isExpanded && node.children.map(child => (
                    <FileTreeItem
                        key={child.path}
                        node={child}
                        depth={depth + 1}
                        activeFile={activeFile}
                        expandedDirs={expandedDirs}
                        onToggleDir={onToggleDir}
                        onSelectFile={onSelectFile}
                    />
                ))}
            </div>
        )
    }

    return (
        <button
            onClick={() => onSelectFile(node.path)}
            className={cn(
                "w-full flex items-center gap-1.5 px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors rounded",
                isActive && "bg-primary/10 text-primary"
            )}
            style={{ paddingLeft: `${depth * 16 + 8}px` }}
            title={node.path}
        >
            <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
            <span className="truncate">{node.name}</span>
        </button>
    )
}

function truncatePath(path: string, maxLength: number = 30): string {
    if (path.length <= maxLength) return path
    const parts = path.split('/')
    if (parts.length <= 2) return '...' + path.slice(-maxLength + 3)
    return parts[0] + '/.../' + parts.slice(-2).join('/')
}

export function FilesPanel() {
    const { papers, selectedPaperId, lastGenCodeResult, updatePaper } = useStudioStore()
    const selectedPaper = useMemo(() =>
        selectedPaperId ? papers.find(p => p.id === selectedPaperId) ?? null : null,
        [papers, selectedPaperId]
    )

    const projectDir = selectedPaper?.outputDir || lastGenCodeResult?.outputDir || ""
    const { files, activeFile, addFile, setActiveFile } = useProjectContext()
    const openFiles = useMemo(() => Object.values(files), [files])

    const [fileIndex, setFileIndex] = useState<string[]>([])
    const [loadingIndex, setLoadingIndex] = useState(false)
    const [query, setQuery] = useState("")
    const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set())
    const [dirDialogOpen, setDirDialogOpen] = useState(false)
    const [newDirPath, setNewDirPath] = useState("")
    const [dirError, setDirError] = useState<string | null>(null)
    const [indexError, setIndexError] = useState<string | null>(null)
    const [settingDir, setSettingDir] = useState(false)

    const filteredFiles = useMemo(() => {
        const q = query.trim().toLowerCase()
        if (!q) return fileIndex
        return fileIndex.filter((p) => p.toLowerCase().includes(q))
    }, [fileIndex, query])

    const fileTree = useMemo(() => buildFileTree(filteredFiles), [filteredFiles])

    const refreshIndex = async () => {
        if (!projectDir) return
        setLoadingIndex(true)
        setIndexError(null)
        try {
            const res = await fetch(`/api/runbook/files?project_dir=${encodeURIComponent(projectDir)}&recursive=true`)
            if (!res.ok) {
                const detail = await readErrorDetail(res)
                setFileIndex([])
                setIndexError(detail)
                return
            }
            const data = (await res.json()) as FileIndexResponse
            setFileIndex(data.files || [])
            const firstLevelDirs = new Set<string>()
            for (const f of data.files || []) {
                const firstPart = f.split('/')[0]
                if (f.includes('/')) firstLevelDirs.add(firstPart)
            }
            setExpandedDirs(firstLevelDirs)
        } catch {
            setFileIndex([])
        } finally {
            setLoadingIndex(false)
        }
    }

    useEffect(() => {
        setFileIndex([])
        setQuery("")
        setExpandedDirs(new Set())
        setIndexError(null)
        if (projectDir) refreshIndex()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectDir])

    const openFile = async (path: string) => {
        if (!projectDir) return
        try {
            const res = await fetch(`/api/runbook/file?project_dir=${encodeURIComponent(projectDir)}&path=${encodeURIComponent(path)}`)
            if (!res.ok) return
            const data = (await res.json()) as { path: string; content: string }
            addFile(data.path, data.content, languageForPath(data.path))
        } catch {
            // ignore
        }
    }

    const toggleDir = (path: string) => {
        setExpandedDirs(prev => {
            const next = new Set(prev)
            if (next.has(path)) next.delete(path)
            else next.add(path)
            return next
        })
    }

    const handleSetDirectory = async () => {
        if (!selectedPaperId || !newDirPath.trim()) return
        setSettingDir(true)
        setDirError(null)
        try {
            const res = await fetch("/api/runbook/project-dir/prepare", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    project_dir: newDirPath.trim(),
                    create_if_missing: true,
                }),
            })
            if (!res.ok) {
                const detail = await readErrorDetail(res)
                setDirError(detail)
                return
            }
            const data = (await res.json()) as PrepareProjectDirResponse
            updatePaper(selectedPaperId, { outputDir: data.project_dir || newDirPath.trim() })
            setDirDialogOpen(false)
            setNewDirPath("")
            setDirError(null)
        } catch {
            setDirError("Failed to validate directory. Please try again.")
        } finally {
            setSettingDir(false)
        }
    }

    const openDirDialog = () => {
        setNewDirPath(projectDir || "")
        setDirError(null)
        setDirDialogOpen(true)
    }

    return (
        <div className="h-full min-w-0 min-h-0 bg-muted/30 dark:bg-zinc-900/50 flex flex-col overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 shrink-0 border-b">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-muted-foreground tracking-wide">FILES</span>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={refreshIndex}
                        disabled={!projectDir || loadingIndex}
                    >
                        <RefreshCw className={cn("h-3.5 w-3.5", loadingIndex && "animate-spin")} />
                    </Button>
                </div>

                {/* Directory Path - Clickable */}
                <button
                    onClick={openDirDialog}
                    disabled={!selectedPaperId}
                    className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-md transition-colors text-left disabled:opacity-50 disabled:cursor-not-allowed"
                    title={projectDir || "Click to set output directory"}
                >
                    <FolderInput className="h-3.5 w-3.5 shrink-0" />
                    <span className="truncate flex-1">
                        {projectDir ? truncatePath(projectDir) : "Set output directory..."}
                    </span>
                </button>
            </div>

            {/* Search */}
            <div className="px-3 py-2">
                <div className="relative">
                    <Search className="absolute left-2.5 top-2 h-4 w-4 text-muted-foreground" />
                    <Input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Filter files..."
                        className="pl-8 h-8 bg-muted/50 dark:bg-zinc-800/50 border-0 rounded-lg focus-visible:ring-1"
                        disabled={!projectDir}
                    />
                </div>
                {indexError && (
                    <p className="mt-2 text-xs text-destructive">{indexError}</p>
                )}
            </div>

            {/* File Tree */}
            <ScrollArea className="flex-1 min-h-0">
                <div className="px-2 pb-4">
                    {projectDir ? (
                        fileTree.length === 0 ? (
                            <div className="text-sm text-muted-foreground text-center py-8">
                                {loadingIndex ? "Loading..." : "No files yet"}
                            </div>
                        ) : (
                            fileTree.map(node => (
                                <FileTreeItem
                                    key={node.path}
                                    node={node}
                                    depth={0}
                                    activeFile={activeFile}
                                    expandedDirs={expandedDirs}
                                    onToggleDir={toggleDir}
                                    onSelectFile={openFile}
                                />
                            ))
                        )
                    ) : openFiles.length > 0 ? (
                        openFiles.map((file) => (
                            <button
                                key={file.name}
                                className={cn(
                                    "w-full flex items-center gap-1.5 px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors rounded",
                                    activeFile === file.name && "bg-primary/10 text-primary"
                                )}
                                onClick={() => setActiveFile(file.name)}
                            >
                                <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                                <span className="truncate">{file.name.split('/').pop()}</span>
                            </button>
                        ))
                    ) : (
                        <div className="flex flex-col items-center justify-center text-muted-foreground py-12 px-4 text-center">
                            <Folder className="h-10 w-10 mb-3 opacity-20" />
                            <p className="text-sm">Set a directory or generate code</p>
                        </div>
                    )}
                </div>
            </ScrollArea>

            {/* Directory Selection Dialog */}
            <Dialog open={dirDialogOpen} onOpenChange={setDirDialogOpen}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle>Set Output Directory</DialogTitle>
                        <DialogDescription>
                            Enter the local directory path where generated code will be saved for this paper.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-2">
                        <div className="space-y-2">
                            <Label htmlFor="dir-path">Directory Path</Label>
                            <Input
                                id="dir-path"
                                value={newDirPath}
                                onChange={(e) => setNewDirPath(e.target.value)}
                                placeholder="/Users/username/projects/paper-code"
                                className="font-mono text-sm"
                            />
                            <p className="text-xs text-muted-foreground">
                                This directory will be used to save generated code and can be accessed by Claude CLI.
                            </p>
                            {dirError && (
                                <p className="text-xs text-destructive">{dirError}</p>
                            )}
                        </div>
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDirDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button onClick={handleSetDirectory} disabled={!newDirPath.trim() || !selectedPaperId || settingDir}>
                            {settingDir ? "Validating..." : "Set Directory"}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    )
}
