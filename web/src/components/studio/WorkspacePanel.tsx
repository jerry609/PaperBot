"use client"

import { useEffect, useMemo, useState } from "react"
import { DeepCodeEditor } from "@/components/studio/DeepCodeEditor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useStudioStore } from "@/lib/store/studio-store"
import { useProjectContext } from "@/lib/store/project-context"
import { cn } from "@/lib/utils"
import { FileText, Folder, RefreshCw, Save, Search, Camera, GitCompare, Undo2 } from "lucide-react"
import { DiffModal } from "@/components/studio/DiffViewer"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"

type FileIndexResponse = {
  project_dir: string
  files: string[]
  directories?: string[]
  truncated?: boolean
}

type ChangesResponse = {
  changed: string[]
  unchanged: string[]
  added: string[]
  removed: string[]
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

export function WorkspacePanel() {
  const { lastGenCodeResult, workspaceSnapshotId, setWorkspaceSnapshotId } = useStudioStore()
  const projectDir = lastGenCodeResult?.outputDir || ""
  const { files, activeFile, addFile } = useProjectContext()

  const [fileIndex, setFileIndex] = useState<string[]>([])
  const [loadingIndex, setLoadingIndex] = useState(false)
  const [query, setQuery] = useState("")
  const [saving, setSaving] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)
  const [diffOpen, setDiffOpen] = useState(false)
  const [diffOld, setDiffOld] = useState("")
  const [diffNew, setDiffNew] = useState("")
  const [diffFile, setDiffFile] = useState<string | undefined>(undefined)
  const [changesOpen, setChangesOpen] = useState(false)
  const [changes, setChanges] = useState<ChangesResponse | null>(null)
  const [loadingChanges, setLoadingChanges] = useState(false)

  const filteredFiles = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return fileIndex
    return fileIndex.filter((p) => p.toLowerCase().includes(q))
  }, [fileIndex, query])

  const refreshIndex = async () => {
    if (!projectDir) return
    setLoadingIndex(true)
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/files?project_dir=${encodeURIComponent(projectDir)}&recursive=true`)
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to list files (${res.status}): ${text}`)
      }
      const data = (await res.json()) as FileIndexResponse
      setFileIndex(data.files || [])
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
      setFileIndex([])
    } finally {
      setLoadingIndex(false)
    }
  }

  useEffect(() => {
    setFileIndex([])
    setQuery("")
    setLastError(null)
    if (projectDir) {
      refreshIndex()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectDir])

  const openFile = async (path: string) => {
    if (!projectDir) return
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/file?project_dir=${encodeURIComponent(projectDir)}&path=${encodeURIComponent(path)}`)
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to read file (${res.status}): ${text}`)
      }
      const data = (await res.json()) as { path: string; content: string }
      addFile(data.path, data.content, languageForPath(data.path))
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    }
  }

  const diffFor = async (path: string) => {
    if (!projectDir || !workspaceSnapshotId) return
    setLastError(null)
    try {
      const res = await fetch(
        `/api/runbook/diff?snapshot_id=${encodeURIComponent(String(workspaceSnapshotId))}&project_dir=${encodeURIComponent(projectDir)}&path=${encodeURIComponent(path)}`
      )
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to diff (${res.status}): ${text}`)
      }
      const data = (await res.json()) as { old: string; new: string; path: string }
      setDiffOld(data.old || "")
      setDiffNew(files[path]?.content ?? data.new ?? "")
      setDiffFile(data.path)
      setDiffOpen(true)
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    }
  }

  const saveActive = async () => {
    if (!projectDir || !activeFile) return
    const vf = files[activeFile]
    if (!vf) return

    setSaving(true)
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_dir: projectDir, path: vf.name, content: vf.content }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to save (${res.status}): ${text}`)
      }
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const createBaseline = async () => {
    if (!projectDir) return
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/snapshots`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_dir: projectDir, label: "baseline" }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to create snapshot (${res.status}): ${text}`)
      }
      const data = (await res.json()) as { snapshot_id: number }
      setWorkspaceSnapshotId(data.snapshot_id)
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    }
  }

  const revertFileToBaseline = async (path: string) => {
    if (!projectDir || !workspaceSnapshotId) return
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/revert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ snapshot_id: workspaceSnapshotId, project_dir: projectDir, path }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to revert (${res.status}): ${text}`)
      }
      // reload file from disk
      await openFile(path)
      setDiffOpen(false)
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    }
  }

  const deleteFile = async (path: string) => {
    if (!projectDir) return
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_dir: projectDir, path }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to delete (${res.status}): ${text}`)
      }
      await refreshIndex()
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    }
  }

  const loadChanges = async () => {
    if (!projectDir || !workspaceSnapshotId) return
    setLoadingChanges(true)
    setLastError(null)
    try {
      const res = await fetch(
        `/api/runbook/changes?snapshot_id=${encodeURIComponent(String(workspaceSnapshotId))}&project_dir=${encodeURIComponent(projectDir)}`
      )
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to compute changes (${res.status}): ${text}`)
      }
      const data = (await res.json()) as ChangesResponse
      setChanges(data)
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
      setChanges(null)
    } finally {
      setLoadingChanges(false)
    }
  }

  const openChanges = async () => {
    setChangesOpen(true)
    await loadChanges()
  }

  const revertAll = async () => {
    if (!projectDir || !workspaceSnapshotId) return
    setLastError(null)
    try {
      const res = await fetch(`/api/runbook/revert-project`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ snapshot_id: workspaceSnapshotId, project_dir: projectDir, delete_added: true }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Failed to revert project (${res.status}): ${text}`)
      }
      await refreshIndex()
      setChangesOpen(false)
    } catch (e) {
      setLastError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="h-full min-w-0 min-h-0 bg-background flex flex-col">
      <div className="border-b px-4 py-3 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold">Workspace</div>
          <div className="text-xs text-muted-foreground truncate">
            {projectDir ? `Project: ${projectDir}` : "Run Paper2Code to create a project directory."}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={createBaseline}
            disabled={!projectDir}
            title="Create a baseline snapshot for diff/revert"
          >
            <Camera className="h-3.5 w-3.5 mr-2" />
            Baseline
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={() => activeFile ? diffFor(activeFile) : undefined}
            disabled={!projectDir || !activeFile || !workspaceSnapshotId}
            title="Compare active file against baseline"
          >
            <GitCompare className="h-3.5 w-3.5 mr-2" />
            Diff
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={() => activeFile ? revertFileToBaseline(activeFile) : undefined}
            disabled={!projectDir || !activeFile || !workspaceSnapshotId}
            title="Revert active file to baseline"
          >
            <Undo2 className="h-3.5 w-3.5 mr-2" />
            Revert
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={openChanges}
            disabled={!projectDir || !workspaceSnapshotId}
            title="View changed/added/removed files against baseline"
          >
            <GitCompare className="h-3.5 w-3.5 mr-2" />
            Changes
          </Button>
          <Button variant="outline" size="sm" className="h-8 text-xs" onClick={refreshIndex} disabled={!projectDir || loadingIndex}>
            <RefreshCw className={cn("h-3.5 w-3.5 mr-2", loadingIndex && "animate-spin")} />
            Refresh
          </Button>
          <Button variant="default" size="sm" className="h-8 text-xs" onClick={saveActive} disabled={!projectDir || !activeFile || saving}>
            <Save className="h-3.5 w-3.5 mr-2" />
            Save
          </Button>
        </div>
      </div>

      {lastError && (
        <div className="border-b bg-red-50 dark:bg-red-950/20 px-4 py-2 text-xs text-red-700 dark:text-red-300">
          {lastError}
        </div>
      )}

      <div className="flex-1 min-h-0 grid grid-cols-[260px_1fr]">
        {/* File Explorer */}
        <div className="border-r bg-muted/5 min-w-0 min-h-0 flex flex-col">
          <div className="p-3 border-b bg-background">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search files…"
                className="pl-8 h-9"
                disabled={!projectDir}
              />
            </div>
            <div className="mt-2 text-[11px] text-muted-foreground flex items-center justify-between">
              <span>{filteredFiles.length} files</span>
              {loadingIndex && <span>indexing…</span>}
            </div>
          </div>

          <ScrollArea className="flex-1 min-h-0">
            <div className="p-2 space-y-1">
              {!projectDir ? (
                <div className="p-3 text-xs text-muted-foreground">No project loaded.</div>
              ) : filteredFiles.length === 0 ? (
                <div className="p-3 text-xs text-muted-foreground">No files.</div>
              ) : (
                filteredFiles.map((p) => (
                  <button
                    key={p}
                    onClick={() => openFile(p)}
                    className={cn(
                      "w-full flex items-center gap-2 px-2.5 py-2 rounded-md text-left text-xs hover:bg-muted/60 transition-colors",
                      activeFile === p && "bg-muted ring-1 ring-border"
                    )}
                    title={p}
                  >
                    {p.includes("/") ? <Folder className="h-3.5 w-3.5 text-muted-foreground" /> : <FileText className="h-3.5 w-3.5 text-muted-foreground" />}
                    <span className="truncate">{p}</span>
                  </button>
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Editor */}
        <div className="min-w-0 min-h-0">
          <DeepCodeEditor />
        </div>
      </div>

      <DiffModal
        isOpen={diffOpen}
        oldValue={diffOld}
        newValue={diffNew}
        filename={diffFile}
        onClose={() => setDiffOpen(false)}
        onReject={() => setDiffOpen(false)}
        onApply={() => diffFile ? revertFileToBaseline(diffFile) : undefined}
        applyLabel="Revert"
        rejectLabel="Keep"
      />

      <Dialog open={changesOpen} onOpenChange={setChangesOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Changes vs Baseline</DialogTitle>
            <DialogDescription>
              Snapshot: {workspaceSnapshotId ?? "—"} {projectDir ? `• ${projectDir}` : ""}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs text-muted-foreground">
                {loadingChanges ? "Computing changes…" : changes ? (
                  <>
                    {changes.changed.length} changed • {changes.added.length} added • {changes.removed.length} removed
                  </>
                ) : "No data"}
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" className="h-8 text-xs" onClick={loadChanges} disabled={loadingChanges}>
                  <RefreshCw className={cn("h-3.5 w-3.5 mr-2", loadingChanges && "animate-spin")} />
                  Refresh
                </Button>
                <Button variant="destructive" size="sm" className="h-8 text-xs" onClick={revertAll} disabled={!changes}>
                  <Undo2 className="h-3.5 w-3.5 mr-2" />
                  Revert All
                </Button>
              </div>
            </div>

            {changes && (
              <ScrollArea className="h-[420px] border rounded-md">
                <div className="p-2 space-y-3">
                  {changes.changed.length > 0 && (
                    <div>
                      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">Changed</div>
                      <div className="space-y-1">
                        {changes.changed.map((p) => (
                          <div key={p} className="flex items-center justify-between gap-2 px-2 py-1 rounded hover:bg-muted/50">
                            <button className="text-xs font-mono truncate text-left" onClick={() => openFile(p)} title={p}>
                              {p}
                            </button>
                            <div className="flex items-center gap-2 shrink-0">
                              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => diffFor(p)}>
                                Diff
                              </Button>
                              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => revertFileToBaseline(p)}>
                                Revert
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {changes.added.length > 0 && (
                    <div>
                      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">Added</div>
                      <div className="space-y-1">
                        {changes.added.map((p) => (
                          <div key={p} className="flex items-center justify-between gap-2 px-2 py-1 rounded hover:bg-muted/50">
                            <button className="text-xs font-mono truncate text-left" onClick={() => openFile(p)} title={p}>
                              {p}
                            </button>
                            <div className="flex items-center gap-2 shrink-0">
                              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => deleteFile(p)}>
                                Delete
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {changes.removed.length > 0 && (
                    <div>
                      <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">Removed</div>
                      <div className="space-y-1">
                        {changes.removed.map((p) => (
                          <div key={p} className="flex items-center justify-between gap-2 px-2 py-1 rounded hover:bg-muted/50">
                            <div className="text-xs font-mono truncate" title={p}>
                              {p}
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => revertFileToBaseline(p)}>
                                Restore
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setChangesOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
