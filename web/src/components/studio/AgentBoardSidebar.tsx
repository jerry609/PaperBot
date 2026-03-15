"use client"

import { useCallback, useEffect, useState } from "react"
import { useStudioStore, type SandboxFileEntry } from "@/lib/store/studio-store"
import { ChevronRight, File, Folder, Clock, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { backendUrl } from "@/lib/backend-url"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

// ---------------------------------------------------------------------------
// File tree
// ---------------------------------------------------------------------------

function FileTreeItem({
  entry,
  parentPath = "",
  depth = 0,
  onFileClick,
}: {
  entry: SandboxFileEntry
  parentPath?: string
  depth?: number
  onFileClick?: (path: string) => void
}) {
  const [expanded, setExpanded] = useState(depth < 1)
  const fullPath = parentPath ? `${parentPath}/${entry.name}` : entry.name

  if (entry.type === "directory") {
    return (
      <div>
        <button
          className="flex items-center gap-1 w-full text-left py-0.5 hover:bg-zinc-100 rounded px-1 transition-colors"
          style={{ paddingLeft: depth * 12 + 4 }}
          onClick={() => setExpanded(!expanded)}
        >
          <ChevronRight
            className={cn(
              "h-3 w-3 text-zinc-400 transition-transform shrink-0",
              expanded && "rotate-90",
            )}
          />
          <Folder className="h-3 w-3 text-zinc-400 shrink-0" />
          <span className="text-[11px] text-zinc-600 truncate">{entry.name}</span>
        </button>
        {expanded && entry.children && (
          <div>
            {entry.children.map((child) => (
              <FileTreeItem
                key={child.name}
                entry={child}
                parentPath={fullPath}
                depth={depth + 1}
                onFileClick={onFileClick}
              />
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <button
      className="flex items-center gap-1 w-full text-left py-0.5 hover:bg-zinc-100 rounded px-1 transition-colors"
      style={{ paddingLeft: depth * 12 + 18 }}
      onClick={() => onFileClick?.(fullPath)}
    >
      <File className="h-3 w-3 text-zinc-400 shrink-0" />
      <span className="text-[11px] text-zinc-600 truncate">{entry.name}</span>
    </button>
  )
}

function countFiles(entries: SandboxFileEntry[]): number {
  let count = 0
  for (const e of entries) {
    if (e.type === "file") count++
    if (e.children) count += countFiles(e.children)
  }
  return count
}

// ---------------------------------------------------------------------------
// File viewer dialog
// ---------------------------------------------------------------------------

type FileViewerResult =
  | { content: string; error?: undefined }
  | { content?: undefined; error: string }

function useSandboxFileContent(
  sessionId: string | null,
  filePath: string | null,
  open: boolean,
) {
  // Track the fetch key to derive "loading" without synchronous setState
  const [requestKey, setRequestKey] = useState<string | null>(null)
  const [result, setResult] = useState<FileViewerResult | null>(null)

  const currentKey = open && filePath && sessionId ? `${sessionId}:${filePath}` : null

  // When the dialog opens (or file changes), bump the request key.
  // This is called from an event handler via handleFileClick, not from an effect,
  // so the state is already set before the effect runs.
  const loading = currentKey !== null && (requestKey !== currentKey || result === null)

  useEffect(() => {
    if (!currentKey || !filePath || !sessionId) return

    setRequestKey(currentKey)
    setResult(null)

    const controller = new AbortController()

    fetch(
      backendUrl(
        `/api/agent-board/sessions/${sessionId}/sandbox/file?path=${encodeURIComponent(filePath)}`
      ),
      { signal: controller.signal },
    )
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text()
          throw new Error(text || `Failed to load file (${res.status})`)
        }
        return res.json() as Promise<{ path: string; content: string }>
      })
      .then((data) => {
        setResult({ content: data.content })
      })
      .catch((err) => {
        if (controller.signal.aborted) return
        setResult({ error: err instanceof Error ? err.message : "Failed to load file" })
      })

    return () => controller.abort()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentKey])

  return { loading, result }
}

function SandboxFileViewerDialog({
  filePath,
  open,
  onOpenChange,
}: {
  filePath: string | null
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const boardSessionId = useStudioStore((s) => s.boardSessionId)
  const { loading, result } = useSandboxFileContent(boardSessionId, filePath, open)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-sm font-mono font-medium truncate">
            {filePath}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 min-h-0 overflow-auto rounded-md border bg-zinc-50">
          {loading && (
            <div className="flex items-center justify-center py-12 text-zinc-400 text-sm">
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
              Loading...
            </div>
          )}
          {!loading && result?.error && (
            <div className="p-4 text-sm text-red-600">{result.error}</div>
          )}
          {!loading && result?.content !== undefined && (
            <pre className="p-4 text-xs leading-relaxed font-mono text-zinc-700 whitespace-pre-wrap break-words">
              {result.content}
            </pre>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Sandbox files section
// ---------------------------------------------------------------------------

function SandboxFilesSection() {
  const sandboxFiles = useStudioStore((s) => s.sandboxFiles)
  const fileCount = countFiles(sandboxFiles)
  const [viewerPath, setViewerPath] = useState<string | null>(null)
  const [viewerOpen, setViewerOpen] = useState(false)

  const handleFileClick = useCallback((path: string) => {
    setViewerPath(path)
    setViewerOpen(true)
  }, [])

  if (sandboxFiles.length === 0) {
    return (
      <div className="px-3 py-4">
        <div className="flex items-center gap-2 mb-3">
          <Folder className="h-3.5 w-3.5 text-zinc-400" />
          <span className="text-xs font-medium text-zinc-600">Sandbox Files</span>
        </div>
        <p className="text-[11px] text-zinc-400">No files generated yet.</p>
      </div>
    )
  }

  return (
    <div className="px-3 py-4">
      <div className="flex items-center gap-2 mb-3">
        <Folder className="h-3.5 w-3.5 text-zinc-400" />
        <span className="text-xs font-medium text-zinc-600">Sandbox Files</span>
        <span className="text-[10px] text-zinc-400 ml-auto">{fileCount} files</span>
      </div>
      <div className="max-h-[300px] overflow-y-auto space-y-0">
        {sandboxFiles.map((entry) => (
          <FileTreeItem
            key={entry.name}
            entry={entry}
            onFileClick={handleFileClick}
          />
        ))}
      </div>
      <SandboxFileViewerDialog
        filePath={viewerPath}
        open={viewerOpen}
        onOpenChange={setViewerOpen}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Time estimate
// ---------------------------------------------------------------------------

function formatDuration(ms: number): string {
  if (ms <= 0) return "0s"
  const secs = Math.floor(ms / 1000)
  const mins = Math.floor(secs / 60)
  const remainSecs = secs % 60
  if (mins === 0) return `${remainSecs}s`
  return `${mins}m ${remainSecs}s`
}

function TimeEstimateSection() {
  const pipelinePhase = useStudioStore((s) => s.pipelinePhase)
  const agentTasks = useStudioStore((s) => s.agentTasks)
  const [elapsed, setElapsed] = useState(0)
  const [initialNow] = useState(() => Date.now())

  // Find the earliest task start time
  const startTime = agentTasks.reduce((earliest, t) => {
    const ts = new Date(t.createdAt).getTime()
    return ts < earliest ? ts : earliest
  }, initialNow)

  const completedTasks = agentTasks.filter((t) => t.status === "done" || t.status === "human_review").length
  const totalTasks = agentTasks.length

  useEffect(() => {
    if (pipelinePhase === "idle" || pipelinePhase === "completed" || pipelinePhase === "failed" || pipelinePhase === "cancelled" || totalTasks === 0) return

    const id = setInterval(() => {
      setElapsed(Date.now() - startTime)
    }, 1000)
    return () => clearInterval(id)
  }, [pipelinePhase, startTime, totalTasks])

  if (totalTasks === 0) {
    return (
      <div className="px-3 py-4 border-t border-zinc-100">
        <div className="flex items-center gap-2 mb-3">
          <Clock className="h-3.5 w-3.5 text-zinc-400" />
          <span className="text-xs font-medium text-zinc-600">Time Estimate</span>
        </div>
        <p className="text-[11px] text-zinc-400">No tasks yet.</p>
      </div>
    )
  }

  const avgTaskMs = completedTasks > 0 ? elapsed / completedTasks : null
  const remainingTasks = totalTasks - completedTasks
  const remainingMs = avgTaskMs ? avgTaskMs * remainingTasks : null
  const progressPct = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0

  const isRunning = pipelinePhase !== "idle" && pipelinePhase !== "completed" && pipelinePhase !== "failed" && pipelinePhase !== "cancelled"

  return (
    <div className="px-3 py-4 border-t border-zinc-100">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="h-3.5 w-3.5 text-zinc-400" />
        <span className="text-xs font-medium text-zinc-600">Time Estimate</span>
        {isRunning && <Loader2 className="h-3 w-3 text-zinc-400 animate-spin ml-auto" />}
      </div>

      {/* Progress bar */}
      <div className="mb-3">
        <div className="flex justify-between text-[10px] text-zinc-400 mb-1">
          <span>
            {completedTasks}/{totalTasks} tasks
          </span>
          <span>{progressPct}%</span>
        </div>
        <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      <div className="space-y-1.5 text-[11px]">
        <div className="flex justify-between">
          <span className="text-zinc-400">Elapsed</span>
          <span className="text-zinc-600 font-mono">{formatDuration(elapsed)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-400">Remaining</span>
          <span className="text-zinc-600 font-mono">
            {remainingMs !== null ? `~${formatDuration(remainingMs)}` : "Calculating..."}
          </span>
        </div>
        {avgTaskMs && (
          <div className="flex justify-between">
            <span className="text-zinc-400">Avg/task</span>
            <span className="text-zinc-600 font-mono">~{formatDuration(avgTaskMs)}</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sidebar wrapper
// ---------------------------------------------------------------------------

export function AgentBoardSidebar({
  backgroundColor = "#f3f3f2",
  className,
}: {
  backgroundColor?: string
  className?: string
}) {
  return (
    <div
      className={cn(
        "overflow-y-auto border-zinc-200",
        className ?? "w-[260px] shrink-0 border-r",
      )}
      style={{ background: backgroundColor }}
    >
      <SandboxFilesSection />
      <TimeEstimateSection />
    </div>
  )
}
