"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  ArrowDown,
  ArrowUp,
  Brain,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Cpu,
  FolderOpen,
  Info,
  Loader2,
  Pin,
  PinOff,
  Wrench,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { getAgentPresentation, isCommanderAssignee } from "@/lib/agent-runtime"
import type { AgentTask, AgentTaskLog } from "@/lib/store/studio-store"
import {
  buildTaskTimelineEntries,
  filterTaskLogs,
  inferTaskLogBlockType,
  type TaskTimelineDisplayMode,
  type TaskTimelineEntry,
  type TaskTimelineFilterMode,
} from "@/lib/studio-task-log-groups"
import { InfoBlock, ThinkBlock, ToolBlock, DiffBlock, ResultBlock } from "./blocks"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusBadge(status: AgentTask["status"]) {
  const config = {
    done: { label: "Done", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
    in_progress: { label: "Running", className: "bg-blue-50 text-blue-700 border-blue-200" },
    repairing: { label: "Repairing", className: "bg-amber-50 text-amber-700 border-amber-200" },
    human_review: { label: "Done", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
    paused: { label: "Paused", className: "bg-amber-50 text-amber-700 border-amber-200" },
    cancelled: { label: "Cancelled", className: "bg-zinc-100 text-zinc-400 border-zinc-200" },
    planning: { label: "Planning", className: "bg-zinc-50 text-zinc-500 border-zinc-200" },
  } as const
  const c = config[status] ?? config.planning
  return (
    <Badge variant="outline" className={cn("text-[10px]", c.className)}>
      {c.label}
    </Badge>
  )
}

function formatElapsed(createdAt: string): string {
  const ms = Date.now() - new Date(createdAt).getTime()
  const secs = Math.floor(ms / 1000)
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m ${secs % 60}s`
  const hrs = Math.floor(mins / 60)
  return `${hrs}h ${mins % 60}m`
}

/** Count workspace stats from execution logs */
function computeWorkspaceStats(logs: AgentTaskLog[]) {
  let filesChanged = 0
  let linesAdded = 0
  const seenFiles = new Set<string>()

  for (const log of logs) {
    const bt = inferTaskLogBlockType(log)
    if (bt === "diff") {
      const fp = log.details?.file_path as string | undefined
      if (fp && !seenFiles.has(fp)) {
        seenFiles.add(fp)
        filesChanged++
      }
      linesAdded += (log.details?.lines_added as number) ?? 0
    }
  }
  return { filesChanged, linesAdded, fileNames: Array.from(seenFiles) }
}

function renderTimelineLog(log: AgentTaskLog, onFileClick?: (path: string) => void) {
  const bt = inferTaskLogBlockType(log)
  switch (bt) {
    case "think":
      return <ThinkBlock key={log.id} log={log} />
    case "tool":
      return <ToolBlock key={log.id} log={log} />
    case "diff":
      return <DiffBlock key={log.id} log={log} onFileClick={onFileClick} />
    case "result":
      return <ResultBlock key={log.id} log={log} />
    default:
      return <InfoBlock key={log.id} log={log} />
  }
}

function groupAppearance(entry: Extract<TaskTimelineEntry, { kind: "group" }>) {
  if (entry.blockType === "think") {
    return {
      icon: Brain,
      panelClass: "border-indigo-200 bg-indigo-50/70",
      iconClass: "text-indigo-600",
      titleClass: "text-indigo-900",
      metaClass: "text-indigo-600",
    }
  }
  if (entry.blockType === "tool") {
    return {
      icon: Wrench,
      panelClass:
        entry.status === "error"
          ? "border-rose-200 bg-rose-50/70"
          : "border-slate-200 bg-[#f8faf6]",
      iconClass: entry.status === "error" ? "text-rose-600" : "text-slate-600",
      titleClass: entry.status === "error" ? "text-rose-900" : "text-slate-900",
      metaClass: entry.status === "error" ? "text-rose-600" : "text-slate-500",
    }
  }
  return {
    icon: Info,
    panelClass: "border-slate-200 bg-white",
    iconClass: "text-slate-500",
    titleClass: "text-slate-900",
    metaClass: "text-slate-500",
  }
}

function GroupedTimelineEntry({
  entry,
  onFileClick,
}: {
  entry: Extract<TaskTimelineEntry, { kind: "group" }>
  onFileClick?: (path: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const appearance = groupAppearance(entry)
  const Icon = appearance.icon

  return (
    <div className="px-3 py-2">
      <div className={cn("overflow-hidden rounded-2xl border shadow-[0_1px_0_rgba(255,255,255,0.75)_inset]", appearance.panelClass)}>
        <button
          type="button"
          className="flex w-full items-start gap-3 px-3 py-3 text-left"
          onClick={() => setExpanded((current) => !current)}
        >
          <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-2xl border border-white/70 bg-white/80">
            <Icon className={cn("h-4 w-4", appearance.iconClass)} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className={cn("text-[12px] font-semibold", appearance.titleClass)}>{entry.title}</span>
              <span className={cn("rounded-full border border-current/15 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.12em]", appearance.metaClass)}>
                {entry.logs.length}
              </span>
            </div>
            {entry.preview.length > 0 ? (
              <div className="mt-1 space-y-1">
                {entry.preview.map((line) => (
                  <p key={line} className="text-[11px] leading-4 text-slate-600">
                    {line}
                  </p>
                ))}
              </div>
            ) : null}
            {entry.toolNames.length > 0 ? (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {entry.toolNames.slice(0, 4).map((toolName) => (
                  <span
                    key={toolName}
                    className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[10px] font-medium text-slate-600"
                  >
                    {toolName}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
          <div className="shrink-0 rounded-full border border-slate-200 bg-white/80 p-1 text-slate-500">
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </div>
        </button>

        {expanded ? (
          <div className="border-t border-slate-200/70 bg-white/75">
            <div className="divide-y divide-slate-100">
              {entry.logs.map((log) => renderTimelineLog(log, onFileClick))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Layer 1: Task Overview
// ---------------------------------------------------------------------------

function TaskOverview({
  task,
  onOpenWorkspace,
}: {
  task: AgentTask
  onOpenWorkspace: () => void
}) {
  const logs = task.executionLog ?? []
  const stats = computeWorkspaceStats(logs)
  const completedSubtasks = task.subtasks.filter((s) => s.done).length

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5 space-y-5">
      {/* Description */}
      {task.description && (
        <section>
          <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
            Description
          </h3>
          <div className="rounded-lg border bg-zinc-50/50 p-3 text-sm text-zinc-700 leading-relaxed">
            {task.description}
          </div>
        </section>
      )}

      {/* Subtasks */}
      {task.subtasks.length > 0 && (
        <section>
          <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
            Subtasks
            <span className="ml-2 text-zinc-400 normal-case">
              {completedSubtasks}/{task.subtasks.length}
            </span>
          </h3>
          <div className="space-y-1.5">
            {task.subtasks.map((sub) => (
              <div
                key={sub.id}
                className="flex items-center gap-2.5 rounded-md border px-3 py-2"
              >
                <span
                  className={cn(
                    "h-2.5 w-2.5 rounded-full shrink-0",
                    sub.done ? "bg-emerald-500" : "bg-zinc-300",
                  )}
                />
                <span className={cn("text-sm", sub.done ? "text-zinc-500 line-through" : "text-zinc-700")}>
                  {sub.title}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Workspace Card */}
      <section>
        <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
          Workspace
        </h3>
        <button
          className="w-full rounded-lg border border-zinc-200 bg-white p-4 text-left hover:border-indigo-300 hover:shadow-sm transition-all group"
          onClick={onOpenWorkspace}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded-lg bg-indigo-50 flex items-center justify-center">
                <FolderOpen className="h-4 w-4 text-indigo-500" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-zinc-800">
                    {stats.filesChanged > 0
                      ? `${stats.filesChanged} file${stats.filesChanged > 1 ? "s" : ""} changed`
                      : "No files yet"}
                  </span>
                  {stats.linesAdded > 0 && (
                    <span className="text-xs font-mono text-emerald-600">+{stats.linesAdded}</span>
                  )}
                </div>
                <p className="text-[11px] text-zinc-400 mt-0.5">
                  {logs.length > 0
                    ? `${logs.length} execution steps · Click to review the summary timeline`
                    : "No execution steps yet"}
                </p>
              </div>
            </div>
            <ChevronRight className="h-4 w-4 text-zinc-300 group-hover:text-indigo-400 transition-colors" />
          </div>
        </button>
      </section>

      {/* AI Review */}
      {task.reviewFeedback && (
        <section>
          <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
            AI Review
          </h3>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50/50 p-3">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 mt-0.5 shrink-0" />
              <p className="text-xs text-emerald-800 whitespace-pre-wrap">{task.reviewFeedback}</p>
            </div>
          </div>
        </section>
      )}

      {/* Error */}
      {task.lastError && (
        <section>
          <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
            Error
          </h3>
          <div className="rounded-lg border border-red-200 bg-red-50/50 p-3 text-xs text-red-700">
            {task.lastError}
          </div>
        </section>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Layer 2: Workspace / Thinking Process View
// ---------------------------------------------------------------------------

function ThinkingTimeline({
  logs,
  filter,
  displayMode,
  onFileClick,
}: {
  logs: AgentTaskLog[]
  filter: TaskTimelineFilterMode
  displayMode: TaskTimelineDisplayMode
  onFileClick?: (path: string) => void
}) {
  const filtered = useMemo(() => filterTaskLogs(logs, filter), [logs, filter])
  const entries = useMemo(
    () => (displayMode === "summary" ? buildTaskTimelineEntries(filtered) : []),
    [displayMode, filtered],
  )

  return (
    <div>
      <div className="divide-y divide-zinc-100">
        {filtered.length === 0 && (
          <div className="py-12 text-center text-sm text-zinc-400">
            {logs.length === 0 ? "No execution steps yet." : "No matching blocks."}
          </div>
        )}
        {displayMode === "summary"
          ? entries.map((entry) =>
              entry.kind === "group" ? (
                <GroupedTimelineEntry key={entry.id} entry={entry} onFileClick={onFileClick} />
              ) : (
                renderTimelineLog(entry.log, onFileClick)
              ),
            )
          : filtered.map((log) => renderTimelineLog(log, onFileClick))}
      </div>
    </div>
  )
}

function WorkspaceView({
  task,
  onBack,
  onFileClick,
}: {
  task: AgentTask
  onBack: () => void
  onFileClick?: (path: string) => void
}) {
  const [filter, setFilter] = useState<TaskTimelineFilterMode>("all")
  const [displayMode, setDisplayMode] = useState<TaskTimelineDisplayMode>("summary")
  const [autoScroll, setAutoScroll] = useState(true)
  const logs = useMemo(() => task.executionLog ?? [], [task.executionLog])
  const scrollRef = useRef<HTMLDivElement>(null)
  const prevLogCount = useRef(logs.length)
  const filteredLogs = useMemo(() => filterTaskLogs(logs, filter), [logs, filter])
  const summaryEntries = useMemo(() => buildTaskTimelineEntries(filteredLogs), [filteredLogs])

  const scrollToTop = useCallback(() => {
    scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" })
  }, [])

  const scrollToBottom = useCallback(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [])

  // Auto-scroll when new logs arrive
  useEffect(() => {
    if (autoScroll && logs.length > prevLogCount.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
    prevLogCount.current = logs.length
  }, [logs.length, autoScroll])

  // Block counts for status bar
  const blockCounts = logs.reduce(
    (acc, log) => {
      const bt = inferTaskLogBlockType(log)
      acc[bt] = (acc[bt] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* Header bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b bg-zinc-50/80 shrink-0">
        <button
          className="text-xs text-zinc-500 hover:text-zinc-800 flex items-center gap-1 transition-colors"
          onClick={onBack}
        >
          <ChevronRight className="h-3 w-3 rotate-180" />
          Back
        </button>
        <span className="text-xs text-zinc-300">/</span>
        <span className="text-xs font-medium text-zinc-600 truncate">Workspace</span>

        <div className="ml-auto flex items-center gap-1.5">
          <div className="inline-flex items-center rounded-md border border-zinc-200 bg-white p-0.5">
            {(["summary", "raw"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setDisplayMode(mode)}
                className={cn(
                  "rounded px-2 py-1 text-[10px] font-medium uppercase tracking-[0.12em] transition-colors",
                  displayMode === mode
                    ? "bg-zinc-900 text-white"
                    : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700",
                )}
              >
                {mode}
              </button>
            ))}
          </div>
          {(["all", "diffs", "thinking"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                "text-[10px] px-2 py-1 rounded-md transition-colors capitalize",
                filter === f
                  ? "bg-indigo-100 text-indigo-700"
                  : "text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600",
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto"
        onScroll={(e) => {
          const el = e.currentTarget
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
          if (atBottom !== autoScroll) setAutoScroll(atBottom)
        }}
      >
        <ThinkingTimeline logs={logs} filter={filter} displayMode={displayMode} onFileClick={onFileClick} />
      </div>

      {/* Bottom action bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-t bg-zinc-50/80 shrink-0">
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-[11px] gap-1.5"
          onClick={scrollToTop}
          title="Scroll to top"
        >
          <ArrowUp className="h-3 w-3" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-[11px] gap-1.5"
          onClick={scrollToBottom}
          title="Scroll to bottom"
        >
          <ArrowDown className="h-3 w-3" />
        </Button>
        <Button
          variant={autoScroll ? "default" : "outline"}
          size="sm"
          className={cn("h-7 text-[11px] gap-1.5", autoScroll && "bg-indigo-600 hover:bg-indigo-700")}
          onClick={() => setAutoScroll(!autoScroll)}
          title={autoScroll ? "Auto-scroll ON" : "Auto-scroll OFF"}
        >
          {autoScroll ? <Pin className="h-3 w-3" /> : <PinOff className="h-3 w-3" />}
        </Button>

        <div className="flex-1" />

        {/* Block type counts */}
        <div className="flex items-center gap-3 text-[10px] text-zinc-400">
          {displayMode === "summary" ? (
            <span title="Summary entries">{summaryEntries.length} entries</span>
          ) : (
            <span title="Raw blocks">{filteredLogs.length} blocks</span>
          )}
          {blockCounts.think ? (
            <span title="Thinking blocks">{blockCounts.think} think</span>
          ) : null}
          {blockCounts.tool ? (
            <span title="Tool calls">{blockCounts.tool} tool</span>
          ) : null}
          {blockCounts.diff ? (
            <span title="File changes" className="text-emerald-600">{blockCounts.diff} diff</span>
          ) : null}
          <span className="text-zinc-300">|</span>
          <span>{logs.length} steps</span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Dialog: Two-layer navigation
// ---------------------------------------------------------------------------

type Layer = "overview" | "workspace"

export function TaskDetailPanel({
  task,
  open,
  onOpenChange,
  onFileClick,
}: {
  task: AgentTask | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onFileClick?: (path: string) => void
}) {
  const [layer, setLayer] = useState<Layer>("overview")
  const [prevTaskId, setPrevTaskId] = useState<string | null>(null)

  // Reset to overview when dialog opens with a new task (no setState in effect)
  if (open && task?.id && task.id !== prevTaskId) {
    setPrevTaskId(task.id)
    if (layer !== "overview") setLayer("overview")
  }

  if (!task) return null

  const assigneePresentation = getAgentPresentation(task.assignee)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[97vw] max-w-[97vw] sm:max-w-5xl h-[88vh] p-0 overflow-hidden !gap-0">
        <div className="h-full min-h-0 flex flex-col">
          {/* Header */}
          <DialogHeader className="px-5 py-4 border-b shrink-0">
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1 min-w-0">
                <DialogTitle className="text-base leading-tight truncate">
                  {task.title}
                </DialogTitle>
                <DialogDescription className="text-xs flex items-center gap-2">
                  <span className="flex items-center gap-1">
                    {isCommanderAssignee(task.assignee) ? (
                      <Bot className="h-3 w-3" />
                    ) : (
                      <Cpu className="h-3 w-3" />
                    )}
                    <span title={task.assignee}>{assigneePresentation.label}</span>
                  </span>
                  <span className="text-zinc-300">·</span>
                  <span>{task.progress}%</span>
                  <span className="text-zinc-300">·</span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatElapsed(task.createdAt)}
                  </span>
                </DialogDescription>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {statusBadge(task.status)}
                {task.status === "in_progress" && (
                  <Loader2 className="h-3.5 w-3.5 text-blue-500 animate-spin" />
                )}
              </div>
            </div>

            {/* Progress bar */}
            {task.progress > 0 && task.progress < 100 && (
              <div className="mt-3">
                <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 rounded-full transition-all"
                    style={{ width: `${task.progress}%` }}
                  />
                </div>
              </div>
            )}
          </DialogHeader>

          {/* Body — switches between Layer 1 and Layer 2 */}
          {layer === "overview" ? (
            <TaskOverview
              task={task}
              onOpenWorkspace={() => setLayer("workspace")}
            />
          ) : (
            <WorkspaceView
              task={task}
              onBack={() => setLayer("overview")}
              onFileClick={onFileClick}
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
