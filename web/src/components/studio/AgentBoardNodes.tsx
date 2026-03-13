"use client"

import { useEffect, useState } from "react"
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Bot, Clock, Cpu, ExternalLink, FileCode, Loader2, Zap, CheckCircle2 } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AgentTask, AgentTaskLog, E2EState } from "@/lib/store/studio-store"

// ---------------------------------------------------------------------------
// Shared card wrapper (Flowith-style)
// ---------------------------------------------------------------------------

function FlowCard({
  children,
  className,
  onClick,
  selected,
  width,
}: {
  children: React.ReactNode
  className?: string
  onClick?: () => void
  selected?: boolean
  width?: number
}) {
  return (
    <div
      className={cn(
        "bg-white rounded-xl border border-zinc-100 p-4",
        "shadow-[0_2px_12px_rgba(0,0,0,0.06)]",
        "hover:shadow-[0_4px_16px_rgba(0,0,0,0.08)] transition-shadow",
        selected && "ring-2 ring-indigo-500/30",
        onClick && "cursor-pointer",
        className,
      )}
      style={width ? { width } : undefined}
      onClick={onClick}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                onClick()
              }
            }
          : undefined
      }
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 1. Commander Node
// ---------------------------------------------------------------------------

export type CommanderNodeData = {
  paperTitle: string
  action: string
  status: "idle" | "working" | "ready"
  onClick?: () => void
  emphasize?: boolean
  showExpandHint?: boolean
}

export function CommanderNode({ data }: NodeProps<Node<CommanderNodeData>>) {
  const isFocusCard = Boolean(data.emphasize)
  const dotColor =
    data.status === "working"
      ? "bg-blue-500 animate-pulse"
      : data.status === "ready"
        ? "bg-emerald-500"
        : "bg-zinc-300"

  return (
    <>
      <FlowCard
        width={isFocusCard ? 430 : 280}
        onClick={data.onClick}
        className={cn(isFocusCard && "border-indigo-200 ring-2 ring-indigo-100")}
      >
        <div className="flex items-center gap-2 mb-2">
          <Bot className={cn("text-indigo-500", isFocusCard ? "h-5 w-5" : "h-4 w-4")} />
          <span className={cn("font-semibold text-zinc-800", isFocusCard ? "text-base" : "text-sm")}>
            Claude Commander
          </span>
          <span className={cn("ml-auto h-2.5 w-2.5 rounded-full shrink-0", dotColor)} />
        </div>
        <p className="text-xs text-zinc-400 truncate">{data.paperTitle || "No paper selected"}</p>
        <p className="text-xs text-zinc-500 mt-1">{data.action}</p>
        {data.showExpandHint && (
          <p className="mt-3 text-xs text-indigo-600 font-medium">
            Click this card to expand all task cards
          </p>
        )}
      </FlowCard>
      <Handle type="source" position={Position.Bottom} className="!bg-zinc-300 !h-2 !w-2 !border-white !border-2" />
    </>
  )
}

// ---------------------------------------------------------------------------
// 2. Task Node — same content as current TaskCard
// ---------------------------------------------------------------------------

export type TaskNodeData = {
  task: AgentTask
  onClick: () => void
}

function taskStatusBadge(status: AgentTask["status"]) {
  if (status === "done" || status === "human_review") {
    return (
      <Badge variant="outline" className="text-[10px] bg-emerald-50 text-emerald-700 border-emerald-200">
        Done
      </Badge>
    )
  }
  if (status === "in_progress") {
    return (
      <Badge variant="outline" className="text-[10px] bg-blue-50 text-blue-700 border-blue-200">
        Running
      </Badge>
    )
  }
  if (status === "repairing") {
    return (
      <Badge variant="outline" className="text-[10px] bg-amber-50 text-amber-700 border-amber-200">
        Repairing
      </Badge>
    )
  }
  if (status === "paused") {
    return (
      <Badge variant="outline" className="text-[10px] bg-amber-50 text-amber-700 border-amber-200">
        Paused
      </Badge>
    )
  }
  if (status === "cancelled") {
    return (
      <Badge variant="outline" className="text-[10px] bg-zinc-100 text-zinc-400 border-zinc-200 line-through">
        Cancelled
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="text-[10px] bg-zinc-50 text-zinc-500 border-zinc-200">
      Planning
    </Badge>
  )
}

function computeCardStats(logs?: AgentTaskLog[]) {
  if (!logs || logs.length === 0) return null
  let filesChanged = 0
  let linesAdded = 0
  const seen = new Set<string>()
  for (const log of logs) {
    const bt = log.blockType ?? (log as unknown as Record<string, unknown>)["block_type"]
    const ev = log.event
    if (bt === "diff" || ev === "file_write" || ev === "file_edit") {
      const fp = log.details?.file_path as string | undefined
      if (fp && !seen.has(fp)) { seen.add(fp); filesChanged++ }
      linesAdded += (log.details?.lines_added as number) ?? 0
    }
  }
  return filesChanged > 0 ? { filesChanged, linesAdded } : null
}

export function TaskNode({ data }: NodeProps<Node<TaskNodeData>>) {
  const task = data.task
  const completedSubtasks = task.subtasks.filter((s) => s.done).length
  const totalSubtasks = task.subtasks.length
  const cardStats = computeCardStats(task.executionLog)

  const [relativeTime, setRelativeTime] = useState("")
  useEffect(() => {
    const compute = () => {
      const diff = Date.now() - new Date(task.updatedAt).getTime()
      const mins = Math.floor(diff / 60000)
      if (mins < 60) {
        setRelativeTime(`${mins}m ago`)
        return
      }
      const hours = Math.floor(mins / 60)
      if (hours < 24) {
        setRelativeTime(`${hours}h ago`)
        return
      }
      setRelativeTime(`${Math.floor(hours / 24)}d ago`)
    }
    compute()
    const id = setInterval(compute, 60_000)
    return () => clearInterval(id)
  }, [task.updatedAt])

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-zinc-300 !h-2 !w-2 !border-white !border-2" />
      <FlowCard width={260} onClick={data.onClick}>
        <div className="space-y-2">
          {/* Title + status badge */}
          <div className="flex items-start justify-between">
            <h4 className="text-sm font-medium leading-tight line-clamp-2 text-zinc-800">{task.title}</h4>
            <div className="flex gap-1 shrink-0 ml-2">{taskStatusBadge(task.status)}</div>
          </div>

          {/* Description */}
          {task.description && (
            <p className="text-xs text-zinc-400 line-clamp-2">{task.description}</p>
          )}

          {/* Tags */}
          {task.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {task.tags.map((tag) => (
                <Badge key={tag} variant="secondary" className="text-[10px]">
                  {tag}
                </Badge>
              ))}
            </div>
          )}

          {/* Progress bar */}
          {task.progress > 0 && (
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-zinc-400">
                <span>Progress</span>
                <span>{task.progress}%</span>
              </div>
              <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all",
                    task.progress >= 100 ? "bg-emerald-500" : "bg-indigo-500",
                  )}
                  style={{ width: `${Math.min(task.progress, 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Subtask dots */}
          {totalSubtasks > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              {task.subtasks.slice(0, 10).map((sub) => (
                <div
                  key={sub.id}
                  className={cn(
                    "h-2 w-2 rounded-full",
                    sub.done ? "bg-emerald-500" : "bg-zinc-200",
                  )}
                  title={sub.title}
                />
              ))}
              {totalSubtasks > 10 && (
                <span className="text-[10px] text-zinc-400">+{totalSubtasks - 10}</span>
              )}
              <span className="text-[10px] text-zinc-400 ml-1">
                {completedSubtasks}/{totalSubtasks}
              </span>
            </div>
          )}

          {/* Workspace stats */}
          {cardStats && (
            <div className="flex items-center gap-2 text-[10px] text-zinc-400 pt-0.5">
              <FileCode className="h-2.5 w-2.5" />
              <span>{cardStats.filesChanged} file{cardStats.filesChanged > 1 ? "s" : ""}</span>
              {cardStats.linesAdded > 0 && (
                <span className="text-emerald-600 font-mono">+{cardStats.linesAdded}</span>
              )}
            </div>
          )}

          {/* Assignee + time */}
          <div className="flex items-center justify-between text-[10px] text-zinc-400 pt-1">
            <div className="flex items-center gap-1">
              {task.assignee === "claude" ? (
                <Bot className="h-3 w-3" />
              ) : (
                <Cpu className="h-3 w-3" />
              )}
              <span>{task.assignee}</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="h-2.5 w-2.5" />
              <span>{relativeTime}</span>
            </div>
          </div>
        </div>
      </FlowCard>
      <Handle type="source" position={Position.Bottom} className="!bg-zinc-300 !h-2 !w-2 !border-white !border-2" />
    </>
  )
}

// ---------------------------------------------------------------------------
// 3. E2E Execution Node
// ---------------------------------------------------------------------------

export type E2ENodeData = {
  e2e: E2EState | null
  pipelinePhase?: string
  onClick: () => void
}

export function E2ENode({ data }: NodeProps<Node<E2ENodeData>>) {
  const e2e = data.e2e
  const isCancelled = data.pipelinePhase === "cancelled"

  const statusLabel =
    isCancelled && (!e2e || e2e.status === "waiting")
      ? "Cancelled"
      : !e2e || e2e.status === "waiting"
        ? "Waiting"
        : e2e.status === "running"
          ? "Running..."
          : e2e.status === "passed"
            ? "Passed"
            : e2e.status === "skipped"
              ? "Skipped"
              : e2e.status === "repairing"
                ? "Repairing..."
                : "Failed"

  const statusColor =
    isCancelled && (!e2e || e2e.status === "waiting")
      ? "text-zinc-400"
      : !e2e || e2e.status === "waiting"
        ? "text-zinc-400"
        : e2e.status === "running"
          ? "text-blue-600"
          : e2e.status === "passed"
            ? "text-emerald-600"
            : e2e.status === "skipped"
              ? "text-zinc-500"
              : e2e.status === "repairing"
                ? "text-amber-600"
                : "text-red-600"

  const dotColor =
    isCancelled && (!e2e || e2e.status === "waiting")
      ? "bg-zinc-300"
      : !e2e || e2e.status === "waiting"
        ? "bg-zinc-300"
        : e2e.status === "running"
          ? "bg-blue-500 animate-pulse"
          : e2e.status === "passed"
            ? "bg-emerald-500"
            : e2e.status === "skipped"
              ? "bg-zinc-400"
              : e2e.status === "repairing"
                ? "bg-amber-500 animate-pulse"
                : "bg-red-500"

  const stdoutLines = (e2e?.lastStdout || "").split("\n").filter(Boolean).slice(-3)

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-zinc-300 !h-2 !w-2 !border-white !border-2" />
      <FlowCard width={380} onClick={data.onClick}>
        <div className="space-y-2.5">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-indigo-500" />
            <span className="text-sm font-semibold text-zinc-800">End-to-End Execution</span>
            <span className={cn("ml-auto h-2.5 w-2.5 rounded-full shrink-0", dotColor)} />
          </div>

          {e2e && e2e.entryPoint && (
            <div className="text-xs text-zinc-500">
              <span className="text-zinc-400">Entry:</span>{" "}
              <code className="bg-zinc-50 px-1 py-0.5 rounded text-[11px]">{e2e.command || e2e.entryPoint}</code>
            </div>
          )}

          {e2e && e2e.maxAttempts > 0 && (
            <div className="text-xs text-zinc-500">
              Attempt: {e2e.attempt + 1} / {e2e.maxAttempts + 1}
            </div>
          )}

          {e2e && (e2e.status === "running" || e2e.status === "repairing") && (
            <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
              <div className="h-full bg-indigo-500 rounded-full animate-pulse w-2/3" />
            </div>
          )}

          {stdoutLines.length > 0 && (
            <div className="bg-zinc-900 rounded-lg p-2 text-[11px] font-mono text-zinc-300 max-h-16 overflow-hidden space-y-0.5">
              {stdoutLines.map((line, i) => (
                <div key={i} className="truncate">
                  <span className="text-zinc-500">{">"}</span> {line}
                </div>
              ))}
            </div>
          )}

          <div className={cn("text-xs font-medium flex items-center gap-1.5", statusColor)}>
            {e2e?.status === "running" && <Loader2 className="h-3 w-3 animate-spin" />}
            {e2e?.status === "passed" && <CheckCircle2 className="h-3 w-3" />}
            {statusLabel}
          </div>
        </div>
      </FlowCard>
      <Handle type="source" position={Position.Bottom} className="!bg-zinc-300 !h-2 !w-2 !border-white !border-2" />
    </>
  )
}

// ---------------------------------------------------------------------------
// 4. Download Node
// ---------------------------------------------------------------------------

export type OpenVSCodeNodeData = {
  directory: string
  pipelinePhase?: string
}

export function OpenVSCodeNode({ data }: NodeProps<Node<OpenVSCodeNodeData>>) {
  const isCancelled = data.pipelinePhase === "cancelled"
  const hasDir = Boolean(data.directory)
  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-zinc-300 !h-2 !w-2 !border-white !border-2" />
      <FlowCard width={320}>
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <ExternalLink className="h-4 w-4 text-indigo-500" />
            <span className="text-sm font-semibold text-zinc-800">Open in VS Code</span>
          </div>

          {hasDir ? (
            <>
              <p className="text-[11px] font-mono text-zinc-400 truncate">{data.directory}</p>
              <Button
                size="sm"
                className="w-full h-8 text-xs gap-1.5 bg-zinc-900 hover:bg-zinc-800 text-white"
                onClick={(e) => {
                  e.stopPropagation()
                  window.open(`vscode://file${data.directory}`, "_blank")
                }}
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Open Workspace
              </Button>
            </>
          ) : (
            <p className="text-xs text-zinc-400">
              {isCancelled ? "Pipeline cancelled" : "Waiting for workspace..."}
            </p>
          )}
        </div>
      </FlowCard>
    </>
  )
}

// ---------------------------------------------------------------------------
// Export node types for ReactFlow
// ---------------------------------------------------------------------------

export const nodeTypes = {
  commander: CommanderNode,
  task: TaskNode,
  e2e: E2ENode,
  download: OpenVSCodeNode,
}
