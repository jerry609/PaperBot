"use client"

import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { AgentTask, AgentTaskStatus } from "@/lib/store/studio-store"

type ColumnDef = {
  id: string
  label: string
  statuses: AgentTaskStatus[]
}

const COLUMNS: ColumnDef[] = [
  { id: "planned", label: "Planned", statuses: ["planning"] },
  { id: "in_progress", label: "In Progress", statuses: ["in_progress", "repairing"] },
  { id: "review", label: "Review", statuses: ["human_review"] },
  { id: "done", label: "Done", statuses: ["done"] },
  { id: "blocked", label: "Blocked", statuses: ["paused", "cancelled"] },
]

type AgentLabelResult = {
  label: string
  variant: "default" | "secondary" | "outline"
}

export function agentLabel(assignee: string): AgentLabelResult {
  if (!assignee || assignee === "claude") {
    return { label: "Claude Code", variant: "default" }
  }
  if (assignee.startsWith("codex-retry")) {
    return { label: "Codex (retry)", variant: "secondary" }
  }
  if (assignee.startsWith("codex")) {
    return { label: "Codex", variant: "secondary" }
  }
  return { label: assignee, variant: "outline" }
}

const CODEX_REASON_LABELS: Record<string, string> = {
  max_iterations_exhausted: "Iteration limit",
  stagnation_detected: "No progress",
  repeated_tool_calls: "Tool loop",
  too_many_tool_errors: "Too many errors",
  timeout: "Timeout",
  sandbox_crash: "Sandbox crash",
}

export function extractCodexFailureReason(task: AgentTask): string | null {
  if (task.executionLog) {
    for (let i = task.executionLog.length - 1; i >= 0; i--) {
      const entry = task.executionLog[i]
      if (entry.event === "task_failed") {
        const details = entry.details as Record<string, unknown> | undefined
        const diag = details?.codex_diagnostics as Record<string, unknown> | undefined
        const reasonCode = diag?.reason_code
        if (typeof reasonCode === "string") {
          return reasonCode
        }
      }
    }
  }
  if (task.lastError) return task.lastError
  return null
}

export function KanbanBoard({ tasks }: { tasks: AgentTask[] }) {
  return (
    <div className="flex h-full gap-3 overflow-x-auto px-3 py-3">
      {COLUMNS.map((col) => {
        const colTasks = tasks.filter((t) => col.statuses.includes(t.status))
        return (
          <div key={col.id} className="w-56 shrink-0 flex flex-col gap-2">
            {/* Column header */}
            <div className="flex items-center justify-between px-1">
              <span className="text-sm font-semibold">{col.label}</span>
              <Badge variant="outline" className="text-xs">
                {colTasks.length}
              </Badge>
            </div>

            {/* Column body */}
            <ScrollArea className="flex-1">
              <div className="flex flex-col gap-2">
                {colTasks.length === 0 ? (
                  <div className="text-xs text-muted-foreground text-center py-4">Empty</div>
                ) : (
                  colTasks.map((task) => {
                    const { label: agentLbl, variant: agentVariant } = agentLabel(task.assignee)
                    const failureReason = extractCodexFailureReason(task)
                    const reasonLabel =
                      failureReason && failureReason in CODEX_REASON_LABELS
                        ? CODEX_REASON_LABELS[failureReason]
                        : null

                    return (
                      <div
                        key={task.id}
                        className="rounded-md border bg-card p-2 flex flex-col gap-1.5"
                      >
                        {/* Task title */}
                        <p className="text-xs font-medium truncate" title={task.title}>
                          {task.title}
                        </p>

                        {/* Badges row */}
                        <div className="flex flex-wrap gap-1">
                          {/* Agent identity badge */}
                          <Badge variant={agentVariant} className="text-xs">
                            {agentLbl}
                          </Badge>

                          {/* Error badge */}
                          {task.lastError && (
                            <Badge variant="destructive" className="text-xs">
                              Error
                            </Badge>
                          )}

                          {/* Codex failure reason label */}
                          {reasonLabel && (
                            <span className="text-xs text-muted-foreground">{reasonLabel}</span>
                          )}
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </div>
        )
      })}
    </div>
  )
}
