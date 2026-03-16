"use client"

import type { CodexDelegationEntry, ToolCallEntry } from "./types"

export type SubagentActivityStatus = "queued" | "running" | "completed" | "failed"

export type SubagentActivityGroup = {
  id: string
  taskId: string
  workerRunId: string
  taskTitle: string
  assignee: string
  sessionId: string
  runtime: string
  controlMode: "mirrored" | "managed"
  interruptible: boolean
  status: SubagentActivityStatus
  startedAt: string
  updatedAt: string
  finishedAt: string | null
  filesGenerated: string[]
  reasonCode: string | null
  error: string | null
  eventCount: number
  toolCount: number
  toolErrorCount: number
  recentTools: ToolCallEntry[]
}

function toMillis(ts: string): number {
  const parsed = new Date(ts).getTime()
  return Number.isFinite(parsed) ? parsed : 0
}

function resolveStatus(
  latestEventType: CodexDelegationEntry["event_type"],
  toolCount: number,
): SubagentActivityStatus {
  if (latestEventType === "codex_failed") return "failed"
  if (latestEventType === "codex_completed") return "completed"
  if (latestEventType === "codex_accepted") return "running"
  return toolCount > 0 ? "running" : "queued"
}

export function buildSubagentActivityGroups(
  delegations: CodexDelegationEntry[],
  toolCalls: ToolCallEntry[],
): SubagentActivityGroup[] {
  const groups = new Map<string, CodexDelegationEntry[]>()

  for (const entry of delegations) {
    const key = entry.task_id || entry.id
    const current = groups.get(key)
    if (current) {
      current.push(entry)
    } else {
      groups.set(key, [entry])
    }
  }

  return Array.from(groups.values())
    .map((entries) => {
      const sortedEvents = [...entries].sort((a, b) => toMillis(a.ts) - toMillis(b.ts))
      const first = sortedEvents[0]
      const latest = sortedEvents[sortedEvents.length - 1]
      const startedAtMs = toMillis(first.ts)
      const finishedAt =
        latest.event_type === "codex_completed" || latest.event_type === "codex_failed"
          ? latest.ts
          : null
      const finishedAtMs = finishedAt ? toMillis(finishedAt) : Number.POSITIVE_INFINITY

      const relatedTools = toolCalls
        .filter((tool) => {
          if (tool.agent_name !== latest.assignee) return false
          const toolTs = toMillis(tool.ts)
          return toolTs >= startedAtMs && toolTs <= finishedAtMs
        })
        .sort((a, b) => toMillis(b.ts) - toMillis(a.ts))

      return {
        id: latest.task_id || latest.id,
        taskId: latest.task_id,
        workerRunId: latest.worker_run_id,
        taskTitle: latest.task_title,
        assignee: latest.assignee,
        sessionId: latest.session_id,
        runtime: latest.runtime,
        controlMode: latest.control_mode,
        interruptible: latest.interruptible,
        status: resolveStatus(latest.event_type, relatedTools.length),
        startedAt: first.ts,
        updatedAt: latest.ts,
        finishedAt,
        filesGenerated: latest.files_generated ?? [],
        reasonCode: latest.reason_code ?? null,
        error: latest.error ?? null,
        eventCount: sortedEvents.length,
        toolCount: relatedTools.length,
        toolErrorCount: relatedTools.filter((tool) => tool.status === "error").length,
        recentTools: relatedTools.slice(0, 3),
      } satisfies SubagentActivityGroup
    })
    .sort((a, b) => toMillis(b.updatedAt) - toMillis(a.updatedAt))
}
