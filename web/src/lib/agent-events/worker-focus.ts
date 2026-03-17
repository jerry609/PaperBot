"use client"

import { buildSubagentActivityGroups, type SubagentActivityGroup } from "./subagent-groups"
import type {
  ActivityFeedItem,
  CodexDelegationEntry,
  FileTouchedEntry,
  ToolCallEntry,
} from "./types"

function toMillis(ts: string): number {
  const value = new Date(ts).getTime()
  return Number.isFinite(value) ? value : 0
}

function getActivityWorkerRunId(item: ActivityFeedItem): string | null {
  const payload = item.raw.payload
  if (!payload || typeof payload !== "object") {
    return typeof item.raw.run_id === "string" && item.raw.run_id.trim().length > 0
      ? item.raw.run_id
      : null
  }

  const workerRunId = (payload as Record<string, unknown>).worker_run_id
  if (typeof workerRunId === "string" && workerRunId.trim().length > 0) {
    return workerRunId.trim()
  }

  return typeof item.raw.run_id === "string" && item.raw.run_id.trim().length > 0
    ? item.raw.run_id
    : null
}

export function resolveSelectedWorkerGroup(
  selectedWorkerRunId: string | null,
  codexDelegations: CodexDelegationEntry[],
  toolCalls: ToolCallEntry[],
): SubagentActivityGroup | null {
  if (!selectedWorkerRunId) return null

  return (
    buildSubagentActivityGroups(codexDelegations, toolCalls).find(
      (group) => group.workerRunId === selectedWorkerRunId,
    ) ?? null
  )
}

export function workerTimestampMatches(
  timestamp: string,
  group: Pick<SubagentActivityGroup, "startedAt" | "finishedAt">,
): boolean {
  const value = toMillis(timestamp)
  if (!value) return false

  const startedAt = toMillis(group.startedAt)
  const finishedAt = group.finishedAt ? toMillis(group.finishedAt) : Number.POSITIVE_INFINITY
  return value >= startedAt && value <= finishedAt
}

export function activityFeedItemMatchesWorker(
  item: ActivityFeedItem,
  group: Pick<SubagentActivityGroup, "workerRunId" | "assignee" | "startedAt" | "finishedAt">,
): boolean {
  const workerRunId = getActivityWorkerRunId(item)
  if (workerRunId === group.workerRunId) return true
  if (item.agent_name !== group.assignee) return false
  return workerTimestampMatches(item.ts, group)
}

export function toolCallMatchesWorker(
  entry: ToolCallEntry,
  group: Pick<SubagentActivityGroup, "assignee" | "startedAt" | "finishedAt">,
): boolean {
  if (entry.agent_name !== group.assignee) return false
  return workerTimestampMatches(entry.ts, group)
}

export function fileTouchedMatchesWorker(
  entry: FileTouchedEntry,
  group: Pick<SubagentActivityGroup, "workerRunId" | "assignee" | "startedAt" | "finishedAt">,
): boolean {
  if (entry.run_id === group.workerRunId) return true
  if (entry.agent_name !== group.assignee) return false
  return workerTimestampMatches(entry.ts, group)
}
