"use client"

import type { ActivityFeedItem, AgentStatus, AgentStatusEntry, AgentEventEnvelopeRaw, ToolCallEntry, FileTouchedEntry } from "./types"

const LIFECYCLE_TYPES = new Set([
  "agent_started",
  "agent_working",
  "agent_completed",
  "agent_error",
])

const TOOL_TYPES = new Set(["tool_result", "tool_error", "tool_call"])

export function parseActivityItem(raw: AgentEventEnvelopeRaw): ActivityFeedItem | null {
  if (!raw.type || !raw.ts) return null
  const id = `${raw.run_id ?? ""}-${raw.ts}`
  const summary = deriveHumanSummary(raw)
  return {
    id,
    type: raw.type,
    agent_name: String(raw.agent_name ?? "unknown"),
    workflow: String(raw.workflow ?? ""),
    stage: String(raw.stage ?? ""),
    ts: String(raw.ts),
    summary,
    raw,
  }
}

function deriveHumanSummary(raw: AgentEventEnvelopeRaw): string {
  const t = raw.type ?? ""
  const payload = (raw.payload ?? {}) as Record<string, unknown>

  if (t === "agent_started") return `${raw.agent_name} started: ${raw.stage}`
  if (t === "agent_working") return `${raw.agent_name} working on: ${raw.stage}`
  if (t === "agent_completed") return `${raw.agent_name} completed: ${raw.stage}`
  if (t === "agent_error") return `${raw.agent_name} error: ${String(payload.detail ?? "")}`
  if (t === "tool_result" || t === "tool_error" || t === "tool_call") {
    const tool = String(payload.tool ?? t)
    return `Tool: ${tool} — ${String(payload.result_summary ?? "").slice(0, 80)}`
  }
  if (t === "job_start") return `Job started: ${raw.stage}`
  if (t === "job_result") return `Job finished: ${raw.stage}`
  if (t === "source_record") return `Source record: ${raw.workflow}/${raw.stage}`
  if (t === "score_update") return `Score update from ${raw.agent_name}`
  if (t === "insight") return `Insight from ${raw.agent_name}`
  return `${t}: ${raw.agent_name ?? ""} / ${raw.stage ?? ""}`
}

export function parseAgentStatus(raw: AgentEventEnvelopeRaw): AgentStatusEntry | null {
  if (!LIFECYCLE_TYPES.has(String(raw.type ?? ""))) return null
  const statusMap: Record<string, AgentStatus> = {
    agent_started: "working",
    agent_working: "working",
    agent_completed: "completed",
    agent_error: "errored",
  }
  return {
    agent_name: String(raw.agent_name ?? "unknown"),
    status: statusMap[raw.type as string] ?? "idle",
    last_stage: String(raw.stage ?? ""),
    last_ts: String(raw.ts ?? ""),
  }
}

export function parseToolCall(raw: AgentEventEnvelopeRaw): ToolCallEntry | null {
  if (!TOOL_TYPES.has(String(raw.type ?? ""))) return null
  const payload = (raw.payload ?? {}) as Record<string, unknown>
  const metrics = (raw.metrics ?? {}) as Record<string, unknown>
  const tool = String(payload.tool ?? raw.stage ?? "unknown")
  return {
    id: `${raw.run_id ?? ""}-${tool}-${raw.ts ?? ""}`,
    tool,
    agent_name: String(raw.agent_name ?? "unknown"),
    arguments: (payload.arguments as Record<string, unknown>) ?? {},
    result_summary: String(payload.result_summary ?? ""),
    error: typeof payload.error === "string" ? payload.error : null,
    duration_ms: typeof metrics.duration_ms === "number" ? metrics.duration_ms : 0,
    ts: String(raw.ts ?? ""),
    status:
      raw.type === "tool_error" || (typeof payload.error === "string" && payload.error)
        ? "error"
        : "ok",
  }
}

const FILE_CHANGE_TYPES = new Set(["file_change"])

export function parseFileTouched(raw: AgentEventEnvelopeRaw): FileTouchedEntry | null {
  const t = String(raw.type ?? "")
  const payload = (raw.payload ?? {}) as Record<string, unknown>

  const isExplicitFileChange = FILE_CHANGE_TYPES.has(t)
  const isWriteFileTool =
    t === "tool_result" &&
    typeof payload.tool === "string" &&
    payload.tool === "write_file"

  if (!isExplicitFileChange && !isWriteFileTool) return null
  if (!raw.run_id || !raw.ts) return null

  const path = String(
    (isExplicitFileChange
      ? payload.path
      : payload.arguments
        ? (payload.arguments as Record<string, unknown>).path
        : undefined) ?? ""
  )
  if (!path) return null

  return {
    run_id: String(raw.run_id),
    path,
    status: (payload.status as "created" | "modified") ?? "modified",
    ts: String(raw.ts),
    linesAdded: typeof payload.lines_added === "number" ? payload.lines_added : undefined,
    linesDeleted: typeof payload.lines_deleted === "number" ? payload.lines_deleted : undefined,
    diff: typeof payload.diff === "string" ? payload.diff : undefined,
    oldContent: typeof payload.old_content === "string" ? payload.old_content : undefined,
    newContent: typeof payload.new_content === "string" ? payload.new_content : undefined,
  }
}
