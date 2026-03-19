"use client"

import type { ActivityFeedItem, AgentStatus, AgentStatusEntry, AgentEventEnvelopeRaw, CodexDelegationEntry, ScoreEdgeEntry, ToolCallEntry, FileTouchedEntry } from "./types"
import { getAgentPresentation } from "@/lib/agent-runtime"

const LIFECYCLE_TYPES = new Set([
  "agent_started",
  "agent_working",
  "agent_completed",
  "agent_error",
])

const TOOL_TYPES = new Set(["tool_result", "tool_error", "tool_call"])

function displayScalar(value: unknown): string {
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  return ""
}

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
  if (t === "agent_error") return `${raw.agent_name} error: ${displayScalar(payload.detail)}`
  if (t === "tool_result" || t === "tool_error" || t === "tool_call") {
    const tool = displayScalar(payload.tool) || t
    return `Tool: ${tool} — ${displayScalar(payload.result_summary).slice(0, 80)}`
  }
  if (t === "codex_dispatched") {
    const assignee = displayScalar(payload.assignee) || String(raw.agent_name ?? "Codex")
    const title = displayScalar(payload.task_title)
    return `Worker dispatched to ${getAgentPresentation(assignee).shortLabel}: ${title}`
  }
  if (t === "codex_accepted") {
    const assignee = displayScalar(payload.assignee) || String(raw.agent_name ?? "Codex")
    const title = displayScalar(payload.task_title)
    return `${getAgentPresentation(assignee).shortLabel} accepted worker run: ${title}`
  }
  if (t === "codex_completed") {
    const assignee = displayScalar(payload.assignee) || String(raw.agent_name ?? "Codex")
    const title = displayScalar(payload.task_title)
    const files = Array.isArray(payload.files_generated) ? payload.files_generated.length : 0
    return `${getAgentPresentation(assignee).shortLabel} completed worker run: ${title} (${files} files)`
  }
  if (t === "codex_failed") {
    const assignee = displayScalar(payload.assignee) || String(raw.agent_name ?? "Codex")
    const title = displayScalar(payload.task_title)
    const reason = displayScalar(payload.reason_code) || "unknown"
    return `${getAgentPresentation(assignee).shortLabel} failed worker run: ${title} (${reason})`
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

const CODEX_DELEGATION_TYPES = new Set([
  "codex_dispatched",
  "codex_accepted",
  "codex_completed",
  "codex_failed",
])

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
    agent_name: String(raw.agent_name ?? "unknown"),
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

export function parseCodexDelegation(raw: AgentEventEnvelopeRaw): CodexDelegationEntry | null {
  const t = String(raw.type ?? "")
  if (!CODEX_DELEGATION_TYPES.has(t)) return null

  const payload = (raw.payload ?? {}) as Record<string, unknown>
  const task_id = String(payload.task_id ?? "")
  const task_title = String(payload.task_title ?? "")
  const assignee = String(payload.assignee ?? raw.agent_name ?? "")
  const session_id = String(payload.session_id ?? "")
  const ts = String(raw.ts ?? "")

  const entry: CodexDelegationEntry = {
    id: `${t}-${task_id}-${ts}`,
    event_type: t as CodexDelegationEntry["event_type"],
    task_id,
    worker_run_id: String(payload.worker_run_id ?? task_id),
    task_title,
    assignee,
    session_id,
    runtime: String(payload.runtime ?? "worker"),
    control_mode: payload.control_mode === "managed" ? "managed" : "mirrored",
    interruptible: payload.interruptible === true,
    ts,
  }

  if (t === "codex_completed" && Array.isArray(payload.files_generated)) {
    entry.files_generated = payload.files_generated as string[]
  }

  if (t === "codex_failed") {
    if (typeof payload.reason_code === "string") entry.reason_code = payload.reason_code
    if (typeof payload.error === "string") entry.error = payload.error
  }

  return entry
}

export function parseScoreEdge(raw: AgentEventEnvelopeRaw): ScoreEdgeEntry | null {
  if (raw.type !== "score_update") return null

  const payload = (raw.payload ?? {}) as Record<string, unknown>
  const scoreObj = (payload.score ?? {}) as Record<string, unknown>

  const stage = typeof scoreObj.stage === "string" ? scoreObj.stage : String(raw.stage ?? "")
  const from_agent = String(raw.stage ?? raw.agent_name ?? "ScoreShareBus")
  const to_agent = String(raw.workflow ?? "pipeline")
  const score = typeof scoreObj.score === "number" ? scoreObj.score : 0
  const id = `${from_agent}-${to_agent}-${stage}`

  return {
    id,
    from_agent,
    to_agent,
    stage,
    score,
    ts: String(raw.ts ?? ""),
  }
}
