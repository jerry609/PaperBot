// TypeScript types mirroring the Python EventType vocabulary from message_schema.py

export type AgentStatus = "idle" | "working" | "completed" | "errored"

export type AgentLifecycleEvent = {
  type: "agent_started" | "agent_working" | "agent_completed" | "agent_error"
  run_id: string
  trace_id: string
  agent_name: string
  workflow: string
  stage: string
  ts: string
  payload: {
    status: string
    agent_name: string
    detail?: string
  }
}

export type ToolCallEvent = {
  type: "tool_result" | "tool_error"
  run_id: string
  trace_id: string
  agent_name: string
  workflow: string
  stage: string
  ts: string
  payload: {
    tool: string
    arguments: Record<string, unknown>
    result_summary: string
    error: string | null
  }
  metrics: {
    duration_ms: number
  }
}

export type AgentEventEnvelopeRaw = Record<string, unknown> & {
  type: string
  run_id?: string
  trace_id?: string
  agent_name?: string
  workflow?: string
  stage?: string
  ts?: string
  payload?: Record<string, unknown>
  metrics?: Record<string, unknown>
}

// Derived display types

export type ActivityFeedItem = {
  id: string // run_id + ts (dedup key)
  type: string
  agent_name: string
  workflow: string
  stage: string
  ts: string
  summary: string // human-readable line (derived from payload)
  raw: AgentEventEnvelopeRaw
}

export type AgentStatusEntry = {
  agent_name: string
  status: AgentStatus
  last_stage: string
  last_ts: string
}

export type ToolCallEntry = {
  id: string // run_id + tool + ts
  tool: string
  agent_name: string
  arguments: Record<string, unknown>
  result_summary: string
  error: string | null
  duration_ms: number
  ts: string
  status: "ok" | "error"
}
