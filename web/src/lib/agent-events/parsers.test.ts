import { describe, expect, it } from "vitest"
import { parseActivityItem, parseAgentStatus, parseToolCall } from "./parsers"
import type { AgentEventEnvelopeRaw } from "./types"

const BASE_ENVELOPE: AgentEventEnvelopeRaw = {
  type: "agent_started",
  run_id: "abc",
  trace_id: "trace-1",
  agent_name: "ResearchAgent",
  workflow: "scholar_pipeline",
  stage: "paper_search",
  ts: "2026-03-15T00:00:00Z",
  payload: { status: "agent_started", agent_name: "ResearchAgent" },
  metrics: {},
}

describe("parseActivityItem", () => {
  it("returns ActivityFeedItem with correct summary for agent_started", () => {
    const result = parseActivityItem(BASE_ENVELOPE)
    expect(result).not.toBeNull()
    expect(result?.summary).toBe("ResearchAgent started: paper_search")
    expect(result?.type).toBe("agent_started")
    expect(result?.agent_name).toBe("ResearchAgent")
    expect(result?.workflow).toBe("scholar_pipeline")
    expect(result?.stage).toBe("paper_search")
    expect(result?.ts).toBe("2026-03-15T00:00:00Z")
  })

  it("returns null for envelope missing type", () => {
    const raw = { ...BASE_ENVELOPE, type: "" }
    expect(parseActivityItem(raw)).toBeNull()
  })

  it("returns null for envelope missing ts", () => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { ts: _ts, ...rawWithoutTs } = BASE_ENVELOPE
    const raw = rawWithoutTs as AgentEventEnvelopeRaw
    expect(parseActivityItem(raw)).toBeNull()
  })

  it("generates id from run_id + ts", () => {
    const result = parseActivityItem(BASE_ENVELOPE)
    expect(result?.id).toBe("abc-2026-03-15T00:00:00Z")
  })
})

describe("parseAgentStatus", () => {
  it("returns working status for agent_started", () => {
    const result = parseAgentStatus(BASE_ENVELOPE)
    expect(result).not.toBeNull()
    expect(result?.status).toBe("working")
    expect(result?.agent_name).toBe("ResearchAgent")
    expect(result?.last_stage).toBe("paper_search")
    expect(result?.last_ts).toBe("2026-03-15T00:00:00Z")
  })

  it("returns working status for agent_working", () => {
    const raw = { ...BASE_ENVELOPE, type: "agent_working" }
    const result = parseAgentStatus(raw)
    expect(result?.status).toBe("working")
  })

  it("returns completed status for agent_completed", () => {
    const raw = { ...BASE_ENVELOPE, type: "agent_completed" }
    const result = parseAgentStatus(raw)
    expect(result?.status).toBe("completed")
  })

  it("returns errored status for agent_error", () => {
    const raw = { ...BASE_ENVELOPE, type: "agent_error" }
    const result = parseAgentStatus(raw)
    expect(result?.status).toBe("errored")
  })

  it("returns null for non-lifecycle event types (tool_result)", () => {
    const raw = { ...BASE_ENVELOPE, type: "tool_result" }
    expect(parseAgentStatus(raw)).toBeNull()
  })

  it("returns null for non-lifecycle event types (score_update)", () => {
    const raw = { ...BASE_ENVELOPE, type: "score_update" }
    expect(parseAgentStatus(raw)).toBeNull()
  })
})

describe("parseToolCall", () => {
  const TOOL_ENVELOPE: AgentEventEnvelopeRaw = {
    type: "tool_result",
    run_id: "run-1",
    trace_id: "trace-2",
    agent_name: "paperbot-mcp",
    workflow: "mcp",
    stage: "tool_call",
    ts: "2026-03-15T01:00:00Z",
    payload: {
      tool: "paper_search",
      arguments: { query: "LLM" },
      result_summary: "Found 5 papers",
      error: null,
    },
    metrics: { duration_ms: 123 },
  }

  it("returns ToolCallEntry with status ok for tool_result", () => {
    const result = parseToolCall(TOOL_ENVELOPE)
    expect(result).not.toBeNull()
    expect(result?.status).toBe("ok")
    expect(result?.tool).toBe("paper_search")
    expect(result?.duration_ms).toBe(123)
    expect(result?.result_summary).toBe("Found 5 papers")
    expect(result?.arguments).toEqual({ query: "LLM" })
    expect(result?.error).toBeNull()
  })

  it("returns ToolCallEntry with status error for tool_error", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...TOOL_ENVELOPE,
      type: "tool_error",
      payload: {
        tool: "paper_search",
        arguments: {},
        result_summary: "",
        error: "timeout",
      },
    }
    const result = parseToolCall(raw)
    expect(result).not.toBeNull()
    expect(result?.status).toBe("error")
    expect(result?.error).toBe("timeout")
  })

  it("returns null for non-tool event types", () => {
    const raw = { ...BASE_ENVELOPE, type: "agent_started" }
    expect(parseToolCall(raw)).toBeNull()
  })

  it("returns null for job_start event type", () => {
    const raw = { ...BASE_ENVELOPE, type: "job_start" }
    expect(parseToolCall(raw)).toBeNull()
  })

  it("generates id from run_id + tool + ts", () => {
    const result = parseToolCall(TOOL_ENVELOPE)
    expect(result?.id).toBe("run-1-paper_search-2026-03-15T01:00:00Z")
  })
})
