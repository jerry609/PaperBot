import { describe, expect, it } from "vitest"
import { parseActivityItem, parseAgentStatus, parseToolCall, parseFileTouched, parseCodexDelegation, parseScoreEdge } from "./parsers"
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

describe("parseFileTouched", () => {
  const FILE_CHANGE_ENVELOPE: AgentEventEnvelopeRaw = {
    type: "file_change",
    run_id: "run-fc-1",
    trace_id: "trace-fc-1",
    agent_name: "CodingAgent",
    workflow: "repro",
    stage: "generation",
    ts: "2026-03-15T02:00:00Z",
    payload: {
      path: "src/main.py",
      status: "modified",
      lines_added: 10,
      lines_deleted: 2,
    },
  }

  it("returns FileTouchedEntry for file_change event with path/status/lines_added", () => {
    const result = parseFileTouched(FILE_CHANGE_ENVELOPE)
    expect(result).not.toBeNull()
    expect(result?.run_id).toBe("run-fc-1")
    expect(result?.agent_name).toBe("CodingAgent")
    expect(result?.path).toBe("src/main.py")
    expect(result?.status).toBe("modified")
    expect(result?.ts).toBe("2026-03-15T02:00:00Z")
    expect(result?.linesAdded).toBe(10)
    expect(result?.linesDeleted).toBe(2)
  })

  it("returns FileTouchedEntry for tool_result with payload.tool=='write_file' (fallback path)", () => {
    const raw: AgentEventEnvelopeRaw = {
      type: "tool_result",
      run_id: "run-fc-2",
      ts: "2026-03-15T02:01:00Z",
      payload: {
        tool: "write_file",
        arguments: { path: "src/utils.py", content: "# code" },
        result_summary: "written",
        error: null,
      },
    }
    const result = parseFileTouched(raw)
    expect(result).not.toBeNull()
    expect(result?.path).toBe("src/utils.py")
    expect(result?.run_id).toBe("run-fc-2")
  })

  it("returns null for lifecycle events (agent_started)", () => {
    const raw = { ...BASE_ENVELOPE, type: "agent_started" }
    expect(parseFileTouched(raw)).toBeNull()
  })

  it("returns null for tool_result with payload.tool!='write_file'", () => {
    const raw: AgentEventEnvelopeRaw = {
      type: "tool_result",
      run_id: "run-fc-3",
      ts: "2026-03-15T02:02:00Z",
      payload: {
        tool: "paper_search",
        arguments: {},
        result_summary: "found",
        error: null,
      },
    }
    expect(parseFileTouched(raw)).toBeNull()
  })

  it("returns null when run_id is missing", () => {
    const raw = { ...FILE_CHANGE_ENVELOPE } as AgentEventEnvelopeRaw
    delete raw.run_id
    expect(parseFileTouched(raw as AgentEventEnvelopeRaw)).toBeNull()
  })

  it("returns null when path is empty or missing", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...FILE_CHANGE_ENVELOPE,
      payload: { ...FILE_CHANGE_ENVELOPE.payload, path: "" },
    }
    expect(parseFileTouched(raw)).toBeNull()
  })
})

describe("parseCodexDelegation", () => {
  const CODEX_DISPATCHED: AgentEventEnvelopeRaw = {
    type: "codex_dispatched",
    run_id: "run-cdx-1",
    trace_id: "trace-cdx-1",
    agent_name: "Orchestrator",
    workflow: "repro",
    stage: "delegation",
    ts: "2026-03-15T03:00:00Z",
    payload: {
      task_id: "task-abc",
      worker_run_id: "worker-run-abc",
      task_title: "Implement auth module",
      assignee: "codex-a1b2",
      session_id: "sess-001",
      runtime: "codex",
      control_mode: "mirrored",
      interruptible: false,
    },
  }

  it("returns CodexDelegationEntry for codex_dispatched", () => {
    const result = parseCodexDelegation(CODEX_DISPATCHED)
    expect(result).not.toBeNull()
    expect(result?.event_type).toBe("codex_dispatched")
    expect(result?.task_id).toBe("task-abc")
    expect(result?.worker_run_id).toBe("worker-run-abc")
    expect(result?.task_title).toBe("Implement auth module")
    expect(result?.assignee).toBe("codex-a1b2")
    expect(result?.session_id).toBe("sess-001")
    expect(result?.runtime).toBe("codex")
    expect(result?.control_mode).toBe("mirrored")
    expect(result?.interruptible).toBe(false)
    expect(result?.ts).toBe("2026-03-15T03:00:00Z")
  })

  it("returns CodexDelegationEntry for codex_accepted", () => {
    const raw: AgentEventEnvelopeRaw = { ...CODEX_DISPATCHED, type: "codex_accepted" }
    const result = parseCodexDelegation(raw)
    expect(result).not.toBeNull()
    expect(result?.event_type).toBe("codex_accepted")
  })

  it("preserves managed control metadata when present", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...CODEX_DISPATCHED,
      payload: {
        ...CODEX_DISPATCHED.payload,
        control_mode: "managed",
        interruptible: true,
      },
    }
    const result = parseCodexDelegation(raw)
    expect(result?.control_mode).toBe("managed")
    expect(result?.interruptible).toBe(true)
  })

  it("returns CodexDelegationEntry with files_generated for codex_completed", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...CODEX_DISPATCHED,
      type: "codex_completed",
      payload: {
        ...CODEX_DISPATCHED.payload,
        files_generated: ["src/auth.ts", "src/auth.test.ts"],
      },
    }
    const result = parseCodexDelegation(raw)
    expect(result).not.toBeNull()
    expect(result?.event_type).toBe("codex_completed")
    expect(result?.files_generated).toEqual(["src/auth.ts", "src/auth.test.ts"])
  })

  it("returns CodexDelegationEntry with reason_code for codex_failed", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...CODEX_DISPATCHED,
      type: "codex_failed",
      payload: {
        ...CODEX_DISPATCHED.payload,
        reason_code: "max_iterations_exhausted",
        error: "Agent exhausted maximum iterations",
      },
    }
    const result = parseCodexDelegation(raw)
    expect(result).not.toBeNull()
    expect(result?.event_type).toBe("codex_failed")
    expect(result?.reason_code).toBe("max_iterations_exhausted")
    expect(result?.error).toBe("Agent exhausted maximum iterations")
  })

  it("returns null for non-codex event types", () => {
    const raw = { ...CODEX_DISPATCHED, type: "agent_started" }
    expect(parseCodexDelegation(raw)).toBeNull()
  })

  it("generates deterministic id from type + task_id + ts", () => {
    const result = parseCodexDelegation(CODEX_DISPATCHED)
    expect(result?.id).toBe("codex_dispatched-task-abc-2026-03-15T03:00:00Z")
  })
})

describe("parseScoreEdge", () => {
  const SCORE_UPDATE_ENVELOPE: AgentEventEnvelopeRaw = {
    type: "score_update",
    run_id: "run-score-1",
    trace_id: "trace-score-1",
    agent_name: "JudgeAgent",
    workflow: "scholar_pipeline",
    stage: "research",
    ts: "2026-03-15T05:00:00Z",
    payload: {
      score: { stage: "research", score: 0.85 },
    },
  }

  it("returns null for agent_started event", () => {
    const raw = { ...SCORE_UPDATE_ENVELOPE, type: "agent_started" }
    expect(parseScoreEdge(raw)).toBeNull()
  })

  it("returns null for tool_result event", () => {
    const raw = { ...SCORE_UPDATE_ENVELOPE, type: "tool_result" }
    expect(parseScoreEdge(raw)).toBeNull()
  })

  it("returns ScoreEdgeEntry with correct id for score_update event", () => {
    const result = parseScoreEdge(SCORE_UPDATE_ENVELOPE)
    expect(result).not.toBeNull()
    // id = `${from_agent}-${to_agent}-${stage}`
    // from_agent = raw.stage = "research", to_agent = raw.workflow = "scholar_pipeline", stage = score.stage = "research"
    expect(result?.id).toBe("research-scholar_pipeline-research")
  })

  it("uses raw.stage as from_agent context", () => {
    const result = parseScoreEdge(SCORE_UPDATE_ENVELOPE)
    expect(result?.from_agent).toBe("research")
  })

  it("uses raw.workflow as to_agent context", () => {
    const result = parseScoreEdge(SCORE_UPDATE_ENVELOPE)
    expect(result?.to_agent).toBe("scholar_pipeline")
  })

  it("extracts score number from payload.score.score", () => {
    const result = parseScoreEdge(SCORE_UPDATE_ENVELOPE)
    expect(result?.score).toBe(0.85)
  })

  it("extracts stage from payload.score.stage", () => {
    const result = parseScoreEdge(SCORE_UPDATE_ENVELOPE)
    expect(result?.stage).toBe("research")
  })

  it("returns ts from raw.ts", () => {
    const result = parseScoreEdge(SCORE_UPDATE_ENVELOPE)
    expect(result?.ts).toBe("2026-03-15T05:00:00Z")
  })

  it("falls back to 'ScoreShareBus' as from_agent when stage and agent_name both missing", () => {
    const raw: AgentEventEnvelopeRaw = {
      type: "score_update",
      run_id: "run-2",
      ts: "2026-03-15T05:01:00Z",
      workflow: "my_workflow",
      payload: { score: { stage: "quality", score: 0.7 } },
    }
    const result = parseScoreEdge(raw)
    expect(result?.from_agent).toBe("ScoreShareBus")
  })

  it("falls back to 'pipeline' as to_agent when workflow is missing", () => {
    const raw: AgentEventEnvelopeRaw = {
      type: "score_update",
      run_id: "run-3",
      ts: "2026-03-15T05:02:00Z",
      stage: "code",
      payload: { score: { stage: "code", score: 0.5 } },
    }
    const result = parseScoreEdge(raw)
    expect(result?.to_agent).toBe("pipeline")
  })

  it("defaults score to 0 when payload.score.score is not a number", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...SCORE_UPDATE_ENVELOPE,
      payload: { score: { stage: "research", score: "not-a-number" } },
    }
    const result = parseScoreEdge(raw)
    expect(result?.score).toBe(0)
  })
})

describe("deriveHumanSummary for score_update (via parseActivityItem)", () => {
  it("returns 'Score update from {agent_name}' for score_update", () => {
    const raw: AgentEventEnvelopeRaw = {
      type: "score_update",
      run_id: "run-su-1",
      agent_name: "JudgeAgent",
      workflow: "scholar_pipeline",
      stage: "research",
      ts: "2026-03-15T05:10:00Z",
      payload: { score: { stage: "research", score: 0.9 } },
    }
    const result = parseActivityItem(raw)
    expect(result?.summary).toBe("Score update from JudgeAgent")
  })
})

describe("deriveHumanSummary for codex events (via parseActivityItem)", () => {
  const BASE: AgentEventEnvelopeRaw = {
    run_id: "run-cdx-2",
    trace_id: "trace-cdx-2",
    agent_name: "codex-a1b2",
    workflow: "repro",
    stage: "delegation",
    ts: "2026-03-15T04:00:00Z",
    payload: {
      task_id: "task-xyz",
      task_title: "Build ML pipeline",
      assignee: "codex-a1b2",
      session_id: "sess-002",
    },
  }

  it("returns 'Task dispatched to Codex: {title}' for codex_dispatched", () => {
    const raw: AgentEventEnvelopeRaw = { ...BASE, type: "codex_dispatched" }
    const result = parseActivityItem(raw)
    expect(result?.summary).toBe("Worker dispatched to Codex: Build ML pipeline")
  })

  it("returns 'Codex accepted task: {title}' for codex_accepted", () => {
    const raw: AgentEventEnvelopeRaw = { ...BASE, type: "codex_accepted" }
    const result = parseActivityItem(raw)
    expect(result?.summary).toBe("Codex accepted worker run: Build ML pipeline")
  })

  it("returns 'Codex completed: {title} (N files)' for codex_completed", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...BASE,
      type: "codex_completed",
      payload: { ...BASE.payload, files_generated: ["a.ts", "b.ts", "c.ts"] },
    }
    const result = parseActivityItem(raw)
    expect(result?.summary).toBe("Codex completed worker run: Build ML pipeline (3 files)")
  })

  it("returns 'Codex failed: {title} (reason_code)' for codex_failed", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...BASE,
      type: "codex_failed",
      payload: { ...BASE.payload, reason_code: "stagnation_detected" },
    }
    const result = parseActivityItem(raw)
    expect(result?.summary).toBe("Codex failed worker run: Build ML pipeline (stagnation_detected)")
  })

  it("renders OpenCode label when assignee is opencode-*", () => {
    const raw: AgentEventEnvelopeRaw = {
      ...BASE,
      type: "codex_completed",
      agent_name: "opencode-a1b2",
      payload: {
        ...BASE.payload,
        assignee: "opencode-a1b2",
        files_generated: ["shell.tsx"],
      },
    }
    const result = parseActivityItem(raw)
    expect(result?.summary).toBe("OpenCode completed worker run: Build ML pipeline (1 files)")
  })
})
