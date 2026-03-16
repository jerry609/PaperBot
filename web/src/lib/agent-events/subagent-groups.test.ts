import { describe, expect, it } from "vitest"

import type { CodexDelegationEntry, ToolCallEntry } from "./types"
import { buildSubagentActivityGroups } from "./subagent-groups"

function makeDelegation(partial: Partial<CodexDelegationEntry> & Pick<CodexDelegationEntry, "id" | "event_type" | "task_id" | "task_title" | "assignee" | "session_id" | "ts">): CodexDelegationEntry {
  return {
    worker_run_id: partial.worker_run_id ?? `worker-${partial.task_id}`,
    runtime: partial.runtime ?? "codex",
    control_mode: partial.control_mode ?? "mirrored",
    interruptible: partial.interruptible ?? false,
    ...partial,
  }
}

function makeToolCall(partial: Partial<ToolCallEntry> & Pick<ToolCallEntry, "id" | "tool" | "agent_name" | "result_summary" | "error" | "duration_ms" | "ts" | "status">): ToolCallEntry {
  return {
    arguments: partial.arguments ?? {},
    ...partial,
  }
}

describe("buildSubagentActivityGroups", () => {
  it("groups delegation events into one running/completed card with related tools", () => {
    const groups = buildSubagentActivityGroups(
      [
        makeDelegation({
          id: "dispatch",
          event_type: "codex_dispatched",
          task_id: "task-1",
          task_title: "Inspect branch",
          assignee: "codex-123",
          session_id: "session-1",
          ts: "2026-03-16T12:00:00Z",
        }),
        makeDelegation({
          id: "complete",
          event_type: "codex_completed",
          task_id: "task-1",
          task_title: "Inspect branch",
          assignee: "codex-123",
          session_id: "session-1",
          ts: "2026-03-16T12:00:08Z",
          files_generated: ["src/app.ts"],
        }),
      ],
      [
        makeToolCall({
          id: "tool-1",
          tool: "Bash",
          agent_name: "codex-123",
          result_summary: "test/milestone-v1.2",
          error: null,
          duration_ms: 310,
          ts: "2026-03-16T12:00:05Z",
          status: "ok",
        }),
        makeToolCall({
          id: "tool-2",
          tool: "Read",
          agent_name: "codex-123",
          result_summary: "Loaded src/app.ts",
          error: null,
          duration_ms: 120,
          ts: "2026-03-16T11:59:59Z",
          status: "ok",
        }),
      ],
    )

    expect(groups).toHaveLength(1)
    expect(groups[0].status).toBe("completed")
    expect(groups[0].workerRunId).toBe("worker-task-1")
    expect(groups[0].runtime).toBe("codex")
    expect(groups[0].interruptible).toBe(false)
    expect(groups[0].toolCount).toBe(1)
    expect(groups[0].filesGenerated).toEqual(["src/app.ts"])
    expect(groups[0].recentTools[0]?.tool).toBe("Bash")
  })

  it("marks a dispatched run as running once worker tools appear", () => {
    const groups = buildSubagentActivityGroups(
      [
        makeDelegation({
          id: "dispatch",
          event_type: "codex_dispatched",
          task_id: "task-2",
          task_title: "Run smoke test",
          assignee: "codex-456",
          session_id: "session-2",
          ts: "2026-03-16T12:10:00Z",
        }),
      ],
      [
        makeToolCall({
          id: "tool-3",
          tool: "Bash",
          agent_name: "codex-456",
          result_summary: "pytest -q",
          error: null,
          duration_ms: 980,
          ts: "2026-03-16T12:10:03Z",
          status: "ok",
        }),
      ],
    )

    expect(groups).toHaveLength(1)
    expect(groups[0].status).toBe("running")
    expect(groups[0].toolCount).toBe(1)
  })

  it("carries failure metadata into the grouped card", () => {
    const groups = buildSubagentActivityGroups(
      [
        makeDelegation({
          id: "dispatch",
          event_type: "codex_dispatched",
          task_id: "task-3",
          task_title: "Fix parser",
          assignee: "codex-789",
          session_id: "session-3",
          ts: "2026-03-16T12:20:00Z",
        }),
        makeDelegation({
          id: "failed",
          event_type: "codex_failed",
          task_id: "task-3",
          task_title: "Fix parser",
          assignee: "codex-789",
          session_id: "session-3",
          ts: "2026-03-16T12:20:06Z",
          reason_code: "tool_error",
          error: "subagent crashed",
        }),
      ],
      [
        makeToolCall({
          id: "tool-4",
          tool: "Bash",
          agent_name: "codex-789",
          result_summary: "",
          error: "subagent crashed",
          duration_ms: 450,
          ts: "2026-03-16T12:20:05Z",
          status: "error",
        }),
      ],
    )

    expect(groups).toHaveLength(1)
    expect(groups[0].status).toBe("failed")
    expect(groups[0].toolErrorCount).toBe(1)
    expect(groups[0].reasonCode).toBe("tool_error")
    expect(groups[0].error).toBe("subagent crashed")
  })
})
