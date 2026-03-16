import { describe, expect, it } from "vitest"

import type { AgentTaskLog } from "@/lib/store/studio-store"

import { buildTaskTimelineEntries, filterTaskLogs, inferTaskLogBlockType, stripLogPrefix } from "./studio-task-log-groups"

function makeLog(partial: Partial<AgentTaskLog> & Pick<AgentTaskLog, "id" | "timestamp" | "event" | "phase" | "level" | "message">): AgentTaskLog {
  return {
    ...partial,
  }
}

describe("studio-task-log-groups", () => {
  it("infers diff blocks from file write events", () => {
    const log = makeLog({
      id: "1",
      timestamp: "2026-03-16T00:00:00Z",
      event: "file_write",
      phase: "executing",
      level: "info",
      message: "[step 2] wrote file",
    })

    expect(inferTaskLogBlockType(log)).toBe("diff")
    expect(stripLogPrefix(log.message)).toBe("wrote file")
  })

  it("collapses contiguous think and tool runs while preserving diff/result blocks", () => {
    const entries = buildTaskTimelineEntries([
      makeLog({
        id: "think-1",
        timestamp: "2026-03-16T00:00:00Z",
        event: "thinking",
        phase: "executing",
        level: "info",
        message: "[step 1] Reviewing the code",
        blockType: "think",
      }),
      makeLog({
        id: "think-2",
        timestamp: "2026-03-16T00:00:01Z",
        event: "thinking",
        phase: "executing",
        level: "info",
        message: "[step 2] Planning the patch",
        blockType: "think",
      }),
      makeLog({
        id: "tool-1",
        timestamp: "2026-03-16T00:00:02Z",
        event: "tool_call",
        phase: "executing",
        level: "info",
        message: "[step 3] run_command: pytest -q",
        blockType: "tool",
        details: { tool: "run_command" },
      }),
      makeLog({
        id: "tool-2",
        timestamp: "2026-03-16T00:00:03Z",
        event: "tool_call",
        phase: "executing",
        level: "error",
        message: "[step 4] read_file: src/app.ts",
        blockType: "tool",
        details: { tool: "read_file" },
      }),
      makeLog({
        id: "diff-1",
        timestamp: "2026-03-16T00:00:04Z",
        event: "file_write",
        phase: "executing",
        level: "info",
        message: "[step 5] wrote src/app.ts",
        blockType: "diff",
      }),
      makeLog({
        id: "result-1",
        timestamp: "2026-03-16T00:00:05Z",
        event: "task_done",
        phase: "executing",
        level: "success",
        message: "[step 6] task_done: fixed parser",
        blockType: "result",
      }),
    ])

    expect(entries).toHaveLength(4)
    expect(entries[0]?.kind).toBe("group")
    expect(entries[0]?.blockType).toBe("think")
    expect(entries[1]?.kind).toBe("group")
    expect(entries[1]?.blockType).toBe("tool")
    if (entries[1]?.kind === "group") {
      expect(entries[1].status).toBe("error")
      expect(entries[1].toolNames).toEqual(["run_command", "read_file"])
    }
    expect(entries[2]?.kind).toBe("log")
    expect(entries[2]?.blockType).toBe("diff")
    expect(entries[3]?.kind).toBe("log")
    expect(entries[3]?.blockType).toBe("result")
  })

  it("filters logs for thinking and diffs", () => {
    const logs: AgentTaskLog[] = [
      makeLog({
        id: "a",
        timestamp: "2026-03-16T00:00:00Z",
        event: "thinking",
        phase: "executing",
        level: "info",
        message: "think",
      }),
      makeLog({
        id: "b",
        timestamp: "2026-03-16T00:00:01Z",
        event: "file_write",
        phase: "executing",
        level: "info",
        message: "write",
      }),
    ]

    expect(filterTaskLogs(logs, "thinking")).toHaveLength(1)
    expect(filterTaskLogs(logs, "diffs")).toHaveLength(1)
    expect(filterTaskLogs(logs, "all")).toHaveLength(2)
  })
})
