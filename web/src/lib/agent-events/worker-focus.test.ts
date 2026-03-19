import { describe, expect, it } from "vitest"

import {
  activityFeedItemMatchesWorker,
  fileTouchedMatchesWorker,
  resolveSelectedWorkerGroup,
  toolCallMatchesWorker,
  workerTimestampMatches,
} from "./worker-focus"
import type {
  ActivityFeedItem,
  CodexDelegationEntry,
  FileTouchedEntry,
  ToolCallEntry,
} from "./types"

const baseDelegation: CodexDelegationEntry = {
  id: "codex-1",
  event_type: "codex_completed",
  task_id: "task-1",
  worker_run_id: "worker-run-1",
  task_title: "Implement worker focus",
  assignee: "codex-a1b2",
  session_id: "sess-1",
  runtime: "codex",
  control_mode: "mirrored",
  interruptible: false,
  ts: "2026-03-17T03:10:00.000Z",
}

const baseToolCall: ToolCallEntry = {
  id: "tool-1",
  tool: "Read",
  agent_name: "codex-a1b2",
  arguments: {},
  result_summary: "opened file",
  error: null,
  duration_ms: 12,
  ts: "2026-03-17T03:05:00.000Z",
  status: "ok",
}

describe("worker-focus helpers", () => {
  it("resolves the selected worker group by worker run id", () => {
    const group = resolveSelectedWorkerGroup("worker-run-1", [baseDelegation], [baseToolCall])

    expect(group?.workerRunId).toBe("worker-run-1")
    expect(group?.assignee).toBe("codex-a1b2")
  })

  it("matches timestamps within the worker lifetime window", () => {
    expect(
      workerTimestampMatches("2026-03-17T03:06:00.000Z", {
        startedAt: "2026-03-17T03:00:00.000Z",
        finishedAt: "2026-03-17T03:10:00.000Z",
      }),
    ).toBe(true)

    expect(
      workerTimestampMatches("2026-03-17T03:12:00.000Z", {
        startedAt: "2026-03-17T03:00:00.000Z",
        finishedAt: "2026-03-17T03:10:00.000Z",
      }),
    ).toBe(false)
  })

  it("matches activity rows by explicit worker_run_id first", () => {
    const item: ActivityFeedItem = {
      id: "feed-1",
      type: "codex_completed",
      agent_name: "claude",
      workflow: "studio",
      stage: "worker",
      ts: "2026-03-17T03:09:00.000Z",
      summary: "worker completed",
      raw: {
        type: "codex_completed",
        ts: "2026-03-17T03:09:00.000Z",
        payload: { worker_run_id: "worker-run-1" },
      },
    }

    expect(
      activityFeedItemMatchesWorker(item, {
        workerRunId: "worker-run-1",
        assignee: "codex-a1b2",
        startedAt: "2026-03-17T03:00:00.000Z",
        finishedAt: "2026-03-17T03:10:00.000Z",
      }),
    ).toBe(true)
  })

  it("falls back to assignee plus time window for tools and files", () => {
    const tool: ToolCallEntry = {
      ...baseToolCall,
      ts: "2026-03-17T03:04:00.000Z",
    }
    const file: FileTouchedEntry = {
      run_id: "outer-run",
      agent_name: "codex-a1b2",
      path: "src/main.ts",
      status: "modified",
      ts: "2026-03-17T03:04:30.000Z",
      linesAdded: 4,
    }

    const group = {
      workerRunId: "worker-run-1",
      assignee: "codex-a1b2",
      startedAt: "2026-03-17T03:00:00.000Z",
      finishedAt: "2026-03-17T03:10:00.000Z",
    }

    expect(toolCallMatchesWorker(tool, group)).toBe(true)
    expect(fileTouchedMatchesWorker(file, group)).toBe(true)
  })
})
