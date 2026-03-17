import { describe, expect, it } from "vitest"

import type { Task } from "@/lib/store/studio-store"
import { findRelatedWorkerThread } from "./studio-worker-links"

function makeTask(partial: Partial<Task> & Pick<Task, "id" | "name" | "status" | "paperId">): Task {
  const now = new Date("2026-03-17T10:00:00Z")
  return {
    id: partial.id,
    name: partial.name,
    kind: partial.kind ?? "chat",
    status: partial.status,
    actions: partial.actions ?? [],
    createdAt: partial.createdAt ?? now,
    updatedAt: partial.updatedAt ?? now,
    history: partial.history ?? [],
    paperId: partial.paperId,
  }
}

describe("findRelatedWorkerThread", () => {
  it("finds a related chat thread by delegation task id", () => {
    const task = makeTask({
      id: "thread-1",
      name: "Telemetry bridge thread",
      status: "completed",
      paperId: "paper-1",
      actions: [
        {
          id: "action-1",
          type: "function_call",
          content: "Agent()",
          timestamp: new Date("2026-03-17T10:01:00Z"),
          metadata: {
            bridgeResult: {
              version: "1",
              executor: "codex",
              taskKind: "code",
              status: "completed",
              summary: "Implemented telemetry bridge.",
              artifacts: [],
              delegation: {
                taskId: "tooluse_agent",
                workerRunId: "worker-run-tooluse_agent",
                taskTitle: "Implement telemetry bridge",
                assignee: "codex-toolus",
                sessionId: "studio-session-1",
                runtime: "codex",
                controlMode: "mirrored",
                interruptible: false,
              },
              payload: {},
              raw: {},
            },
          },
        },
      ],
    })

    const related = findRelatedWorkerThread([task], {
      paperId: "paper-1",
      delegationTaskId: "tooluse_agent",
      workerRunId: null,
    })

    expect(related?.task.id).toBe("thread-1")
    expect(related?.latestBridgeResult?.summary).toBe("Implemented telemetry bridge.")
    expect(related?.pendingApproval).toBe(false)
  })

  it("falls back to worker run id matching", () => {
    const task = makeTask({
      id: "thread-2",
      name: "Worker run fallback",
      status: "running",
      paperId: "paper-1",
      actions: [
        {
          id: "action-2",
          type: "function_call",
          content: "Agent()",
          timestamp: new Date("2026-03-17T10:02:00Z"),
          metadata: {
            bridgeResult: {
              version: "1",
              executor: "codex",
              taskKind: "research",
              status: "partial",
              summary: "Gathered codebase evidence.",
              artifacts: [],
              delegation: null,
              payload: {
                worker_run_id: "worker-run-123",
              },
              raw: {},
            },
          },
        },
      ],
    })

    const related = findRelatedWorkerThread([task], {
      paperId: "paper-1",
      delegationTaskId: null,
      workerRunId: "worker-run-123",
    })

    expect(related?.task.id).toBe("thread-2")
    expect(related?.latestBridgeResult?.taskKind).toBe("research")
  })

  it("prefers the most recent matching action and marks pending approval", () => {
    const olderTask = makeTask({
      id: "thread-older",
      name: "Older match",
      status: "completed",
      paperId: "paper-1",
      actions: [
        {
          id: "action-old",
          type: "function_call",
          content: "Agent()",
          timestamp: new Date("2026-03-17T10:03:00Z"),
          metadata: {
            bridgeResult: {
              version: "1",
              executor: "codex",
              taskKind: "code",
              status: "completed",
              summary: "Older result.",
              artifacts: [],
              delegation: {
                taskId: "tooluse_agent",
                workerRunId: "worker-run-tooluse_agent",
                taskTitle: "Older task",
                assignee: "codex-old",
                sessionId: "studio-session-1",
                runtime: "codex",
                controlMode: "mirrored",
                interruptible: false,
              },
              payload: {},
              raw: {},
            },
          },
        },
      ],
    })

    const newerTask = makeTask({
      id: "thread-newer",
      name: "Newer approval",
      status: "running",
      paperId: "paper-1",
      actions: [
        {
          id: "action-new",
          type: "approval_request",
          content: "Need approval.",
          timestamp: new Date("2026-03-17T10:05:00Z"),
          metadata: {
            approvalRequest: {
              message: "Need approval to continue.",
              command: "git status",
              cliSessionId: "cli-session-1",
              bridgeResult: {
                version: "1",
                executor: "codex",
                taskKind: "approval_required",
                status: "approval_required",
                summary: "Need approval to continue.",
                artifacts: [],
                delegation: {
                  taskId: "tooluse_agent",
                  workerRunId: "worker-run-tooluse_agent",
                  taskTitle: "Approval task",
                  assignee: "codex-new",
                  sessionId: "studio-session-1",
                  runtime: "codex",
                  controlMode: "mirrored",
                  interruptible: false,
                },
                payload: {},
                raw: {},
              },
            },
          },
        },
      ],
    })

    const related = findRelatedWorkerThread([olderTask, newerTask], {
      paperId: "paper-1",
      delegationTaskId: "tooluse_agent",
      workerRunId: null,
    })

    expect(related?.task.id).toBe("thread-newer")
    expect(related?.pendingApproval).toBe(true)
    expect(related?.latestApprovalAction?.id).toBe("action-new")
  })
})
