import { describe, expect, it } from "vitest"

import type { AgentAction } from "@/lib/store/studio-store"

import { collapseToolActivityActions } from "./studio-chat-activity"

function makeAction(partial: Partial<AgentAction> & Pick<AgentAction, "id" | "type" | "content">): AgentAction {
  return {
    id: partial.id,
    type: partial.type,
    content: partial.content,
    timestamp: partial.timestamp ?? new Date("2026-03-16T00:00:00Z"),
    metadata: partial.metadata,
  }
}

describe("collapseToolActivityActions", () => {
  it("collapses contiguous tool actions into one summary card", () => {
    const actions = collapseToolActivityActions([
      makeAction({ id: "u1", type: "user", content: "Inspect the codebase" }),
      makeAction({
        id: "t1",
        type: "function_call",
        content: "Glob()",
        metadata: { functionName: "Glob", params: { path: "src/**" }, result: "12 files" },
      }),
      makeAction({
        id: "t2",
        type: "function_call",
        content: "Read()",
        metadata: { functionName: "Read", params: { path: "src/app.ts" }, result: "ok" },
      }),
      makeAction({ id: "a1", type: "text", content: "I found the entry points." }),
    ])

    expect(actions).toHaveLength(3)
    expect(actions[1].type).toBe("activity_summary")
    expect(actions[1].content).toBe("Scanning the workspace")
    expect(actions[1].metadata?.activitySummary?.totalTools).toBe(2)
    expect(actions[1].metadata?.activitySummary?.counts.read).toBe(2)
    expect(actions[2].type).toBe("text")
  })

  it("preserves non-tool actions and keeps separate tool runs apart", () => {
    const actions = collapseToolActivityActions([
      makeAction({
        id: "t1",
        type: "function_call",
        content: "Read()",
        metadata: { functionName: "Read", params: { path: "src/a.ts" }, result: "ok" },
      }),
      makeAction({ id: "x1", type: "thinking", content: "Reviewing findings" }),
      makeAction({
        id: "t2",
        type: "function_call",
        content: "Bash()",
        metadata: { functionName: "Bash", params: { command: "pytest -q" } },
      }),
    ])

    expect(actions).toHaveLength(3)
    expect(actions[0].type).toBe("activity_summary")
    expect(actions[1].type).toBe("thinking")
    expect(actions[2].type).toBe("activity_summary")
    expect(actions[2].metadata?.activitySummary?.status).toBe("running")
    expect(actions[2].metadata?.activitySummary?.counts.command).toBe(1)
  })

  it("labels delegation-heavy runs as subagent coordination", () => {
    const actions = collapseToolActivityActions([
      makeAction({
        id: "d1",
        type: "function_call",
        content: "task()",
        metadata: { functionName: "task", params: { assignee: "codex-worker", task_title: "Inspect branch" } },
      }),
      makeAction({
        id: "d2",
        type: "function_call",
        content: "spawn_agent()",
        metadata: { functionName: "spawn_agent", params: { runtime: "codex", message: "Inspect branch" } },
      }),
    ])

    expect(actions).toHaveLength(1)
    expect(actions[0].type).toBe("activity_summary")
    expect(actions[0].content).toBe("Coordinating subagents")
    expect(actions[0].metadata?.activitySummary?.counts.delegation).toBe(2)
  })

  it("treats Claude Agent plus nested worker tools as delegated activity", () => {
    const actions = collapseToolActivityActions([
      makeAction({
        id: "d1",
        type: "function_call",
        content: "Agent()",
        metadata: {
          functionName: "Agent",
          params: {
            subagent_type: "codex-worker",
            description: "Get current git branch",
          },
          result: "delegated",
        },
      }),
      makeAction({
        id: "d2",
        type: "function_call",
        content: "Bash()",
        metadata: {
          functionName: "Bash",
          params: { command: "git branch --show-current" },
          result: "test/milestone-v1.2",
        },
      }),
    ])

    expect(actions).toHaveLength(1)
    expect(actions[0].type).toBe("activity_summary")
    expect(actions[0].content).toBe("Coordinating subagents")
    expect(actions[0].metadata?.activitySummary?.counts.delegation).toBe(1)
    expect(actions[0].metadata?.activitySummary?.counts.command).toBe(1)
  })
})
