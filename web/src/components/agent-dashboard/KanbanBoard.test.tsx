import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { KanbanBoard, agentLabel, extractCodexFailureReason } from "./KanbanBoard"
import type { AgentTask } from "@/lib/store/studio-store"

function makeTask(overrides: Partial<AgentTask> = {}): AgentTask {
  return {
    id: "task-default",
    title: "Default Task",
    description: "A test task",
    status: "planning",
    assignee: "claude",
    progress: 0,
    tags: [],
    createdAt: "2026-03-15T00:00:00Z",
    updatedAt: "2026-03-15T00:00:00Z",
    subtasks: [],
    ...overrides,
  }
}

describe("KanbanBoard column rendering", () => {
  it("renders five column headers: Planned, In Progress, Review, Done, Blocked", () => {
    render(<KanbanBoard tasks={[]} />)
    expect(screen.getAllByText("Planned").length).toBeGreaterThan(0)
    expect(screen.getAllByText("In Progress").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Review").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Done").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Blocked").length).toBeGreaterThan(0)
  })

  it("task with status 'planning' appears in Planned column", () => {
    const task = makeTask({ id: "t1", title: "Plan Me", status: "planning" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Plan Me").length).toBeGreaterThan(0)
  })

  it("task with status 'in_progress' appears in In Progress column", () => {
    const task = makeTask({ id: "t2", title: "In Progress Task", status: "in_progress" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("In Progress Task").length).toBeGreaterThan(0)
  })

  it("task with status 'done' appears in Done column", () => {
    const task = makeTask({ id: "t3", title: "Done Task", status: "done" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Done Task").length).toBeGreaterThan(0)
  })

  it("task with status 'paused' appears in Blocked column", () => {
    const task = makeTask({ id: "t4", title: "Paused Task", status: "paused" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Paused Task").length).toBeGreaterThan(0)
  })

  it("empty column shows 'Empty' text", () => {
    render(<KanbanBoard tasks={[]} />)
    const empties = screen.getAllByText("Empty")
    expect(empties.length).toBeGreaterThanOrEqual(5)
  })

  it("column header shows task count badge", () => {
    const tasks = [
      makeTask({ id: "t5", title: "Task A", status: "planning" }),
      makeTask({ id: "t6", title: "Task B", status: "planning" }),
    ]
    render(<KanbanBoard tasks={tasks} />)
    expect(screen.getAllByText("2").length).toBeGreaterThan(0)
  })
})

describe("KanbanBoard agent identity badges", () => {
  it("task with assignee 'claude' shows 'Claude Code' badge", () => {
    const task = makeTask({ id: "t7", title: "Claude Task", assignee: "claude" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Claude Code").length).toBeGreaterThan(0)
  })

  it("task with assignee 'codex-a1b2' shows 'Codex' badge", () => {
    const task = makeTask({ id: "t8", title: "Codex Task", assignee: "codex-a1b2" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Codex").length).toBeGreaterThan(0)
  })

  it("task with assignee 'codex-retry-c3d4' shows 'Codex (retry)' badge", () => {
    const task = makeTask({ id: "t9", title: "Retry Task", assignee: "codex-retry-c3d4" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Codex (retry)").length).toBeGreaterThan(0)
  })

  it("task with lastError shows red 'Error' badge", () => {
    const task = makeTask({ id: "t10", title: "Failed Task", lastError: "Something went wrong" })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Error").length).toBeGreaterThan(0)
  })

  it("task with executionLog containing task_failed entry with codex_diagnostics.reason_code shows reason label", () => {
    const task = makeTask({
      id: "t11",
      title: "Exhausted Task",
      lastError: "Iteration limit reached",
      executionLog: [
        {
          id: "log-1",
          timestamp: "2026-03-15T00:00:00Z",
          event: "task_failed",
          phase: "execution",
          level: "error",
          message: "Task failed",
          details: {
            codex_diagnostics: {
              reason_code: "max_iterations_exhausted",
            },
          },
        },
      ],
    })
    render(<KanbanBoard tasks={[task]} />)
    expect(screen.getAllByText("Iteration limit").length).toBeGreaterThan(0)
  })
})

describe("agentLabel helper", () => {
  it("returns 'Claude Code' for 'claude' assignee", () => {
    const { label } = agentLabel("claude")
    expect(label).toBe("Claude Code")
  })

  it("returns 'Codex (retry)' for assignee starting with 'codex-retry'", () => {
    const { label } = agentLabel("codex-retry-c3d4")
    expect(label).toBe("Codex (retry)")
  })

  it("returns 'Codex' for assignee starting with 'codex'", () => {
    const { label } = agentLabel("codex-a1b2")
    expect(label).toBe("Codex")
  })

  it("returns assignee as label for unknown assignees", () => {
    const { label } = agentLabel("opencode-xyz")
    expect(label).toBe("opencode-xyz")
  })
})

describe("extractCodexFailureReason helper", () => {
  it("extracts reason_code from executionLog task_failed entry", () => {
    const task = makeTask({
      executionLog: [
        {
          id: "log-1",
          timestamp: "2026-03-15T00:00:00Z",
          event: "task_failed",
          phase: "execution",
          level: "error",
          message: "Task failed",
          details: { codex_diagnostics: { reason_code: "stagnation_detected" } },
        },
      ],
    })
    expect(extractCodexFailureReason(task)).toBe("stagnation_detected")
  })

  it("falls back to task.lastError when no executionLog entry matches", () => {
    const task = makeTask({ lastError: "timeout error" })
    expect(extractCodexFailureReason(task)).toBe("timeout error")
  })

  it("returns null when no executionLog and no lastError", () => {
    const task = makeTask()
    expect(extractCodexFailureReason(task)).toBeNull()
  })
})
