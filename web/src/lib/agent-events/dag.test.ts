import { describe, expect, it } from "vitest"
import { computeTaskDepths, buildDagNodes, buildDagEdges, taskStatusToDagStyle } from "./dag"
import type { AgentTask, AgentTaskStatus } from "@/lib/store/studio-store"
import type { ScoreEdgeEntry } from "./types"

function makeTask(
  id: string,
  title: string,
  status: AgentTaskStatus = "planning",
  depends_on?: string[],
): AgentTask {
  return {
    id,
    title,
    description: "Test task",
    status,
    assignee: "claude",
    progress: 0,
    tags: [],
    createdAt: "2026-03-15T00:00:00Z",
    updatedAt: "2026-03-15T00:00:00Z",
    subtasks: [],
    depends_on,
  }
}

function makeScoreEdge(fromAgent: string, toAgent: string, stage: string, score: number): ScoreEdgeEntry {
  return {
    id: `${fromAgent}-${toAgent}-${stage}`,
    from_agent: fromAgent,
    to_agent: toAgent,
    stage,
    score,
    ts: "2026-03-15T05:00:00Z",
  }
}

describe("computeTaskDepths", () => {
  it("returns depth 0 for tasks with no depends_on", () => {
    const tasks = [makeTask("t1", "Task 1"), makeTask("t2", "Task 2")]
    const depths = computeTaskDepths(tasks)
    expect(depths.get("t1")).toBe(0)
    expect(depths.get("t2")).toBe(0)
  })

  it("returns depth 1 for tasks depending on a depth-0 task", () => {
    const tasks = [makeTask("t1", "Task 1"), makeTask("t2", "Task 2", "planning", ["t1"])]
    const depths = computeTaskDepths(tasks)
    expect(depths.get("t1")).toBe(0)
    expect(depths.get("t2")).toBe(1)
  })

  it("returns depth 2 for a task depending on a depth-1 task (transitive)", () => {
    const tasks = [
      makeTask("t1", "Task 1"),
      makeTask("t2", "Task 2", "planning", ["t1"]),
      makeTask("t3", "Task 3", "planning", ["t2"]),
    ]
    const depths = computeTaskDepths(tasks)
    expect(depths.get("t1")).toBe(0)
    expect(depths.get("t2")).toBe(1)
    expect(depths.get("t3")).toBe(2)
  })

  it("handles missing depends_on gracefully (treats as depth 0)", () => {
    const task = makeTask("t1", "Task 1")
    // depends_on is undefined
    const depths = computeTaskDepths([task])
    expect(depths.get("t1")).toBe(0)
  })

  it("handles empty depends_on array (treats as depth 0)", () => {
    const task = makeTask("t1", "Task 1", "planning", [])
    const depths = computeTaskDepths([task])
    expect(depths.get("t1")).toBe(0)
  })

  it("handles depends_on referencing non-existent task ids (defaults dependency to 0)", () => {
    const tasks = [makeTask("t1", "Task 1", "planning", ["missing-id"])]
    const depths = computeTaskDepths(tasks)
    // missing-id not in task list → treated as 0 depth, so t1 = max(0)+1 = 1
    expect(depths.get("t1")).toBe(1)
  })

  it("uses max depth when task depends on multiple tasks at different depths", () => {
    const tasks = [
      makeTask("t1", "Task 1"),
      makeTask("t2", "Task 2", "planning", ["t1"]),
      makeTask("t3", "Task 3", "planning", ["t1", "t2"]),
    ]
    const depths = computeTaskDepths(tasks)
    // t3 depends on t1 (depth 0) and t2 (depth 1) → max = 1, so t3 = 2
    expect(depths.get("t3")).toBe(2)
  })
})

describe("taskStatusToDagStyle", () => {
  it("returns green style for 'done'", () => {
    expect(taskStatusToDagStyle("done")).toBe("border-green-500 bg-green-50")
  })

  it("returns green style for 'human_review'", () => {
    expect(taskStatusToDagStyle("human_review")).toBe("border-green-500 bg-green-50")
  })

  it("returns blue style for 'in_progress'", () => {
    expect(taskStatusToDagStyle("in_progress")).toBe("border-blue-500 bg-blue-50")
  })

  it("returns amber style for 'repairing'", () => {
    expect(taskStatusToDagStyle("repairing")).toBe("border-amber-500 bg-amber-50")
  })

  it("returns slate-100 style for 'cancelled'", () => {
    expect(taskStatusToDagStyle("cancelled")).toBe("border-slate-300 bg-slate-100")
  })

  it("returns slate-100 style for 'paused'", () => {
    expect(taskStatusToDagStyle("paused")).toBe("border-slate-300 bg-slate-100")
  })

  it("returns white style for 'planning' (default)", () => {
    expect(taskStatusToDagStyle("planning")).toBe("border-slate-300 bg-white")
  })
})

describe("buildDagNodes", () => {
  it("returns one Node per task", () => {
    const tasks = [makeTask("t1", "Task 1"), makeTask("t2", "Task 2")]
    const nodes = buildDagNodes(tasks)
    expect(nodes).toHaveLength(2)
  })

  it("sets node type to 'taskDag'", () => {
    const tasks = [makeTask("t1", "Task 1")]
    const nodes = buildDagNodes(tasks)
    expect(nodes[0].type).toBe("taskDag")
  })

  it("positions depth-0 tasks at x=0", () => {
    const tasks = [makeTask("t1", "Task 1")]
    const nodes = buildDagNodes(tasks)
    expect(nodes[0].position.x).toBe(0)
  })

  it("positions depth-1 tasks at x=240 (COL_X)", () => {
    const tasks = [
      makeTask("t1", "Task 1"),
      makeTask("t2", "Task 2", "planning", ["t1"]),
    ]
    const nodes = buildDagNodes(tasks)
    const t2Node = nodes.find((n) => n.id === "t2")
    expect(t2Node?.position.x).toBe(240)
  })

  it("positions tasks at y=rowIndex * 120 (ROW_Y) within same depth", () => {
    const tasks = [
      makeTask("t1", "Task 1"),
      makeTask("t2", "Task 2"),
    ]
    const nodes = buildDagNodes(tasks)
    const ys = nodes.map((n) => n.position.y).sort((a, b) => a - b)
    expect(ys[0]).toBe(0)
    expect(ys[1]).toBe(120)
  })

  it("sets node id to task.id", () => {
    const tasks = [makeTask("my-task-id", "My Task")]
    const nodes = buildDagNodes(tasks)
    expect(nodes[0].id).toBe("my-task-id")
  })

  it("sets node data to { task }", () => {
    const task = makeTask("t1", "Task 1")
    const nodes = buildDagNodes([task])
    expect(nodes[0].data).toEqual({ task })
  })

  it("sets draggable=false and selectable=false", () => {
    const tasks = [makeTask("t1", "Task 1")]
    const nodes = buildDagNodes(tasks)
    expect(nodes[0].draggable).toBe(false)
    expect(nodes[0].selectable).toBe(false)
  })

  it("returns empty array for empty task list", () => {
    expect(buildDagNodes([])).toEqual([])
  })
})

describe("buildDagEdges", () => {
  it("produces dependency edges with type 'smoothstep' from depends_on entries", () => {
    const tasks = [
      makeTask("t1", "Task 1"),
      makeTask("t2", "Task 2", "planning", ["t1"]),
    ]
    const edges = buildDagEdges(tasks, [])
    const depEdge = edges.find((e) => e.id === "dep-t1-t2")
    expect(depEdge).toBeDefined()
    expect(depEdge?.type).toBe("smoothstep")
    expect(depEdge?.source).toBe("t1")
    expect(depEdge?.target).toBe("t2")
  })

  it("produces ScoreShareBus edges with type 'scoreFlow' from ScoreEdgeEntry[]", () => {
    const tasks = [makeTask("t1", "Task 1")]
    const scoreEdge = makeScoreEdge("research", "scholar_pipeline", "research", 0.85)
    const edges = buildDagEdges(tasks, [scoreEdge])
    const flowEdge = edges.find((e) => e.id === "score-research-scholar_pipeline-research")
    expect(flowEdge).toBeDefined()
    expect(flowEdge?.type).toBe("scoreFlow")
    expect(flowEdge?.source).toBe("research")
    expect(flowEdge?.target).toBe("scholar_pipeline")
  })

  it("skips dependency edge if target task id is not in task list", () => {
    const tasks = [makeTask("t2", "Task 2", "planning", ["nonexistent"])]
    const edges = buildDagEdges(tasks, [])
    expect(edges.filter((e) => e.type === "smoothstep")).toHaveLength(0)
  })

  it("score edges include label with stage and score", () => {
    const tasks: AgentTask[] = []
    const scoreEdge = makeScoreEdge("code", "pipeline", "code", 0.75)
    const edges = buildDagEdges(tasks, [scoreEdge])
    const flowEdge = edges.find((e) => e.type === "scoreFlow")
    expect(flowEdge?.label).toBe("code: 0.75")
  })

  it("score edges have strokeDasharray style", () => {
    const tasks: AgentTask[] = []
    const scoreEdge = makeScoreEdge("code", "pipeline", "code", 0.75)
    const edges = buildDagEdges(tasks, [scoreEdge])
    const flowEdge = edges.find((e) => e.type === "scoreFlow")
    expect((flowEdge?.style as Record<string, unknown>)?.strokeDasharray).toBe("5 3")
  })

  it("score edges have animated=false", () => {
    const tasks: AgentTask[] = []
    const scoreEdge = makeScoreEdge("code", "pipeline", "code", 0.75)
    const edges = buildDagEdges(tasks, [scoreEdge])
    const flowEdge = edges.find((e) => e.type === "scoreFlow")
    expect(flowEdge?.animated).toBe(false)
  })

  it("returns empty array for empty tasks and empty scoreEdges", () => {
    expect(buildDagEdges([], [])).toEqual([])
  })

  it("produces no dependency edges for tasks with no depends_on", () => {
    const tasks = [makeTask("t1", "Task 1"), makeTask("t2", "Task 2")]
    const edges = buildDagEdges(tasks, [])
    expect(edges.filter((e) => e.type === "smoothstep")).toHaveLength(0)
  })
})
