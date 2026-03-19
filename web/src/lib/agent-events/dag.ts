import type { Node, Edge } from "@xyflow/react"
import type { AgentTask, AgentTaskStatus } from "@/lib/store/studio-store"
import type { ScoreEdgeEntry } from "./types"

const COL_X = 240
const ROW_Y = 120

export function taskStatusToDagStyle(status: AgentTaskStatus): string {
  switch (status) {
    case "done":
    case "human_review":
      return "border-green-500 bg-green-50"
    case "in_progress":
      return "border-blue-500 bg-blue-50"
    case "repairing":
      return "border-amber-500 bg-amber-50"
    case "cancelled":
    case "paused":
      return "border-slate-300 bg-slate-100"
    default:
      return "border-slate-300 bg-white"
  }
}

export function computeTaskDepths(tasks: AgentTask[]): Map<string, number> {
  const idSet = new Set(tasks.map((t) => t.id))
  const depthMap = new Map<string, number>()

  // Iterative depth computation: keep resolving until stable
  // Initialize all to 0
  for (const task of tasks) {
    depthMap.set(task.id, 0)
  }

  // Topological resolution with a cap to handle circular references
  const maxIterations = tasks.length + 1
  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false
    for (const task of tasks) {
      const deps = task.depends_on ?? []
      if (deps.length === 0) continue
      let maxDepDepth = -1
      for (const depId of deps) {
        if (idSet.has(depId)) {
          const depDepth = depthMap.get(depId) ?? 0
          if (depDepth > maxDepDepth) maxDepDepth = depDepth
        } else {
          // Missing dependency treated as depth 0
          if (maxDepDepth < 0) maxDepDepth = 0
        }
      }
      const newDepth = maxDepDepth + 1
      if ((depthMap.get(task.id) ?? 0) !== newDepth) {
        depthMap.set(task.id, newDepth)
        changed = true
      }
    }
    if (!changed) break
  }

  return depthMap
}

export function buildDagNodes(tasks: AgentTask[]): Node[] {
  const depthMap = computeTaskDepths(tasks)

  // Group tasks by depth to assign row positions
  const byDepth = new Map<number, string[]>()
  for (const task of tasks) {
    const depth = depthMap.get(task.id) ?? 0
    if (!byDepth.has(depth)) byDepth.set(depth, [])
    byDepth.get(depth)!.push(task.id)
  }

  // Build row index lookup: taskId -> rowIndex within its depth column
  const rowIndex = new Map<string, number>()
  for (const [, ids] of byDepth) {
    ids.forEach((id, idx) => rowIndex.set(id, idx))
  }

  return tasks.map((task) => {
    const depth = depthMap.get(task.id) ?? 0
    const row = rowIndex.get(task.id) ?? 0
    return {
      id: task.id,
      type: "taskDag",
      position: { x: depth * COL_X, y: row * ROW_Y },
      data: { task },
      draggable: false,
      selectable: false,
    }
  })
}

export function buildDagEdges(tasks: AgentTask[], scoreEdges: ScoreEdgeEntry[]): Edge[] {
  const idSet = new Set(tasks.map((t) => t.id))
  const edges: Edge[] = []

  // Dependency edges
  for (const task of tasks) {
    const deps = task.depends_on ?? []
    for (const depId of deps) {
      if (!idSet.has(depId)) continue
      edges.push({
        id: `dep-${depId}-${task.id}`,
        source: depId,
        target: task.id,
        type: "smoothstep",
      })
    }
  }

  // ScoreShareBus edges
  for (const entry of scoreEdges) {
    edges.push({
      id: `score-${entry.id}`,
      source: entry.from_agent,
      target: entry.to_agent,
      type: "scoreFlow",
      label: `${entry.stage}: ${entry.score.toFixed(2)}`,
      style: { strokeDasharray: "5 3" },
      animated: false,
    })
  }

  return edges
}
