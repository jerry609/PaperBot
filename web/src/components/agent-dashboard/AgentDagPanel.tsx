"use client"

import { useMemo } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { useAgentEventStore } from "@/lib/agent-events/store"
import { useStudioStore } from "@/lib/store/studio-store"
import { buildDagNodes, buildDagEdges, taskStatusToDagStyle } from "@/lib/agent-events/dag"
import { buildSubagentActivityGroups } from "@/lib/agent-events/subagent-groups"
import { Handle, Position } from "@xyflow/react"
import { Badge } from "@/components/ui/badge"
import type { AgentTask } from "@/lib/store/studio-store"

/* ── TaskDagNode ── */

function TaskDagNode({
  data,
}: {
  data: {
    task: AgentTask
    workerRunId?: string | null
    onOpenWorker?: ((workerRunId: string) => void) | null
  }
}) {
  const { task } = data
  const style = taskStatusToDagStyle(task.status)
  const interactive = Boolean(data.workerRunId && data.onOpenWorker)

  return (
    <button
      type="button"
      onClick={() => {
        if (data.workerRunId && data.onOpenWorker) {
          data.onOpenWorker(data.workerRunId)
        }
      }}
      disabled={!interactive}
      className={`rounded-lg border-2 px-3 py-2 min-w-[160px] max-w-[220px] text-left ${style} ${
        interactive ? "cursor-pointer transition-shadow hover:shadow-md" : "cursor-default"
      }`}
      title={interactive ? "Open worker details" : task.title}
    >
      <Handle type="target" position={Position.Left} className="!bg-slate-400" />
      <div className="text-xs font-semibold truncate">{task.title}</div>
      <div className="flex items-center gap-1.5 mt-1">
        <Badge variant="outline" className="text-[10px] px-1 py-0">
          {task.status}
        </Badge>
        {task.assignee && (
          <span className="text-[10px] text-muted-foreground truncate">
            {task.assignee}
          </span>
        )}
      </div>
      {interactive ? (
        <div className="mt-1.5 text-[10px] text-muted-foreground">
          Open worker detail
        </div>
      ) : null}
      <Handle type="source" position={Position.Right} className="!bg-slate-400" />
    </button>
  )
}

const nodeTypes = { taskDag: TaskDagNode }
const edgeTypes = {}

/* ── AgentDagPanel ── */

export function AgentDagPanel() {
  const studioTasks = useStudioStore((s) => s.agentTasks)
  const eventKanbanTasks = useAgentEventStore((s) => s.kanbanTasks)
  const scoreEdges = useAgentEventStore((s) => s.scoreEdges)
  const codexDelegations = useAgentEventStore((s) => s.codexDelegations)
  const toolCalls = useAgentEventStore((s) => s.toolCalls)
  const openWorkerRun = useAgentEventStore((s) => s.openWorkerRun)

  const tasks = studioTasks.length > 0 ? studioTasks : eventKanbanTasks
  const workerRunIdByTaskId = useMemo(() => {
    const groups = buildSubagentActivityGroups(codexDelegations, toolCalls)
    return new Map(groups.map((group) => [group.taskId, group.workerRunId]))
  }, [codexDelegations, toolCalls])

  const nodes: Node[] = useMemo(
    () =>
      buildDagNodes(tasks).map((node) => {
        const task = (node.data as { task: AgentTask }).task
        return {
          ...node,
          data: {
            task,
            workerRunId: workerRunIdByTaskId.get(task.id) ?? null,
            onOpenWorker: openWorkerRun,
          },
        }
      }),
    [tasks, workerRunIdByTaskId, openWorkerRun],
  )
  const edges: Edge[] = useMemo(() => buildDagEdges(tasks, scoreEdges), [tasks, scoreEdges])

  if (tasks.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        No tasks to visualize
      </div>
    )
  }

  return (
    <div className="h-full w-full" data-testid="agent-dag-panel">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
