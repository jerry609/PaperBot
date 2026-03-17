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
import { resolveSelectedWorkerGroup } from "@/lib/agent-events/worker-focus"
import { Handle, Position } from "@xyflow/react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { AgentTask } from "@/lib/store/studio-store"
import { getAgentPresentation } from "@/lib/agent-runtime"

/* ── TaskDagNode ── */

function TaskDagNode({
  data,
}: {
  data: {
    task: AgentTask
    workerRunId?: string | null
    onOpenWorker?: ((workerRunId: string) => void) | null
    isFocused?: boolean
    isDimmed?: boolean
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
      className={`min-w-[160px] max-w-[220px] rounded-lg border-2 px-3 py-2 text-left ${style} ${
        interactive ? "cursor-pointer transition-shadow hover:shadow-md" : "cursor-default"
      } ${data.isFocused ? "ring-2 ring-slate-900 ring-offset-2" : ""} ${data.isDimmed ? "opacity-45" : ""}`}
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
          {data.isFocused ? "Focused worker" : "Open worker detail"}
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
  const selectedWorkerRunId = useAgentEventStore((s) => s.selectedWorkerRunId)
  const setSelectedWorkerRunId = useAgentEventStore((s) => s.setSelectedWorkerRunId)

  const tasks = studioTasks.length > 0 ? studioTasks : eventKanbanTasks
  const workerGroups = useMemo(
    () => buildSubagentActivityGroups(codexDelegations, toolCalls),
    [codexDelegations, toolCalls],
  )
  const workerRunIdByTaskId = useMemo(() => {
    return new Map(workerGroups.map((group) => [group.taskId, group.workerRunId]))
  }, [workerGroups])
  const selectedWorkerGroup = useMemo(
    () => resolveSelectedWorkerGroup(selectedWorkerRunId, codexDelegations, toolCalls),
    [codexDelegations, selectedWorkerRunId, toolCalls],
  )
  const focusedTaskId = selectedWorkerGroup?.taskId ?? null
  const focusedPresentation = selectedWorkerGroup
    ? getAgentPresentation(selectedWorkerGroup.assignee)
    : null

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
            isFocused: focusedTaskId === task.id,
            isDimmed: Boolean(focusedTaskId) && focusedTaskId !== task.id,
          },
        }
      }),
    [tasks, workerRunIdByTaskId, openWorkerRun, focusedTaskId],
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
    <div className="relative h-full w-full" data-testid="agent-dag-panel">
      {selectedWorkerGroup ? (
        <div className="absolute left-3 top-3 z-10 flex max-w-[340px] items-center justify-between gap-2 rounded-2xl border border-slate-200 bg-white/95 px-3 py-2 shadow-sm backdrop-blur">
          <div className="min-w-0">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
              Focused worker
            </div>
            <div className="truncate text-[12px] font-medium text-slate-800">
              {focusedPresentation?.label ? `${focusedPresentation.label} · ` : ""}
              {selectedWorkerGroup.taskTitle || "Untitled task"}
            </div>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 rounded-full px-2 text-[11px] text-slate-600"
            onClick={() => setSelectedWorkerRunId(null)}
          >
            Clear
          </Button>
        </div>
      ) : null}
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
