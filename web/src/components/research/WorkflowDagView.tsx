"use client"

import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

type StepStatus = "pending" | "running" | "done" | "error" | "skipped"

type DagNodeData = {
  label: string
  status: StepStatus
  meta?: string
}

type WorkflowDagViewProps = {
  statuses: Record<string, StepStatus>
  queriesCount: number
  hitCount: number
  uniqueCount: number
  llmEnabled: boolean
  judgeEnabled: boolean
}

function statusStyle(status: StepStatus) {
  if (status === "done") {
    return "border-green-500 bg-green-50"
  }
  if (status === "running") {
    return "border-blue-500 bg-blue-50"
  }
  if (status === "error") {
    return "border-red-500 bg-red-50"
  }
  if (status === "skipped") {
    return "border-slate-300 bg-slate-100"
  }
  return "border-slate-300 bg-white"
}

function statusText(status: StepStatus) {
  if (status === "done") return "done"
  if (status === "running") return "running"
  if (status === "error") return "error"
  if (status === "skipped") return "skipped"
  return "pending"
}

function StatusNode({ data }: NodeProps<Node<DagNodeData>>) {
  return (
    <div className={`min-w-[170px] rounded-md border px-3 py-2 text-xs shadow-sm ${statusStyle(data.status)}`}>
      <Handle type="target" position={Position.Left} className="!h-2 !w-2" />
      <div className="font-medium text-sm">{data.label}</div>
      <div className="text-[11px] text-muted-foreground">{statusText(data.status)}</div>
      {data.meta ? <div className="mt-1 text-[11px] text-muted-foreground">{data.meta}</div> : null}
      <Handle type="source" position={Position.Right} className="!h-2 !w-2" />
    </div>
  )
}

const nodeTypes = { statusNode: StatusNode }

function buildNodes(args: WorkflowDagViewProps): Array<Node<DagNodeData>> {
  return [
    {
      id: "source",
      type: "statusNode",
      position: { x: 20, y: 90 },
      data: {
        label: "Sources",
        status: args.statuses.source,
        meta: args.queriesCount > 0 ? `queries=${args.queriesCount}` : "queries=0",
      },
    },
    {
      id: "normalize",
      type: "statusNode",
      position: { x: 230, y: 90 },
      data: {
        label: "Normalize",
        status: args.statuses.normalize,
        meta: "alias + tokenization",
      },
    },
    {
      id: "search",
      type: "statusNode",
      position: { x: 440, y: 90 },
      data: {
        label: "Search",
        status: args.statuses.search,
        meta: args.hitCount > 0 ? `hits=${args.hitCount}` : "await search",
      },
    },
    {
      id: "rank",
      type: "statusNode",
      position: { x: 650, y: 90 },
      data: {
        label: "Dedupe + Rank",
        status: args.statuses.rank,
        meta: args.uniqueCount > 0 ? `unique=${args.uniqueCount}` : "await ranking",
      },
    },
    {
      id: "llm",
      type: "statusNode",
      position: { x: 860, y: 20 },
      data: {
        label: "LLM Enrichment",
        status: args.statuses.llm,
        meta: args.llmEnabled ? "summary/trends/insight" : "disabled",
      },
    },
    {
      id: "judge",
      type: "statusNode",
      position: { x: 860, y: 160 },
      data: {
        label: "LLM Judge",
        status: args.statuses.judge,
        meta: args.judgeEnabled ? "multidim scoring" : "disabled",
      },
    },
    {
      id: "report",
      type: "statusNode",
      position: { x: 1080, y: 90 },
      data: {
        label: "DailyPaper Report",
        status: args.statuses.report,
        meta: "markdown + json",
      },
    },
    {
      id: "scheduler",
      type: "statusNode",
      position: { x: 1290, y: 90 },
      data: {
        label: "Scheduler/Feed",
        status: args.statuses.scheduler,
        meta: "cron + feed bridge",
      },
    },
  ]
}

function buildEdges(): Edge[] {
  return [
    ["source", "normalize"],
    ["normalize", "search"],
    ["search", "rank"],
    ["rank", "llm"],
    ["rank", "judge"],
    ["llm", "report"],
    ["judge", "report"],
    ["report", "scheduler"],
  ].map(([source, target]) => ({
    id: `${source}-${target}`,
    source,
    target,
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
    animated: source === "rank",
  }))
}

export default function WorkflowDagView(props: WorkflowDagViewProps) {
  const nodes = buildNodes(props)
  const edges = buildEdges()

  return (
    <div className="h-[280px] w-full rounded-md border bg-muted/20">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ maxZoom: 1.2, minZoom: 0.6 }}
        nodesConnectable={false}
        nodesDraggable={false}
        elementsSelectable={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} size={1} />
        <MiniMap pannable zoomable />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
