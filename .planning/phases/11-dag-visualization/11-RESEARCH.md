# Phase 11: DAG Visualization - Research

**Researched:** 2026-03-15
**Domain:** @xyflow/react 12.x — real-time DAG visualization in Next.js 16 / React 19
**Confidence:** HIGH

## Summary

Phase 11 adds an interactive task-dependency DAG to the Agent Dashboard with two distinguishing features: (1) node colors update in real-time based on task status derived from the existing SSE event stream already flowing through Zustand, and (2) a second class of edges visualizes ScoreShareBus data flow (cross-agent evaluation context sharing) overlaid on the same canvas.

The codebase already uses @xyflow/react 12.10.0 (installed, proven) in two production files: `AgentBoard.tsx` (rich full-page board with `buildFlowNodes`/`buildFlowEdges` pattern and custom node/edge types) and `WorkflowDagView.tsx` (lightweight inline read-only DAG). Both are "use client" components. Pattern and API are fully established — there is nothing novel to discover about the library integration.

The core challenge is data model: `AgentTask` already carries `status` and `subtasks`, but it does not carry a `depends_on` list in the frontend type (`AgentTask` in `studio-store.ts`). The Python backend `TaskCheckpoint` model *does* have `depends_on: List[str]`, and the `TaskCheckpoint` TypeScript mirror in `p2c.ts` also has `depends_on: string[]`. The gap is that `AgentTask` (used by the Kanban board and store) has no `depends_on` field. Phase 11 must bridge this: either add `depends_on` to `AgentTask` or derive dependency edges from a parallel `TaskCheckpoint[]` data source. ScoreShareBus edges require a new event type (`score_update`) that already exists as `EventType.SCORE_UPDATE = "score_update"` in Python and is handled in `parsers.ts` as an activity feed item — but currently produces no structured graph edge data.

**Primary recommendation:** Add a new `AgentDagPanel` component under `web/src/components/agent-dashboard/` that reads from the existing Zustand stores, derives nodes from `kanbanTasks`, derives dependency edges from task `depends_on` (added to `AgentTask`), and derives ScoreShareBus edges from `score_update` events in the feed. Mount it as a fourth view mode on the agent-dashboard page alongside panels/kanban.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VIZ-01 | User can view an agent task dependency DAG with real-time status color updates | Covered by @xyflow/react node types + Zustand `kanbanTasks`/`agentStatuses` stores. Status-to-color mapping follows existing `AgentBoard.tsx` patterns. Real-time updates come from existing SSE hook `useAgentEvents()`. |
| VIZ-02 | User can see cross-agent context sharing (ScoreShareBus data flow) in the dashboard | Covered by parsing `score_update` events from the feed. New `ScoreEdge` entry type needed in Zustand store to track (from_agent, to_agent) sharing pairs. Backend already emits `score_update` via `ScoreShareBus.publish_score()` when `event_log` is wired. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| @xyflow/react | 12.10.0 (installed) | DAG rendering, node/edge management | Already used in AgentBoard and WorkflowDagView; no new deps needed |
| zustand | 5.0.9 (installed) | State management for DAG data | All agent event state already in Zustand; consistent with store pattern |
| React | 19.2.3 (installed) | Component runtime | Project standard |
| Next.js | 16.1.0 (installed) | "use client" boundary | All visualization is client-side |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @xyflow/react Background | bundled | Dot/grid background | Use `<Background gap={16} size={1} />` matching WorkflowDagView |
| @xyflow/react Controls | bundled | Zoom/pan controls | Use `showInteractive={false}` (read-only view) |
| @xyflow/react MarkerType | bundled | Arrowhead markers | Already used in both AgentBoard and WorkflowDagView |
| lucide-react | 0.562.0 (installed) | Icons inside custom nodes | Consistent with existing node components |
| tailwind-merge / clsx | installed | Conditional classes | Used everywhere in the codebase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Adding `AgentDagPanel` | Re-using AgentBoard directly | AgentBoard is P2C-pipeline-specific (has Commander, E2E, Download nodes); a purpose-built panel is cleaner for agent-task DAG |
| Deriving edges from `depends_on` | Inferring order from task creation timestamps | `depends_on` is explicit and matches backend TaskCheckpoint model; timestamps are unreliable |
| New ScoreEdge store slice | Filtering feed for score_update events at render time | Store slice is consistent with other structured data (codexDelegations, toolCalls) and avoids re-scanning feed on every render |

**No new npm packages required.** Everything needed is already installed.

## Architecture Patterns

### Recommended Project Structure
```
web/src/
├── components/agent-dashboard/
│   ├── AgentDagPanel.tsx         # New: top-level DAG component (VIZ-01 + VIZ-02)
│   ├── AgentDagNodes.tsx         # New: TaskDagNode custom node type
│   └── AgentDagEdges.tsx         # New: ScoreFlowEdge custom edge type
├── lib/agent-events/
│   ├── store.ts                  # Extend: add scoreEdges + addScoreEdge
│   ├── types.ts                  # Extend: add ScoreEdgeEntry type
│   ├── parsers.ts                # Extend: add parseScoreEdge() function
│   └── useAgentEvents.ts         # Extend: call parseScoreEdge and addScoreEdge
└── app/agent-dashboard/
    └── page.tsx                  # Extend: add "dag" viewMode option + AgentDagPanel
```

`AgentTask` in `studio-store.ts` gains an optional `depends_on?: string[]` field (matches Python `TaskCheckpoint.depends_on`).

### Pattern 1: Custom Node Type (TaskDagNode)
**What:** Read-only node that renders task title, status badge, and assignee icon; border color reflects task status.
**When to use:** For VIZ-01 — one node per `AgentTask`, colors update when Zustand `kanbanTasks` changes.
**Example:**
```typescript
// Pattern from AgentBoardNodes.tsx and WorkflowDagView.tsx
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"

export type TaskDagNodeData = {
  task: AgentTask
}

export function TaskDagNode({ data }: NodeProps<Node<TaskDagNodeData>>) {
  const borderColor = taskStatusToColor(data.task.status)
  return (
    <>
      <Handle type="target" position={Position.Left} className="!h-2 !w-2" />
      <div className={`rounded-md border-2 px-3 py-2 text-xs shadow-sm ${borderColor} bg-white min-w-[160px]`}>
        <div className="font-medium text-sm truncate">{data.task.title}</div>
        <div className="text-[11px] text-muted-foreground">{data.task.status}</div>
      </div>
      <Handle type="source" position={Position.Right} className="!h-2 !w-2" />
    </>
  )
}
```

### Pattern 2: ScoreShareBus Edge Type (ScoreFlowEdge)
**What:** Dashed animated edge representing ScoreShareBus data flow between agents, labeled with the stage name.
**When to use:** For VIZ-02 — one edge per unique (from_agent, to_agent, stage) triple derived from `score_update` events.
**Example:**
```typescript
// Pattern from AgentBoardEdges.tsx — getSmoothStepPath + animateMotion
import { getSmoothStepPath, EdgeLabelRenderer, type EdgeProps } from "@xyflow/react"

export function ScoreFlowEdge({
  id, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, markerEnd, label,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition, borderRadius: 8,
  })
  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke="#8b5cf6"
        strokeWidth={1.5}
        strokeDasharray="5 3"
        markerEnd={markerEnd as string}
      />
      <EdgeLabelRenderer>
        <div
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}
          className="absolute text-[10px] text-purple-500 bg-white border border-purple-200 rounded px-1 pointer-events-none"
        >
          {label as string}
        </div>
      </EdgeLabelRenderer>
    </>
  )
}
```

### Pattern 3: Node/Edge Derivation with useMemo
**What:** Compute `nodes[]` and `edges[]` from Zustand state using `useMemo` to avoid re-computing on every render.
**When to use:** Consistent with AgentBoard.tsx — all node/edge building is pure functions over Zustand state, called inside `useMemo`.
**Example:**
```typescript
// Source: AgentBoard.tsx lines 659-688
const kanbanTasks = useAgentEventStore((s) => s.kanbanTasks)
const scoreEdges = useAgentEventStore((s) => s.scoreEdges)

const nodes = useMemo(() => buildDagNodes(kanbanTasks), [kanbanTasks])
const edges = useMemo(
  () => buildDagEdges(kanbanTasks, scoreEdges),
  [kanbanTasks, scoreEdges]
)
```

### Pattern 4: Score Edge Zustand Slice
**What:** Store unique (from_agent, to_agent, stage) triples extracted from `score_update` feed events. Dedup by composite key.
**When to use:** VIZ-02 — avoids scanning the entire feed on every render; consistent with `codexDelegations` slice pattern.

New Zustand fields in `store.ts`:
```typescript
// Mirroring the codexDelegations slice pattern
export type ScoreEdgeEntry = {
  id: string          // `${from_agent}-${to_agent}-${stage}` (dedup key)
  from_agent: string  // ScoreShareBus agent_name (producer)
  to_agent: string    // subscriber/consumer agent (via workflow/stage)
  stage: string       // score.stage (research/code/quality/influence)
  score: number
  ts: string
}

scoreEdges: ScoreEdgeEntry[]
addScoreEdge: (entry: ScoreEdgeEntry) => void  // dedup by id, upsert latest
```

### Pattern 5: Positional Layout for Task Nodes
**What:** Simple horizontal/grid layout for task nodes. Tasks without `depends_on` go in column 0; tasks with `depends_on` go in subsequent columns based on their depth in the dependency tree.
**When to use:** Static layout computed once in `buildDagNodes`. No dagre/elk layout library needed for the expected scale (< 20 tasks).

```typescript
function buildDagNodes(tasks: AgentTask[]): Node[] {
  // Compute depth of each task in the dependency graph
  const depthMap = computeTaskDepths(tasks)  // Map<taskId, depth>
  const colBuckets = new Map<number, AgentTask[]>()
  for (const task of tasks) {
    const depth = depthMap.get(task.id) ?? 0
    const col = colBuckets.get(depth) ?? []
    col.push(task)
    colBuckets.set(depth, col)
  }
  const nodes: Node[] = []
  const COL_X = 240, ROW_Y = 160
  for (const [depth, bucket] of colBuckets) {
    bucket.forEach((task, i) => {
      nodes.push({
        id: task.id,
        type: "taskDag",
        position: { x: depth * COL_X, y: i * ROW_Y },
        data: { task },
        draggable: false,
        selectable: false,
      })
    })
  }
  return nodes
}
```

### Pattern 6: Score Update Parser
**What:** `parseScoreEdge()` extracts a `ScoreEdgeEntry` from a `score_update` envelope.
**When to use:** Called in `useAgentEvents.ts` alongside existing parsers.

```typescript
// In parsers.ts
export function parseScoreEdge(raw: AgentEventEnvelopeRaw): ScoreEdgeEntry | null {
  if (raw.type !== "score_update") return null
  const payload = (raw.payload ?? {}) as Record<string, unknown>
  const score = (payload.score ?? {}) as Record<string, unknown>
  const stage = String(score.stage ?? raw.stage ?? "")
  const from_agent = String(raw.agent_name ?? "ScoreShareBus")
  // to_agent: derive from workflow/stage context; default to "all" until
  // backend emits explicit subscriber_agent in payload
  const to_agent = String(payload.subscriber_agent ?? raw.workflow ?? "pipeline")
  const scoreVal = typeof score.score === "number" ? score.score : 0
  const id = `${from_agent}-${to_agent}-${stage}`
  return { id, from_agent, to_agent, stage, score: scoreVal, ts: String(raw.ts ?? "") }
}
```

### Anti-Patterns to Avoid
- **Importing `dagre` or `elkjs`:** Not installed, not needed for < 20 tasks; use simple column-depth layout.
- **Storing nodes/edges in Zustand:** ReactFlow manages node/edge state internally via `useNodesState`/`useEdgesState` when dynamic. For read-only views (like ours), pass nodes/edges as props computed by `useMemo` — matching WorkflowDagView.tsx and AgentBoard.tsx patterns.
- **Using `ReactFlowProvider` unnecessarily:** Not required when `<ReactFlow>` is the top-level element and no sibling component calls `useReactFlow()`.
- **Duplicate SSE connections:** `useAgentEvents()` is already mounted at page root in `AgentDashboardPage`. Never mount it again inside `AgentDagPanel`. Read Zustand store directly.
- **Animating edges via `animated: true` prop:** The project uses custom edge types with `animateMotion` SVG element (see `AgentBoardEdges.tsx`) — use the same pattern for consistency, not the `animated` boolean prop.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph rendering | Custom SVG DAG renderer | @xyflow/react (already installed) | Pan/zoom, handles, markers, edge paths are complex; ReactFlow handles all of it |
| Node layout algorithm | Custom topology sort + layout | Simple column-depth layout (< 20 nodes) OR @dagrejs/dagre (not needed for v1) | At this scale, column depth layout is sufficient and avoids a new dependency |
| Real-time updates | Polling or second SSE connection | Read existing Zustand stores populated by `useAgentEvents()` | SSE connection already exists; adding a second one doubles load on server |
| Edge path geometry | Manual Bezier math | `getSmoothStepPath` from @xyflow/react | Already used in `AgentBoardEdges.tsx` |
| Status color mapping | Inline ternary chains | Pure function `taskStatusToColor(status)` | Same pattern as `statusStyle()` in WorkflowDagView.tsx |

**Key insight:** The entire infrastructure (SSE, Zustand, @xyflow/react) already exists. Phase 11 is purely compositional — wire existing data into new node/edge types.

## Common Pitfalls

### Pitfall 1: `depends_on` field missing from `AgentTask`
**What goes wrong:** `kanbanTasks` in Zustand has type `AgentTask[]` which has no `depends_on` field, so dependency edges cannot be derived from Kanban data.
**Why it happens:** `AgentTask` was defined for the Kanban board in Phase 10; `depends_on` lives in `TaskCheckpoint` (P2C model), a separate type.
**How to avoid:** Add `depends_on?: string[]` to `AgentTask` interface in `studio-store.ts`. When backend emits `task_update` events with `depends_on`, parse and store it. For the initial implementation, fall back gracefully: if `depends_on` is empty/absent, render tasks as flat (no inter-task dependency edges — only show ScoreShareBus edges).
**Warning signs:** Dependency edges all missing from DAG even when tasks have clear ordering.

### Pitfall 2: ScoreShareBus agent_name is "ScoreShareBus", not a real agent
**What goes wrong:** The Python `ScoreShareBus.publish_score()` emits events with `agent_name="ScoreShareBus"` and `role="system"`. Treating "ScoreShareBus" as both source and destination agent conflates the bus with actual agent actors.
**Why it happens:** The bus is middleware, not an agent. The event payload contains `paper_id` and the `score` dict but no explicit subscriber_agent field.
**How to avoid:** In `parseScoreEdge()`, use `raw.workflow` or `raw.stage` to infer the producer context. For VIZ-02, the edge represents ScoreShareBus broadcasting a score — model it as `from_agent=score.stage` (the pipeline stage that produced the score) to `to_agent=workflow` (the overall pipeline). This gives meaningful edge labels like "research → scholar_pipeline". Document this interpretation clearly.
**Warning signs:** All ScoreShareBus edges appear as "ScoreShareBus → ScoreShareBus" self-loops.

### Pitfall 3: @xyflow/react CSS not imported in new component files
**What goes wrong:** ReactFlow renders but nodes/edges look broken (missing default styles, invisible handles).
**Why it happens:** `"@xyflow/react/dist/style.css"` must be imported in the same file (or a parent) that renders `<ReactFlow>`. Both `WorkflowDagView.tsx` and `AgentBoard.tsx` import it.
**How to avoid:** Add `import "@xyflow/react/dist/style.css"` to `AgentDagPanel.tsx`.
**Warning signs:** Handles invisible, edges not connecting, zoom controls missing.

### Pitfall 4: `nodesDraggable={false}` vs ReactFlow internal state
**What goes wrong:** With `nodesDraggable={false}` and read-only ReactFlow, passing controlled `nodes`/`edges` props is correct. But if `useNodesState()` is used, changes from ReactFlow internals conflict with Zustand-derived nodes.
**Why it happens:** `WorkflowDagView.tsx` and AgentBoard both pass `nodes`/`edges` as derived props (not `useNodesState`), which is correct for read-only views.
**How to avoid:** Pass `nodes` and `edges` as computed props from `useMemo`. Do not use `useNodesState` unless the DAG needs to be draggable/connectable.
**Warning signs:** Node positions drift or reset unexpectedly.

### Pitfall 5: `proOptions={{ hideAttribution: true }}` needed for non-commercial use
**What goes wrong:** ReactFlow attribution overlay appears in the bottom right of the canvas.
**Why it happens:** Free tier requires attribution unless `proOptions.hideAttribution` is set.
**How to avoid:** Add `proOptions={{ hideAttribution: true }}` — consistent with `WorkflowDagView.tsx`.
**Warning signs:** "React Flow" watermark appears on canvas.

### Pitfall 6: Real-time color updates require Zustand selector granularity
**What goes wrong:** If `AgentDagPanel` subscribes to the entire `kanbanTasks` array, it re-renders on every task event even when the status hasn't changed.
**Why it happens:** Zustand returns a new array reference on every update.
**How to avoid:** Accept the re-render for now — `useMemo` in the node builder prevents expensive re-computation if the tasks array is reference-stable. The existing store pattern is acceptable for the scale (< 20 tasks).
**Warning signs:** Performance degradation with many events — not a concern at current scale.

## Code Examples

Verified patterns from existing codebase:

### Status Color Mapping (from WorkflowDagView.tsx)
```typescript
// Source: web/src/components/research/WorkflowDagView.tsx lines 33-47
type StepStatus = "pending" | "running" | "done" | "error" | "skipped"

function statusStyle(status: StepStatus) {
  if (status === "done") return "border-green-500 bg-green-50"
  if (status === "running") return "border-blue-500 bg-blue-50"
  if (status === "error") return "border-red-500 bg-red-50"
  if (status === "skipped") return "border-slate-300 bg-slate-100"
  return "border-slate-300 bg-white"
}

// Map AgentTaskStatus to this:
function taskStatusToDagStyle(status: AgentTaskStatus): string {
  if (status === "done" || status === "human_review") return "border-green-500 bg-green-50"
  if (status === "in_progress") return "border-blue-500 bg-blue-50"
  if (status === "repairing") return "border-amber-500 bg-amber-50"
  if (status === "cancelled") return "border-slate-300 bg-slate-100"
  return "border-slate-300 bg-white"
}
```

### Minimal ReactFlow Setup (from WorkflowDagView.tsx)
```typescript
// Source: web/src/components/research/WorkflowDagView.tsx lines 175-198
<div className="h-[400px] w-full rounded-md border bg-muted/20">
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
    <Controls showInteractive={false} />
  </ReactFlow>
</div>
```

### Custom Edge with getSmoothStepPath (from AgentBoardEdges.tsx)
```typescript
// Source: web/src/components/studio/AgentBoardEdges.tsx
import { getSmoothStepPath, type EdgeProps } from "@xyflow/react"

const [edgePath] = getSmoothStepPath({
  sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, borderRadius: 12,
})
// Then render <path d={edgePath} ... />
```

### Zustand slice pattern for new entry type (from store.ts — codexDelegations pattern)
```typescript
// Source: web/src/lib/agent-events/store.ts lines 8-9, 93-98
const SCORE_EDGES_MAX = 200  // one per unique (from, to, stage) triple

scoreEdges: ScoreEdgeEntry[]
addScoreEdge: (entry: ScoreEdgeEntry) => void

// Implementation:
addScoreEdge: (entry) =>
  set((s) => {
    // Upsert by id (dedup)
    const idx = s.scoreEdges.findIndex((e) => e.id === entry.id)
    if (idx !== -1) {
      const next = [...s.scoreEdges]
      next[idx] = entry
      return { scoreEdges: next }
    }
    return { scoreEdges: [entry, ...s.scoreEdges].slice(0, SCORE_EDGES_MAX) }
  }),
```

### Page-level view mode extension (from agent-dashboard/page.tsx)
```typescript
// Source: web/src/app/agent-dashboard/page.tsx
// Currently: "panels" | "kanban"
// Extend to: "panels" | "kanban" | "dag"

const [viewMode, setViewMode] = useState<"panels" | "kanban" | "dag">("panels")

// Add DAG button in header toolbar + conditional render:
{viewMode === "dag" && <AgentDagPanel />}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `animated: true` boolean on edges | Custom edge types with SVG `animateMotion` | @xyflow/react 11+ | Full control over animation style, color, dashing |
| `ReactFlowProvider` always required | Only needed when sibling components call `useReactFlow()` | @xyflow/react 11+ | Simpler setup for isolated DAG panels |
| `useNodesState`/`useEdgesState` for static graphs | Pass nodes/edges as computed props | v11+ | Cleaner for read-only views |
| Default layout only | `fitView` with `fitViewOptions` | v10+ | Consistent with both WorkflowDagView and AgentBoard |

**The @xyflow/react 12.x API is identical to the patterns already used in this codebase. No migration required.**

## Open Questions

1. **Does `AgentTask.depends_on` get populated in practice?**
   - What we know: `TaskCheckpoint` (Python) has `depends_on: List[str]`. The P2C stages emit tasks with `depends_on`. The Kanban `AgentTask` (TypeScript) does not have this field.
   - What's unclear: Whether the SSE task events from `/api/agent-board/sessions/{id}/run` include `depends_on` in their payloads.
   - Recommendation: Add `depends_on?: string[]` to `AgentTask` defensively. In `normalizeAgentTaskFromBackend()`, parse it if present. For the Phase 11 initial release, render a flat layout if no `depends_on` data arrives — do not block on this.

2. **Should ScoreShareBus edges show a "to_agent" or just broadcast arrows?**
   - What we know: `ScoreShareBus.publish_score()` notifies subscribers via callbacks, not via named agent identifiers. The event only has `agent_name="ScoreShareBus"` and `workflow`/`stage` fields.
   - What's unclear: Whether a meaningful `to_agent` can be derived without backend changes.
   - Recommendation: For VIZ-02, model ScoreShareBus edges as flowing from `stage` node to a central "Pipeline" sentinel node, or from agent to agent using `workflow` and `stage` as proxy. Add an optional `subscriber_agent` field to the `score_update` payload in Python for future precision — but do not block Phase 11 on this backend change. The visual representation is valid with available data.

3. **How wide is the DAG likely to be in practice?**
   - What we know: P2C pipelines have 5-15 tasks in practice. Scholar pipeline has 4-8 stages.
   - What's unclear: Whether the dashboard will show P2C tasks, scholar pipeline stages, or both.
   - Recommendation: Build for `AgentTask` (Kanban tasks) as the primary data source. Use `fitView` to auto-scale. The panel height should be at least 400px.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | vitest 2.1.4 |
| Config file | `web/vitest.config.ts` |
| Quick run command | `cd web && npm test -- --reporter=verbose src/lib/agent-events/` |
| Full suite command | `cd web && npm test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VIZ-01 | `buildDagNodes(tasks)` maps task status to correct border class | unit | `cd web && npm test -- src/lib/agent-events/dag.test.ts` | Wave 0 |
| VIZ-01 | `buildDagNodes(tasks)` places tasks with `depends_on` in correct column | unit | `cd web && npm test -- src/lib/agent-events/dag.test.ts` | Wave 0 |
| VIZ-02 | `parseScoreEdge()` returns null for non-score_update events | unit | `cd web && npm test -- src/lib/agent-events/parsers.test.ts` | Extend existing |
| VIZ-02 | `parseScoreEdge()` returns ScoreEdgeEntry with correct id for score_update | unit | `cd web && npm test -- src/lib/agent-events/parsers.test.ts` | Extend existing |
| VIZ-02 | `addScoreEdge()` deduplicates by id, upserts latest score | unit | `cd web && npm test -- src/lib/agent-events/store.test.ts` | Extend existing |
| VIZ-01+02 | `AgentDagPanel` renders ReactFlow with nodes and edges | smoke (jsdom) | `cd web && npm test -- src/components/agent-dashboard/AgentDagPanel.test.tsx` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cd web && npm test -- src/lib/agent-events/`
- **Per wave merge:** `cd web && npm test`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `web/src/lib/agent-events/dag.test.ts` — covers VIZ-01 node building logic
- [ ] `web/src/components/agent-dashboard/AgentDagPanel.test.tsx` — smoke render test (jsdom environment, mocks ReactFlow)
- Extend `web/src/lib/agent-events/parsers.test.ts` with `parseScoreEdge` cases
- Extend `web/src/lib/agent-events/store.test.ts` with `addScoreEdge` dedup cases

No new test infrastructure install needed — vitest + jsdom already configured in `web/vitest.config.ts`.

## Sources

### Primary (HIGH confidence)
- Existing codebase `web/src/components/studio/AgentBoard.tsx` — node building pattern, edge types, useMemo, SSE integration
- Existing codebase `web/src/components/research/WorkflowDagView.tsx` — minimal ReactFlow setup, custom node type, status colors
- Existing codebase `web/src/components/studio/AgentBoardEdges.tsx` — custom edge with getSmoothStepPath, animateMotion
- Existing codebase `web/src/components/studio/AgentBoardNodes.tsx` — Handle, FlowCard, badge patterns
- Existing codebase `web/src/lib/agent-events/store.ts` — Zustand slice pattern, CODEX_DELEGATIONS_MAX cap, upsert
- Existing codebase `web/src/lib/agent-events/parsers.ts` — parser function signature, score_update handling
- Existing codebase `web/src/lib/agent-events/useAgentEvents.ts` — SSE hook, parse dispatch pattern
- Existing codebase `web/src/lib/store/studio-store.ts` — AgentTask type, kanbanTasks
- Existing codebase `src/paperbot/core/collaboration/score_bus.py` — ScoreShareBus.publish_score, event payload shape
- Existing codebase `src/paperbot/application/collaboration/message_schema.py` — EventType.SCORE_UPDATE constant
- Installed package `@xyflow/react` 12.10.0 — confirmed ESM export list including `getSmoothStepPath`, `EdgeLabelRenderer`, `MarkerType`, `Background`, `Controls`, `Handle`, `Position`
- `web/vitest.config.ts` — test environment config (node default, jsdom for src/components/**/*.test.tsx)

### Secondary (MEDIUM confidence)
- `web/package.json` — confirmed @xyflow/react ^12.10.0, zustand ^5.0.9, no dagre/elk installed
- `.planning/config.json` — `workflow.nyquist_validation` not set (key absent) → validation section enabled

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries installed and already used in production files
- Architecture: HIGH — all patterns verified against existing working code
- Pitfalls: HIGH — all pitfalls derived from reading actual source files, not speculation
- Data model gap (depends_on): MEDIUM — confirmed gap; mitigation strategy defined

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable library, slow-moving domain)
