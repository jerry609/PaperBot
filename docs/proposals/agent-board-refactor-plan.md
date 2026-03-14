# Agent Board Refactoring Plan: Kanban → Flow DAG

## Overview

Replace the current 5-column Kanban board with a top-to-bottom DAG flow visualization
inspired by Flowith.io's canvas aesthetic. The new design uses `@xyflow/react` (already
a project dependency) to render an interactive pipeline showing the full paper reproduction
lifecycle.

---

## 1. Visual Design

### 1.1 Canvas

- **Background**: Warm light gray (`#f7f7f5` or similar off-white), NOT pure white
- **Dot grid pattern**: Subtle repeating dot grid via CSS (`radial-gradient`), similar to
  Flowith's "flo mode" canvas
- **ReactFlow config**: `panOnDrag`, `zoomOnScroll` enabled; `fitView` on mount;
  `minZoom=0.5`, `maxZoom=1.5`

### 1.2 Card/Node Style (imitating Flowith)

Cards float on the canvas with:

- **Background**: Pure white (`#ffffff`)
- **Border radius**: `rounded-xl` (12px)
- **Shadow**: Soft, diffuse shadow (`shadow-[0_2px_12px_rgba(0,0,0,0.06)]`)
- **Border**: Very subtle, near-invisible (`border border-zinc-100`)
- **Padding**: Generous (`p-4`)
- **Hover**: Slight elevation increase (`hover:shadow-[0_4px_16px_rgba(0,0,0,0.08)]`)
- **Selected state**: Thin primary-color ring (`ring-2 ring-primary/30`)
- **Typography**: Clean, system font stack, muted secondary text

### 1.3 Edge/Arrow Style

- **Default edges**: Thin gray lines (`stroke: #d4d4d8`, `strokeWidth: 1.5`)
- **Animated edges**: CSS dash animation for active flows
- **Repair loop edge**: Red dashed, animated, curved back from E2E node to the specific
  task node being repaired
- **Success edge**: Green tint when E2E passes → Download
- **Edge type**: `smoothstep` for clean bends

### 1.4 Color Palette (muted, Flowith-inspired)

| Element          | Color                                  |
|------------------|----------------------------------------|
| Canvas BG        | `#f7f7f5` warm gray                    |
| Card BG          | `#ffffff` white                        |
| Card border      | `#f4f4f5` zinc-100                     |
| Card shadow      | `rgba(0,0,0,0.06)`                     |
| Primary accent   | `#6366f1` indigo-500 (for active)      |
| Success          | `#22c55e` green-500                    |
| Warning/Repair   | `#f59e0b` amber-500                    |
| Error            | `#ef4444` red-500                      |
| Muted text       | `#a1a1aa` zinc-400                     |
| Edge default     | `#d4d4d8` zinc-300                     |

---

## 2. Layout: DAG Flow (Top → Bottom)

```
                 ┌──────────────────────┐
                 │   Claude Commander   │   ← CommanderNode
                 └──────────────────────┘
                  ↙        ↓         ↘       ← arrows animate L→R
           ┌─────────┐ ┌─────────┐ ┌─────────┐
           │  Task 1  │ │  Task 2  │ │  Task 3  │  ← TaskNode (planning)
           └─────────┘ └─────────┘ └─────────┘
                ↓            ↓           ↓       ← arrows appear simultaneously
           ┌─────────┐ ┌─────────┐ ┌─────────┐
           │  Task 1  │ │  Task 2  │ │  Task 3  │  ← TaskNode (in_progress)
           │ (running)│ │ (running)│ │ (running)│     same card, status changes
           └─────────┘ └─────────┘ └─────────┘
                 ↘         ↓        ↙    ↗ repair (animated red dashed loop)
                 ┌──────────────────────┐
                 │  E2E Execution       │   ← E2ENode
                 │  python main.py      │
                 │  Attempt 1/3 ✓       │
                 └──────────────────────┘
                            ↓ pass
                 ┌──────────────────────┐
                 │  Download Complete   │   ← DownloadNode
                 │  42 files            │
                 │  [Open in VS Code]   │
                 └──────────────────────┘
```

### 2.1 Node Positioning (auto-computed)

- **Row 0** (y=0): CommanderNode — centered
- **Row 1** (y=180): TaskNode[] — spread horizontally, equal spacing
  - x = centerX + (i - (n-1)/2) * cardSpacingX
- **Row 2** (y=420): E2ENode — centered, wider than task cards
- **Row 3** (y=600): DownloadNode — centered

Note: No separate "Planning" vs "In Progress" rows. Each task is a single node
whose visual state changes (planning → in_progress → done). The row represents
tasks, and their status badge + progress bar shows current phase.

### 2.2 Phase Labels

Render as non-interactive annotation nodes or ReactFlow `<Panel>` overlays:
- Left side labels: "Planning", "Executing", "Verification", "Download"
- Rendered as small text positioned to the left of each row

---

## 3. Node Types

### 3.1 CommanderNode

```
┌─────────────────────────────────────┐
│  🤖  Claude Commander               │
│                                     │
│  Decomposing 5 tasks...      ●      │  ← status dot (green=ready, blue=working)
│  Paper: "Attention Is All..."       │
└─────────────────────────────────────┘
```

Content:
- Icon + "Claude Commander" title
- Current action text (decomposing / reviewing / directing repair)
- Paper title (truncated)
- Status indicator dot

### 3.2 TaskNode

Displays ALL the same content as the current `TaskCard`:

```
┌─────────────────────────────────┐
│  Implement attention model       │  ← title (line-clamp-2)
│                                  │
│  Create the multi-head...        │  ← description (line-clamp-2, muted)
│                                  │
│  ┌────┐ ┌────┐                   │  ← tag badges (if any)
│  │NN  │ │core│                   │
│  └────┘ └────┘                   │
│                                  │
│  Progress          45%           │  ← progress bar (0-100%)
│  ████████░░░░░░░░░░░░            │
│                                  │
│  ● ● ● ○ ○ ○     3/6            │  ← subtask dots + count
│                                  │
│  🤖 codex-a1b2     ·   2m ago   │  ← assignee + relative time
│                            Done ▪│  ← status badge
└─────────────────────────────────┘
```

Content (identical to current TaskCard):
- **Title** — `line-clamp-2`, `text-sm font-medium`
- **Description** — `line-clamp-2`, `text-xs text-muted-foreground`
- **Tags** — small badges
- **Progress bar** — only shown when `0 < progress < 100`
- **Subtask dots** — colored circles + `completed/total` count
- **Assignee** — bot or cpu icon + name
- **Relative time** — clock icon + "2m ago"
- **Status badge** — Planning / Running / Done (no AI Review, no Human Review)

**Click behavior**: Opens the existing `TaskDetailDialog` with tabs:
- Overview (description, feedback, errors)
- Subtasks (checklist)
- Logs (terminal-style execution log)
- Files (generated file list)

**Repair state**: When E2E fails and Commander directs repair on this task:
- Status badge changes to "Repairing" (amber)
- An animated looping arrow edge appears (red dashed, from E2E back to this node)
- Progress bar resets and re-animates
- After repair completes, badge reverts to "Done" and loop arrow disappears

### 3.3 E2ENode

```
┌──────────────────────────────────────────────┐
│  ⚡ End-to-End Execution                      │
│                                               │
│  Entry: python main.py                        │
│  Attempt: 2/3                                 │
│                                               │
│  ████████████████░░░░  Running...             │  ← animated progress
│                                               │
│  > Epoch 1/10: loss=0.342, acc=0.81           │  ← last stdout line (live)
│  > Epoch 2/10: loss=0.198, acc=0.89           │
│                                               │
│                                  ● Passed ✓   │  ← or ● Failed ✗
└──────────────────────────────────────────────┘
```

Content:
- Title: "End-to-End Execution"
- Entry point + command being run
- Current attempt / max attempts
- Live stdout preview (last 2-3 lines, scrolling)
- Status: Waiting / Running / Passed / Failed / Repairing

**Click behavior**: Opens a modal/slide-out with:
- Full scrollable stdout/stderr output (terminal style)
- Repair history (collapsible past attempts)
- Entry point and command details

### 3.4 DownloadNode

```
┌──────────────────────────────────────────────┐
│  📦 Download Complete                         │
│                                               │
│  42 files downloaded to local                 │
│  /tmp/paperbot-workspace/attn-a9b1/           │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │       Open in VS Code    →           │     │  ← primary button
│  └──────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

Content:
- File count
- Local directory path
- "Open in VS Code" button

---

## 4. Left Sidebar

A narrow panel (width ~260px) on the left side of the Agent Board, outside the
ReactFlow canvas.  Uses the same warm-gray background as the canvas.

### 4.1 Sandbox Files Section

```
┌─────────────────────────────┐
│  📁 Sandbox Files            │
│                              │
│  ▸ src/                      │
│    model.py                  │
│    train.py                  │
│    utils.py                  │
│  ▸ tests/                    │
│    test_model.py             │
│  requirements.txt            │
│  main.py                     │
│                              │
│  12 files · 3.2 KB           │
└─────────────────────────────┘
```

- Fetches from `GET /sessions/{id}/sandbox/files?paper_slug=xxx`
- Collapsible directory tree
- Click a file → opens content via `GET /sessions/{id}/sandbox/file?path=xxx`
  in a modal or the existing Files tab
- Auto-refreshes when tasks complete (on `executor_finished` SSE events)

### 4.2 Time Estimate Section

```
┌─────────────────────────────┐
│  ⏱ Reproduction Estimate    │
│                              │
│  ██████████░░░░░  3/5 tasks  │
│                              │
│  Elapsed:    4m 32s          │
│  Remaining: ~6m 15s          │
│  Total est: ~10m 47s         │
│                              │
│  Avg task:  ~2m 8s           │
└─────────────────────────────┘
```

- Computed client-side from task timestamps and progress
- `elapsed` = now - first task start time
- `avgTaskDuration` = elapsed / completedTasks
- `remaining` = avgTaskDuration * remainingTasks + E2E estimate
- Updates every second via `setInterval`
- Shows "Calculating..." until at least one task completes

---

## 5. Edge Animation Behavior

Per the user's diagram notes:

| Phase | Edge Behavior |
|-------|---------------|
| Planning | Commander → Task edges appear **one by one, left to right** (staggered 300ms delay per edge) |
| Executing | Task edges to next phase appear **simultaneously** |
| Task completion | Each done task draws an edge toward E2E node |
| E2E running | All task→E2E edges visible, E2E node pulses |
| E2E failure | Red dashed animated edge: E2E → specific task node(s) being repaired |
| E2E success | Green edge: E2E → Download node |
| Download | Download node activates, "Open in VS Code" enabled |

Implementation:
- Edge visibility controlled by state (`edges` array in ReactFlow)
- Staggered appearance via `setTimeout` + state updates
- Animated edges use `animated: true` + custom CSS
- Repair loop edge: `style: { stroke: '#ef4444', strokeDasharray: '5 5' }` + `animated: true`

---

## 6. State Model Changes (`studio-store.ts`)

### 6.1 New Types

```ts
type PipelinePhase =
  | 'idle'
  | 'planning'       // Commander decomposing tasks
  | 'executing'      // Codex workers running sub-tasks
  | 'e2e_running'    // Running full project code
  | 'e2e_repairing'  // Commander-directed repair after E2E fail
  | 'downloading'    // Pulling files from sandbox to local
  | 'completed'      // All done
  | 'failed'         // Unrecoverable failure

interface E2EState {
  status: 'waiting' | 'running' | 'passed' | 'failed' | 'repairing'
  attempt: number
  maxAttempts: number
  entryPoint: string | null
  command: string | null
  lastExitCode: number | null
  lastStdout: string
  lastStderr: string
  history: Array<{
    attempt: number
    success: boolean
    exitCode: number
    duration: number
    stdoutPreview: string
  }>
}

interface SandboxFileEntry {
  name: string
  type: 'file' | 'directory'
  children?: SandboxFileEntry[]
}

interface TimeEstimate {
  elapsedMs: number
  remainingMs: number | null
  avgTaskMs: number | null
  completedTasks: number
  totalTasks: number
}
```

### 6.2 New Store Fields & Actions

```ts
// Fields
pipelinePhase: PipelinePhase
e2eState: E2EState | null
sandboxFiles: SandboxFileEntry[]
timeEstimate: TimeEstimate | null

// Actions
setPipelinePhase: (phase: PipelinePhase) => void
setE2EState: (state: Partial<E2EState>) => void
setSandboxFiles: (files: SandboxFileEntry[]) => void
setTimeEstimate: (estimate: TimeEstimate) => void
```

### 6.3 Remove

- Remove `human_review` from `AgentTask.status` type
- Remove `humanReviews` field from `AgentTask`
- Remove `reviewFeedback` from `AgentTask` (or keep as read-only for AI review logs)

### 6.4 Status Simplification

Task statuses: `'planning' | 'in_progress' | 'repairing' | 'done'`

- `ai_review` → removed (replaced by E2E execution)
- `human_review` → removed entirely
- `repairing` → new status for tasks being fixed during E2E repair loop

---

## 7. SSE Event Handling

Update `runAllWithWorkspace()` to handle new events from the 6-stage backend pipeline:

| SSE Event | Store Action |
|-----------|-------------|
| `sandbox_init` | `setPipelinePhase('executing')` |
| `executor_started` | Update task status to `in_progress`, add edge |
| `tool_step` | Update task progress |
| `executor_finished` | Update task, add edge to E2E |
| `executor_failed` | Update task status, mark error |
| `verify_started/finished` | Silent — only append to task execution log |
| `e2e_started` | `setPipelinePhase('e2e_running')`, init E2E state |
| `e2e_attempt` | Update E2E state (attempt, stdout, exit code) |
| `e2e_finished` (success) | `setPipelinePhase('downloading')`, add E2E→Download edge |
| `e2e_finished` (fail) | `setPipelinePhase('e2e_repairing')`, add repair loop edge |
| `e2e_error` | `setPipelinePhase('failed')` |
| `download_complete` | `setPipelinePhase('completed')`, activate Download node |
| `download_skipped` | `setPipelinePhase('completed')` |

---

## 8. Removals

### 8.1 Delete from `AgentBoard.tsx`

- `COLUMNS` array and Kanban column layout
- `tasksByColumn` computation
- Human review: `reviewNotes`, `reviewSubmitting`, `reviewError` state
- `handleHumanReviewDecision()` function
- Human review section in `TaskDetailDialog` (the review textarea + approve/request changes buttons)
- `HumanReviewRequest` type
- `ai_review` badge variant in `taskStatusBadge()`
- `human_review` badge variant in `taskStatusBadge()`

### 8.2 Keep from `AgentBoard.tsx`

- `TaskCard` content (reused inside `TaskNode` in the DAG)
- `TaskDetailDialog` (opened on node click) — minus the human review section
- `taskStatusBadge()` — simplified to Planning / Running / Repairing / Done
- `extractPossibleFiles()` utility
- `runAllWithWorkspace()` — updated with new SSE handlers
- `upsertTaskFromEvent()` — updated for new statuses
- `appendInlineTaskLog()` — unchanged

---

## 9. File Changes

| File | Change |
|------|--------|
| `web/src/lib/store/studio-store.ts` | Add `PipelinePhase`, `E2EState`, `SandboxFileEntry`, `TimeEstimate`; add new store fields/actions; simplify task statuses |
| `web/src/lib/store/studio-store.test.ts` | Update tests for new statuses, remove human_review tests |
| `web/src/components/studio/AgentBoard.tsx` | Major refactor: replace Kanban with ReactFlow DAG canvas + left sidebar; remove human review UI; add E2E + Download node rendering; update SSE handlers |
| `web/src/components/studio/AgentBoardNodes.tsx` | **New** — custom ReactFlow node components: `CommanderNode`, `TaskNode`, `E2ENode`, `DownloadNode` |
| `web/src/components/studio/AgentBoardEdges.tsx` | **New** — custom edge components with animation CSS |
| `web/src/components/studio/AgentBoardSidebar.tsx` | **New** — left sidebar: sandbox file tree + time estimate |
| `web/src/components/studio/ReproductionLog.tsx` | No change (already renders `<AgentBoard>`) |
| `web/src/app/studio/page.tsx` | No change |

---

## 10. Implementation Order

1. **Store changes** — Add new types, fields, actions; simplify statuses
2. **Node components** — `AgentBoardNodes.tsx` with Flowith-style cards
3. **Edge components** — `AgentBoardEdges.tsx` with animation
4. **Sidebar** — `AgentBoardSidebar.tsx` with file tree + time estimate
5. **Main refactor** — `AgentBoard.tsx`: replace Kanban with ReactFlow canvas + sidebar layout + SSE handler updates
6. **Cleanup** — Remove human review code, update tests
