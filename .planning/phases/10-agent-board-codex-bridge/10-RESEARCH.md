# Phase 10: Agent Board + Codex Bridge - Research

**Researched:** 2026-03-15
**Domain:** React Kanban board (pure CSS), Codex delegation via `.claude/agents/` definition, SSE delegation events, Paper2Code overflow routing, Codex-specific error surfacing
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DASH-02 | User can manage agent tasks via Kanban board showing Claude Code and Codex agent identity | `AgentBoard.tsx` already exists in `web/src/components/studio/` with ReactFlow DAG view; Phase 10 adds a Kanban column view alongside it. `AgentTask.assignee` field encodes agent identity (`"claude"` vs `"codex-{hex}"`). No new drag-and-drop library needed — column layout is display-only (task cards move via API, not drag). |
| DASH-03 | User can see Codex-specific error states (timeout, sandbox crash) surfaced prominently | `CodexResult.diagnostics.reason_code` carries structured failure codes (`timeout`, `stagnation_detected`, `max_iterations_exhausted`, etc.). `_format_codex_failure()` already maps these. Frontend needs a `CodexErrorBadge` or failure section in the task card that reads `task.lastError` and `task.executionLog` for structured codes. |
| CDX-01 | Claude Code can delegate tasks to Codex via custom agent definition (codex-worker.md) | `.claude/agents/codex-worker.md` is a Claude Code custom agent definition file. It calls `POST /api/agent-board/tasks/{task_id}/dispatch` to mark a task as dispatched to a Codex worker. The backend `CodexDispatcher` then runs the actual Codex execution. The file does not exist yet — Phase 10 creates it. |
| CDX-02 | Paper2Code pipeline stages can overflow from Claude Code to Codex when workload is high | `repro/orchestrator.py` runs `PipelineStage`s (PLANNING, CODING, VERIFICATION, DEBUGGING). A lightweight overflow guard in the orchestrator checks a workload condition and re-routes the CODING/DEBUGGING stages to `agent_board` via `CodexDispatcher` instead of the local LLM executor. |
| CDX-03 | User can observe Codex delegation events (dispatched, accepted, completed, failed) in activity feed | New `EventType` constants (`CODEX_DISPATCHED`, `CODEX_ACCEPTED`, `CODEX_COMPLETED`, `CODEX_FAILED`) emitted from `agent_board.py` into the `EventBusEventLog`. Existing SSE fan-out (`/api/events/stream`) delivers them. `parsers.ts` extended with `parseCodexDelegation()`. ActivityFeed renders them as a new item type. |
</phase_requirements>

---

## Summary

Phase 10 delivers two parallel concerns: (1) a Kanban board view in the agent dashboard that shows tasks by status column with agent identity badges, and (2) the Codex Bridge — the mechanism by which Claude Code delegates tasks to Codex workers and users can observe those delegation events in the activity feed.

The Kanban board requires no new drag-and-drop dependencies. `AgentTask` objects from `agent_board.py` already carry `status` (planning, in_progress, ai_review, human_review, done, paused, cancelled) and `assignee` (either `"claude"` for Claude-owned tasks or `"codex-{hex}"` for dispatched tasks). A pure CSS column layout over the existing `AgentBoard` or a new `KanbanBoard` component grouped by these status values is sufficient. The existing `studio-store.ts` / `AgentTask` types and the `TasksPanel` from Phase 9 are the anchoring data models.

The Codex Bridge has two parts. On the agent side, `.claude/agents/codex-worker.md` is a Claude Code sub-agent definition file (markdown with YAML front-matter) that Claude Code loads and invokes when it wants to delegate. The sub-agent calls the existing `POST /api/agent-board/tasks/{task_id}/dispatch` endpoint, which triggers `CodexDispatcher` to run the task via the OpenAI API. A key architectural decision already locked in STATE.md: "Codex bridge is a `.claude/agents/` file, not PaperBot server code." On the event side, new `EventType` constants for delegation lifecycle (`CODEX_DISPATCHED`, `CODEX_ACCEPTED`, `CODEX_COMPLETED`, `CODEX_FAILED`) are added to `message_schema.py`, emitted from `agent_board.py`, and parsed on the frontend.

**Primary recommendation:** Build the Kanban board as a new `KanbanBoard` component in `web/src/components/agent-dashboard/` that reads from `useAgentEventStore` and groups tasks by `assignee`-derived column. Create `.claude/agents/codex-worker.md` using the Claude Code sub-agent definition format. Add four new `EventType` constants and emit them from `agent_board.py` at the four delegation lifecycle points. Extend `parsers.ts` with a `parseCodexDelegation()` function. No new npm packages needed.

---

## Standard Stack

### Core (zero new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `useAgentEventStore` | Phase 8/9 deliverable | Zustand store: feed, agentStatuses, filesTouched | All Phase 9 components read from this store; KanbanBoard does the same |
| `AgentTask` / `AgentBoard` | `studio-store.ts` + `agent_board.py` | Task model with `status`, `assignee`, `executionLog` | Already the authoritative task model; `assignee` encodes agent identity |
| `EventType` class | `message_schema.py` | String constants for event types | Phase 8/9 pattern: add new constants, don't invent new schemas |
| `make_event()` | `message_schema.py` | Factory for `AgentEventEnvelope` | Already used by agent_board.py for SSE emission |
| `EventBusEventLog` | Phase 7 deliverable | AsyncIO queue fan-out for SSE | All delegation events flow through this; no new transport needed |
| `CodexDispatcher` | `infrastructure/swarm/codex_dispatcher.py` | OpenAI API task execution | Already implements full tool-loop, diagnostics, timeout handling |
| `tailwindcss` | ^4 | Column layout styling | Project-wide CSS framework |
| `lucide-react` | ^0.562.0 | Status icons, agent badge icons | Already the project icon library |
| `@radix-ui/react-scroll-area` | ^1.2.10 | Scrollable Kanban column content | Already installed, used by Phase 8/9 components |
| `zustand` | ^5.0.9 | State for Kanban column data | Already used project-wide |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `framer-motion` | ^12.23.26 | Task card entrance animation when task moves column | Already installed; optional, use for card slide-in on status transition |
| `@radix-ui/react-tooltip` | ^1.2.8 | Tooltip for Codex error detail on task card hover | Already installed; use for compact error display without expanding card |
| `@radix-ui/react-dialog` | ^1.1.15 | Codex error detail modal when user clicks failed task | Already installed; matches existing studio dialog pattern |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure CSS column Kanban | `@dnd-kit/core` (drag-and-drop) | Task status changes happen via API (`PATCH /api/agent-board/tasks/{id}`); drag-and-drop adds ~15KB and complexity. The board is primarily observational — drag is not in requirements. |
| New `KanbanBoard` component | Extending existing `AgentBoard.tsx` ReactFlow DAG | AgentBoard.tsx is already 1300+ lines. A new file keeps concerns separated and avoids bloating the DAG component. The Kanban view is a separate mental model (columns) from the DAG view (nodes/edges). |
| `EventType` string constants | Python Enum | `EventType` is already a plain class with string annotations (Phase 8 decision). Constants are usable as `str` without `.value` unwrapping. Continue this pattern. |
| Separate SSE endpoint for delegation events | Reuse `/api/events/stream` | The existing fan-out already delivers all `EventBusEventLog` events. No new endpoint needed — just emit into the existing bus. |

**Installation:** No new packages needed.

---

## Architecture Patterns

### Recommended File Structure

```
web/src/
├── app/
│   └── agent-dashboard/
│       └── page.tsx                        # EXISTING: Phase 9 page (no change)
├── components/
│   └── agent-dashboard/
│       ├── TasksPanel.tsx                  # EXISTING: Phase 9 left rail
│       ├── FileListPanel.tsx               # EXISTING: Phase 9 right panel
│       ├── InlineDiffPanel.tsx             # EXISTING: Phase 9 diff view
│       └── KanbanBoard.tsx                 # NEW: Kanban column board
└── lib/
    └── agent-events/
        ├── store.ts                        # MODIFIED: add kanbanTasks, codex delegation fields
        ├── types.ts                        # MODIFIED: add CodexDelegationEntry type
        └── parsers.ts                      # MODIFIED: add parseCodexDelegation()

src/paperbot/
├── application/
│   └── collaboration/
│       └── message_schema.py               # MODIFIED: add CODEX_DISPATCHED/ACCEPTED/COMPLETED/FAILED
└── api/
    └── routes/
        └── agent_board.py                  # MODIFIED: emit delegation events via event_log

.claude/
└── agents/
    └── codex-worker.md                     # NEW: Claude Code sub-agent definition
```

### Pattern 1: Kanban Board — Column Layout (display-only)

The Kanban board groups tasks by status column. No drag-and-drop is required — task status changes happen exclusively via the backend API. The board is a read-only view of `AgentTask[]` state that the SSE stream keeps current.

**Column mapping:**

| Column | Statuses | Agent Badge Color |
|--------|----------|------------------|
| Planned | `planning` | grey (unassigned) |
| In Progress | `in_progress` | blue (claude) / purple (codex) |
| Review | `ai_review`, `human_review` | amber |
| Done | `done` | green |
| Blocked | `paused`, `cancelled` | red |

**Agent identity from `assignee`:**
- `"claude"` → "Claude Code" badge
- `"codex-{hex4}"` or starts with `"codex"` → "Codex" badge
- `"codex-retry-{hex4}"` → "Codex (retry)" badge

```typescript
// web/src/components/agent-dashboard/KanbanBoard.tsx
"use client"

import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import type { AgentTask } from "@/lib/store/studio-store"

type KanbanColumn = {
  id: string
  label: string
  statuses: AgentTask["status"][]
}

const COLUMNS: KanbanColumn[] = [
  { id: "planned",     label: "Planned",     statuses: ["planning"] },
  { id: "in_progress", label: "In Progress", statuses: ["in_progress"] },
  { id: "review",      label: "Review",      statuses: ["ai_review", "human_review"] },
  { id: "done",        label: "Done",        statuses: ["done"] },
  { id: "blocked",     label: "Blocked",     statuses: ["paused", "cancelled"] },
]

function agentLabel(assignee: string): { label: string; variant: "default" | "secondary" | "destructive" | "outline" } {
  if (!assignee || assignee === "claude") return { label: "Claude Code", variant: "default" }
  if (assignee.startsWith("codex-retry")) return { label: "Codex (retry)", variant: "secondary" }
  if (assignee.startsWith("codex")) return { label: "Codex", variant: "secondary" }
  return { label: assignee, variant: "outline" }
}

export function KanbanBoard({ tasks }: { tasks: AgentTask[] }) {
  return (
    <div className="flex h-full gap-3 overflow-x-auto px-3 py-3">
      {COLUMNS.map((col) => {
        const columnTasks = tasks.filter((t) => col.statuses.includes(t.status))
        return (
          <div key={col.id} className="flex w-56 shrink-0 flex-col rounded-lg border bg-muted/40">
            <div className="flex items-center justify-between px-3 py-2 border-b">
              <span className="text-xs font-semibold">{col.label}</span>
              <Badge variant="outline" className="text-[10px]">{columnTasks.length}</Badge>
            </div>
            <ScrollArea className="flex-1">
              <ul className="space-y-2 p-2">
                {columnTasks.map((task) => {
                  const agent = agentLabel(task.assignee)
                  const hasError = !!task.lastError
                  return (
                    <li
                      key={task.id}
                      className={`rounded-md border bg-background p-2 text-xs shadow-sm ${
                        hasError ? "border-red-300" : ""
                      }`}
                    >
                      <p className="font-medium truncate mb-1">{task.title}</p>
                      <div className="flex items-center gap-1">
                        <Badge variant={agent.variant} className="text-[10px] shrink-0">
                          {agent.label}
                        </Badge>
                        {hasError && (
                          <Badge variant="destructive" className="text-[10px] shrink-0">
                            Error
                          </Badge>
                        )}
                      </div>
                      {hasError && task.lastError && (
                        <p className="mt-1 text-[10px] text-red-500 truncate">{task.lastError}</p>
                      )}
                    </li>
                  )
                })}
                {columnTasks.length === 0 && (
                  <li className="py-4 text-center text-[10px] text-muted-foreground">Empty</li>
                )}
              </ul>
            </ScrollArea>
          </div>
        )
      })}
    </div>
  )
}
```

### Pattern 2: Wiring Kanban into the Agent Dashboard Page

The Kanban board can be added as a tab view within the existing `TasksPanel` left rail, or as a new view mode on the agent dashboard page. The simplest approach: add a toggle button in the dashboard header that switches between the existing three-panel view and a full-width Kanban view. The same `useAgentEventStore` feeds both views.

```typescript
// web/src/app/agent-dashboard/page.tsx — MODIFIED
// Add a "view" toggle state ("panels" | "kanban") in the header
// When view === "kanban", render <KanbanBoard tasks={...} /> instead of <SplitPanels />
// Tasks derived from: useAgentEventStore(s => s.kanbanTasks)
```

### Pattern 3: Codex-Specific Error State Display (DASH-03)

The backend `CodexResult.diagnostics.reason_code` values that must be surfaced prominently:

| reason_code | User-Facing Label | UI Treatment |
|-------------|------------------|--------------|
| `max_iterations_exhausted` | "Iteration limit reached" | Red badge on task card |
| `stagnation_detected` | "No progress detected" | Red badge on task card |
| `timeout` (asyncio.TimeoutError) | "Codex timeout" | Red badge + error icon |
| `repeated_tool_calls` | "Stuck in tool loop" | Red badge |
| `too_many_tool_errors` | "Too many errors" | Red badge |
| `sandbox_crash` | "Sandbox crashed" | Red badge + skull icon |

**Where this data lives:** `task.executionLog` entries with `event === "task_failed"` carry `details.codex_diagnostics.reason_code`. The `lastError` field carries the human-readable message from `_format_codex_failure()`.

**Implementation:** In `KanbanBoard`, when a task is in the "Blocked" column (status `cancelled`) or failed with a `codex_` assignee, check `task.executionLog` for the last `task_failed` entry and display the `reason_code` as a colour-coded badge. This is display-only — no new API calls.

### Pattern 4: codex-worker.md — Claude Code Sub-Agent Definition

Claude Code custom agents live in `.claude/agents/` as markdown files with YAML front-matter. The file instructs Claude Code how to invoke the Codex worker by calling the agent board API.

```markdown
---
name: codex-worker
description: Delegates a coding task to a Codex worker via the PaperBot agent board API.
  Use when: the current workload is high, a task is parallelizable, or explicitly requested.
tools:
  - Bash
  - Read
---

# Codex Worker Sub-Agent

You are a Codex delegation coordinator. Your job is to dispatch a task to a Codex worker
via the PaperBot agent board API and monitor the result.

## When to use

Delegate to this sub-agent when:
1. You have identified a self-contained coding task that Codex can complete independently
2. The task has clear acceptance criteria (subtasks)
3. Current workload is high (multiple tasks running simultaneously)

## Delegation Protocol

### Step 1: Confirm the task exists on the agent board

```bash
curl -s http://localhost:8000/api/agent-board/sessions/{session_id}
```

### Step 2: Dispatch the task to a Codex worker

```bash
curl -s -X POST http://localhost:8000/api/agent-board/tasks/{task_id}/dispatch
```

This marks the task `status: in_progress` and assigns `assignee: codex-{hex4}`.

### Step 3: Stream execution events

```bash
curl -s http://localhost:8000/api/agent-board/tasks/{task_id}/execute
```

### Step 4: Report result

On success: "Task {task_id} completed by Codex. Files: {files_generated}"
On failure: "Task {task_id} failed. Reason: {reason_code}. Error: {error}"

## Error Handling

- `OPENAI_API_KEY not set`: Cannot delegate. Inform the user.
- Timeout: The Codex worker timed out. Task is in `human_review`.
- Sandbox crash: Report `reason_code: sandbox_crash`. Task is in `human_review`.
```

**Key insight:** The `.claude/agents/` file format is read by Claude Code at startup. The file name becomes the agent name. Claude Code invokes it as a sub-agent using the `Task` tool. The file uses existing `Bash` and `Read` tools — no new tooling needed.

### Pattern 5: Delegation Event Types

Add four new constants to `EventType` in `message_schema.py`:

```python
# src/paperbot/application/collaboration/message_schema.py — APPEND to EventType class

# --- Codex delegation events (CDX-03) ---
CODEX_DISPATCHED: str = "codex_dispatched"
# Payload: task_id, task_title, assignee, session_id
CODEX_ACCEPTED: str = "codex_accepted"
# Payload: task_id, assignee, model
CODEX_COMPLETED: str = "codex_completed"
# Payload: task_id, assignee, files_generated, output_preview
CODEX_FAILED: str = "codex_failed"
# Payload: task_id, assignee, reason_code, error, diagnostics
```

**Where to emit:** In `agent_board.py`, at the four delegation lifecycle points:
1. `CODEX_DISPATCHED`: after `task.assignee = f"codex-{uuid...}"` is set
2. `CODEX_ACCEPTED`: when the `CodexDispatcher` begins execution (first `on_step` callback)
3. `CODEX_COMPLETED`: when `result.success == True` after dispatch
4. `CODEX_FAILED`: when `result.success == False` after dispatch

**Bus access pattern:** `agent_board.py` currently does not import the EventBus. Add a module-level `_get_event_log()` helper that lazily reads from `app.state.event_log` — but `agent_board.py` is called from FastAPI route functions, not from request handlers that have access to `app.state`. The correct pattern: inject `event_log` via FastAPI `Depends`, or use the global `Container.instance()`.

The simplest approach: use `Container.instance().event_log` (already the DI pattern for the entire project). Add a `_get_event_log()` helper:

```python
# In agent_board.py — module-level helper
def _get_event_log():
    """Lazily get event_log from the DI container."""
    from ...core.di.container import Container
    return Container.instance().event_log

async def _emit_codex_event(event_type: str, task: "AgentTask", session: "BoardSession", extra: dict):
    """Emit a Codex delegation lifecycle event into the EventBus."""
    from ...application.collaboration.message_schema import make_event, new_run_id, new_trace_id
    el = _get_event_log()
    if el is None:
        return
    env = make_event(
        run_id=new_run_id(),
        trace_id=new_trace_id(),
        workflow="agent_board",
        stage=task.id,
        attempt=0,
        agent_name=task.assignee or "codex",
        role="worker",
        type=event_type,
        payload={
            "task_id": task.id,
            "task_title": task.title,
            "session_id": session.session_id,
            **extra,
        },
    )
    el.append(env)
```

### Pattern 6: Frontend Parser for Delegation Events

```typescript
// web/src/lib/agent-events/parsers.ts — APPEND

const CODEX_DELEGATION_TYPES = new Set([
  "codex_dispatched",
  "codex_accepted",
  "codex_completed",
  "codex_failed",
])

export type CodexDelegationEntry = {
  id: string
  event_type: "codex_dispatched" | "codex_accepted" | "codex_completed" | "codex_failed"
  task_id: string
  task_title: string
  assignee: string
  session_id: string
  ts: string
  // For completed:
  files_generated?: string[]
  // For failed:
  reason_code?: string
  error?: string
}

export function parseCodexDelegation(raw: AgentEventEnvelopeRaw): CodexDelegationEntry | null {
  const t = String(raw.type ?? "")
  if (!CODEX_DELEGATION_TYPES.has(t)) return null
  const payload = (raw.payload ?? {}) as Record<string, unknown>
  if (!payload.task_id || !raw.ts) return null
  return {
    id: `${t}-${String(payload.task_id)}-${String(raw.ts)}`,
    event_type: t as CodexDelegationEntry["event_type"],
    task_id: String(payload.task_id),
    task_title: String(payload.task_title ?? ""),
    assignee: String(raw.agent_name ?? payload.assignee ?? "codex"),
    session_id: String(payload.session_id ?? ""),
    ts: String(raw.ts),
    files_generated: Array.isArray(payload.files_generated)
      ? (payload.files_generated as string[])
      : undefined,
    reason_code: typeof payload.reason_code === "string" ? payload.reason_code : undefined,
    error: typeof payload.error === "string" ? payload.error : undefined,
  }
}
```

The `useAgentEvents` hook already dispatches all raw events through the parsers. Add `parseCodexDelegation` to the dispatch chain in `useAgentEvents.ts`.

### Pattern 7: Paper2Code Overflow (CDX-02)

The Paper2Code pipeline in `repro/orchestrator.py` runs stages sequentially (PLANNING → CODING → VERIFICATION → DEBUGGING). For CDX-02, the overflow condition is checked before the CODING stage:

```python
# src/paperbot/repro/orchestrator.py — in the run() method, before CODING stage

def _should_overflow_to_codex(self) -> bool:
    """Check if current workload warrants Codex delegation."""
    # Simple threshold: check CODEX_OVERFLOW_THRESHOLD env var (default: disabled)
    threshold = os.getenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "").strip()
    if not threshold:
        return False  # Overflow disabled by default
    try:
        # In the future: check active task count against threshold
        # For Phase 10: simple flag-based overflow
        return threshold.lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False
```

For Phase 10, the overflow is opt-in via environment variable. When enabled, the orchestrator's CODING stage calls the `agent_board` API to create a task and delegates to `CodexDispatcher` instead of running the local coding agents. The exact multi-task overflow mechanism (capacity-based routing) is deferred to a future phase; Phase 10 establishes the flag and the code path stub.

### Anti-Patterns to Avoid

- **Drag-and-drop on the Kanban board:** Task status changes happen via the backend API. The board is observational. Adding DnD adds complexity without fulfilling any requirement.
- **Separate SSE endpoint for Codex delegation events:** All events flow through the existing `EventBusEventLog` → `/api/events/stream`. Don't add a new endpoint.
- **Importing `app.state` from `agent_board.py`:** Route functions have access to `request.app.state`, but the delegation emit helpers are called from nested async functions that don't receive `request`. Use `Container.instance().event_log` instead.
- **Putting Codex execution logic in `codex-worker.md`:** The `.claude/agents/` file is an instruction document, not executable code. It calls the existing `POST /api/agent-board/tasks/{id}/dispatch` endpoint — it does not implement the CodexDispatcher logic.
- **Rendering all `task.executionLog` entries in the Kanban card:** The card must be compact. Show only the most recent failure entry's `reason_code`. Link to the full `TaskDetailPanel` for the complete log.
- **Creating a new AgentTask type for Phase 10:** `AgentTask` in `studio-store.ts` and `agent_board.py` already models everything Phase 10 needs. Do not fork the type.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Kanban column layout | Custom grid/flex CSS from scratch | Tailwind `flex gap-3 overflow-x-auto` with fixed-width `w-56 shrink-0` columns | Two lines of Tailwind; no layout library needed |
| Task status change persistence | Local React state for column membership | `PATCH /api/agent-board/tasks/{id}` → the task's canonical state is in `_board_store` (SQLite) | Columns derive from stored `status`; mutating local state diverges from server truth |
| Codex execution from the `.claude/agents/` file | Running `CodexDispatcher` from a shell script | Call `POST /api/agent-board/tasks/{task_id}/execute` (SSE stream) | The full tool-loop, diagnostics, retry logic, sandbox management, and event emission are already in `agent_board.py` |
| Delegation event schema | Custom event envelope | `make_event()` + new `EventType` constants | Consistent with Phase 8 vocabulary; EventBus delivers for free |
| Activity feed rendering of delegation events | New ActivityFeed component variant | Extend `deriveHumanSummary()` in `parsers.ts` to handle `codex_dispatched`/`codex_completed`/`codex_failed` types | The `ActivityFeed` already renders all `ActivityFeedItem` entries uniformly; adding a new type case is a one-liner |

**Key insight:** The Kanban board is a view into already-existing data (`AgentTask[]`). The Codex Bridge is a documentation artifact (`.claude/agents/codex-worker.md`) plus four event emission points in already-existing code. Phase 10 is primarily about wiring and surfacing, not building new infrastructure.

---

## Common Pitfalls

### Pitfall 1: KanbanBoard Receives Stale Task Data

**What goes wrong:** The Kanban board receives a snapshot of tasks at mount time and does not update as the SSE stream delivers task status changes.

**Why it happens:** `AgentTask[]` objects from `studio-store.ts` are updated by the SSE event handler in `AgentBoard.tsx`, which uses `updateAgentTask()`. The new `KanbanBoard` component does not have access to this update path unless it reads from the same store.

**How to avoid:** The `KanbanBoard` must read from `useStudioStore` (for tasks from active studio sessions) or from `useAgentEventStore` (for tasks derived from SSE events). For Phase 10, when the dashboard embeds the Kanban view, the tasks come from `useAgentEventStore` extended with a `kanbanTasks: AgentTask[]` field populated from `codex_dispatched`/`task_dispatched` events. Alternatively, read directly from `useStudioStore.getState().tasks` if the Kanban is embedded within the studio context.

**Warning signs:** Task cards do not move between columns after pipeline events. All tasks stay in "Planned" regardless of SSE updates.

### Pitfall 2: `Container.instance().event_log` Returns None in Tests

**What goes wrong:** The `_emit_codex_event()` helper calls `Container.instance().event_log`, but in unit tests the container has not been initialized with an event_log.

**Why it happens:** `Container._instance` is reset to `None` in test `setup_method`. The `event_log` attribute on a fresh container may not be set.

**How to avoid:** In `_emit_codex_event()`, guard with `if el is None: return`. The delegation events are best-effort — if the event_log is unavailable (offline tests), silently skip. Unit tests for `agent_board.py` that verify event emission should inject a mock EventLog via monkeypatch.

**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'append'` in agent_board route tests.

### Pitfall 3: codex-worker.md Tool Names Must Match Claude Code's Available Tools

**What goes wrong:** The `.claude/agents/codex-worker.md` YAML front-matter lists tools that Claude Code does not have installed or has under different names, causing the agent to fail at load time.

**Why it happens:** Claude Code validates tool names listed in the agent definition against its loaded tool set.

**How to avoid:** Only list `Bash` and `Read` in the `tools:` section — these are always available in Claude Code. The agent makes HTTP calls via `curl` inside `Bash`, so no extra tool permissions are needed.

**Warning signs:** Claude Code reports "Unknown tool: X" or refuses to invoke the sub-agent.

### Pitfall 4: Four Delegation Event Constants Create Orphan Events in the Activity Feed

**What goes wrong:** `parseActivityItem()` in `parsers.ts` has a fallback branch `return \`${t}: ${raw.agent_name}\`` for unknown event types. Delegation events appear in the activity feed as raw type strings rather than readable summaries.

**Why it happens:** `deriveHumanSummary()` does not know about the four new `codex_*` event types.

**How to avoid:** Add cases in `deriveHumanSummary()`:
```typescript
if (t === "codex_dispatched") return `Task dispatched to ${raw.agent_name}: ${payload.task_title}`
if (t === "codex_accepted") return `Codex accepted task: ${payload.task_title}`
if (t === "codex_completed") return `Codex completed: ${payload.task_title} (${(payload.files_generated as string[] ?? []).length} files)`
if (t === "codex_failed") return `Codex failed: ${payload.task_title} (${payload.reason_code})`
```

**Warning signs:** Activity feed shows entries like `codex_dispatched: codex-a1b2 / task-abc123` instead of readable summaries.

### Pitfall 5: Paper2Code Overflow Flag Has No Visible Effect Without Active Codex Dispatcher

**What goes wrong:** Setting `PAPERBOT_CODEX_OVERFLOW_THRESHOLD=true` enables the overflow code path, but if `OPENAI_API_KEY` is not set, `CodexDispatcher` returns an immediate `success=False` with `error="OPENAI_API_KEY not set"`. The overflow silently degrades back to local execution without surfacing the failure reason to the user.

**Why it happens:** `CodexDispatcher.dispatch_auto()` checks for the key at the start of execution and returns an error result — it does not raise an exception. The overflow caller must check `result.success` and handle gracefully.

**How to avoid:** After the overflow dispatch returns `result.success == False`, the orchestrator logs the failure and falls back to local agent execution. Add a warning log: `"Codex overflow failed (reason: {result.error}); falling back to local execution"`. Emit a `CODEX_FAILED` event to notify the user.

**Warning signs:** No Codex events appear in the activity feed even though `PAPERBOT_CODEX_OVERFLOW_THRESHOLD=true` is set.

### Pitfall 6: KanbanBoard Mounted Inside SplitPanels Breaks the Layout

**What goes wrong:** If `KanbanBoard` is placed as a fourth panel inside `SplitPanels`, the horizontal overflow of Kanban columns fights with the panel's `overflow-hidden` constraint.

**Why it happens:** `SplitPanels` sets `overflow-hidden` on panels to prevent content from overflowing panel boundaries. `KanbanBoard`'s horizontal column scroll requires `overflow-x-auto`.

**How to avoid:** Do not put `KanbanBoard` inside `SplitPanels`. Instead, toggle the entire `SplitPanels` component with a full-width `KanbanBoard` via a view-mode state in the page header.

**Warning signs:** Kanban columns are clipped at the right edge of the panel; horizontal scroll does not work.

---

## Code Examples

### Deriving Agent Identity from `assignee`

```typescript
// Source: agent_board.py line 755 — assignee pattern
// "claude" = Claude Code tasks
// "codex-{hex4}" = Codex dispatched tasks
// "codex-retry-{hex4}" = Codex retry tasks

function isCodexTask(assignee: string): boolean {
  return typeof assignee === "string" && assignee.startsWith("codex")
}

function agentDisplayName(assignee: string): string {
  if (!assignee || assignee === "claude") return "Claude Code"
  if (assignee.startsWith("codex-retry")) return "Codex (retry)"
  if (assignee.startsWith("codex")) return "Codex"
  return assignee
}
```

### Codex Failure Detection from executionLog

```typescript
// Source: agent_board.py _format_codex_failure() + task.execution_log structure
// task.executionLog entries with event "task_failed" carry details.codex_diagnostics

function extractCodexFailureReason(task: AgentTask): string | null {
  if (!task.executionLog) return null
  for (let i = task.executionLog.length - 1; i >= 0; i--) {
    const entry = task.executionLog[i]
    if (entry.event === "task_failed") {
      const diag = (entry.details?.codex_diagnostics ?? {}) as Record<string, unknown>
      const code = String(diag.reason_code ?? "")
      if (code) return code
      if (entry.message) return entry.message
    }
  }
  return task.lastError ?? null
}

// User-facing labels for reason codes:
const CODEX_REASON_LABELS: Record<string, string> = {
  max_iterations_exhausted: "Iteration limit reached",
  stagnation_detected: "No progress detected",
  repeated_tool_calls: "Stuck in tool loop",
  too_many_tool_errors: "Too many errors",
  terminated_finish_reason: "Model stopped early",
  timeout: "Codex timeout",
  sandbox_crash: "Sandbox crashed",
}
```

### Emitting Delegation Events in agent_board.py

```python
# Source: agent_board.py pattern — emit at each delegation lifecycle point
# Place after task.assignee is set (dispatched), after first on_step (accepted),
# after result check (completed/failed)

import asyncio

async def _emit_codex_event(
    event_type: str,
    task: "AgentTask",
    session: "BoardSession",
    extra: dict,
) -> None:
    """Emit a Codex delegation lifecycle event. Best-effort; never raises."""
    try:
        from ...core.di.container import Container
        from ...application.collaboration.message_schema import (
            make_event, new_run_id, new_trace_id, EventType,
        )
        el = Container.instance().event_log
        if el is None:
            return
        env = make_event(
            run_id=new_run_id(),
            trace_id=new_trace_id(),
            workflow="agent_board",
            stage=task.id,
            attempt=0,
            agent_name=task.assignee or "codex",
            role="worker",
            type=event_type,
            payload={
                "task_id": task.id,
                "task_title": task.title,
                "session_id": session.session_id,
                **extra,
            },
        )
        el.append(env)
    except Exception:
        log.debug("Failed to emit codex event %s for task %s", event_type, task.id, exc_info=True)
```

### EventType Constants to Add

```python
# Source: src/paperbot/application/collaboration/message_schema.py — APPEND to EventType class

# --- Codex delegation events (Phase 10 / CDX-03) ---
CODEX_DISPATCHED: str = "codex_dispatched"
CODEX_ACCEPTED: str = "codex_accepted"
CODEX_COMPLETED: str = "codex_completed"
CODEX_FAILED: str = "codex_failed"
```

### AgentEventStore Extension for Kanban Tasks

```typescript
// web/src/lib/agent-events/store.ts — additional fields

// In the interface:
kanbanTasks: AgentTask[]
upsertKanbanTask: (task: AgentTask) => void

// In the create() call:
kanbanTasks: [],
upsertKanbanTask: (task) =>
  set((s) => {
    const idx = s.kanbanTasks.findIndex((t) => t.id === task.id)
    if (idx === -1) {
      return { kanbanTasks: [...s.kanbanTasks, task] }
    }
    const updated = [...s.kanbanTasks]
    updated[idx] = { ...updated[idx], ...task }
    return { kanbanTasks: updated }
  }),
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| AgentBoard uses ReactFlow DAG only | Kanban column view added as alternative view mode | Phase 10 | Users see columnar task flow matching familiar Kanban mental model; DAG view remains available |
| Codex execution invisible to activity feed | Four delegation lifecycle events emitted to EventBus | Phase 10 | Users see `codex_dispatched → codex_accepted → codex_completed/failed` in real-time |
| Codex error info buried in task log | `reason_code` surfaced as badge on Kanban card | Phase 10 | Failed Codex tasks are immediately visible in the board column and labelled with their failure reason |
| Claude Code delegates to Codex via ad-hoc prompting | `.claude/agents/codex-worker.md` defines the delegation protocol | Phase 10 | Claude Code has a stable, documented sub-agent definition for Codex delegation |

**Deprecated/outdated:**
- None — Phase 10 adds to Phase 9's established three-panel layout; nothing is replaced.

---

## Open Questions

1. **Should `KanbanBoard` read from `useStudioStore` (studio session tasks) or `useAgentEventStore` (SSE-derived tasks)?**
   - What we know: Studio session tasks live in `useStudioStore.tasks` and are updated by `AgentBoard.tsx`'s SSE handler. The agent dashboard uses `useAgentEventStore`.
   - What's unclear: Whether the Kanban view is meant to show studio session tasks (Paper2Code context) or generic agent tasks (any SSE run).
   - Recommendation: For Phase 10, embed the Kanban board in the studio agent-board context (reads from `useStudioStore`). The agent dashboard page remains the three-panel layout. This avoids the complexity of syncing two stores.

2. **What is the exact trigger condition for Paper2Code overflow (CDX-02)?**
   - What we know: The requirement says "when workload exceeds capacity." Phase 10 implements an env var flag (`PAPERBOT_CODEX_OVERFLOW_THRESHOLD`) rather than dynamic capacity measurement.
   - What's unclear: Whether the requirements team expects real-time capacity tracking (active task count) or a simple manual flag.
   - Recommendation: Phase 10 implements a simple boolean flag. A follow-up phase can add dynamic threshold checking. Document the flag clearly so users can enable it.

3. **Where in the agent dashboard does the Kanban board appear?**
   - What we know: Phase 9 delivered a three-panel layout at `/agent-dashboard`. DASH-02 says "Kanban board showing Claude Code and Codex agent identity."
   - What's unclear: Does the board replace TasksPanel (left rail), replace the whole layout, or appear as a separate page?
   - Recommendation: Add a view-mode toggle (icon buttons) in the dashboard page header: "Panels" | "Kanban". When "Kanban" is selected, replace the `SplitPanels` component with a full-width `KanbanBoard`. This is the cleanest approach — no nested panels fighting horizontal scroll.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | vitest 2.1.4 (frontend); pytest + pytest-asyncio (backend) |
| Config file | `web/vitest.config.ts` — environment: "node", alias: "@" → "./src" |
| Quick run command | `cd web && npm test -- agent-dashboard KanbanBoard` |
| Full suite command | `cd web && npm test -- agent-dashboard agent-events` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DASH-02 | `KanbanBoard` renders columns "Planned", "In Progress", "Review", "Done", "Blocked" | unit (vitest) | `cd web && npm test -- KanbanBoard` | Wave 0 |
| DASH-02 | `KanbanBoard` shows "Claude Code" badge for tasks with `assignee: "claude"` | unit (vitest) | `cd web && npm test -- KanbanBoard` | Wave 0 |
| DASH-02 | `KanbanBoard` shows "Codex" badge for tasks with `assignee: "codex-a1b2"` | unit (vitest) | `cd web && npm test -- KanbanBoard` | Wave 0 |
| DASH-02 | `KanbanBoard` shows task count badge per column | unit (vitest) | `cd web && npm test -- KanbanBoard` | Wave 0 |
| DASH-03 | `extractCodexFailureReason()` returns `reason_code` from last `task_failed` log entry | unit (vitest) | `cd web && npm test -- KanbanBoard` | Wave 0 |
| DASH-03 | Failed Codex task card shows red Error badge and `reason_code` label | unit (vitest) | `cd web && npm test -- KanbanBoard` | Wave 0 |
| CDX-01 | `.claude/agents/codex-worker.md` exists and has valid YAML front-matter with `name: codex-worker` | file existence check (Bash) | `test -f .claude/agents/codex-worker.md && head -3 .claude/agents/codex-worker.md` | Wave 0 |
| CDX-03 | `EventType.CODEX_DISPATCHED == "codex_dispatched"` etc. (all four) | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -x` | Extend existing |
| CDX-03 | `_emit_codex_event()` calls `el.append()` with correct payload | unit (pytest, mock event_log) | `PYTHONPATH=src pytest tests/unit/test_agent_board_route.py -k codex_event -x` | Wave 0 |
| CDX-03 | `parseCodexDelegation()` returns `CodexDelegationEntry` for `codex_dispatched` events | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| CDX-03 | `parseCodexDelegation()` returns `null` for lifecycle events | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| CDX-03 | `deriveHumanSummary()` returns readable string for all four `codex_*` types | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| CDX-02 | `_should_overflow_to_codex()` returns `False` when `PAPERBOT_CODEX_OVERFLOW_THRESHOLD` unset | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_codex_overflow.py -x` | Wave 0 (new file) |
| CDX-02 | `_should_overflow_to_codex()` returns `True` when env var is `"true"` | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_codex_overflow.py -x` | Wave 0 (new file) |

### Sampling Rate

- **Per task commit:** `cd web && npm test -- KanbanBoard agent-events 2>&1 | tail -10`
- **Per wave merge:** `cd web && npm test -- KanbanBoard agent-events && PYTHONPATH=src pytest tests/unit/test_agent_board_route.py tests/unit/test_codex_overflow.py -q`
- **Phase gate:** Full vitest suite green (`cd web && npm test`) + Python unit suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `web/src/components/agent-dashboard/KanbanBoard.tsx` — new component (renders columns, agent badges, error badges)
- [ ] `web/src/components/agent-dashboard/KanbanBoard.test.tsx` — unit tests (columns, badges, error state)
- [ ] `web/src/lib/agent-events/parsers.ts` — MODIFIED: add `parseCodexDelegation()`, extend `deriveHumanSummary()` for 4 codex types
- [ ] `web/src/lib/agent-events/parsers.test.ts` — EXTENDED: `parseCodexDelegation` test cases
- [ ] `web/src/lib/agent-events/types.ts` — MODIFIED: add `CodexDelegationEntry` type
- [ ] `web/src/lib/agent-events/store.ts` — MODIFIED: add `kanbanTasks`, `upsertKanbanTask`
- [ ] `.claude/agents/codex-worker.md` — new Claude Code sub-agent definition
- [ ] `src/paperbot/application/collaboration/message_schema.py` — MODIFIED: add 4 `CODEX_*` EventType constants
- [ ] `src/paperbot/api/routes/agent_board.py` — MODIFIED: add `_emit_codex_event()` and 4 emission points
- [ ] `tests/unit/test_codex_overflow.py` — new: `_should_overflow_to_codex()` env var tests
- [ ] `tests/unit/test_agent_board_route.py` — EXTENDED: `_emit_codex_event()` tests with mock event_log
- [ ] Extend `tests/unit/test_agent_events_vocab.py` — verify 4 new EventType constants

*(No new framework install needed — vitest and pytest already configured)*

---

## Sources

### Primary (HIGH confidence)

- Codebase direct read: `web/src/components/studio/AgentBoard.tsx` — confirmed `AgentTask.assignee` pattern (`"codex-{hex4}"`, `"codex-retry-{hex4}"`, `"claude"`), existing task status set, ReactFlow DAG structure
- Codebase direct read: `web/src/lib/store/studio-store.ts` — confirmed `AgentTask` interface with `assignee`, `status`, `executionLog`, `lastError` fields
- Codebase direct read: `src/paperbot/api/routes/agent_board.py` — confirmed `AgentTask` Pydantic model, `_format_codex_failure()`, `BoardSession`, `CodexResult.diagnostics`, dispatch patterns
- Codebase direct read: `src/paperbot/infrastructure/swarm/codex_dispatcher.py` — confirmed `CodexResult` dataclass with `diagnostics: Dict[str, Any]`, timeout error handling, reason codes (`stagnation_detected`, `max_iterations_exhausted`, etc.)
- Codebase direct read: `src/paperbot/application/collaboration/message_schema.py` — confirmed `EventType` class pattern (plain class, string constants), `make_event()` factory, existing constants (FILE_CHANGE already added in Phase 9)
- Codebase direct read: `web/src/lib/agent-events/store.ts` — confirmed Phase 9 store shape, `filesTouched` extension pattern
- Codebase direct read: `web/src/lib/agent-events/parsers.ts` — confirmed parser function signatures, `parseFileTouched()` pattern for new parser addition
- Codebase direct read: `web/src/lib/agent-events/types.ts` — confirmed TypeScript type conventions
- Codebase direct read: `web/src/components/agent-dashboard/TasksPanel.tsx` — confirmed Phase 9 component shape, `useAgentEventStore` read pattern
- Codebase direct read: `.planning/STATE.md` decision log — confirmed "Codex bridge is a `.claude/agents/` file, not PaperBot server code"
- Codebase direct read: `docs/proposals/codex-loop-iteration-policy-plan-zh.md` — confirmed `ToolLoopPolicy`, `LoopProgressTracker`, `diagnostics.reason_code` field structure
- Codebase direct read: `web/package.json` — confirmed no drag-and-drop libraries installed; `@xyflow/react` present (ReactFlow for DAG)
- Codebase direct read: `tests/unit/test_agent_board_route.py` — confirmed test patterns for `agent_board` route (monkeypatch `_board_store`, `_isolated_board_store` fixture)
- Codebase direct read: `.planning/REQUIREMENTS.md` — confirmed DASH-02, DASH-03, CDX-01, CDX-02, CDX-03 requirements verbatim

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` accumulated context section — confirmed key architectural decisions for Codex bridge and event vocabulary patterns
- `docs/proposals/codex-loop-iteration-policy-plan-zh.md` — confirmed `reason_code` vocabulary and plan to surface failure diagnostics to frontend

### Tertiary (LOW confidence)

- Claude Code custom agents format inferred from `.claude/agents/` directory convention (internal project decision); official Claude Code documentation format not directly verified. The YAML front-matter with `name:`, `description:`, `tools:` fields is consistent with the `.claude/skills/` SKILL.md format already in use by this project.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies confirmed from `web/package.json`; no new installs required
- Architecture: HIGH — KanbanBoard pattern derived from direct inspection of `AgentBoard.tsx`, `studio-store.ts`, `agent_board.py`; delegation pattern confirmed from `message_schema.py` and Phase 8/9 patterns
- Pitfalls: HIGH — SSE fan-out pattern, Container.instance() event_log access, and layout constraints verified from existing source code

**Research date:** 2026-03-15
**Valid until:** 2026-09-15 (all dependencies pinned; re-verify if Next.js, Zustand, or Claude Code agent format changes major version)
