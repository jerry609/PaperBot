# Phase 8: Agent Event Vocabulary - Research

**Researched:** 2026-03-15
**Domain:** Event type vocabulary + frontend real-time activity feed (Python + React/Zustand)
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVNT-01 | User can view a real-time scrolling activity feed showing agent events as they happen | useAgentEvents hook consuming /api/events/stream; feed component bound to Zustand activity list |
| EVNT-02 | User can see each agent's lifecycle status (idle, working, completed, errored) at a glance | agent_started/agent_completed/agent_error typed events update per-agent status map in Zustand store |
| EVNT-03 | User can view a structured tool call timeline showing tool name, arguments, result summary, and duration | tool_call/tool_result typed events with standardized payload shape; existing _audit.py already emits compatible data |
</phase_requirements>

---

## Summary

Phase 8 has two distinct halves: a **Python back-end vocabulary layer** (standardized event types and a helper API for emitting lifecycle/tool-call events) and a **React front-end rendering layer** (SSE consumer hook + activity feed + agent status badge + tool call timeline components).

The back-end work is small. `AgentEventEnvelope` already exists and already flows through `EventBusEventLog` to `/api/events/stream`. What is missing is a **defined set of `type` string values** that the dashboard understands, plus convenience helpers for emitting them. The existing `_audit.py` already emits `tool_result` events with the right payload shape; this phase standardizes the vocabulary and adds the missing lifecycle event types (`agent_started`, `agent_working`, `agent_completed`, `agent_error`).

The front-end work is the bulk of the phase. A `useAgentEvents` hook connects to `/api/events/stream` via the existing `readSSE()` utility, parses incoming events, and writes them into a Zustand store. The store maintains: a bounded activity feed list (capped at ~200 entries to avoid unbounded growth), a per-agent status map keyed by `agent_name`, and a per-tool-call timeline. Three display components consume the store: `ActivityFeed`, `AgentStatusBadge`/`AgentStatusPanel`, and `ToolCallTimeline`.

**Primary recommendation:** Define the vocabulary as constants in `message_schema.py`, add four lifecycle event helpers in a new `agent_events.py` helper module, and build a self-contained `useAgentEvents` hook + three components in `web/src/lib/agent-events/` and `web/src/components/agent-events/`.

---

## Standard Stack

### Core (zero new dependencies — all already present)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `asyncio.Queue` / `EventBusEventLog` | stdlib / Phase 7 | Already delivers events to `/api/events/stream` | Phase 7 is complete; bus is live |
| `zustand` | ^5.0.9 (in `web/package.json`) | Client-side store for activity feed + agent status | Already the project's state management library |
| `readSSE()` in `web/src/lib/sse.ts` | project code | Parses `data: {...}\n\n` frames from SSE stream | Already used by P2C generation hook; proven pattern |
| React `useEffect` + `useRef` | React 19 | Connection lifecycle management in the hook | Standard; no third-party SSE library needed |
| `lucide-react` | ^0.562.0 (in `web/package.json`) | Status icons (CircleDot, CheckCircle2, XCircle, etc.) | Already the project's icon library |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `@radix-ui/react-scroll-area` | ^1.2.10 | Scrollable activity feed container | Already installed; use for the feed panel |
| `framer-motion` | ^12.23.26 | Subtle entrance animation for feed rows | Already installed; optional, keep lightweight |
| `tailwindcss` | ^4 | All component styling | Already the project's CSS framework |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Plain `readSSE()` loop | `EventSource` browser API | `EventSource` does not support POST or custom headers; `readSSE()` works with the existing `fetch` flow and is already tested in the project |
| Zustand store | React Context + useReducer | Zustand is already used for `useWorkflowStore` and `useStudioStore`; consistency matters |
| Inline type constants | New Python enum | Enums are harder to extend for new event types and introduce import coupling; string constants in a `VOCAB` dict are lighter and consistent with the existing `type: str = ""` field on `AgentEventEnvelope` |

**Installation:** No new packages needed. All primitives already in `pyproject.toml` and `web/package.json`.

---

## Architecture Patterns

### Recommended File Structure

```
src/paperbot/
├── application/
│   └── collaboration/
│       ├── message_schema.py          # MODIFIED: add vocabulary constants
│       └── agent_events.py            # NEW: make_lifecycle_event(), make_tool_call_event()
└── ...

web/src/
├── lib/
│   └── agent-events/
│       ├── types.ts                   # NEW: TypeScript types mirroring Python vocab
│       ├── store.ts                   # NEW: useAgentEventStore (Zustand)
│       └── useAgentEvents.ts          # NEW: SSE consumer hook
└── components/
    └── agent-events/
        ├── ActivityFeed.tsx           # NEW: scrolling list (EVNT-01)
        ├── AgentStatusPanel.tsx       # NEW: per-agent badge grid (EVNT-02)
        └── ToolCallTimeline.tsx       # NEW: tool call rows (EVNT-03)

tests/
└── unit/
    └── test_agent_events_vocab.py     # NEW: vocabulary helpers unit tests
```

### Pattern 1: Python Vocabulary Constants in message_schema.py

Add a `EventType` namespace object (not an enum — consistent with existing `type: str = ""`):

```python
# src/paperbot/application/collaboration/message_schema.py
# Append after existing dataclass definitions

class EventType:
    """
    Canonical event type strings for AgentEventEnvelope.type.

    All new code MUST use these constants rather than raw string literals.
    Existing callers (arq_worker, connectors) may be migrated over time.
    """
    # Lifecycle
    AGENT_STARTED   = "agent_started"   # Agent began processing a stage
    AGENT_WORKING   = "agent_working"   # Agent is actively executing (heartbeat)
    AGENT_COMPLETED = "agent_completed" # Agent finished successfully
    AGENT_ERROR     = "agent_error"     # Agent encountered an unrecoverable error

    # Tool calls (standardize existing ad-hoc "tool_result" / "error" usage)
    TOOL_CALL       = "tool_call"       # Tool invocation started (optional pre-event)
    TOOL_RESULT     = "tool_result"     # Tool returned a result
    TOOL_ERROR      = "tool_error"      # Tool call failed

    # Existing types (document for completeness — do NOT rename callers this phase)
    JOB_START       = "job_start"
    JOB_RESULT      = "job_result"
    JOB_ENQUEUE     = "job_enqueue"
    STAGE_EVENT     = "stage_event"
    SOURCE_RECORD   = "source_record"
    SCORE_UPDATE    = "score_update"
    INSIGHT         = "insight"
```

**Why not an Enum:** The existing field is `type: str = ""`. Changing callers to `EventType.TOOL_RESULT` works without any protocol change. Enums would require `.value` everywhere, breaking the existing `make_event(type=...)` call sites.

### Pattern 2: Lifecycle Event Helper

```python
# src/paperbot/application/collaboration/agent_events.py
from __future__ import annotations

from typing import Any, Dict, Optional
from .message_schema import AgentEventEnvelope, EventType, make_event, new_run_id, new_trace_id


def make_lifecycle_event(
    *,
    status: str,                # one of EventType.AGENT_* constants
    agent_name: str,
    run_id: str,
    trace_id: str,
    workflow: str,
    stage: str,
    attempt: int = 0,
    role: str = "worker",
    detail: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> AgentEventEnvelope:
    """
    Emit an agent lifecycle status event.

    Payload shape (stable contract consumed by frontend):
        {
          "status": "agent_started" | "agent_working" | "agent_completed" | "agent_error",
          "agent_name": str,
          "detail": str | null
        }
    """
    payload: Dict[str, Any] = {
        "status": status,
        "agent_name": agent_name,
    }
    if detail is not None:
        payload["detail"] = detail
    return make_event(
        run_id=run_id,
        trace_id=trace_id,
        workflow=workflow,
        stage=stage,
        attempt=attempt,
        agent_name=agent_name,
        role=role,
        type=status,
        payload=payload,
        metrics=metrics or {},
        tags=tags or {},
    )


def make_tool_call_event(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    result_summary: str,
    duration_ms: float,
    run_id: str,
    trace_id: str,
    workflow: str = "mcp",
    stage: str = "tool_call",
    agent_name: str = "paperbot-mcp",
    role: str = "system",
    error: Optional[str] = None,
) -> AgentEventEnvelope:
    """
    Emit a structured tool call result event.

    Payload shape (stable contract consumed by ToolCallTimeline):
        {
          "tool": str,
          "arguments": dict,
          "result_summary": str,
          "error": str | null,
          "duration_ms": float
        }
    """
    return make_event(
        run_id=run_id,
        trace_id=trace_id,
        workflow=workflow,
        stage=stage,
        attempt=0,
        agent_name=agent_name,
        role=role,
        type=EventType.TOOL_ERROR if error else EventType.TOOL_RESULT,
        payload={
            "tool": tool_name,
            "arguments": arguments,
            "result_summary": result_summary,
            "error": error,
        },
        metrics={"duration_ms": duration_ms},
    )
```

**Note on `_audit.py`:** The existing `log_tool_call()` in `_audit.py` already emits a compatible payload. Phase 8 should update `_audit.py` to use `EventType.TOOL_RESULT` / `EventType.TOOL_ERROR` constants instead of raw strings. No payload structure changes needed — the frontend types will match the existing shape.

### Pattern 3: TypeScript Types (mirroring Python vocab)

```typescript
// web/src/lib/agent-events/types.ts

export type AgentStatus = "idle" | "working" | "completed" | "errored"

export type AgentLifecycleEvent = {
  type: "agent_started" | "agent_working" | "agent_completed" | "agent_error"
  run_id: string
  trace_id: string
  agent_name: string
  workflow: string
  stage: string
  ts: string
  payload: {
    status: string
    agent_name: string
    detail?: string
  }
}

export type ToolCallEvent = {
  type: "tool_result" | "tool_error"
  run_id: string
  trace_id: string
  agent_name: string
  workflow: string
  stage: string
  ts: string
  payload: {
    tool: string
    arguments: Record<string, unknown>
    result_summary: string
    error: string | null
  }
  metrics: {
    duration_ms: number
  }
}

export type AgentEventEnvelopeRaw = Record<string, unknown> & {
  type: string
  run_id?: string
  trace_id?: string
  agent_name?: string
  workflow?: string
  stage?: string
  ts?: string
  payload?: Record<string, unknown>
  metrics?: Record<string, unknown>
}

// Derived display types
export type ActivityFeedItem = {
  id: string            // run_id + ts (dedup key)
  type: string
  agent_name: string
  workflow: string
  stage: string
  ts: string
  summary: string       // human-readable line (derived from payload)
  raw: AgentEventEnvelopeRaw
}

export type AgentStatusEntry = {
  agent_name: string
  status: AgentStatus
  last_stage: string
  last_ts: string
}

export type ToolCallEntry = {
  id: string            // run_id + tool + ts
  tool: string
  agent_name: string
  arguments: Record<string, unknown>
  result_summary: string
  error: string | null
  duration_ms: number
  ts: string
  status: "ok" | "error"
}
```

### Pattern 4: Zustand Store for Agent Events

```typescript
// web/src/lib/agent-events/store.ts
import { create } from "zustand"
import type { ActivityFeedItem, AgentStatusEntry, ToolCallEntry } from "./types"

const FEED_MAX = 200          // cap activity feed to avoid unbounded memory
const TOOL_TIMELINE_MAX = 100

interface AgentEventState {
  // SSE connection status
  connected: boolean
  setConnected: (c: boolean) => void

  // Activity feed — newest first, capped at FEED_MAX
  feed: ActivityFeedItem[]
  addFeedItem: (item: ActivityFeedItem) => void
  clearFeed: () => void

  // Per-agent status map
  agentStatuses: Record<string, AgentStatusEntry>
  updateAgentStatus: (entry: AgentStatusEntry) => void

  // Tool call timeline — newest first, capped at TOOL_TIMELINE_MAX
  toolCalls: ToolCallEntry[]
  addToolCall: (entry: ToolCallEntry) => void
  clearToolCalls: () => void
}

export const useAgentEventStore = create<AgentEventState>((set) => ({
  connected: false,
  setConnected: (c) => set({ connected: c }),

  feed: [],
  addFeedItem: (item) =>
    set((s) => ({
      feed: [item, ...s.feed].slice(0, FEED_MAX),
    })),
  clearFeed: () => set({ feed: [] }),

  agentStatuses: {},
  updateAgentStatus: (entry) =>
    set((s) => ({
      agentStatuses: { ...s.agentStatuses, [entry.agent_name]: entry },
    })),

  toolCalls: [],
  addToolCall: (entry) =>
    set((s) => ({
      toolCalls: [entry, ...s.toolCalls].slice(0, TOOL_TIMELINE_MAX),
    })),
  clearToolCalls: () => set({ toolCalls: [] }),
}))
```

### Pattern 5: useAgentEvents Hook

The hook connects to `/api/events/stream`, parses each incoming `AgentEventEnvelope`, and dispatches to the Zustand store. Connection management uses `useEffect` + `AbortController`.

```typescript
// web/src/lib/agent-events/useAgentEvents.ts
"use client"

import { useEffect, useRef } from "react"
import { readSSE } from "@/lib/sse"
import { useAgentEventStore } from "./store"
import { parseActivityItem, parseAgentStatus, parseToolCall } from "./parsers"
import type { AgentEventEnvelopeRaw } from "./types"

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export function useAgentEvents() {
  const { setConnected, addFeedItem, updateAgentStatus, addToolCall } = useAgentEventStore()
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller

    async function connect() {
      try {
        const res = await fetch(`${BACKEND_URL}/api/events/stream`, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        })
        if (!res.ok || !res.body) return
        setConnected(true)

        for await (const msg of readSSE(res.body)) {
          const raw = msg as AgentEventEnvelopeRaw
          if (!raw?.type) continue

          const feedItem = parseActivityItem(raw)
          if (feedItem) addFeedItem(feedItem)

          const statusEntry = parseAgentStatus(raw)
          if (statusEntry) updateAgentStatus(statusEntry)

          const toolCall = parseToolCall(raw)
          if (toolCall) addToolCall(toolCall)
        }
      } catch (err) {
        if ((err as Error)?.name !== "AbortError") {
          console.warn("[useAgentEvents] disconnected, will retry in 3s", err)
          setTimeout(connect, 3000)
        }
      } finally {
        setConnected(false)
      }
    }

    connect()
    return () => {
      controller.abort()
    }
  }, [setConnected, addFeedItem, updateAgentStatus, addToolCall])
}
```

**Why no `EventSource`:** The existing project uses `fetch` + `readSSE()` for all SSE connections (see `useContextPackGeneration.ts`). Consistency with this pattern avoids introducing a second paradigm. The `readSSE()` utility already handles keep-alive comments (it skips lines not starting with `data:`).

### Pattern 6: Parser Helpers

A pure `parsers.ts` module converts raw envelope dicts to typed display objects:

```typescript
// web/src/lib/agent-events/parsers.ts
import type {
  ActivityFeedItem, AgentStatusEntry, AgentStatus, ToolCallEntry, AgentEventEnvelopeRaw
} from "./types"

const LIFECYCLE_TYPES = new Set([
  "agent_started", "agent_working", "agent_completed", "agent_error"
])

const TOOL_TYPES = new Set(["tool_result", "tool_error", "error"])

export function parseActivityItem(raw: AgentEventEnvelopeRaw): ActivityFeedItem | null {
  if (!raw.type || !raw.ts) return null
  const id = `${raw.run_id ?? ""}-${raw.ts}`
  const payload = raw.payload ?? {}
  const summary = deriveHumanSummary(raw)
  return {
    id,
    type: raw.type,
    agent_name: String(raw.agent_name ?? "unknown"),
    workflow: String(raw.workflow ?? ""),
    stage: String(raw.stage ?? ""),
    ts: String(raw.ts),
    summary,
    raw,
  }
}

function deriveHumanSummary(raw: AgentEventEnvelopeRaw): string {
  const t = raw.type ?? ""
  const payload = (raw.payload ?? {}) as Record<string, unknown>

  if (t === "agent_started") return `${raw.agent_name} started: ${raw.stage}`
  if (t === "agent_working") return `${raw.agent_name} working on: ${raw.stage}`
  if (t === "agent_completed") return `${raw.agent_name} completed: ${raw.stage}`
  if (t === "agent_error") return `${raw.agent_name} error: ${String(payload.detail ?? "")}`
  if (t === "tool_result" || t === "tool_error") {
    const tool = String(payload.tool ?? t)
    return `Tool: ${tool} — ${String(payload.result_summary ?? "").slice(0, 80)}`
  }
  if (t === "job_start") return `Job started: ${raw.stage}`
  if (t === "job_result") return `Job finished: ${raw.stage}`
  if (t === "source_record") return `Source record: ${raw.workflow}/${raw.stage}`
  if (t === "score_update") return `Score update from ${raw.agent_name}`
  if (t === "insight") return `Insight from ${raw.agent_name}`
  // Fallback: stringify the type
  return `${t}: ${raw.agent_name ?? ""} / ${raw.stage ?? ""}`
}

export function parseAgentStatus(raw: AgentEventEnvelopeRaw): AgentStatusEntry | null {
  if (!LIFECYCLE_TYPES.has(String(raw.type ?? ""))) return null
  const statusMap: Record<string, AgentStatus> = {
    agent_started: "working",
    agent_working: "working",
    agent_completed: "completed",
    agent_error: "errored",
  }
  return {
    agent_name: String(raw.agent_name ?? "unknown"),
    status: statusMap[raw.type as string] ?? "idle",
    last_stage: String(raw.stage ?? ""),
    last_ts: String(raw.ts ?? ""),
  }
}

export function parseToolCall(raw: AgentEventEnvelopeRaw): ToolCallEntry | null {
  if (!TOOL_TYPES.has(String(raw.type ?? ""))) return null
  const payload = (raw.payload ?? {}) as Record<string, unknown>
  const metrics = (raw.metrics ?? {}) as Record<string, unknown>
  const tool = String(payload.tool ?? raw.stage ?? "unknown")
  return {
    id: `${raw.run_id ?? ""}-${tool}-${raw.ts ?? ""}`,
    tool,
    agent_name: String(raw.agent_name ?? "unknown"),
    arguments: (payload.arguments as Record<string, unknown>) ?? {},
    result_summary: String(payload.result_summary ?? ""),
    error: typeof payload.error === "string" ? payload.error : null,
    duration_ms: typeof metrics.duration_ms === "number" ? metrics.duration_ms : 0,
    ts: String(raw.ts ?? ""),
    status: raw.type === "tool_error" || (typeof payload.error === "string" && payload.error) ? "error" : "ok",
  }
}
```

### Anti-Patterns to Avoid

- **Adding a new top-level envelope field for status:** The `type` field already conveys event kind. Adding a `status` field to `AgentEventEnvelope` itself would break the clean `type` contract and confuse consumers. Use `payload.status` for lifecycle payload details.
- **Creating a parallel event schema:** The success criterion explicitly prohibits this. All new event types MUST be emitted as `AgentEventEnvelope` instances via `make_event()`.
- **Polling `/api/events` from multiple components:** Mount `useAgentEvents` exactly once (at the layout or page root). Multiple mounts create multiple SSE connections. Components read from the Zustand store, not from the hook directly.
- **Unbounded activity feed array:** Without a cap (`FEED_MAX = 200`), long-running sessions will accumulate thousands of events in memory. Cap in the Zustand `addFeedItem` action.
- **Re-connecting on every re-render:** The `useEffect` dependency array must be stable (store action references are stable in Zustand — they do not change on re-renders), so the `AbortController` is not re-created unnecessarily.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSE frame parsing | Custom TextDecoder loop | `readSSE()` from `web/src/lib/sse.ts` | Already handles keep-alive comments, `[DONE]` sentinel, partial frames, and JSON parse errors — all edge cases that bite custom implementations |
| Status derivation from events | Custom state machine class | Zustand store + `parseAgentStatus()` | A pure reducer pattern in Zustand is simpler than a class-based state machine and avoids mutable shared state |
| Scrollable list | Custom overflow-scroll div | `@radix-ui/react-scroll-area` | Already installed; handles cross-browser scrollbar normalization and keyboard accessibility |
| Connection retry | Manual `setTimeout` polling | 3-second retry in `catch` block of `connect()` | Simple, effective for SSE reconnect; matches browser `EventSource` default backoff behaviour without a library |
| Duration formatting | Custom `ms → "2.3s"` util | Inline in component (`(ms/1000).toFixed(1) + "s"`) | Too small to warrant a utility; inline is readable |

---

## Common Pitfalls

### Pitfall 1: Multiple SSE Connections from Multiple `useAgentEvents` Mounts

**What goes wrong:** Two components both call `useAgentEvents()`. The browser opens two connections to `/api/events/stream`. Both connections consume memory on the server (two `asyncio.Queue` instances in `EventBusEventLog`). The Zustand store receives every event twice, duplicating feed items.

**Why it happens:** Each mount of the hook creates its own `useEffect` with its own `AbortController` and `fetch` call.

**How to avoid:** Mount `useAgentEvents` exactly once — at the root of the agent dashboard page/layout. Child components subscribe to `useAgentEventStore` directly. This is the same pattern used for `useWorkflowStore` and `useStudioStore`.

**Warning signs:** Feed items appear duplicated; `EventBusEventLog._queues` set has more entries than expected.

### Pitfall 2: Activity Feed Grows Unboundedly

**What goes wrong:** Without a cap, the Zustand `feed` array accumulates all events since mount. On a busy server with ARQ cron jobs, connectors, and MCP calls, this can grow to thousands of items rapidly.

**Why it happens:** Events are appended without a max-length check.

**How to avoid:** Cap in `addFeedItem`: `[item, ...s.feed].slice(0, FEED_MAX)` where `FEED_MAX = 200`. This keeps the newest 200 items.

**Warning signs:** Browser memory usage grows steadily; React DevTools shows a very large `feed` array.

### Pitfall 3: Agent Status Stuck on "working" After Error

**What goes wrong:** An agent emits `agent_started` and `agent_working`, then crashes without emitting `agent_error`. The status badge stays "working" forever.

**Why it happens:** The back-end code path that handles the exception doesn't emit an `agent_error` event.

**How to avoid:** Wrap agent hot-paths in `try/finally` that emits `agent_error` on exception. The `make_lifecycle_event(status=EventType.AGENT_ERROR, ...)` helper makes this a one-liner. Document this as a convention in `agent_events.py` docstring.

**Warning signs:** Status badge shows "working" for an agent that has not emitted events in > 30 seconds.

### Pitfall 4: Tool Arguments Contain Sensitive Data in the Frontend

**What goes wrong:** MCP tool arguments (e.g., API keys passed as parameters) are logged in the event payload and rendered verbatim in `ToolCallTimeline`.

**Why it happens:** `_audit.py` already sanitizes arguments via `_sanitize_arguments()` before emitting. However, if new callers use `make_tool_call_event()` directly without sanitization, raw sensitive data ends up in the SSE stream.

**How to avoid:** `make_tool_call_event()` should accept pre-sanitized arguments only. Document this requirement. The frontend `ToolCallTimeline` should collapse deep argument objects (show first-level keys, expand on click) rather than rendering full nested JSON.

**Warning signs:** API key strings visible in the browser's developer tools network tab.

### Pitfall 5: Importing `useAgentEvents` in a Server Component

**What goes wrong:** Next.js App Router Server Components cannot run hooks. Importing `useAgentEvents` or `useAgentEventStore` in a Server Component causes a build error: `Error: Hooks can only be called inside a Client Component`.

**Why it happens:** The hook uses `useEffect`, `useRef`, and Zustand's `create()` — all client-only APIs.

**How to avoid:** Mark all files in `web/src/lib/agent-events/` and `web/src/components/agent-events/` with `"use client"` at the top. The dashboard page (Phase 9) will be a Client Component.

**Warning signs:** Build error: `You're importing a component that needs useState...`.

---

## Code Examples

### Python: Emitting an Agent Lifecycle Event

```python
# Source: agent_events.py pattern (new Phase 8 helper)
from paperbot.application.collaboration.agent_events import make_lifecycle_event
from paperbot.application.collaboration.message_schema import EventType

# In any agent or pipeline stage:
event_log.append(
    make_lifecycle_event(
        status=EventType.AGENT_STARTED,
        agent_name="ResearchAgent",
        run_id=run_id,
        trace_id=trace_id,
        workflow="scholar_pipeline",
        stage="paper_search",
    )
)
```

### Python: Updating _audit.py to Use Constants

```python
# Before (in _audit.py):
type="error" if error is not None else "tool_result",

# After (Phase 8 migration):
from paperbot.application.collaboration.message_schema import EventType
...
type=EventType.TOOL_ERROR if error is not None else EventType.TOOL_RESULT,
```

### TypeScript: Reading Agent Status in a Component

```typescript
// Source: Zustand selector pattern (consistent with useWorkflowStore in the project)
import { useAgentEventStore } from "@/lib/agent-events/store"

function AgentStatusPanel() {
  const statuses = useAgentEventStore((s) => s.agentStatuses)
  const connected = useAgentEventStore((s) => s.connected)

  return (
    <div>
      {!connected && <span className="text-amber-500 text-xs">Connecting...</span>}
      {Object.values(statuses).map((entry) => (
        <AgentStatusBadge key={entry.agent_name} entry={entry} />
      ))}
    </div>
  )
}
```

### TypeScript: SSE Reconnect Pattern

```typescript
// Source: pattern adapted from useContextPackGeneration.ts (existing project hook)
async function connect() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/events/stream`, {
      signal: controller.signal,
      headers: { Accept: "text/event-stream" },
    })
    if (!res.ok || !res.body) throw new Error(`SSE connect failed: ${res.status}`)
    setConnected(true)
    for await (const msg of readSSE(res.body)) {
      // dispatch to store ...
    }
  } catch (err) {
    if ((err as Error)?.name !== "AbortError") {
      setTimeout(connect, 3000)   // retry after 3s
    }
  } finally {
    setConnected(false)
  }
}
```

### TypeScript: ActivityFeed Component Skeleton

```typescript
// web/src/components/agent-events/ActivityFeed.tsx
"use client"
import { useAgentEventStore } from "@/lib/agent-events/store"
import * as ScrollArea from "@radix-ui/react-scroll-area"

export function ActivityFeed() {
  const feed = useAgentEventStore((s) => s.feed)

  return (
    <ScrollArea.Root className="h-full overflow-hidden">
      <ScrollArea.Viewport className="h-full w-full">
        <ul className="space-y-1 px-2 py-2">
          {feed.map((item) => (
            <ActivityFeedRow key={item.id} item={item} />
          ))}
        </ul>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar orientation="vertical" />
    </ScrollArea.Root>
  )
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ad-hoc `type` strings scattered across callers | `EventType` constants class in `message_schema.py` | Phase 8 | New code uses constants; existing callers unchanged (backward compatible) |
| `_audit.py` raw `"tool_result"` / `"error"` strings | `EventType.TOOL_RESULT` / `EventType.TOOL_ERROR` | Phase 8 | Consistent naming; frontend parser reliable |
| No agent lifecycle events | `agent_started`/`agent_working`/`agent_completed`/`agent_error` via `make_lifecycle_event()` | Phase 8 | Dashboard can derive per-agent status |
| No frontend SSE consumer for global events bus | `useAgentEvents` hook + `useAgentEventStore` | Phase 8 | Enables EVNT-01, EVNT-02, EVNT-03 |

**Deprecated/outdated:**

- Raw string literals for `type=` in new code: use `EventType.*` constants going forward.
- The `PROGRESS_LIKE` set in `web/src/lib/sse.ts` was designed for workflow-scoped SSE (analyze, search). The global events bus carries `AgentEventEnvelope` fields directly, so `normalizeSSEMessage()` from `sse.ts` is NOT used in `useAgentEvents` — parse the raw envelope directly.

---

## Open Questions

1. **Where to mount `useAgentEvents` in Phase 8?**
   - What we know: Phase 9 builds the three-panel IDE layout. Phase 8 builds the components but there is no dedicated page yet.
   - What's unclear: Should Phase 8 add a standalone `/agent-events` route for testing, or mount in the existing `/studio` page?
   - Recommendation: Create a minimal `web/src/app/agent-events/page.tsx` as a test harness for Phase 8. Phase 9 will integrate the components into the full layout. The test harness can be removed or repurposed later.

2. **Should `make_lifecycle_event()` automatically emit via `event_log`?**
   - What we know: `_audit.py`'s `log_tool_call()` fetches `event_log` from `Container` and appends internally — a "fire and forget" audit function. `make_event()` returns an envelope and the caller appends.
   - What's unclear: Which pattern is better for agent lifecycle events?
   - Recommendation: Return the envelope (same as `make_event()`). Let the caller append. This is more testable and doesn't couple the helper to `Container`. Document the call pattern: `event_log.append(make_lifecycle_event(...))`.

3. **Agent status reset on new run?**
   - What we know: `agent_completed` and `agent_error` set terminal status. But the same agent may run again on the next request.
   - What's unclear: Should receiving `agent_started` reset a "completed" status to "working"?
   - Recommendation: Yes. The `parseAgentStatus` parser maps `agent_started` → `"working"`, overwriting any prior terminal status. This is correct because a new run starts fresh.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio 0.21+ (backend); vitest 2.1.4 (frontend) |
| Config file | `pyproject.toml` — `asyncio_mode = "strict"` |
| Quick run command | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -q` |
| Full suite command | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py tests/integration/test_events_sse_endpoint.py -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVNT-01 | `parseActivityItem()` returns ActivityFeedItem for any event type | unit (frontend vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| EVNT-01 | Feed capped at FEED_MAX after many appends | unit (frontend vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| EVNT-02 | `parseAgentStatus()` returns correct AgentStatus for each lifecycle type | unit (frontend vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| EVNT-02 | `make_lifecycle_event()` produces correct AgentEventEnvelope type field | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_lifecycle_event_types -x` | Wave 0 |
| EVNT-03 | `parseToolCall()` extracts tool, arguments, result_summary, duration_ms | unit (frontend vitest) | `cd web && npm test -- agent-events` | Wave 0 |
| EVNT-03 | `make_tool_call_event()` sets `type=tool_error` when error is provided | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_tool_call_event_error_type -x` | Wave 0 |
| EVNT-01/02/03 | `EventType` constants are unique strings and non-empty | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_event_type_constants -x` | Wave 0 |
| EVNT-03 | `_audit.py` uses `EventType.TOOL_RESULT` / `EventType.TOOL_ERROR` (no raw strings) | unit (pytest — import + grep assert) | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_audit_uses_constants -x` | Wave 0 |

**Note:** Frontend components (ActivityFeed, AgentStatusPanel, ToolCallTimeline) are render-tested via vitest + React Testing Library if available; otherwise verified manually in the test harness page.

### Sampling Rate

- **Per task commit:** `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -q`
- **Per wave merge:** `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py tests/integration/test_events_sse_endpoint.py -q && cd web && npm test`
- **Phase gate:** Full CI suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/test_agent_events_vocab.py` — covers EventType constants, make_lifecycle_event, make_tool_call_event, _audit.py constant usage
- [ ] `web/src/lib/agent-events/parsers.test.ts` — covers parseActivityItem, parseAgentStatus, parseToolCall with fixture envelopes
- [ ] `web/src/lib/agent-events/store.test.ts` — covers feed cap, addFeedItem, updateAgentStatus, addToolCall (adapt existing `studio-store.test.ts` pattern)

*(No framework install needed — pytest-asyncio already present; vitest already in `web/package.json`)*

---

## Sources

### Primary (HIGH confidence)

- Codebase direct read: `src/paperbot/application/collaboration/message_schema.py` — `AgentEventEnvelope`, `make_event()`, existing type field semantics
- Codebase direct read: `src/paperbot/application/collaboration/messages.py` — `MessageType` enum, `AgentMessage`
- Codebase direct read: `src/paperbot/mcp/tools/_audit.py` — existing `log_tool_call()` and payload shape
- Codebase direct read: `src/paperbot/infrastructure/event_log/event_bus_event_log.py` — Phase 7 implementation, `subscribe()`/`unsubscribe()` API
- Codebase direct read: `src/paperbot/api/routes/events.py` — GET `/api/events/stream` endpoint; confirmed live
- Codebase direct read: `src/paperbot/infrastructure/queue/arq_worker.py` — existing event types: `job_start`, `job_result`, `job_enqueue`
- Codebase direct read: `src/paperbot/infrastructure/connectors/*.py` — existing `source_record` event type
- Codebase direct read: `web/src/lib/sse.ts` — `readSSE()` async generator, `SSEMessage` shape
- Codebase direct read: `web/src/hooks/useContextPackGeneration.ts` — SSE connection + reconnect pattern used in the project
- Codebase direct read: `web/src/lib/stores/workflow-store.ts` — Zustand 5 `create()` pattern with persist middleware
- Codebase direct read: `web/src/lib/store/studio-store.ts` — Zustand store shape, bounded arrays pattern, `AgentTask` type
- Codebase direct read: `web/package.json` — all front-end dependency versions verified
- Codebase direct read: `.planning/phases/07-eventbus-sse-foundation/07-01-SUMMARY.md`, `07-02-SUMMARY.md` — Phase 7 completed work

### Secondary (MEDIUM confidence)

- `pyproject.toml` — asyncio_mode strict, pytest-asyncio version confirmed
- Zustand 5 docs (verified via package.json version `^5.0.9`): `create()` API is identical to v4 for basic usage; `persist` middleware unchanged

### Tertiary (LOW confidence)

- None — all research based on direct codebase inspection

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all dependencies confirmed present in package.json / pyproject.toml; no new installs
- Architecture: HIGH — patterns derived from direct inspection of existing Phase 7 code and existing hooks; the vocabulary constants and helper functions are straightforward extensions of code already in place
- Pitfalls: HIGH for front-end (unbounded array, multiple mounts, server component); HIGH for back-end (_audit.py migration); MEDIUM for agent stuck-in-working (depends on caller discipline)

**Research date:** 2026-03-15
**Valid until:** 2026-09-15 (all dependencies are pinned; re-verify if Zustand or Next.js version changes significantly)
