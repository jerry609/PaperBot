---
phase: 08-agent-event-vocabulary
plan: "02"
subsystem: ui
tags: [typescript, zustand, react, sse, next-js, vitest, agent-events, radix-ui]

# Dependency graph
requires:
  - phase: 07-eventbus-sse-foundation
    provides: /api/events/stream SSE endpoint delivering AgentEventEnvelope dicts
  - phase: 08-agent-event-vocabulary (plan 01)
    provides: Python EventType constants and make_lifecycle_event/make_tool_call_event helpers

provides:
  - TypeScript types mirroring Python EventType vocabulary (types.ts — 7 exported types)
  - Pure parser functions converting raw SSE envelopes to typed display objects (parsers.ts)
  - Zustand store for agent event state with bounded feed (200) and tool timeline (100) (store.ts)
  - SSE consumer hook connecting to /api/events/stream with 3s reconnect (useAgentEvents.ts)
  - ActivityFeed component: real-time scrolling event list with color-coded badges
  - AgentStatusPanel component: per-agent idle/working/completed/errored badge grid
  - ToolCallTimeline component: structured tool call rows with name, duration, args, summary
  - Test harness page at /agent-events mounting all three components

affects:
  - 09-three-panel-dashboard (will integrate ActivityFeed, AgentStatusPanel, ToolCallTimeline into main layout)

# Tech tracking
tech-stack:
  added: []  # No new dependencies — all packages already in web/package.json
  patterns:
    - Zustand 5 non-persisted store with create<T>() single-call form (no persist middleware for ephemeral SSE data)
    - SSE hook mounted once at page root; child components read Zustand store directly (no duplicate connections)
    - TDD RED/GREEN cycle for types+parsers+store before implementing hook and components
    - Test reset via useAgentEventStore.getInitialState() (same as studio-store.test.ts pattern)

key-files:
  created:
    - web/src/lib/agent-events/types.ts
    - web/src/lib/agent-events/parsers.ts
    - web/src/lib/agent-events/parsers.test.ts
    - web/src/lib/agent-events/store.ts
    - web/src/lib/agent-events/store.test.ts
    - web/src/lib/agent-events/useAgentEvents.ts
    - web/src/components/agent-events/ActivityFeed.tsx
    - web/src/components/agent-events/AgentStatusPanel.tsx
    - web/src/components/agent-events/ToolCallTimeline.tsx
    - web/src/app/agent-events/page.tsx
  modified: []

key-decisions:
  - "Zustand 5 create() uses single-call form (no curry) for non-persisted stores — consistent with project; actions are stable references so useEffect deps array is safe"
  - "useAgentEvents mounted exactly once at page root — not in individual components — to prevent multiple SSE connections to EventBusEventLog"
  - "parsers.ts marked use client (will be imported by client components); types.ts has no pragma (pure types only)"
  - "Store test reset uses getInitialState() not manual setState with plain object — avoids wiping action functions (consistent with studio-store.test.ts)"
  - "tool_call type added to TOOL_TYPES set in parsers.ts (in addition to tool_result and tool_error) to handle pre-result event"

patterns-established:
  - "Pattern: SSE hook owns connection + cleanup via AbortController; store owns all state; components are pure Zustand consumers"
  - "Pattern: Zustand store reset in tests via store.getInitialState() with replace=true flag"
  - "Pattern: Parser functions return null for non-matching event types (safe to call all three parsers on every event)"

requirements-completed: [EVNT-01, EVNT-02, EVNT-03]

# Metrics
duration: 6min
completed: 2026-03-15
---

# Phase 08 Plan 02: Frontend Agent Event Consumer Layer Summary

**SSE consumer hook, Zustand store with bounded feed/timeline, and three real-time display components (ActivityFeed, AgentStatusPanel, ToolCallTimeline) connecting to /api/events/stream**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-15T10:34:14Z
- **Completed:** 2026-03-15T10:41:05Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- TDD-built parser layer: 27 vitest tests covering all parser and store behavior cases pass (GREEN)
- Zustand store caps activity feed at 200 items and tool timeline at 100 items — prevents unbounded memory growth
- SSE hook connects to /api/events/stream via fetch + readSSE(), reconnects after 3s on error, dispatches to store
- Three display components: ActivityFeed (Radix ScrollArea, color-coded by event type), AgentStatusPanel (lucide-react status icons with animate-spin for working state), ToolCallTimeline (structured rows with collapsed args)
- Test harness page at /agent-events mounts hook once, renders all three components in two-column layout
- Next.js build: zero type errors, zero server component violations

## Task Commits

Each task was committed atomically:

1. **Task 1: Types + Parsers + Store with TDD tests** - `2951688` (feat)
2. **Task 2: SSE hook + three display components + test harness page** - `4fe7841` (feat)

**Plan metadata:** (docs commit — pending)

_Note: Task 1 used TDD (RED then GREEN). Store test reset required one auto-fix (Rule 1 - Bug)._

## Files Created/Modified
- `web/src/lib/agent-events/types.ts` - 7 TypeScript types mirroring Python EventType vocabulary
- `web/src/lib/agent-events/parsers.ts` - parseActivityItem, parseAgentStatus, parseToolCall + deriveHumanSummary
- `web/src/lib/agent-events/parsers.test.ts` - 15 vitest tests covering all parser behavior cases
- `web/src/lib/agent-events/store.ts` - useAgentEventStore with FEED_MAX=200, TOOL_TIMELINE_MAX=100
- `web/src/lib/agent-events/store.test.ts` - 12 vitest tests for store actions, caps, and status keying
- `web/src/lib/agent-events/useAgentEvents.ts` - SSE hook with AbortController cleanup and 3s reconnect
- `web/src/components/agent-events/ActivityFeed.tsx` - Scrollable event list with Radix ScrollArea
- `web/src/components/agent-events/AgentStatusPanel.tsx` - Per-agent status badge grid with lucide icons
- `web/src/components/agent-events/ToolCallTimeline.tsx` - Tool call rows with duration, collapsed args, error badges
- `web/src/app/agent-events/page.tsx` - Test harness page at /agent-events

## Decisions Made
- Zustand `create<T>()` single-call form (no curry) for non-persisted store — the workflow-store.ts pattern uses `create<T>()(persist(...))` (curry for middleware), but a plain store needs just `create<T>((...) => ({...}))`.
- SSE hook mounted exactly once at page root to prevent multiple EventBusEventLog queue registrations.
- `tool_call` added to TOOL_TYPES alongside `tool_result` and `tool_error` for completeness (pre-result events).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Store test reset used plain object (no action functions)**
- **Found during:** Task 1 (store.test.ts failing)
- **Issue:** Test's `resetStore` used `setState({...plain data...}, true)` which replaced the full Zustand state including action functions with a plain data object. `getState().setConnected` returned undefined instead of a function.
- **Fix:** Changed `resetStore` to `useAgentEventStore.setState(useAgentEventStore.getInitialState(), true)` — same pattern used in studio-store.test.ts
- **Files modified:** web/src/lib/agent-events/store.test.ts
- **Verification:** All 12 store tests pass
- **Committed in:** `2951688` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Essential test infrastructure fix. No scope creep.

## Issues Encountered
- None beyond the auto-fixed store test reset issue above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All three display components ready to be integrated into Phase 9 three-panel IDE layout
- useAgentEvents hook must be mounted exactly once at the dashboard layout root (not in individual components)
- /agent-events test harness page can be used for visual verification during Phase 9 integration

---
*Phase: 08-agent-event-vocabulary*
*Completed: 2026-03-15*
