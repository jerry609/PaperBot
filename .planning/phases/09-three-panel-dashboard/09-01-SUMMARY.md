---
phase: 09-three-panel-dashboard
plan: "01"
subsystem: ui
tags: [zustand, typescript, python, sse, file-tracking, tdd, agent-events]

# Dependency graph
requires:
  - phase: 08-agent-event-vocabulary
    provides: "AgentEventEnvelopeRaw, useAgentEventStore, parsers.ts, useAgentEvents.ts"

provides:
  - "FileTouchedEntry and FileChangeStatus types (types.ts)"
  - "parseFileTouched() parser for file_change and write_file tool_result events"
  - "Zustand store file tracking: filesTouched Record, addFileTouched (dedup + 20-run eviction)"
  - "selectedRunId and selectedFile state + setters in useAgentEventStore"
  - "SSE hook wired to dispatch parseFileTouched results to store"
  - "EventType.FILE_CHANGE constant in Python message_schema.py"

affects:
  - 09-02-three-panel-dashboard
  - future-file-list-panel
  - future-inline-diff-panel

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "parseFileTouched handles both explicit file_change events and write_file tool_result fallback"
    - "Store 20-run eviction: Object.keys(updated)[0] deleted when keys > 20"
    - "Path dedup within run_id: some((e) => e.path === entry.path) guard before push"
    - "TDD: tests written RED before implementation (12 new tests: 6 parser + 6 store)"

key-files:
  created: []
  modified:
    - web/src/lib/agent-events/types.ts
    - web/src/lib/agent-events/parsers.ts
    - web/src/lib/agent-events/parsers.test.ts
    - web/src/lib/agent-events/store.ts
    - web/src/lib/agent-events/store.test.ts
    - web/src/lib/agent-events/useAgentEvents.ts
    - src/paperbot/application/collaboration/message_schema.py
    - tests/unit/test_agent_events_vocab.py

key-decisions:
  - "parseFileTouched handles two event shapes: explicit file_change type and tool_result with payload.tool=='write_file' (fallback path for agents that emit write_file tool results)"
  - "Eviction uses Object.keys(updated)[0] (insertion order) — oldest run_id deleted when >20 keys"
  - "Path dedup within run_id ignores second write to same path (first-wins, preserves original metadata)"
  - "addFileTouched added to useEffect dependency array in SSE hook for React hook correctness"

patterns-established:
  - "File event parsing: check explicit type first, then tool_result fallback — null if no run_id/ts/path"
  - "Store bounded collections: use Object.keys length check + delete first key for FIFO eviction"

requirements-completed: [DASH-01, FILE-01, FILE-02]

# Metrics
duration: 3min
completed: 2026-03-15
---

# Phase 09 Plan 01: Three-Panel Dashboard File Tracking Data Layer Summary

**FileTouchedEntry type + parseFileTouched parser + Zustand store file tracking (dedup, 20-run eviction) + EventType.FILE_CHANGE — full data contract for the file list and diff panels**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-15T03:14:59Z
- **Completed:** 2026-03-15T03:17:57Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Added `FileTouchedEntry` and `FileChangeStatus` TypeScript types defining the file change data contract
- Implemented `parseFileTouched()` handling both explicit `file_change` events and `write_file` tool_result fallback
- Extended Zustand store with `filesTouched` (dedup + bounded 20-run eviction), `selectedRunId`, and `selectedFile`
- Added `EventType.FILE_CHANGE = "file_change"` to Python message_schema.py EventType class
- Wired SSE hook to dispatch `parseFileTouched()` results alongside existing parsers
- 12 new TDD tests (6 parser + 6 store) added — all 47 tests pass (Python 8, vitest 39)

## Task Commits

Each task was committed atomically:

1. **Task 1: FileTouchedEntry type, parseFileTouched parser, store file tracking (TDD RED + GREEN)** - `09a1e37` (feat)
2. **Task 2: EventType.FILE_CHANGE in Python, SSE hook extension, backend test** - `dd08497` (feat)

_Note: TDD task 1 had RED (tests written first, all failing) then GREEN (implementation, all passing) in a single commit per plan instructions._

## Files Created/Modified
- `web/src/lib/agent-events/types.ts` - Added `FileChangeStatus` and `FileTouchedEntry` types
- `web/src/lib/agent-events/parsers.ts` - Added `parseFileTouched()` function and `FILE_CHANGE_TYPES` set
- `web/src/lib/agent-events/parsers.test.ts` - Added 6 tests for `parseFileTouched`
- `web/src/lib/agent-events/store.ts` - Extended with `filesTouched`, `addFileTouched`, `selectedRunId`, `setSelectedRunId`, `selectedFile`, `setSelectedFile`
- `web/src/lib/agent-events/store.test.ts` - Added 6 tests for file tracking store state
- `web/src/lib/agent-events/useAgentEvents.ts` - Imported `parseFileTouched`, destructured `addFileTouched`, added dispatch in for-await loop and dependency array
- `src/paperbot/application/collaboration/message_schema.py` - Added `FILE_CHANGE: str = "file_change"` to `EventType` class
- `tests/unit/test_agent_events_vocab.py` - Added `test_file_change_event_type()` test

## Decisions Made
- `parseFileTouched` handles two event shapes: explicit `file_change` type and `tool_result` with `payload.tool=='write_file'` (fallback path for agents that emit write_file tool results)
- Eviction uses `Object.keys(updated)[0]` (insertion order) — oldest run_id deleted when >20 keys
- Path dedup within run_id ignores second write to same path (first-wins, preserves original metadata)
- `addFileTouched` added to `useEffect` dependency array in SSE hook for React hook correctness

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All data contracts are in place for Plan 02 (UI components): `FileTouchedEntry` type, `filesTouched` store field, `selectedRunId`, `selectedFile`
- Plan 02 can import these types and read from `useAgentEventStore` to render the file list panel and inline diff panel
- No blockers

---
*Phase: 09-three-panel-dashboard*
*Completed: 2026-03-15*
