---
phase: 08-agent-event-vocabulary
plan: "01"
subsystem: api
tags: [event-vocabulary, agent-events, mcp, sse, constants, message-schema]

# Dependency graph
requires:
  - phase: 07-eventbus-sse-foundation
    provides: AgentEventEnvelope, make_event(), EventBusEventLog, SSE streaming foundation
provides:
  - EventType constants class in message_schema.py (14 named constants)
  - make_lifecycle_event() helper in agent_events.py
  - make_tool_call_event() helper in agent_events.py
  - _audit.py migrated to EventType constants (no raw string literals)
affects:
  - 08-02 (frontend event type consumers need these constants)
  - 09-agent-board (dashboard will consume lifecycle event types)
  - any caller of _audit.py or log_tool_call()

# Tech tracking
tech-stack:
  added: []
  patterns:
    - EventType plain class (not enum) pattern for string constants — avoids .value unwrapping
    - make_lifecycle_event / make_tool_call_event wrappers delegate to make_event() for envelope consistency
    - TDD RED/GREEN cycle for vocabulary contract

key-files:
  created:
    - src/paperbot/application/collaboration/agent_events.py
    - tests/unit/test_agent_events_vocab.py
  modified:
    - src/paperbot/application/collaboration/message_schema.py
    - src/paperbot/mcp/tools/_audit.py

key-decisions:
  - "EventType is a plain class with string annotations, not an enum — constants usable as str anywhere without .value"
  - "make_tool_call_event auto-generates run_id/trace_id if not provided (optional kwargs) — callers can omit for standalone tool logging"
  - "agent_events.py imports only from message_schema — no circular dependency risk"
  - "_audit.py migration: only the type= argument changed; all other logic and sanitization unchanged"

patterns-established:
  - "EventType.TOOL_ERROR / EventType.TOOL_RESULT: use constants, never raw strings, in type= fields"
  - "make_lifecycle_event: single call site for all agent lifecycle events"
  - "make_tool_call_event: single call site for all MCP tool audit events"

requirements-completed: [EVNT-01, EVNT-02, EVNT-03]

# Metrics
duration: 2min
completed: 2026-03-15
---

# Phase 8 Plan 01: Agent Event Vocabulary Summary

**EventType constants class with 14 named constants, make_lifecycle_event/make_tool_call_event helpers in agent_events.py, and _audit.py migrated from raw "error"/"tool_result" strings to EventType.TOOL_ERROR/TOOL_RESULT**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-15T02:33:58Z
- **Completed:** 2026-03-15T02:35:40Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Added `EventType` plain-class constants to `message_schema.py` (4 lifecycle + 3 tool + 7 existing-type aliases)
- Created `agent_events.py` with `make_lifecycle_event()` and `make_tool_call_event()` helpers
- Migrated `_audit.py` to use `EventType.TOOL_ERROR` / `EventType.TOOL_RESULT` instead of raw strings
- Full TDD cycle: 7 failing tests committed (RED), then all 7 pass GREEN with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: TDD RED — Write test scaffold** - `bf418f6` (test)
2. **Task 2: Implement EventType, helpers, migrate _audit.py** - `db59a2d` (feat)

_Note: TDD tasks have separate RED and GREEN commits_

## Files Created/Modified
- `src/paperbot/application/collaboration/message_schema.py` - Added EventType class after make_event()
- `src/paperbot/application/collaboration/agent_events.py` - New module with make_lifecycle_event and make_tool_call_event helpers
- `src/paperbot/mcp/tools/_audit.py` - Added EventType import; replaced raw type strings with constants
- `tests/unit/test_agent_events_vocab.py` - 7 unit tests covering constants, lifecycle events, tool call events, and _audit.py migration

## Decisions Made
- **EventType as plain class, not Enum:** Constants are `str` annotations directly — callers use `EventType.AGENT_STARTED` anywhere a `str` is expected without `.value` unwrapping. Consistent with project pattern in message_schema (all types were raw strings before).
- **make_tool_call_event optional run_id/trace_id:** Marked `Optional[str] = None` and auto-generated when absent, matching the pattern in `log_tool_call()` — callers that only have a tool name can still produce a valid envelope.
- **No changes to _audit.py sanitization logic:** Only the `type=` assignment line changed — all existing argument redaction, truncation, and error handling left intact to avoid regression risk.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- EventType constants and helpers are ready for Phase 8 Plan 02 (frontend event type consumers)
- Phase 9 (Agent Board dashboard) can reference EventType constants for lifecycle event filtering
- All existing Phase 7 SSE integration tests still pass (2/2 green)

---
*Phase: 08-agent-event-vocabulary*
*Completed: 2026-03-15*
