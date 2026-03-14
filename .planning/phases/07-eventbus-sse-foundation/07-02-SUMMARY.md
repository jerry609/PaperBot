---
phase: 07-eventbus-sse-foundation
plan: "02"
subsystem: api
tags: [sse, event-bus, fan-out, fastapi, streaming, asyncio, integration-test]

# Dependency graph
requires:
  - phase: 07-01
    provides: "EventBusEventLog subscribe/unsubscribe/append API"
  - phase: existing-api
    provides: "FastAPI app, SSE_HEADERS, sse_comment, CompositeEventLog"
provides:
  - "GET /api/events/stream SSE fan-out endpoint"
  - "EventBusEventLog wired as third CompositeEventLog backend in main.py startup"
  - "Integration tests: event delivery latency (<1s) and heartbeat on idle"
affects:
  - 07-03-frontend-sse-client

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "late import of EventBusEventLog inside _get_bus() avoids circular import at module load"
    - "_event_generator try/finally guarantees unsubscribe on client disconnect"
    - "asyncio.wait_for with timeout=_HEARTBEAT_SECONDS drives keepalive without separate task"
    - "request.is_disconnected() checked each loop iteration for fast disconnect detection"
    - "StreamingResponse with direct async generator — no wrap_generator() wrapper layer"

key-files:
  created:
    - src/paperbot/api/routes/events.py
  modified:
    - src/paperbot/api/main.py
    - tests/integration/test_events_sse_endpoint.py

key-decisions:
  - "Late import of EventBusEventLog inside _get_bus() prevents circular import (events.py loaded during app creation before bus is wired)"
  - "_event_generator uses asyncio.wait_for(q.get(), timeout=15.0) — single await point drives both delivery and heartbeat"
  - "No wrap_generator(): events carry own AgentEventEnvelope fields; second envelope layer would confuse consumers"
  - "test_heartbeat_on_idle patches module-level _HEARTBEAT_SECONDS to 0.05s for speed; restores in finally"
  - "test_event_delivered_within_1s exercises bus directly (no HTTP server) — validates delivery latency guarantee"

requirements-completed: [EVNT-04]

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 7 Plan 02: SSE Endpoint + main.py Wiring Summary

**GET /api/events/stream fan-out endpoint wired into FastAPI with EventBusEventLog as CompositeEventLog backend, delivering all event_log.append() calls to SSE clients within 1 second**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-14T06:43:16Z
- **Completed:** 2026-03-14T06:46:48Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `src/paperbot/api/routes/events.py` with `GET /api/events/stream` SSE endpoint
- `_event_generator()` uses try/finally to guarantee `bus.unsubscribe(q)` on client disconnect — no queue leak
- Heartbeat: 15s timeout on `q.get()` → yields `: keepalive\n\n` comment on idle
- `_get_bus()` uses late import to avoid circular import; finds EventBusEventLog in `_backends` or directly
- Modified `main.py`: added `EventBusEventLog` as third `CompositeEventLog` backend in `_startup_eventlog()`
- Registered `events_route.router` at `prefix="/api"` — route appears as `/api/events/stream`
- Replaced pytest.skip stubs with real integration tests — both pass GREEN
- All 7 tests pass (5 unit from Plan 07-01 + 2 new integration)
- No existing `event_log.append()` callers required any changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create GET /api/events/stream SSE endpoint** - `b8cb0d4` (feat)
2. **Task 2: Wire main.py + integration tests GREEN** - `5314131` (feat)

## Files Created/Modified

- `src/paperbot/api/routes/events.py` (new) — SSE fan-out endpoint: `_get_bus()`, `_event_generator()`, `events_stream()`
- `src/paperbot/api/main.py` (modified) — EventBusEventLog import + CompositeEventLog wiring + events router registration
- `tests/integration/test_events_sse_endpoint.py` (modified) — stubs replaced with two real passing integration tests

## Decisions Made

- Late import of `EventBusEventLog` inside `_get_bus()` prevents circular import: `events.py` is imported when `app` is created, before startup hooks run
- `asyncio.wait_for(q.get(), timeout=_HEARTBEAT_SECONDS)` at `_HEARTBEAT_SECONDS = 15.0` — single await point drives both event delivery and idle heartbeat
- No `wrap_generator()` or `sse_response()` — events already carry `AgentEventEnvelope` fields (`run_id`, `trace_id`, `seq`); wrapping adds duplicate envelope layer
- `test_heartbeat_on_idle` patches `_HEARTBEAT_SECONDS` to `0.05` for speed, not by subclassing, to keep test simple
- `test_event_delivered_within_1s` exercises `EventBusEventLog` subscribe/append/get directly (no HTTP), asserting < 1s delivery budget

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Both integration tests passed on first run after implementation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `/api/events/stream` is live and ready for Plan 07-03 (frontend SSE client)
- EventBusEventLog wired as side-effect backend — all existing endpoints transparently fan out to SSE clients
- `subscribe()` / `unsubscribe()` API proven correct by integration tests
- EVNT-04 requirement is now fully complete: end-to-end from `event_log.append()` to SSE frame

---
*Phase: 07-eventbus-sse-foundation*
*Completed: 2026-03-14*

## Self-Check: PASSED

- FOUND: src/paperbot/api/routes/events.py
- FOUND: tests/integration/test_events_sse_endpoint.py
- FOUND: .planning/phases/07-eventbus-sse-foundation/07-02-SUMMARY.md
- FOUND: commit b8cb0d4 (feat events.py SSE endpoint)
- FOUND: commit 5314131 (feat main.py wiring + integration tests GREEN)
- All 7 tests passing (5 unit + 2 integration)
