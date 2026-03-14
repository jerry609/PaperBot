---
phase: 07-eventbus-sse-foundation
plan: "01"
subsystem: infra
tags: [asyncio, event-bus, sse, fan-out, ring-buffer, backpressure, event-log]

# Dependency graph
requires:
  - phase: existing-event-log
    provides: "EventLogPort protocol, CompositeEventLog, AgentEventEnvelope"
provides:
  - "EventBusEventLog — in-process asyncio fan-out ring buffer backend"
  - "subscribe()/unsubscribe() API for SSE client queue management"
  - "Ring buffer catch-up burst on connect (configurable maxlen=200)"
  - "Drop-oldest backpressure — producer never blocks"
  - "Unit test suite (5 tests) and integration stubs (2 skipped)"
affects:
  - 07-02-sse-endpoint
  - 07-03-frontend-sse-client

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "fan-out via asyncio.Queue.put_nowait() — sync append(), no await in hot path"
    - "drop-oldest backpressure: get_nowait() then put_nowait() on full queue"
    - "ring buffer catch-up: pre-load new subscriber queue from deque snapshot"
    - "list(self._queues) snapshot in _fan_out() guards against concurrent unsubscribe"

key-files:
  created:
    - src/paperbot/infrastructure/event_log/event_bus_event_log.py
    - tests/unit/test_event_bus_event_log.py
    - tests/integration/test_events_sse_endpoint.py
  modified: []

key-decisions:
  - "collections.deque(maxlen=200) ring buffer — configurable via ring_buffer_size param"
  - "asyncio.Queue(maxsize=256) per subscriber — configurable via client_queue_size param"
  - "drop-oldest backpressure: evict oldest via get_nowait() then insert newest — never block producer"
  - "AgentEventEnvelope serialized once via .to_dict() in append(); fan-out distributes dict"
  - "stream() returns iter(()) — bus does not support run_id historical replay"
  - "integration test stubs remain skipped until Plan 07-02 wires the SSE endpoint"

patterns-established:
  - "EventBusEventLog plugs into CompositeEventLog as a side-effect backend — no changes to existing event log"
  - "TDD RED-GREEN: test file committed before implementation file"

requirements-completed: [EVNT-04]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 7 Plan 01: EventBusEventLog Fan-out Ring Buffer Summary

**asyncio fan-out ring buffer (EventBusEventLog) with drop-oldest backpressure and ring buffer catch-up burst for SSE delivery, implementing EVNT-04 without any new dependencies**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T06:37:58Z
- **Completed:** 2026-03-14T06:40:25Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- EventBusEventLog implements EventLogPort using stdlib only (asyncio, collections)
- Drop-oldest backpressure via get_nowait()+put_nowait() keeps producer non-blocking in all cases
- subscribe() pre-loads new queue with ring buffer contents for immediate catch-up on SSE connect
- TDD: RED state (ImportError) confirmed before implementation; GREEN (5/5 passing) after
- Integration test stubs created and skipped pending Plan 07-02 endpoint wiring

## Task Commits

Each task was committed atomically:

1. **Task 1: Write Wave-0 test scaffolds (RED)** - `f5ffe62` (test)
2. **Task 2: Implement EventBusEventLog (GREEN)** - `b756693` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified
- `src/paperbot/infrastructure/event_log/event_bus_event_log.py` - EventBusEventLog: subscribe/unsubscribe/append/stream/close + _fan_out/_put_nowait_drop_oldest
- `tests/unit/test_event_bus_event_log.py` - 5 unit tests: fan-out, ring buffer catch-up, backpressure, unsubscribe cleanup, composite wiring
- `tests/integration/test_events_sse_endpoint.py` - 2 integration stubs (skipped): SSE delivery within 1s, heartbeat on idle

## Decisions Made
- Used `collections.deque(maxlen=200)` for the ring buffer — oldest auto-evicted on overflow, configurable
- Used `asyncio.Queue(maxsize=256)` per subscriber with drop-oldest (not drop-newest, not block) backpressure
- Serialized `AgentEventEnvelope` exactly once in `append()` via `.to_dict()`, then distributed the dict — avoids repeated serialization in `_fan_out()`
- `stream()` returns `iter(())` — the bus is a live delivery channel, not a historical store
- `list(self._queues)` snapshot in `_fan_out()` to guard against concurrent `unsubscribe()` in same event-loop tick

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. All 5 unit tests passed on first run after implementation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- EventBusEventLog is complete and correct; ready for Plan 07-02 to wire it into the FastAPI SSE endpoint
- subscribe()/unsubscribe() API is stable — Plan 07-02 will call these from the async SSE route handler
- CompositeEventLog integration confirmed working (test_composite_includes_bus passes)
- Integration test stubs in tests/integration/test_events_sse_endpoint.py will be fleshed out in Plan 07-02

---
*Phase: 07-eventbus-sse-foundation*
*Completed: 2026-03-14*

## Self-Check: PASSED

- FOUND: src/paperbot/infrastructure/event_log/event_bus_event_log.py
- FOUND: tests/unit/test_event_bus_event_log.py
- FOUND: tests/integration/test_events_sse_endpoint.py
- FOUND: .planning/phases/07-eventbus-sse-foundation/07-01-SUMMARY.md
- FOUND: commit f5ffe62 (test RED scaffolds)
- FOUND: commit b756693 (feat GREEN implementation)
- All 5 unit tests passing
