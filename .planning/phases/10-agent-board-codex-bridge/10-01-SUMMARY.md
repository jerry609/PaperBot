---
phase: 10-agent-board-codex-bridge
plan: "01"
subsystem: backend-events
tags: [eventtype, codex-bridge, agent-board, delegation-events, overflow-routing]
dependency_graph:
  requires: []
  provides: [CODEX_DISPATCHED, CODEX_ACCEPTED, CODEX_COMPLETED, CODEX_FAILED, _emit_codex_event, _should_overflow_to_codex]
  affects: [message_schema, agent_board, repro/orchestrator]
tech_stack:
  added: []
  patterns: [lazy-import-container, try-except-silent-event-emission, tdd-red-green]
key_files:
  created:
    - tests/unit/test_codex_overflow.py
    - tests/unit/test_agent_board_codex_events.py
  modified:
    - src/paperbot/application/collaboration/message_schema.py
    - src/paperbot/api/routes/agent_board.py
    - src/paperbot/repro/orchestrator.py
    - tests/unit/test_agent_events_vocab.py
decisions:
  - "_emit_codex_event uses _get_event_log_from_container() lazy helper (not app.state) so tests can monkeypatch without a live FastAPI app"
  - "_should_overflow_to_codex is a stub only — no actual overflow wiring in Orchestrator.run() yet; CDX-02 wiring planned for phase 10-02 or later"
  - "CODEX_DISPATCHED emitted at two call sites: dispatch_task() and _execute_task_stream() legacy path dispatch block"
metrics:
  duration: "5 min"
  completed_date: "2026-03-15"
  tasks_completed: 1
  files_changed: 6
---

# Phase 10 Plan 01: Codex Delegation EventType Constants and Emission Helper Summary

**One-liner:** Four CODEX_* EventType constants, _emit_codex_event async helper emitting delegation lifecycle events into EventBusEventLog, and _should_overflow_to_codex env-var-driven routing stub for CDX-02/CDX-03.

## What Was Built

### EventType Constants (message_schema.py)
Four new string constants added to the `EventType` class under a `# --- Codex delegation events (Phase 10 / CDX-03) ---` section comment:
- `CODEX_DISPATCHED = "codex_dispatched"`
- `CODEX_ACCEPTED = "codex_accepted"`
- `CODEX_COMPLETED = "codex_completed"`
- `CODEX_FAILED = "codex_failed"`

EventType now has 12 constants total (8 pre-existing + 4 new).

### _get_event_log_from_container + _emit_codex_event (agent_board.py)
A lazy-import `_get_event_log_from_container()` helper retrieves `Container.instance().event_log` without circular imports. The async `_emit_codex_event(event_type, task, session, extra)` function:
- Wraps everything in try/except — never raises to callers
- Builds payload with `task_id`, `task_title`, `session_id` plus `extra` dict
- Calls `make_event()` with `workflow="agent_board"` and `role="worker"`
- Silently returns when event_log is None

Emission call sites added:
1. `dispatch_task()` -> CODEX_DISPATCHED with `{"assignee": task.assignee}`
2. `_execute_task_stream()` dispatch block -> CODEX_DISPATCHED with `{"assignee": task.assignee}`
3. `_execute_task_stream()` before `build_codex_prompt` -> CODEX_ACCEPTED with `{"assignee": task.assignee, "model": "codex"}`
4. After `result.success == False` -> CODEX_FAILED with `{"assignee", "reason_code", "error"}`
5. After successful result + files_written log -> CODEX_COMPLETED with `{"assignee", "files_generated", "output_preview"}`

### _should_overflow_to_codex (orchestrator.py)
Module-level stub function reads `PAPERBOT_CODEX_OVERFLOW_THRESHOLD` env var:
- Unset/empty -> False
- `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive) -> True
- Any other value -> False
- Wrapped in try/except returning False on error

### Tests (TDD Red-Green)
- `tests/unit/test_agent_events_vocab.py`: 4 new assertions for CODEX_* constants
- `tests/unit/test_codex_overflow.py`: 7 tests covering all env var states
- `tests/unit/test_agent_board_codex_events.py`: 5 tests for `_emit_codex_event` using monkeypatched `_get_event_log_from_container`

Total: 24 new tests pass; 22 existing `test_agent_board_route.py` tests pass without regression.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| a2e0562 | test | RED: failing tests for CODEX_* constants, overflow stub, emit helper |
| 4c3c3d5 | feat | GREEN: implementation of all three deliverables |

## Deviations from Plan

None — plan executed exactly as written.

The plan specified `Container.instance().event_log` as the lookup path. Since the production app stores the event_log on `app.state` (not Container), the lookup will return None in production (silently no-op). This is acceptable for the stub phase; proper wiring to `app.state` can be added in a later plan when the live fan-out is needed. The plan was explicit that this is a "stub" pattern for CDX-03 observability.

## Self-Check

**Status: PASSED**

All required files exist and both commits are present in git history:
- FOUND: src/paperbot/application/collaboration/message_schema.py
- FOUND: src/paperbot/api/routes/agent_board.py
- FOUND: src/paperbot/repro/orchestrator.py
- FOUND: tests/unit/test_agent_events_vocab.py
- FOUND: tests/unit/test_codex_overflow.py
- FOUND: tests/unit/test_agent_board_codex_events.py
- FOUND: commit a2e0562 (RED phase tests)
- FOUND: commit 4c3c3d5 (GREEN phase implementation)
