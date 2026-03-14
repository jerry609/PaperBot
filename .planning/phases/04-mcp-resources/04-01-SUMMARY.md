---
phase: 04-mcp-resources
plan: 01
subsystem: api
tags: [mcp, fastmcp, resources, anyio, tdd, json, track, scholars, memory]

# Dependency graph
requires:
  - phase: 03-remaining-mcp-tools
    provides: MCP tool pattern (_impl + register + lazy singleton + anyio.to_thread.run_sync)
provides:
  - track_metadata resource: paperbot://track/{track_id} (MCP-06)
  - track_papers resource: paperbot://track/{track_id}/papers (MCP-07)
  - track_memory resource: paperbot://track/{track_id}/memory (MCP-08)
  - scholars resource: paperbot://scholars (MCP-09)
  - Unit tests covering normal, not-found, invalid-id, empty-result, scope-filter, and FileNotFoundError behaviors
affects: [04-mcp-resources, mcp-server-registration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MCP resource registration: @mcp.resource(uri, mime_type='application/json') with _impl + register pattern"
    - "Lazy singleton _store/_service module-level var with _get_store()/_get_service() for production injection"
    - "Test injection via module-level singleton override: mod._store = FakeStore(); try/finally reset to None"
    - "anyio.to_thread.run_sync(lambda: store.method(...)) for all synchronous store/service calls"
    - "Fresh SubscriptionService instantiation per call (no caching) for always-fresh config file reads"
    - "Graceful error handling: invalid track_id and missing config return JSON error objects, not exceptions"

key-files:
  created:
    - src/paperbot/mcp/resources/__init__.py
    - src/paperbot/mcp/resources/track_metadata.py
    - src/paperbot/mcp/resources/track_papers.py
    - src/paperbot/mcp/resources/track_memory.py
    - src/paperbot/mcp/resources/scholars.py
    - tests/unit/test_mcp_track_metadata.py
    - tests/unit/test_mcp_track_papers.py
    - tests/unit/test_mcp_track_memory.py
    - tests/unit/test_mcp_scholars.py
  modified: []

key-decisions:
  - "Track resources use anyio.to_thread.run_sync() because SqlAlchemyResearchStore and SqlAlchemyMemoryStore are synchronous"
  - "track_memory passes both scope_type='track' and scope_id=str(tid) to filter correctly (not global scope)"
  - "scholars.py instantiates fresh SubscriptionService() each call (no singleton caching) for always-fresh YAML reads"
  - "_service in scholars.py is test-injection-only variable, not a lazy singleton"

patterns-established:
  - "MCP resource _impl pattern: async def _X_impl(param: str) -> str (returns JSON string)"
  - "MCP resource register pattern: def register(mcp) with @mcp.resource(uri, mime_type='application/json')"
  - "Static vs template resources: scholars uses static URI, track resources use {track_id} template"

requirements-completed: [MCP-06, MCP-07, MCP-08, MCP-09]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 04 Plan 01: MCP Resource Modules Summary

**4 read-only paperbot:// MCP resources (track metadata, papers, memory, scholars) wrapping sync stores via anyio.to_thread.run_sync with 12 unit tests**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T04:59:49Z
- **Completed:** 2026-03-14T05:02:20Z
- **Tasks:** 2 (each TDD: RED + GREEN)
- **Files modified:** 9

## Accomplishments

- `track_metadata` resource (`paperbot://track/{track_id}`) wraps `SqlAlchemyResearchStore.get_track_by_id()` returning full track JSON with name, description, keywords, venues, methods
- `track_papers` resource (`paperbot://track/{track_id}/papers`) wraps `list_track_feed(user_id="default")` returning up to 50 papers with metadata
- `track_memory` resource (`paperbot://track/{track_id}/memory`) wraps `list_memories(scope_type="track", scope_id=str(tid))` for correctly scoped memory retrieval
- `scholars` resource (`paperbot://scholars`) wraps `SubscriptionService.get_scholar_configs()` with FileNotFoundError handled gracefully
- 12 unit tests pass covering all must-have behaviors: normal return, not-found, invalid-id, empty-result, scope-filter, and missing-config-file

## Task Commits

Each task was committed atomically (TDD split into test then feat):

1. **RED: failing track tests** - `2e68e84` (test)
2. **GREEN: track resource implementations** - `877f984` (feat)
3. **RED: failing scholars test** - `5ef18e7` (test)
4. **GREEN: scholars resource implementation** - `45e3d37` (feat)

_Note: TDD tasks split into two commits each (test -> feat)_

## Files Created/Modified

- `src/paperbot/mcp/resources/__init__.py` - Package marker for resources directory
- `src/paperbot/mcp/resources/track_metadata.py` - paperbot://track/{track_id}, lazy singleton _store, anyio wrapping, invalid-id and not-found error handling
- `src/paperbot/mcp/resources/track_papers.py` - paperbot://track/{track_id}/papers, lazy singleton _store, anyio wrapping, invalid-id error handling
- `src/paperbot/mcp/resources/track_memory.py` - paperbot://track/{track_id}/memory, lazy singleton _store, anyio wrapping, scope_type="track" filtering
- `src/paperbot/mcp/resources/scholars.py` - paperbot://scholars static resource, fresh SubscriptionService per call, FileNotFoundError handling
- `tests/unit/test_mcp_track_metadata.py` - 3 tests: normal metadata, not-found error, invalid-id error
- `tests/unit/test_mcp_track_papers.py` - 3 tests: normal items, empty items, invalid-id error
- `tests/unit/test_mcp_track_memory.py` - 4 tests: normal memories, empty list, scope_type="track" verification, invalid-id error
- `tests/unit/test_mcp_scholars.py` - 2 tests: normal scholar list, FileNotFoundError handling

## Decisions Made

- Track resources use `anyio.to_thread.run_sync()` because `SqlAlchemyResearchStore` and `SqlAlchemyMemoryStore` are synchronous (same pattern as Phase 03 analyze_trends)
- `track_memory` passes both `scope_type="track"` and `scope_id=str(tid)` to filter memories correctly to the specific track (not global scope)
- `scholars.py` instantiates a fresh `SubscriptionService()` each call rather than caching -- ensures config file changes are picked up immediately; the `_service` variable is test-injection-only
- All invalid track_id inputs return JSON error objects (not exceptions) for safe agent consumption

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 4 MCP resource modules follow established _impl + register pattern
- Resources directory `src/paperbot/mcp/resources/` ready for additional resources in subsequent plans
- Each resource's `register()` function ready for wiring into `server.py` in a future registration task
- 12 unit tests green, all requirement behaviors verified

## Self-Check: PASSED

- FOUND: src/paperbot/mcp/resources/__init__.py
- FOUND: src/paperbot/mcp/resources/track_metadata.py
- FOUND: src/paperbot/mcp/resources/track_papers.py
- FOUND: src/paperbot/mcp/resources/track_memory.py
- FOUND: src/paperbot/mcp/resources/scholars.py
- FOUND: tests/unit/test_mcp_track_metadata.py
- FOUND: tests/unit/test_mcp_track_papers.py
- FOUND: tests/unit/test_mcp_track_memory.py
- FOUND: tests/unit/test_mcp_scholars.py
- FOUND commit: 2e68e84 (test RED track)
- FOUND commit: 877f984 (feat GREEN track)
- FOUND commit: 5ef18e7 (test RED scholars)
- FOUND commit: 45e3d37 (feat GREEN scholars)
- All 12 tests: PASSED

---
*Phase: 04-mcp-resources*
*Completed: 2026-03-14*
