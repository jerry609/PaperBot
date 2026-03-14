---
phase: 03-remaining-mcp-tools
plan: 01
subsystem: api
tags: [mcp, fastmcp, trend-analyzer, semantic-scholar, anyio, tdd]

# Dependency graph
requires:
  - phase: 02-core-paper-tools
    provides: MCP tool pattern (_impl + register + lazy singleton + log_tool_call audit)
provides:
  - analyze_trends MCP tool wrapping sync TrendAnalyzer via anyio.to_thread.run_sync()
  - check_scholar MCP tool wrapping async SemanticScholarClient
  - Unit tests covering normal, degraded, and audit paths for both tools
affects: [03-remaining-mcp-tools, mcp-server-registration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy singleton with module-level _var + _get_var() function for MCP tool dependencies"
    - "anyio.to_thread.run_sync() for wrapping synchronous services in async MCP tools"
    - "Degraded detection: empty string from LLM triggers degraded=True response"
    - "Direct async client usage in MCP tools (no thread wrapping needed for async services)"

key-files:
  created:
    - src/paperbot/mcp/tools/analyze_trends.py
    - src/paperbot/mcp/tools/check_scholar.py
    - tests/unit/test_mcp_analyze_trends.py
    - tests/unit/test_mcp_check_scholar.py
  modified: []

key-decisions:
  - "analyze_trends uses anyio.to_thread.run_sync() because TrendAnalyzer.analyze() is synchronous"
  - "check_scholar awaits SemanticScholarClient directly (no thread wrapping needed - already async)"
  - "Degraded detection for analyze_trends: empty/whitespace-only string signals LLM unavailability"
  - "check_scholar returns degraded=True with candidates=[] on empty author search (not exception)"

patterns-established:
  - "Sync LLM service wrapping: anyio.to_thread.run_sync(lambda: service.method(...))"
  - "Async client wrapping: direct await, no thread overhead"

requirements-completed: [MCP-01, MCP-02]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 03 Plan 01: analyze_trends and check_scholar MCP Tools Summary

**Two MCP tools wrapping TrendAnalyzer (sync/anyio) and SemanticScholarClient (async) with degraded-mode detection and audit logging**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T04:23:23Z
- **Completed:** 2026-03-14T04:25:05Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments

- `analyze_trends` MCP tool wraps synchronous `TrendAnalyzer.analyze()` with `anyio.to_thread.run_sync()`, detecting degraded state when LLM returns empty string
- `check_scholar` MCP tool awaits async `SemanticScholarClient.search_authors()` and `get_author_papers()`, returning degraded result when scholar not found
- 6 unit tests covering normal, degraded, and audit paths for both tools - all passing

## Task Commits

Each task was committed atomically:

1. **RED phase: failing tests** - `ada2c56` (test)
2. **GREEN phase: both tool implementations** - `214a027` (feat)

_Note: TDD task split into two commits (test -> feat)_

## Files Created/Modified

- `src/paperbot/mcp/tools/analyze_trends.py` - analyze_trends MCP tool: lazy singleton _analyzer, anyio thread wrapping, degraded detection, log_tool_call audit
- `src/paperbot/mcp/tools/check_scholar.py` - check_scholar MCP tool: lazy singleton _client, async S2 client calls, not-found degraded path, log_tool_call audit
- `tests/unit/test_mcp_analyze_trends.py` - 3 tests: normal result, degraded (empty LLM), audit log
- `tests/unit/test_mcp_check_scholar.py` - 3 tests: normal result, degraded (scholar not found), audit log

## Decisions Made

- `analyze_trends` uses `anyio.to_thread.run_sync()` because `TrendAnalyzer.analyze()` is a synchronous method
- `check_scholar` awaits `SemanticScholarClient` directly since it is already async (no thread overhead)
- Degraded detection for `analyze_trends` uses empty/whitespace-only string check (matches TrendAnalyzer's behavior when LLM is unavailable)
- `check_scholar` returns `degraded=True` with `candidates=[]` on empty author search rather than raising an exception (graceful degradation)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Two MCP tool patterns now established: sync-wrapped (anyio) and direct-async
- Remaining tools in Phase 03 can follow the same two patterns
- Both tools ready for registration in MCP server `__init__.py`

## Self-Check: PASSED

- FOUND: src/paperbot/mcp/tools/analyze_trends.py
- FOUND: src/paperbot/mcp/tools/check_scholar.py
- FOUND: tests/unit/test_mcp_analyze_trends.py
- FOUND: tests/unit/test_mcp_check_scholar.py
- FOUND: .planning/phases/03-remaining-mcp-tools/03-01-SUMMARY.md
- FOUND commit: ada2c56 (test - RED phase)
- FOUND commit: 214a027 (feat - GREEN phase)
- All 6 tests: PASSED

---
*Phase: 03-remaining-mcp-tools*
*Completed: 2026-03-14*
