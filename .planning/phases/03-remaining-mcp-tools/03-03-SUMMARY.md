---
phase: 03-remaining-mcp-tools
plan: 03
subsystem: mcp
tags: [mcp, fastmcp, server-registration, integration-tests, tdd]

# Dependency graph
requires:
  - phase: 03-remaining-mcp-tools/03-01
    provides: analyze_trends and check_scholar MCP tool implementations
  - phase: 03-remaining-mcp-tools/03-02
    provides: get_research_context, save_to_memory, export_to_obsidian MCP tool implementations
provides:
  - "MCP server registering all 9 tools via FastMCP (paper_search, paper_judge, paper_summarize, relevance, analyze_trends, check_scholar, get_research_context, save_to_memory, export_to_obsidian)"
  - "Integration test suite with 31 tests covering discovery, schema, invocation, and audit logging for all 9 tools"
affects: [phase-04, phase-05, phase-06, mcp-client-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "All 9 MCP tools share consistent lazy singleton + _impl + register + log_tool_call pattern"
    - "Integration tests inject module-level singletons directly (not DI container) for tool isolation"
    - "4-class test structure: Listing, Schemas, Invocation, EventLogging"

key-files:
  created:
    - tests/integration/test_mcp_tool_calls.py
  modified:
    - src/paperbot/mcp/server.py
    - tests/integration/test_mcp_tool_calls.py

key-decisions:
  - "No architectural changes needed: 5 new tools follow exactly the same registration pattern as existing 4"
  - "Integration tests inject fakes via module-level _var singletons (same pattern as unit tests), not via DI container"
  - "Consistent structure test validates all 9 tools emit workflow='mcp', stage='tool_call', agent_name='paperbot-mcp' with duration_ms"

patterns-established:
  - "MCP server registration: single try/import block, all tools imported and registered in sequence"
  - "Integration test 4-class pattern: TestMCPToolListing, TestMCPToolSchemas, TestMCPToolInvocation, TestMCPToolEventLogging"

requirements-completed: [MCP-01, MCP-02, MCP-03, MCP-04, MCP-05]

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 03 Plan 03: MCP Server Registration + Integration Tests Summary

**9-tool FastMCP server with 31 integration tests covering discovery, schema validation, invocation, and audit logging — completing Phase 3's MCP tool surface**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T04:27:05Z
- **Completed:** 2026-03-14T04:32:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Registered all 5 new tools (analyze_trends, check_scholar, get_research_context, save_to_memory, export_to_obsidian) in server.py alongside existing 4 — 9 total discoverable via MCP tools/list
- Extended integration test file from 4 tools / ~16 tests to 9 tools / 31 tests with full coverage across all 4 test classes (Listing, Schemas, Invocation, EventLogging)
- Phase 3 gate passes: 46 tests total (15 unit + 31 integration), all green

## Task Commits

Each task was committed atomically:

1. **Task 1: Register 5 new tools in server.py** - `e16b2e0` (feat)
2. **Task 2: Extend integration tests to all 9 tools** - `f1b8828` (feat)

## Files Created/Modified

- `src/paperbot/mcp/server.py` - Added 5 new import+register calls; now registers all 9 MCP tools in single FastMCP server
- `tests/integration/test_mcp_tool_calls.py` - Expanded from ~16 to 31 tests covering all 9 tools across 4 test classes; added fakes for _FakeTrendAnalyzer, _FakeS2Client, _FakeContextEngine, _FakeMemoryStore, _FakeExporter

## Decisions Made

- No architectural changes were required: the 5 new tools follow exactly the same import-and-register pattern as the original 4, making server.py extension trivial
- Integration test fakes inject directly into module-level singletons (e.g., `at_mod._analyzer = _FakeTrendAnalyzer()`) rather than via DI container, keeping the same isolation pattern as the unit tests from Plans 01 and 02
- The consistent structure test validates all 9 tools together in a single call sequence, confirming event log shape (workflow/stage/agent_name/payload/metrics) is maintained across the entire tool surface

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 9 Phase 3 MCP tools are registered, tested, and auditable
- Phase 3 is complete: Plans 01 (analyze_trends, check_scholar), 02 (get_research_context, save_to_memory, export_to_obsidian), 03 (server wiring + integration tests) all done
- Requirements MCP-01 through MCP-05 all completed
- Ready for Phase 4 (Scholar Tracking) or any follow-on MCP work

## Self-Check: PASSED

- FOUND: src/paperbot/mcp/server.py (with 9 registrations)
- FOUND: tests/integration/test_mcp_tool_calls.py (31 tests)
- FOUND commit: e16b2e0 (feat - Task 1: server registration)
- FOUND commit: f1b8828 (feat - Task 2: integration tests)
- Phase gate: 46/46 tests PASSED

---
*Phase: 03-remaining-mcp-tools*
*Completed: 2026-03-14*
