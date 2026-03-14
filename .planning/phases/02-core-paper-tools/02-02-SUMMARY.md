---
phase: 02-core-paper-tools
plan: 02
subsystem: mcp
tags: [mcp, paper-judge, paper-summarize, relevance, llm, anyio, degraded-mode]

# Dependency graph
requires:
  - "Shared log_tool_call() audit helper for MCP tool event logging (02-01)"
  - "Tool registration pattern register(mcp) from 02-01"
  - "MCP server.py with FastMCP instance from 02-01"
provides:
  - "paper_judge MCP tool wrapping PaperJudge with degraded LLM detection"
  - "paper_summarize MCP tool wrapping PaperSummarizer with empty output detection"
  - "relevance_assess MCP tool wrapping RelevanceAssessor with fallback scoring detection"
  - "All 4 tools registered on MCP server (paper_search + these 3)"
affects: [02-core-paper-tools]

# Tech tracking
tech-stack:
  added: [anyio]
  patterns: [anyio.to_thread.run_sync() for sync-to-async wrapping, degraded output detection per tool, module-level _impl function for testability]

key-files:
  created:
    - src/paperbot/mcp/tools/paper_judge.py
    - src/paperbot/mcp/tools/paper_summarize.py
    - src/paperbot/mcp/tools/relevance.py
    - tests/unit/test_mcp_paper_judge.py
    - tests/unit/test_mcp_paper_summarize.py
    - tests/unit/test_mcp_relevance.py
  modified:
    - src/paperbot/mcp/server.py

key-decisions:
  - "Used module-level _impl async functions for all three tools, matching paper_search pattern from Plan 01"
  - "Degraded detection is tool-specific: judge checks judge_model empty, summarize checks empty output, relevance checks Fallback in reason"
  - "All sync service calls wrapped with anyio.to_thread.run_sync(lambda: ...) pattern"

patterns-established:
  - "Degraded output detection: each tool checks for specific indicators of LLM unavailability and annotates result with degraded=True"
  - "Module-level _impl function: async implementation exposed at module level for direct test invocation"
  - "Consistent error handling: try/except wrapping with log_tool_call in both success and error paths"

requirements-completed: [R2.2, R2.3, R2.4]

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 02 Plan 02: LLM-based Paper Tools Summary

**Three LLM-based MCP tools (paper_judge, paper_summarize, relevance_assess) with sync-to-async wrapping via anyio, degraded output detection, and 10 TDD unit tests**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T02:31:46Z
- **Completed:** 2026-03-14T02:40:10Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Built paper_judge tool wrapping PaperJudge.judge_single() with abstract-to-snippet parameter mapping and degraded detection when judge_model is empty
- Built paper_summarize tool wrapping PaperSummarizer.summarize_item() with degraded detection on empty LLM output
- Built relevance_assess tool wrapping RelevanceAssessor.assess() with fallback scoring detection when reason contains "Fallback"
- All three tools use anyio.to_thread.run_sync() for sync service calls and log via log_tool_call()
- MCP server now registers all 4 tools; 23 total MCP tests passing (7 audit + 3 search + 4 judge + 3 summarize + 3 relevance + 3 bootstrap)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement paper_judge, paper_summarize, and relevance_assess tools with tests** - `3bff474` (test, RED) + `c2ec8d6` (feat, GREEN)
2. **Task 2: Register all three tools in MCP server** - `b962259` (feat)

_Note: Task 1 followed TDD (RED/GREEN) -- tests written first (10 failing), then implementation (10 passing)._

## Files Created/Modified
- `src/paperbot/mcp/tools/paper_judge.py` - paper_judge MCP tool wrapping PaperJudge with degraded LLM detection
- `src/paperbot/mcp/tools/paper_summarize.py` - paper_summarize MCP tool wrapping PaperSummarizer with empty output detection
- `src/paperbot/mcp/tools/relevance.py` - relevance_assess MCP tool wrapping RelevanceAssessor with fallback scoring detection
- `src/paperbot/mcp/server.py` - Updated to register all 4 tools
- `tests/unit/test_mcp_paper_judge.py` - 4 unit tests for paper_judge (normal, degraded, abstract mapping, audit)
- `tests/unit/test_mcp_paper_summarize.py` - 3 unit tests for paper_summarize (normal, degraded, audit)
- `tests/unit/test_mcp_relevance.py` - 3 unit tests for relevance_assess (normal, fallback, audit)

## Decisions Made
- Used module-level _impl async functions for all three tools, following the established paper_search pattern from Plan 01. This enables direct test invocation without requiring FastMCP.
- Degraded detection is tool-specific: paper_judge checks if judge_model is empty string, paper_summarize checks if summary output is empty/whitespace, relevance_assess checks if reason contains "Fallback". Each approach matches how the underlying service signals LLM unavailability.
- All sync service calls use `anyio.to_thread.run_sync(lambda: ...)` pattern rather than functools.partial, as closures work better with the service method pattern.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 paper-related tools are now registered on the MCP server
- Plan 03 can add the remaining tools (scholar_track, etc.)
- When Python is upgraded to 3.10+, install the mcp package to enable full FastMCP server functionality

---
*Phase: 02-core-paper-tools*
*Completed: 2026-03-14*

## Self-Check: PASSED

All 7 created/modified files verified present. All 3 task commits (3bff474, c2ec8d6, b962259) verified in git log.
