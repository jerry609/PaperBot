---
phase: 02-core-paper-tools
plan: 01
subsystem: mcp
tags: [mcp, audit, event-log, paper-search, fastmcp]

# Dependency graph
requires: []
provides:
  - "Shared log_tool_call() audit helper for MCP tool event logging"
  - "paper_search MCP tool wrapping PaperSearchService"
  - "Tool registration pattern (register(mcp) function per tool module)"
  - "MCP server.py with FastMCP instance and tool registration"
affects: [02-core-paper-tools]

# Tech tracking
tech-stack:
  added: []
  patterns: [register(mcp) tool pattern, module-level lazy singleton for services, graceful degradation on missing DI dependencies]

key-files:
  created:
    - src/paperbot/mcp/__init__.py
    - src/paperbot/mcp/server.py
    - src/paperbot/mcp/tools/__init__.py
    - src/paperbot/mcp/tools/_audit.py
    - src/paperbot/mcp/tools/paper_search.py
    - tests/unit/test_mcp_audit.py
    - tests/unit/test_mcp_paper_search.py
    - tests/unit/test_mcp_bootstrap.py
  modified: []

key-decisions:
  - "Used try/except ImportError for FastMCP import to handle Python 3.9 where mcp package is unavailable"
  - "Exposed _paper_search_impl() as module-level function for direct test invocation without FastMCP dependency"
  - "Created bootstrap test to verify server module imports and tool registration functions exist"

patterns-established:
  - "register(mcp) pattern: each tool module exports a register() function that receives the FastMCP instance, avoiding circular imports"
  - "Module-level _service lazy singleton: tools use a global _service variable that tests can override directly"
  - "Graceful degradation: _get_event_log() catches all exceptions when resolving EventLogPort, never blocks tool execution"

requirements-completed: [R6.1, R6.2, R2.1]

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 02 Plan 01: Audit Helper and Paper Search Tool Summary

**Shared MCP audit helper (log_tool_call) with event logging and paper_search tool wrapping PaperSearchService, validated with 13 TDD unit tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-14T02:21:45Z
- **Completed:** 2026-03-14T02:25:32Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Created log_tool_call() audit helper that creates AgentEventEnvelope events with workflow="mcp", stage="tool_call" and degrades gracefully when EventLogPort is missing
- Built paper_search MCP tool that wraps PaperSearchService, converts results to dicts, and logs all calls via the audit helper
- Established the register(mcp) tool registration pattern that remaining tools will follow
- All 13 unit tests passing (7 audit, 3 paper_search, 3 bootstrap)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create audit helper and paper_search tool with tests** - `fa16e09` (feat)
2. **Task 2: Register paper_search in MCP server** - `9d912d7` (feat)

_Note: Task 1 followed TDD (RED/GREEN) - tests written first, then implementation._

## Files Created/Modified
- `src/paperbot/mcp/__init__.py` - Package marker for MCP module
- `src/paperbot/mcp/server.py` - FastMCP instance with paper_search registered
- `src/paperbot/mcp/tools/__init__.py` - Package marker for tools submodule
- `src/paperbot/mcp/tools/_audit.py` - Shared log_tool_call() helper for all tools
- `src/paperbot/mcp/tools/paper_search.py` - paper_search MCP tool wrapping PaperSearchService
- `tests/unit/test_mcp_audit.py` - 7 unit tests for audit helper
- `tests/unit/test_mcp_paper_search.py` - 3 unit tests for paper_search tool
- `tests/unit/test_mcp_bootstrap.py` - 3 unit tests for server bootstrap

## Decisions Made
- Used try/except ImportError for FastMCP import in server.py to handle Python 3.9 compatibility where the mcp package is not installable (requires Python 3.10+). The server degrades to mcp=None, but all tool modules remain fully importable and testable.
- Exposed _paper_search_impl() as a module-level async function separate from the @mcp.tool() decorated wrapper, enabling direct test invocation without requiring FastMCP.
- Created a bootstrap test to verify the server module, register functions, and audit helper are all importable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created MCP package structure from scratch**
- **Found during:** Task 1 (audit helper creation)
- **Issue:** Plan referenced Phase 1 creating src/paperbot/mcp/ directory and server.py, but neither existed
- **Fix:** Created __init__.py package markers for mcp/ and mcp/tools/ directories
- **Files modified:** src/paperbot/mcp/__init__.py, src/paperbot/mcp/tools/__init__.py
- **Verification:** Module imports succeed
- **Committed in:** fa16e09 (Task 1 commit)

**2. [Rule 3 - Blocking] Created server.py with FastMCP compatibility fallback**
- **Found during:** Task 2 (server registration)
- **Issue:** mcp package not installable on Python 3.9.7; server.py did not exist from Phase 1
- **Fix:** Created server.py with try/except ImportError for FastMCP, falling back to mcp=None
- **Files modified:** src/paperbot/mcp/server.py
- **Verification:** Server module imports cleanly; all tests pass
- **Committed in:** 9d912d7 (Task 2 commit)

**3. [Rule 3 - Blocking] Created missing bootstrap test**
- **Found during:** Task 2 (verification step references test_mcp_bootstrap.py)
- **Issue:** Plan verification references tests/unit/test_mcp_bootstrap.py but no such file existed
- **Fix:** Created bootstrap test with 3 test cases verifying module imports
- **Files modified:** tests/unit/test_mcp_bootstrap.py
- **Verification:** All 3 bootstrap tests pass
- **Committed in:** 9d912d7 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3 blocking)
**Impact on plan:** All auto-fixes necessary for execution. No scope creep -- these were missing prerequisites that should have been created by Phase 1.

## Issues Encountered
- The mcp Python package (providing FastMCP) requires Python 3.10+ and cannot be installed on this environment's Python 3.9.7. This was handled by wrapping the import in try/except so the server degrades gracefully while all tool modules remain fully functional and testable.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Audit helper and tool registration pattern are ready for plans 02 and 03 to add paper_analyze, paper_review, and scholar_track tools
- When Python is upgraded to 3.10+, install the mcp package to enable full FastMCP server functionality

---
*Phase: 02-core-paper-tools*
*Completed: 2026-03-14*

## Self-Check: PASSED

All 8 created files verified present. Both task commits (fa16e09, 9d912d7) verified in git log.
