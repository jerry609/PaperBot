---
phase: 03-remaining-mcp-tools
plan: 02
subsystem: mcp
tags: [mcp, fastmcp, context-engine, memory-store, obsidian, anyio, tdd]

# Dependency graph
requires:
  - phase: 02-core-paper-tools
    provides: "MCP tool pattern (lazy singleton, _impl, register, log_tool_call audit)"
  - phase: 03-remaining-mcp-tools/03-01
    provides: "Phase 3 TDD pattern established for analyze_trends and check_scholar"
provides:
  - "get_research_context MCP tool wrapping ContextEngine.build_context_pack() in offline mode"
  - "save_to_memory MCP tool wrapping SqlAlchemyMemoryStore.add_memories() via anyio.to_thread"
  - "export_to_obsidian MCP tool with in-memory markdown rendering via ObsidianFilesystemExporter"
  - "9 unit tests covering normal, edge case, and audit paths for all three tools"
affects: [mcp-server-registration, phase-04, phase-05, phase-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Direct async await for ContextEngine (already async-native, no anyio wrapper needed)"
    - "anyio.to_thread.run_sync() for synchronous store and exporter methods"
    - "MemoryKind validation with silent default to 'note' on invalid input"
    - "In-memory Obsidian rendering by calling _render_paper_note() + _yaml_frontmatter() directly"

key-files:
  created:
    - src/paperbot/mcp/tools/get_research_context.py
    - src/paperbot/mcp/tools/save_to_memory.py
    - src/paperbot/mcp/tools/export_to_obsidian.py
    - tests/unit/test_mcp_get_research_context.py
    - tests/unit/test_mcp_save_to_memory.py
    - tests/unit/test_mcp_export_to_obsidian.py
  modified: []

key-decisions:
  - "get_research_context uses ContextEngineConfig(offline=True, paper_limit=0) as default to avoid side effects during tool calls"
  - "save_to_memory defaults invalid MemoryKind values to 'note' with a logger.warning rather than raising an error"
  - "export_to_obsidian uses _render_paper_note() private method directly (documented in research as intentional) with no filesystem I/O"

patterns-established:
  - "Async-native tools: use direct await; sync-wrapped tools: use anyio.to_thread.run_sync()"
  - "Kind validation pattern: frozenset of allowed values, silent default with warning on invalid input"
  - "In-memory export pattern: call renderer method directly and return markdown string without filesystem writes"

requirements-completed: [MCP-03, MCP-04, MCP-05]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 3 Plan 2: Remaining MCP Tools Summary

**Three MCP tools added: get_research_context (async ContextEngine), save_to_memory (anyio-wrapped MemoryStore with MemoryKind validation), and export_to_obsidian (in-memory Jinja2 rendering with YAML frontmatter)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T04:23:35Z
- **Completed:** 2026-03-14T04:25:54Z
- **Tasks:** 1 (TDD: RED commit + GREEN commit)
- **Files modified:** 6

## Accomplishments
- Implemented get_research_context wrapping ContextEngine.build_context_pack() with offline=True default for side-effect-free tool calls
- Implemented save_to_memory with MemoryKind validation (11 allowed values), defaulting invalid input to "note" with a warning
- Implemented export_to_obsidian performing pure in-memory rendering via ObsidianFilesystemExporter._render_paper_note() and _yaml_frontmatter()
- 9 unit tests pass covering: context pack dict return, user_id/track_id passthrough, invalid kind default, frontmatter presence, and audit logging for all three tools

## Task Commits

Each task was committed atomically:

1. **RED phase (failing tests)** - `d82e0d4` (test)
2. **GREEN phase (implementations)** - `302edcf` (feat)

*Note: TDD task split into RED (failing tests) and GREEN (implementation) commits.*

## Files Created/Modified
- `src/paperbot/mcp/tools/get_research_context.py` - Async MCP tool wrapping ContextEngine with lazy singleton
- `src/paperbot/mcp/tools/save_to_memory.py` - MCP tool wrapping SqlAlchemyMemoryStore with MemoryKind validation via anyio.to_thread
- `src/paperbot/mcp/tools/export_to_obsidian.py` - MCP tool for in-memory Obsidian markdown rendering via anyio.to_thread
- `tests/unit/test_mcp_get_research_context.py` - 3 tests: context pack, passthrough, audit
- `tests/unit/test_mcp_save_to_memory.py` - 3 tests: counts, invalid kind, audit
- `tests/unit/test_mcp_export_to_obsidian.py` - 3 tests: markdown key, frontmatter+title, audit

## Decisions Made
- **ContextEngine defaults to offline mode:** `ContextEngineConfig(offline=True, paper_limit=0)` prevents network calls during MCP tool execution by default; callers can override via environment if needed
- **Invalid MemoryKind defaults silently to 'note':** Raising a validation error would break agent workflows passing loose strings; warning + default is safer for MCP tool callers
- **Private method _render_paper_note() called directly:** This is intentional per plan research notes - it performs pure template rendering with no filesystem I/O, making it safe for MCP tool use

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 9 MCP tools now implemented (paper_judge, paper_search, paper_summarize, relevance, analyze_trends, check_scholar + the 3 from this plan)
- All tools follow consistent lazy singleton + _impl + register + log_tool_call pattern
- Ready for Phase 3 Plan 3 (MCP server registration) to wire all tools into a single FastMCP server

## Self-Check: PASSED

All artifacts verified:
- FOUND: src/paperbot/mcp/tools/get_research_context.py
- FOUND: src/paperbot/mcp/tools/save_to_memory.py
- FOUND: src/paperbot/mcp/tools/export_to_obsidian.py
- FOUND: tests/unit/test_mcp_get_research_context.py
- FOUND: tests/unit/test_mcp_save_to_memory.py
- FOUND: tests/unit/test_mcp_export_to_obsidian.py
- FOUND: .planning/phases/03-remaining-mcp-tools/03-02-SUMMARY.md
- FOUND: d82e0d4 (RED commit)
- FOUND: 302edcf (GREEN commit)

---
*Phase: 03-remaining-mcp-tools*
*Completed: 2026-03-14*
