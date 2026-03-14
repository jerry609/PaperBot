---
phase: 06-agent-skills
plan: 01
subsystem: agent-skills
tags: [skill-md, mcp, claude-code, agent-discovery, workflow]

# Dependency graph
requires:
  - phase: 05-transport-entry-point
    provides: FastMCP HTTP/stdio transport and serve.py entry point
  - phase: 04-mcp-resources
    provides: MCP resources (track metadata, papers, memory, scholars)
  - phase: 03-remaining-mcp-tools
    provides: analyze_trends, check_scholar, get_research_context, save_to_memory, export_to_obsidian
  - phase: 02-core-paper-tools
    provides: paper_search, paper_judge, paper_summarize, relevance_assess MCP tools
provides:
  - Four SKILL.md agent skill files in .claude/skills/ for Claude Code/Codex discovery
  - literature-review workflow skill (search -> filter -> judge -> summarize -> export -> save)
  - paper-reproduction workflow skill (find -> reproducibility-judge -> summarize -> plan -> export)
  - trend-analysis workflow skill (context -> search -> analyze_trends -> save)
  - scholar-monitoring workflow skill (check_scholar -> optional trends -> save)
  - Structural validation tests (tests/unit/test_agent_skills.py) covering all MCP-13 assertions
affects:
  - 07-eventbus-sse (v1.1 phase — agent skills complete the v1.0 MCP Server milestone)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SKILL.md frontmatter: name (matches directory), description (4-6 trigger phrases), tools list"
    - "Skill body: imperative form, numbered workflow steps, tool parameters, Degraded Mode section"
    - "Progressive disclosure: lean body (80-200 lines), references/ subdirectory for detail if needed"
    - "TDD: write structural tests first (RED), then create files (GREEN)"

key-files:
  created:
    - .claude/skills/literature-review/SKILL.md
    - .claude/skills/paper-reproduction/SKILL.md
    - .claude/skills/trend-analysis/SKILL.md
    - .claude/skills/scholar-monitoring/SKILL.md
    - tests/unit/test_agent_skills.py
  modified: []

key-decisions:
  - "Skill tool names copied verbatim from @mcp.tool() source to prevent name mismatch bugs"
  - "Degraded Mode section included in all four skills — LLM tools return degraded=True when API key missing"
  - "tools frontmatter field included as advisory documentation, not enforced security boundary"
  - "version field omitted from frontmatter — not required by skill discovery mechanism"
  - "No references/ subdirectories needed — all four skills fit within 80-200 line target"

patterns-established:
  - "SKILL.md trigger phrases: include 4-6 quoted user-utterance phrases covering casual and formal phrasings"
  - "MCP tool references in skill bodies use backtick code formatting for exact tool names"

requirements-completed: [MCP-13]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 6 Plan 01: Agent Skills Summary

**Four SKILL.md workflow skills (.claude/skills/) enabling Claude Code/Codex agents to discover and execute PaperBot MCP tool chains for literature review, paper reproduction, trend analysis, and scholar monitoring**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T06:16:00Z
- **Completed:** 2026-03-14T06:19:20Z
- **Tasks:** 2 (TDD: test file + skill files)
- **Files modified:** 5 (1 test file + 4 SKILL.md files)

## Accomplishments

- Created `tests/unit/test_agent_skills.py` with 6 structural tests (TDD red phase confirmed failing before skill files existed)
- Created 4 SKILL.md files covering the full PaperBot MCP tool surface (all 9 tools referenced across skills)
- All 6 structural tests pass; full CI suite (66 tests) passes with no regressions
- Completed the v1.0 MCP Server milestone (phases 1-6 all plans done)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create structural validation tests (TDD red)** - `c6bb8c7` (test)
2. **Task 2: Create four SKILL.md agent skill files (TDD green)** - `830cc81` (feat)

**Plan metadata:** (docs commit follows)

_Note: TDD task 1 commit was test-only (red phase). Task 2 commit created all SKILL.md files (green phase)._

## Files Created/Modified

- `.claude/skills/literature-review/SKILL.md` - 6-step literature review workflow: paper_search -> relevance_assess -> paper_judge -> paper_summarize -> export_to_obsidian -> save_to_memory
- `.claude/skills/paper-reproduction/SKILL.md` - 5-step paper reproduction workflow: paper_search -> paper_judge(rubric=reproducibility) -> paper_summarize -> save_to_memory -> export_to_obsidian
- `.claude/skills/trend-analysis/SKILL.md` - 4-step trend analysis workflow: get_research_context -> paper_search -> analyze_trends -> save_to_memory
- `.claude/skills/scholar-monitoring/SKILL.md` - 3-step scholar monitoring workflow: check_scholar -> analyze_trends -> save_to_memory
- `tests/unit/test_agent_skills.py` - 6 structural tests covering: directory exists, file existence, frontmatter validity, name-directory match, tool references, trigger phrase count

## Decisions Made

- Tool names copied verbatim from `@mcp.tool()` source (verified in 06-RESEARCH.md) to prevent silent name mismatch bugs that would cause "tool not found" at agent runtime
- `Degraded Mode` section included in all four skills because all LLM-backed tools (`paper_judge`, `paper_summarize`, `relevance_assess`, `analyze_trends`) return `degraded=True` when API keys are missing — critical for user diagnosis
- `tools` frontmatter field included as advisory documentation (helps agents pre-allow tools), not enforced as a security boundary
- `version` field omitted — not required by Claude Code skill loader, keeps frontmatter minimal
- No `references/` subdirectories created — all workflow bodies fit within the 80-200 line target

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Skills are static files; no build steps needed.

## Next Phase Readiness

- v1.0 MCP Server milestone complete (phases 1-6 all done)
- Skills are discoverable by Claude Code agents pointing at the PaperBot repo
- Agents can trigger skills with natural language matching the trigger phrases in each description field
- Ready for v1.1 Agent Orchestration Dashboard (Phase 7: EventBus + SSE Foundation)

## Self-Check

Verified:
- `c6bb8c7` exists: test commit confirmed
- `830cc81` exists: feat commit confirmed
- All 4 SKILL.md files present at `.claude/skills/*/SKILL.md`
- `tests/unit/test_agent_skills.py` exists with 128 lines (6 test functions)
- `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -v` → 6 passed

---
*Phase: 06-agent-skills*
*Completed: 2026-03-14*
