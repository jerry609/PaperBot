---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: MCP Server
status: planning
stopped_at: Completed 05-transport-entry-point/05-01-PLAN.md
last_updated: "2026-03-14T05:51:03.467Z"
last_activity: 2026-03-14 -- Roadmap created for v1.1 milestone (phases 7-11)
progress:
  total_phases: 9
  completed_phases: 3
  total_plans: 8
  completed_plans: 6
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** Paper-specific capability layer surfaced as standard MCP tools + agent orchestration dashboard
**Current focus:** v1.1 Agent Orchestration Dashboard -- Phase 7 (EventBus + SSE Foundation)

## Current Position

Phase: 7 of 11 (EventBus + SSE Foundation)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-03-14 -- Roadmap created for v1.1 milestone (phases 7-11)

Progress: [█████░░░░░] 50%

## Upcoming Milestone: v2.0 PostgreSQL Migration & Data Layer Refactoring

Status: Defining requirements
Target: PG migration + async data layer + systematic model refactoring

## Performance Metrics

**Velocity:**
- Total plans completed: 2 (from v1.0 phase 02)
- Average duration: 6 min
- Total execution time: 0.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 2/3 | 12min | 6min |

**Recent Trend:**
- Last 2 plans: 4min, 8min
- Trend: Stable
| Phase 03-remaining-mcp-tools P01 | 2 | 1 tasks | 4 files |
| Phase 03-remaining-mcp-tools P02 | 2 | 1 tasks | 6 files |
| Phase 03-remaining-mcp-tools P03 | 5 | 2 tasks | 2 files |
| Phase 04-mcp-resources P01 | 3 | 2 tasks | 9 files |
| Phase 04-mcp-resources P02 | 2 | 2 tasks | 2 files |
| Phase 05-transport-entry-point P01 | 3min | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1 init] EventBus as CompositeEventLog backend -- extends existing event system, not parallel
- [v1.1 init] Codex bridge is a .claude/agents/ file, not PaperBot server code
- [v1.1 init] Zero new dependencies -- builds on existing packages
- [Phase 03-remaining-mcp-tools]: analyze_trends uses anyio.to_thread.run_sync() to wrap sync TrendAnalyzer; check_scholar awaits async SemanticScholarClient directly
- [Phase 03-remaining-mcp-tools]: Degraded detection for analyze_trends: empty/whitespace-only LLM response triggers degraded=True
- [Phase 03-remaining-mcp-tools]: get_research_context uses ContextEngineConfig(offline=True) default to avoid network side-effects in MCP tool calls
- [Phase 03-remaining-mcp-tools]: save_to_memory defaults invalid MemoryKind to 'note' with logger.warning rather than raising, for safer agent workflows
- [Phase 03-remaining-mcp-tools]: export_to_obsidian calls _render_paper_note() directly (private method, intentional) for pure in-memory rendering with no filesystem I/O
- [Phase 03-remaining-mcp-tools]: All 9 MCP tools registered in single FastMCP server via sequential import+register pattern
- [Phase 04-mcp-resources]: Track resources use anyio.to_thread.run_sync() because SqlAlchemyResearchStore and SqlAlchemyMemoryStore are synchronous
- [Phase 04-mcp-resources]: track_memory passes scope_type='track' and scope_id=str(tid) to filter memories to specific track (not global scope)
- [Phase 04-mcp-resources]: scholars.py instantiates fresh SubscriptionService() each call (no singleton caching) for always-fresh YAML config reads
- [Phase 04-mcp-resources]: Resources registered with same import+register pattern as tools in server.py try: block
- [Phase 04-mcp-resources]: Integration tests use inspect.getsource() for resource registration verification (Python 3.9 FastMCP constraint)
- [v2.0 init] Migrate SQLite → PostgreSQL with Docker for local dev
- [v2.0 init] Async data layer: sync Session → AsyncSession (asyncpg), all stores
- [v2.0 init] Systematic model refactoring: normalize, add constraints, merge/split tables
- [v2.0 init] Use PG-native features: tsvector (replace FTS5), JSONB (replace JSON text)
- [Phase 05-transport-entry-point]: [Phase 05-01]: _get_mcp() helper for testable mcp lazy import; default HTTP port 8001 avoids FastAPI conflict; serve.py redirects logging to stderr for stdio purity

### Pending Todos

None yet.

### Blockers/Concerns

- v1.0 MCP server (phases 1-6) must be functional before v1.1 work begins
- Codex CLI JSONL output format should be tested with real `codex exec --json` before building parser
- [v2.0] FTS5 virtual tables (memory_items, document_chunks) need tsvector migration strategy
- [v2.0] sqlite-vec extension needs PG vector equivalent (pgvector)
- [v2.0] All 17+ store classes need sync→async conversion
- [v2.0] Existing tests rely on SQLite in-memory databases — need PG test strategy

## Session Continuity

Last session: 2026-03-14T05:51:03.462Z
Stopped at: Completed 05-transport-entry-point/05-01-PLAN.md
Resume file: None
