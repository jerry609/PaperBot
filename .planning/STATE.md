---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: PostgreSQL Migration & Data Layer Refactoring
status: planning
stopped_at: Milestone initialized
last_updated: "2026-03-14"
last_activity: 2026-03-14 -- Milestone v2.0 started (PG migration + async + model refactoring)
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** Paper-specific capability layer surfaced as standard MCP tools + agent orchestration dashboard
**Current focus:** v2.0 PostgreSQL Migration & Data Layer Refactoring

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-14 — Milestone v2.0 started

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 5 (from v1.0)
- Average duration: 6 min
- Total execution time: 0.5 hours

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v2.0 init] Migrate SQLite → PostgreSQL with Docker for local dev
- [v2.0 init] Async data layer: sync Session → AsyncSession (asyncpg), all stores
- [v2.0 init] Systematic model refactoring: normalize, add constraints, merge/split tables
- [v2.0 init] Use PG-native features: tsvector (replace FTS5), JSONB (replace JSON text)
- [v2.0 init] v2.0 runs as independent epic, parallel to v1.0/v1.1 remaining phases

### Pending Todos

None yet.

### Blockers/Concerns

- FTS5 virtual tables (memory_items, document_chunks) need tsvector migration strategy
- sqlite-vec extension needs PG vector equivalent (pgvector)
- All 17+ store classes need sync→async conversion
- Existing tests rely on SQLite in-memory databases — need PG test strategy

## Session Continuity

Last session: 2026-03-14
Stopped at: Milestone v2.0 initialized
Resume file: None
