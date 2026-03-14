# PaperBot

## What This Is

PaperBot is a multi-agent research workflow framework for academic paper discovery, analysis, and reproduction. It provides a FastAPI backend with SSE streaming, a Next.js web dashboard, and a terminal CLI. The platform is evolving toward a Skill-Driven Architecture where PaperBot acts as a capability provider, exposing paper-specific tools via MCP and providing an agent orchestration dashboard for Claude Code and Codex.

## Core Value

Paper-specific capability layer: understanding, reproduction, verification, and context — surfaced as standard MCP tools that any agent can consume, with a visual dashboard for agent orchestration.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from existing codebase. -->

- ✓ Paper search and discovery (arxiv, openalex, semantic scholar, reddit, HF daily, paperscool)
- ✓ Paper analysis pipeline: judge, summarize, trend analysis, relevance assessment
- ✓ Scholar tracking and monitoring
- ✓ Paper2Code reproduction pipeline (planning → blueprint → generation → verification)
- ✓ CodeRAG pattern retrieval and CodeMemory cross-file context
- ✓ Research context engine and track routing
- ✓ Memory module (save/retrieve research context)
- ✓ DailyPaper cron workflow (ARQ-backed)
- ✓ FastAPI server with SSE streaming
- ✓ Next.js web dashboard (studio, wiki, papers, research, scholars, workflows)
- ✓ DI container and pipeline framework
- ✓ Event logging and audit trail
- ✓ Authentication (JWT, email/password, API key middleware)
- ✓ Agent infrastructure (codex_dispatcher.py, claude_commander.py in swarm/)

### Active

<!-- Current scope: v1.1 Agent Orchestration Dashboard -->

- [ ] Codex subagent bridge for Claude Code (custom agent definition)
- [ ] Agent orchestration dashboard (replaces studio page)
- [ ] Agent event logging via MCP (lifecycle, tool calls, file changes, task status)
- [ ] Three-panel IDE layout (tasks | agent activity | files)
- [ ] Live SSE streaming for real-time agent activity
- [ ] Paper2Code overflow delegation workflow (Claude Code → Codex)

### Out of Scope

- Custom agent orchestration runtime — host agents (Claude Code) own orchestration
- Per-host adapters — one MCP surface serves all
- Business logic duplication — tools must reuse existing services
- Building Codex itself — uses existing Codex CLI

## Context

- Architecture pivot from AgentSwarm to Skill-Driven Architecture (2026-03-13)
- Existing `codex_dispatcher.py` and `claude_commander.py` in infrastructure/swarm/
- Existing `AgentEventEnvelope` with run_id/trace_id/span_id in application/collaboration/
- Studio page exists with Monaco editor and XTerm terminal
- @xyflow/react already in web dashboard for DAG visualization
- MCP server (v1.0 milestone) is prerequisite — provides tool surface for agent integration
- Dev branch synced to origin/dev at 2e5173d (2026-03-14)

## Constraints

- **MCP prerequisite**: v1.0 MCP server must be functional before agent orchestration
- **Reuse**: Event logging must extend existing AgentEventEnvelope, not create parallel system
- **Claude Code bridge**: Codex integration is a Claude Code agent definition, not PaperBot server code
- **Studio integration**: Dashboard integrates with existing Monaco/XTerm, not replaces them
- **Transport**: SSE for live updates (existing infrastructure)

## Current Milestone: v2.0 PostgreSQL Migration & Data Layer Refactoring

**Goal:** Migrate from SQLite to PostgreSQL, refactor all 46 data models systematically, and convert the entire data access layer from synchronous to async (asyncpg + AsyncSession).

**Target features:**
- Full PostgreSQL migration with Docker-based local development
- Async data layer (AsyncSession + asyncpg) across all stores
- Systematic model refactoring: normalization, constraints, redundancy removal
- PG-native features: tsvector full-text search (replacing FTS5), JSONB columns, proper indexing
- Alembic migration path from SQLite to PostgreSQL
- Data migration tooling for existing SQLite databases

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PaperBot = Skill Provider, not runtime | Host agents (Claude Code, Codex) already own orchestration | ✓ Good |
| Codex via custom agent definition | `.claude/agents/codex-worker.md` — simplest, uses existing Claude Code infrastructure | — Pending |
| Replace studio page with agent dashboard | Studio's Monaco/XTerm integrate into agent view | — Pending |
| Extend AgentEventEnvelope | Reuse existing run_id/trace_id/span_id schema | — Pending |
| Overflow delegation model | Claude Code does everything, delegates to Codex when workload is high | — Pending |
| MCP event log as data flow | Agent activity → MCP event log → dashboard reads | — Pending |
| Live SSE streaming | Real-time updates using existing SSE infrastructure | — Pending |

| PG migration over SQLite | SQLite concurrency limits, lack of PG features (tsvector, JSONB), production readiness | — Pending |
| Async data layer (asyncpg) | FastAPI is async; sync DB calls block event loop; do it together with PG migration | — Pending |
| Systematic model refactoring | 46 models accumulated organically; normalize, add constraints, remove redundancy | — Pending |
| Docker PG for local dev | Standard dev setup, matches production topology | — Pending |

---
*Last updated: 2026-03-14 after v2.0 milestone initialization*
