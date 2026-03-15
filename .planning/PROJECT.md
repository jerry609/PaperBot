# PaperBot

## What This Is

PaperBot is a multi-agent research workflow framework for academic paper discovery, analysis, and reproduction. It provides a FastAPI backend with SSE streaming, a Next.js web dashboard, and a terminal CLI. The platform follows a Skill-Driven Architecture where PaperBot acts as a capability provider, exposing paper-specific tools via MCP. The web dashboard (DeepCode) serves as an agent-agnostic visualization and control surface — proxying chat to whichever code agent the user configures (Claude Code, Codex, OpenCode, etc.) and displaying real-time agent activity, team decomposition, and file changes.

## Core Value

Paper-specific capability layer: understanding, reproduction, verification, and context — surfaced as standard MCP tools that any agent can consume, with an agent-agnostic dashboard that visualizes and controls whatever code agent the user runs.

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

<!-- Current scope: v1.1 Agent Orchestration Dashboard + v1.2 DeepCode Agent Dashboard + v2.0 PG Migration -->

- [ ] Codex subagent bridge for Claude Code (custom agent definition)
- [ ] Agent orchestration dashboard (replaces studio page)
- [ ] Agent event logging via MCP (lifecycle, tool calls, file changes, task status)
- [ ] Three-panel IDE layout (tasks | agent activity | files)
- [ ] Live SSE streaming for real-time agent activity
- [ ] Paper2Code overflow delegation workflow (Claude Code → Codex)
- [ ] Agent-agnostic proxy layer (chat proxies to user-configured agent: Claude Code, Codex, OpenCode)
- [ ] Multi-agent adapter layer (unified interface for different code agents)
- [ ] Agent activity discovery (hybrid: agent pushes events + dashboard discovers independently)
- [ ] Team visualization (agent-initiated team decomposition reflected in dashboard)
- [ ] Dashboard control surface (send commands/tasks to agents from web UI)
- [ ] PostgreSQL migration (replace SQLite)
- [ ] Async data layer (AsyncSession + asyncpg)
- [ ] Systematic data model refactoring
- [ ] PG-native features (tsvector, JSONB)

### Out of Scope

- Custom agent orchestration runtime — host agents own orchestration, PaperBot visualizes
- Building any code agent (Claude Code, Codex, OpenCode) — uses existing tools
- Business logic duplication — tools must reuse existing services
- Hardcoded agent pipeline logic — agent decides team composition and delegation
- Per-agent custom UI — one unified dashboard serves all agents

## Context

- Architecture pivot from AgentSwarm to Skill-Driven Architecture (2026-03-13)
- Further pivot: DeepCode as agent-agnostic dashboard, not Claude Code-specific (2026-03-15)
- Problem identified: chat mode split between Claude Code CLI connection vs direct API Codex calls — needs unification
- Existing `codex_dispatcher.py` and `claude_commander.py` in infrastructure/swarm/ — to be replaced by unified adapter
- Existing `AgentEventEnvelope` with run_id/trace_id/span_id in application/collaboration/
- Studio page exists with Monaco editor and XTerm terminal
- @xyflow/react already in web dashboard for DAG visualization
- MCP server (v1.0 milestone) provides tool surface for agent integration
- v1.1 EventBus + SSE foundation (phases 7-8) partially built
- Dev branch synced to origin/dev at 2e5173d (2026-03-14)
- Current DB: SQLite with 46 models, sync Session, FTS5 virtual tables, optional sqlite-vec

## Constraints

- **MCP prerequisite**: v1.0 MCP server must be functional before agent orchestration
- **Reuse**: Event logging must extend existing AgentEventEnvelope, not create parallel system
- **Agent-agnostic**: Dashboard must work with any code agent, not hardcode Claude Code or Codex specifics
- **No orchestration logic**: PaperBot does NOT decompose tasks — the host agent does; PaperBot visualizes
- **Studio integration**: Dashboard integrates with existing Monaco/XTerm, not replaces them
- **Transport**: SSE for live updates (existing infrastructure)

## Current Milestone: v1.2 DeepCode Agent Dashboard

**Goal:** Unify the agent interaction model into a single agent-agnostic architecture where PaperBot's web UI (DeepCode) proxies chat to the user's chosen code agent, visualizes agent activity (teams, tasks, files) in real-time, and provides control commands — without hardcoding orchestration logic.

**Target features:**
- Agent-agnostic proxy layer (chat → Claude Code / Codex / OpenCode / etc.)
- Multi-agent adapter layer (unified interface abstracting agent-specific APIs/CLIs)
- Hybrid activity discovery (agent pushes events via MCP + dashboard discovers independently)
- Team visualization (agent-initiated team decomposition rendered in dashboard)
- Dashboard control surface (send commands/tasks back to agents)
- Real-time agent activity stream (builds on v1.1 EventBus/SSE)

## Previous Milestone: v1.1 Agent Orchestration Dashboard

**Goal:** Build a Codex subagent bridge for Claude Code and a real-time agent orchestration dashboard in PaperBot's web UI, enabling the Paper2Code overflow delegation workflow.

**Target features:**
- Codex subagent bridge (`.claude/agents/codex-worker.md`)
- Three-panel agent dashboard (replaces studio page)
- Agent event logging (lifecycle, tools, files, tasks)
- Live SSE streaming for real-time updates
- Paper2Code workflow with Codex overflow delegation

## Planned Milestone: v2.0 PostgreSQL Migration & Data Layer Refactoring

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

| DeepCode = agent-agnostic dashboard | Chat split (CLI vs API) was wrong; unify into proxy model where PaperBot doesn't care which agent | — Pending |
| Agent-initiated team decomposition | Agent decides how to split work; dashboard visualizes, doesn't orchestrate | — Pending |
| Hybrid activity discovery | Agent pushes structured events + dashboard can discover independently | — Pending |
| Dashboard + control (not pure display) | Users need to send commands/tasks, not just watch | — Pending |

---
*Last updated: 2026-03-15 after v1.2 milestone added*
