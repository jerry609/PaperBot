# Roadmap: PaperBot

## Milestones

- ✅ **v1.0 MCP Server** - Phases 1-6 (complete)
- 📋 **v1.1 Agent Orchestration Dashboard** - Phases 7-11 (planned)
- 📋 **v1.2 DeepCode Agent Dashboard** - Phases 18-23 (planned)
- 📋 **v2.0 PostgreSQL Migration & Data Layer Refactoring** - Phases 12-17 (planned)

## Phases

<details>
<summary>v1.0 MCP Server (Phases 1-6) - Complete</summary>

**Milestone Goal:** Complete the MCP server with all 9 tools, 4 resources, transport configuration, and Agent Skills — making PaperBot's full capability surface available to any MCP-compatible agent.

- [x] **Phase 1: MCP Server Setup** - FastMCP instance, package structure (shipped)
- [x] **Phase 2: Core Paper Tools** - paper_search, paper_judge, paper_summarize, relevance_assess + audit helper (shipped)
- [x] **Phase 3: Remaining MCP Tools** - analyze_trends, check_scholar, get_research_context, save_to_memory, export_to_obsidian (completed 2026-03-14)
- [x] **Phase 4: MCP Resources** - 4 resource URIs for track/scholar data access (completed 2026-03-14)
- [x] **Phase 5: Transport & Entry Point** - stdio + Streamable HTTP transports, CLI command (completed 2026-03-14)
- [x] **Phase 6: Agent Skills** - SKILL.md files for core workflows (completed 2026-03-14)

### Phase 3: Remaining MCP Tools
**Goal**: All 9 MCP tools are registered and callable, completing the tool surface
**Depends on**: Phase 2 (audit helper and registration pattern)
**Requirements**: MCP-01, MCP-02, MCP-03, MCP-04, MCP-05
**Success Criteria** (what must be TRUE):
  1. Agent can call `analyze_trends` and receive trend analysis for a set of papers
  2. Agent can call `check_scholar` and receive a scholar's recent publications
  3. Agent can call `get_research_context` and receive context for a research track
  4. Agent can call `save_to_memory` and persist research findings retrievable later
  5. Agent can call `export_to_obsidian` and receive Obsidian-formatted markdown
  6. All 9 tools appear in MCP tools/list
  7. All tools log calls via audit helper
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — analyze_trends + check_scholar tools with TDD
- [ ] 03-02-PLAN.md — get_research_context + save_to_memory + export_to_obsidian tools with TDD
- [ ] 03-03-PLAN.md — Server registration + integration tests for all 9 tools

### Phase 4: MCP Resources
**Goal**: Agents can read PaperBot data via MCP resource URIs without tool calls
**Depends on**: Phase 2 (server instance)
**Requirements**: MCP-06, MCP-07, MCP-08, MCP-09
**Success Criteria** (what must be TRUE):
  1. Agent can read `paperbot://track/{id}` and receive track metadata
  2. Agent can read `paperbot://track/{id}/papers` and receive paper list for a track
  3. Agent can read `paperbot://track/{id}/memory` and receive saved research memory
  4. Agent can read `paperbot://scholars` and receive scholar subscription list
  5. All 4 resources appear in MCP resources/list
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — TDD resource implementations (track_metadata, track_papers, track_memory, scholars) with unit tests
- [ ] 04-02-PLAN.md — Server registration + integration tests for all 4 resources

### Phase 5: Transport & Entry Point
**Goal**: MCP server is runnable via stdio (local) and Streamable HTTP (remote) with a CLI command
**Depends on**: Phase 3, Phase 4 (all tools and resources registered)
**Requirements**: MCP-10, MCP-11, MCP-12
**Success Criteria** (what must be TRUE):
  1. `paperbot mcp serve --stdio` starts MCP server on stdio transport
  2. `paperbot mcp serve --http` starts MCP server on Streamable HTTP transport
  3. Claude Code can connect to PaperBot MCP server via stdio in `claude_desktop_config.json`
  4. Remote agent can connect via HTTP and call tools
**Plans**: 1 plan

Plans:
- [ ] 05-01-PLAN.md — Transport dispatch (serve.py), CLI mcp serve subcommand, packaging (pyproject.toml scripts + mcp dep), unit tests

### Phase 6: Agent Skills
**Goal**: Core PaperBot workflows are available as SKILL.md files for agent discovery
**Depends on**: Phase 3 (tools must exist for skills to reference)
**Requirements**: MCP-13
**Success Criteria** (what must be TRUE):
  1. `.claude/skills/` directory contains SKILL.md files for literature-review, paper-reproduction, trend-analysis, scholar-monitoring
  2. Each SKILL.md has valid YAML frontmatter (name, description, tools) and markdown instructions
  3. Skills reference MCP tools by name and provide multi-step workflow guidance
**Plans**: 1 plan

Plans:
- [ ] 06-01-PLAN.md — Structural tests + four SKILL.md agent skill files (literature-review, paper-reproduction, trend-analysis, scholar-monitoring)

</details>

### v1.1 Agent Orchestration Dashboard

**Milestone Goal:** Build a real-time agent orchestration dashboard and Codex subagent bridge, enabling users to observe and manage Claude Code and Codex agent activity from PaperBot's web UI.

**Phase Numbering:**
- Integer phases (7, 8, 9...): Planned milestone work
- Decimal phases (7.1, 7.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 7: EventBus + SSE Foundation** - In-process event bus with SSE subscription endpoint for real-time push (completed 2026-03-14)
- [ ] **Phase 8: Agent Event Vocabulary** - Extend AgentEventEnvelope types and build activity feed, lifecycle indicators, and tool call timeline
- [x] **Phase 9: Three-Panel Dashboard** - Frontend store, SSE hook, and three-panel IDE layout with file visualization (completed 2026-03-15)
- [x] **Phase 10: Agent Board + Codex Bridge** - Kanban board generalization, Codex worker agent definition, and overflow delegation (completed 2026-03-15)
- [ ] **Phase 11: DAG Visualization** - Task dependency DAG and cross-agent context sharing visualization

### v1.2 DeepCode Agent Dashboard

**Milestone Goal:** Unify the agent interaction model into a single agent-agnostic architecture where PaperBot's web UI (DeepCode) proxies chat to the user's chosen code agent, visualizes agent activity (teams, tasks, files) in real-time, and provides control commands — without hardcoding orchestration logic.

**Phase Numbering:**
- Integer phases (18, 19, 20...): Planned milestone work
- Decimal phases (18.1, 18.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 18: Adapter Foundation** - BaseAgentAdapter ABC, ClaudeCodeAdapter (persistent REPL mode), AgentProxyService, agent config, hybrid discovery
- [ ] **Phase 19: Activity Stream + Session Management** - SSE reliability fixes, run-scoped filtering, ActivityFeed, session list/detail, token/cost tracking
- [ ] **Phase 20: Chat + Control Surface** - AgentChatPanel, interrupt/cancel control, session replay with timeline scrubber, checkpoint/restore
- [ ] **Phase 21: Visualization Panels** - TeamDAGPanel (@xyflow/react), Monaco file diff panel, agent card grid, swim lane timeline
- [ ] **Phase 22: Additional Agent Adapters** - CodexAdapter (JSONL), OpenCodeAdapter (HTTP/ACP), agent selection settings UI
- [ ] **Phase 23: HITL + Domain Enrichment** - Human-in-the-loop approval modal, Paper2Code enriched view, PaperBot MCP tool card rendering

### v2.0 PostgreSQL Migration & Data Layer Refactoring

**Milestone Goal:** Migrate from SQLite to PostgreSQL, convert all 17 stores to async SQLAlchemy (asyncpg + AsyncSession), add PG-native features (tsvector, JSONB, pgvector), and systematically refactor all 46 data models.

**Phase Numbering:**
- Integer phases (12, 13, 14...): Planned milestone work
- Decimal phases (12.1, 12.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 12: PG Infrastructure & Schema** - Docker Compose PostgreSQL, Alembic dual-path env.py, is_sqlite guards, tsvector/JSONB/pgvector Alembic migrations
- [ ] **Phase 13: Test Infrastructure** - testcontainers PostgreSQL CI fixture, async pytest infrastructure, full test suite passing on PG, performance benchmarks
- [ ] **Phase 14: Async Data Layer** - Single shared AsyncEngine, AsyncSessionProvider, all 17 stores converted to async, lazy="raise" + selectinload audit, MCP anyio wrappers removed, async ARQ worker
- [ ] **Phase 15: PG-Native Features** - Hybrid tsvector+pgvector search with RRF, JSONB GIN indexes, HNSW vector index, JSON helper method removal
- [ ] **Phase 16: Model Refactoring** - relationship lazy="raise" enforcement, NOT NULL/CHECK/UNIQUE constraints, is_active Boolean migration, Author normalization
- [ ] **Phase 17: Data Migration & CI** - pgloader SQLite->PG tooling, sqlite-vec->pgvector embedding re-encoding, CI PostgreSQL service container, Alembic migration validation, slow query logging, health endpoint, pool metrics API

## Phase Details

### Phase 7: EventBus + SSE Foundation
**Goal**: Agent events are pushed to connected clients in real-time without polling
**Depends on**: Phase 6 (v1.0 MCP server provides tool surface)
**Requirements**: EVNT-04
**Success Criteria** (what must be TRUE):
  1. A dashboard client connected via SSE receives agent events within 1 second of emission
  2. Multiple simultaneous SSE clients each receive all events independently
  3. Existing event_log.append() calls automatically push to SSE subscribers with zero changes to calling code
  4. SSE connections clean up gracefully on client disconnect (no leaked queues or background tasks)
**Plans**: 2 plans

Plans:
- [ ] 07-01-PLAN.md — EventBusEventLog TDD (ring buffer + fan-out + backpressure)
- [ ] 07-02-PLAN.md — SSE endpoint + main.py wiring + integration tests

### Phase 8: Agent Event Vocabulary
**Goal**: Users can see meaningful, structured agent activity as it happens
**Depends on**: Phase 7 (events must be pushable before they can be rendered)
**Requirements**: EVNT-01, EVNT-02, EVNT-03
**Success Criteria** (what must be TRUE):
  1. User sees a scrolling activity feed that updates in real-time as agents emit events
  2. User can see at a glance whether each agent is idle, working, completed, or errored
  3. User can view a structured tool call timeline showing tool name, arguments, result summary, and duration for each call
  4. New event types extend AgentEventEnvelope (no parallel event schema created)
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md — Python EventType vocabulary constants, lifecycle/tool-call helpers, _audit.py migration with TDD
- [x] 08-02-PLAN.md — Frontend TypeScript types, parsers, Zustand store, SSE hook, display components (ActivityFeed, AgentStatusPanel, ToolCallTimeline), test harness page

### Phase 9: Three-Panel Dashboard
**Goal**: Users can observe agent work in a three-panel IDE layout with file-level detail
**Depends on**: Phase 8 (activity feed and lifecycle events must exist before dashboard renders them)
**Requirements**: DASH-01, DASH-04, FILE-01, FILE-02
**Success Criteria** (what must be TRUE):
  1. User sees a three-panel layout (tasks | agent activity | files) when opening the agent dashboard
  2. User can drag panel dividers to resize each panel and the layout persists across page navigation
  3. User can view inline diffs showing exactly what an agent changed in each file
  4. User can see a per-task file list with created/modified indicators for every file an agent touched
  5. Dashboard state is managed by a Zustand store fed by the SSE event stream (no polling)
**Plans**: 2 plans

Plans:
- [ ] 09-01-PLAN.md — File tracking data layer: EventType.FILE_CHANGE, FileTouchedEntry type, parseFileTouched parser, store extension, SSE hook wiring with TDD
- [ ] 09-02-PLAN.md — Three-panel UI: TasksPanel, FileListPanel, InlineDiffPanel components, /agent-dashboard page with SplitPanels, sidebar nav link

### Phase 10: Agent Board + Codex Bridge
**Goal**: Users can manage agent tasks on a Kanban board and Claude Code can delegate work to Codex
**Depends on**: Phase 9 (dashboard layout must exist for board embedding; Phase 7 SSE for delegation events)
**Requirements**: DASH-02, DASH-03, CDX-01, CDX-02, CDX-03
**Success Criteria** (what must be TRUE):
  1. User can view and manage agent tasks on a Kanban board that shows which tasks belong to Claude Code vs Codex
  2. User sees Codex-specific error states (timeout, sandbox crash) surfaced prominently on failed tasks
  3. Claude Code can delegate tasks to Codex via the codex-worker.md custom agent definition
  4. Paper2Code pipeline stages can overflow from Claude Code to Codex when workload exceeds capacity
  5. User can observe Codex delegation events (dispatched, accepted, completed, failed) in the activity feed
**Plans**: 3 plans

Plans:
- [ ] 10-01-PLAN.md — Backend: Codex EventType constants, delegation event emission in agent_board.py, Paper2Code overflow stub (TDD)
- [ ] 10-02-PLAN.md — Frontend: CodexDelegationEntry type, parseCodexDelegation parser, KanbanBoard component with agent badges and error states (TDD)
- [ ] 10-03-PLAN.md — Glue: codex-worker.md sub-agent definition, dashboard Panels/Kanban view toggle, human verification

### Phase 11: DAG Visualization
**Goal**: Users can see task dependencies and cross-agent data flow visually
**Depends on**: Phase 10 (tasks and agents must exist before visualizing their relationships)
**Requirements**: VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. User can view an interactive task dependency DAG where node colors update in real-time to reflect task status
  2. User can see ScoreShareBus data flow edges in the DAG showing which agents shared evaluation context with which other agents
  3. DAG renders using existing @xyflow/react (no new visualization dependencies)
**Plans**: TBD

Plans:
- [ ] 11-01: TBD

### Phase 18: Adapter Foundation
**Goal**: The dashboard can connect to and control Claude Code via a stable, agent-agnostic adapter interface — with persistent sessions, typed events, and config-driven agent selection
**Depends on**: Phase 11 (v1.1 milestone complete; EventBus/SSE infrastructure and event vocabulary available)
**Requirements**: ADAPT-01, ADAPT-02, ADAPT-03, ADAPT-06
**Success Criteria** (what must be TRUE):
  1. User can select Claude Code as the active agent in settings and the selection persists in `.paperbot/agent.yaml`
  2. Sending a message to the agent uses a persistent REPL/stdin-mode subprocess — not a new process per message — so context is retained across turns without reloading full context each time
  3. Structured agent events (FILE_CHANGED, TEAM_UPDATE, TASK_UPDATE, CHAT_DELTA, CHAT_DONE) arrive in the EventBus within 1 second of emission with no duplicate events on SSE reconnect (SSE `id:` field emitted, seq-based recovery)
  4. Dashboard discovers agent activity through both agent-pushed MCP events and independent discovery (filesystem watch or polling), whichever arrives first
  5. Adding a second adapter type requires only a new class in `infrastructure/adapters/agent/` with no changes to AgentProxyService, API routes, or frontend components
**Plans**: TBD

Plans:
- [ ] 18-01: TBD
- [ ] 18-02: TBD
- [ ] 18-03: TBD

### Phase 19: Activity Stream + Session Management
**Goal**: Users can see a reliable real-time activity stream scoped to their current session, navigate session history, and track token cost per run
**Depends on**: Phase 18 (adapter must deliver events before stream or session views have meaningful content)
**Requirements**: MONIT-01, MONIT-02, MONIT-03, MONIT-04, MONIT-05, SESS-01, SESS-02, SESS-03
**Success Criteria** (what must be TRUE):
  1. User sees a scrolling ActivityFeed showing only events for the active run_id that auto-scrolls to the latest event and can be paused without dropping events
  2. User sees a tool call log entry for every tool invocation showing tool name, arguments, result status (success/error), and elapsed duration
  3. User sees an agent status badge (running / waiting / complete / error / idle) and a connection status indicator (connected / reconnecting / disconnected) that update without page refresh
  4. Errors are surfaced prominently: failed tool calls render in red, an error badge appears on the agent status indicator, and a toast notification fires on agent-level failure
  5. User can open a session list showing all past and active runs with agent type, status, start time, and estimated cost per session
  6. User can click any session to view its full ordered event timeline filtered to that run_id
  7. User can see input token count, output token count, and estimated dollar cost for each session
**Plans**: TBD

Plans:
- [ ] 19-01: TBD
- [ ] 19-02: TBD
- [ ] 19-03: TBD

### Phase 20: Chat + Control Surface
**Goal**: Users can dispatch tasks to the agent, interrupt running work, and step through or restore completed sessions
**Depends on**: Phase 19 (session management must exist for replay and restore; Phase 18 bidirectional adapter required for interrupt and checkpoint)
**Requirements**: CTRL-01, CTRL-02, SESS-04, SESS-05
**Success Criteria** (what must be TRUE):
  1. User can type a task in the chat input and submit it to the configured agent — the message routes through AgentProxyService to the adapter without the dashboard containing any agent-specific dispatch logic
  2. User can click an interrupt/cancel button during a running agent turn and the agent stops within 5 seconds; the button is disabled when no agent is actively running
  3. User can replay a completed session using a timeline scrubber that steps through events in their original sequence, pausing at any step to inspect state
  4. User can save a checkpoint of the current session and restore from any prior checkpoint, with each checkpoint addressable independently
**Plans**: TBD

Plans:
- [ ] 20-01: TBD
- [ ] 20-02: TBD

### Phase 21: Visualization Panels
**Goal**: Users can see live team decomposition as a DAG, file diffs in Monaco, per-agent cost cards, and agent swim lanes — all driven by a single extended Zustand store with no duplicate SSE connections
**Depends on**: Phase 19 (clean, session-scoped event stream required before panels have meaningful data; Phase 20 studio layout as panel anchor)
**Requirements**: VIS-01, VIS-02, VIS-03, VIS-04
**Success Criteria** (what must be TRUE):
  1. User sees a live DAG of agent-reported team decomposition (nodes = agents, edges = delegation relationships) that updates in real-time as TEAM_UPDATE events arrive, rendered with @xyflow/react
  2. User sees a Monaco diff editor showing the before/after file content for any file the agent modified, triggered automatically by FILE_CHANGED events
  3. User sees a card grid where each active agent has a card showing: cost so far, context window usage percentage with a color-graded bar, status badge, and latest action text
  4. User sees agent swim lanes where each agent occupies a vertical lane and events are plotted chronologically along a shared timeline axis
**Plans**: TBD

Plans:
- [ ] 21-01: TBD
- [ ] 21-02: TBD

### Phase 22: Additional Agent Adapters
**Goal**: Users can switch between Claude Code, Codex, and OpenCode via a settings toggle, and the dashboard renders correctly regardless of which agent is active — validating that the adapter abstraction does not collapse to lowest-common-denominator
**Depends on**: Phase 18 (ClaudeCodeAdapter must be proven stable before the abstraction is validated against two more concrete agents)
**Requirements**: ADAPT-04, ADAPT-05
**Success Criteria** (what must be TRUE):
  1. User can select Codex in settings and send a task that routes through CodexAdapter using subprocess + JSONL (`codex exec --json`); session context persists across turns via thread resumption without reloading full context
  2. User can select OpenCode in settings and send a task that routes through OpenCodeAdapter using HTTP REST+SSE; events normalize to the same AgentEventEnvelope schema as ClaudeCode and Codex
  3. All three adapters produce identical event shapes in ActivityFeed and TeamDAGPanel — no conditional adapter-specific rendering code exists in any frontend component
**Plans**: TBD

Plans:
- [ ] 22-01: TBD
- [ ] 22-02: TBD

### Phase 23: HITL + Domain Enrichment
**Goal**: Users can approve or reject proposed agent actions before execution, and Paper2Code sessions and PaperBot MCP tool calls surface paper-specific context automatically
**Depends on**: Phase 21 (frontend panels must be stable before adding modal overlay; Phase 18 IDLE/PROCESSING/AWAITING_INPUT state machine required to prevent command injection during approval wait)
**Requirements**: CTRL-03, DOMAIN-01, DOMAIN-02
**Success Criteria** (what must be TRUE):
  1. When an agent emits a HUMAN_APPROVAL_REQUIRED event, an approval modal appears showing the proposed action; user can approve or reject and the agent resumes or aborts accordingly without timing out the session
  2. When the active session has run_type paper2code, the dashboard shows an enriched header: paper title, abstract snippet, reproduction stage progress bar, and current stage name alongside the standard activity feed
  3. When PaperBot MCP tool calls appear in the tool call log, they render with a paper-specific card (title, venue, quality score badge) rather than raw JSON arguments
**Plans**: TBD

Plans:
- [ ] 23-01: TBD
- [ ] 23-02: TBD

### Phase 12: PG Infrastructure & Schema
**Goal**: PaperBot runs against PostgreSQL with a complete, PG-compatible schema — tsvector, JSONB, and pgvector columns in place — without crashing on any SQLite-only code path
**Depends on**: Phase 11 (v1.1 milestone completes before v2.0 begins)
**Requirements**: PGINFRA-01, PGINFRA-02
**Success Criteria** (what must be TRUE):
  1. Developer can run `docker-compose up` and get a working PostgreSQL + pgvector environment with no manual extension setup steps
  2. `alembic upgrade head` completes without error against a fresh PostgreSQL database and the dual-path env.py correctly routes async PG vs sync SQLite URLs
  3. All FTS5-dependent and sqlite_master-querying code paths are guarded with `is_sqlite` checks so the application starts and accepts requests on a PostgreSQL URL without crashing
  4. Alembic migrations add tsvector columns + GIN indexes on memory_items and document_chunks, JSONB type casts (with explicit USING col::jsonb) on all 84 JSON columns, and a pgvector Vector(1536) column on memory_items
  5. The existing sync store suite passes against PostgreSQL (minus FTS5 and sqlite-vec paths, which are guarded)
**Plans**: TBD

Plans:
- [ ] 12-01: TBD
- [ ] 12-02: TBD

### Phase 13: Test Infrastructure
**Goal**: All store tests run against real PostgreSQL, async pytest infrastructure is established, and CI greenlight is meaningful for PG-specific behavior
**Depends on**: Phase 12 (PostgreSQL schema must exist before PG-targeted tests can pass)
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. Running `pytest -m postgres` spins up a real PostgreSQL container via testcontainers, runs all store integration tests against it, and tears it down with no manual database setup required
  2. Every existing store integration test passes against PostgreSQL — JSONB operators, tsvector queries, and pgvector distance operators all exercise real PG behavior
  3. Async test fixtures use `pytest-asyncio` with per-test transaction rollback so each test starts from a clean database state with no residual data from prior tests
  4. A performance benchmark test exists that measures full-text search, vector search, and JSONB query latency against a seeded dataset and records a repeatable baseline
**Plans**: TBD

Plans:
- [ ] 13-01: TBD
- [ ] 13-02: TBD

### Phase 14: Async Data Layer
**Goal**: All 17 stores use async SQLAlchemy with a single shared AsyncEngine; no sync DB calls block the event loop; MCP tools await store methods directly; ARQ jobs each own their session
**Depends on**: Phase 13 (PG test fixture must be in place before any async store ships; Phase 12 for PG schema)
**Requirements**: ASYNC-01, ASYNC-02, ASYNC-03, ASYNC-04, ASYNC-05
**Success Criteria** (what must be TRUE):
  1. FastAPI starts with a single shared AsyncEngine registered in the DI container — no per-store engine or independent connection pool exists anywhere in the codebase
  2. All 17 stores expose only `async def` methods using `async with` session context; no synchronous `Session` import remains in any store file
  3. Accessing any ORM relationship attribute without an explicit `selectinload` or `joinedload` raises `MissingGreenlet` immediately (lazy="raise" enforced on all relationships before any store conversion begins)
  4. All MCP tools call `await store.method()` directly — zero `anyio.to_thread.run_sync` wrappers remain in any file under `src/paperbot/mcp/`
  5. ARQ jobs each get an `AsyncSession` created in `on_job_start` and closed in `on_job_complete`; no session object is shared across concurrent jobs
**Plans**: TBD

Plans:
- [ ] 14-01: TBD
- [ ] 14-02: TBD
- [ ] 14-03: TBD
- [ ] 14-04: TBD

### Phase 15: PG-Native Features
**Goal**: Search uses PostgreSQL's tsvector and pgvector capabilities with production-quality indexing; JSONB columns are queryable natively and all JSON helper methods are eliminated
**Depends on**: Phase 14 (async memory_store must be complete before hybrid search queries are added; tsvector and pgvector columns established in Phase 12)
**Requirements**: PGNAT-01, PGNAT-02, PGNAT-03
**Success Criteria** (what must be TRUE):
  1. Full-text search on memory_items and document_chunks uses tsvector + GIN index; all FTS5 virtual-table DDL and sqlite_master query code has been deleted (not just guarded)
  2. All 84 formerly-Text JSON columns are stored as JSONB and all `get_*/set_*` JSON helper methods have been removed — application code accesses JSONB attributes directly
  3. Vector search on memory_items uses pgvector `<=>` operator with an HNSW index; all sqlite-vec LargeBinary float-encoding code has been deleted
  4. Hybrid tsvector + pgvector search with Reciprocal Rank Fusion (RRF) is available in memory_store as a single server-side SQL CTE, replacing the Python-side `_hybrid_merge()` function
**Plans**: TBD

Plans:
- [ ] 15-01: TBD
- [ ] 15-02: TBD

### Phase 16: Model Refactoring
**Goal**: All 46 data models have explicit relationship loading, correct column types, and normalized schema — no schema debt from the organic SQLite-era growth remains
**Depends on**: Phase 15 (PG-native features must be stable; data must already be on PostgreSQL before NOT NULL constraints are enforced against migrated rows)
**Requirements**: MODEL-01, MODEL-02, MODEL-03
**Success Criteria** (what must be TRUE):
  1. Every ORM relationship has `lazy="raise"` and every code path that accesses a relationship uses explicit `selectinload` or `joinedload` — confirmed by a test that loads every model relationship and asserts no `MissingGreenlet` is raised
  2. The `is_active` column is stored as Boolean (not Integer) with all 5 call sites in research_store updated; all nullable columns that are semantically required have NOT NULL constraints; status, confidence, and pii_risk columns have CHECK constraints
  3. Author rows are stored in a normalized Authors table with FK references replacing inline denormalized author strings; all redundant columns identified in the refactoring audit are removed or merged into canonical columns
**Plans**: TBD

Plans:
- [ ] 16-01: TBD
- [ ] 16-02: TBD

### Phase 17: Data Migration & CI
**Goal**: Existing SQLite users can migrate their data to PostgreSQL without data loss; CI validates all migrations and runs every test against PostgreSQL; operational observability is in place
**Depends on**: Phase 16 (final schema must be stable before migration tooling targets it; Phase 13 for CI test infrastructure)
**Requirements**: PGINFRA-03, PGINFRA-04, CI-01, CI-02, CI-03, MON-01, MON-02, MON-03
**Success Criteria** (what must be TRUE):
  1. A developer with an existing SQLite database can run a documented pgloader command and land all relational data in PostgreSQL with FK integrity verified before (PRAGMA integrity_check) and after (FK violation report)
  2. Embedding vectors stored as sqlite-vec LargeBinary float bytes are re-encoded as pgvector float arrays during migration — vector search returns the same top-K results before and after on a sample query set
  3. GitHub Actions runs all tests against a PostgreSQL service container and `pytest -m postgres` is a required CI gate that blocks merge on failure
  4. CI validates `alembic upgrade head` on a fresh database and `alembic downgrade -1` (one revision rollback) without error on every push
  5. `GET /api/health/db` returns current connection status and Alembic migration revision; any SQL query exceeding the configured slow-query threshold emits a WARNING-level log entry with query text and duration
  6. AsyncEngine pool metrics (pool_size, checkedout, overflow) are queryable via an API endpoint
**Plans**: TBD

Plans:
- [ ] 17-01: TBD
- [ ] 17-02: TBD
- [ ] 17-03: TBD

## Progress

**Execution Order:**
Phases execute in milestone order: 1-6 (v1.0) -> 7-11 (v1.1) -> 18-23 (v1.2) -> 12-17 (v2.0)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. MCP Server Setup | v1.0 | — | Complete | 2026-03-13 |
| 2. Core Paper Tools | v1.0 | 3/3 | Complete | 2026-03-14 |
| 3. Remaining MCP Tools | v1.0 | 3/3 | Complete | 2026-03-14 |
| 4. MCP Resources | v1.0 | 2/2 | Complete | 2026-03-14 |
| 5. Transport & Entry Point | v1.0 | 1/1 | Complete | 2026-03-14 |
| 6. Agent Skills | v1.0 | 1/1 | Complete | 2026-03-14 |
| 7. EventBus + SSE Foundation | v1.1 | 2/2 | Complete | 2026-03-14 |
| 8. Agent Event Vocabulary | 1/2 | In Progress|  | - |
| 9. Three-Panel Dashboard | 2/2 | Complete   | 2026-03-15 | - |
| 10. Agent Board + Codex Bridge | 3/3 | Complete   | 2026-03-15 | - |
| 11. DAG Visualization | v1.1 | 0/? | Not started | - |
| 18. Adapter Foundation | v1.2 | 0/? | Not started | - |
| 19. Activity Stream + Session Management | v1.2 | 0/? | Not started | - |
| 20. Chat + Control Surface | v1.2 | 0/? | Not started | - |
| 21. Visualization Panels | v1.2 | 0/? | Not started | - |
| 22. Additional Agent Adapters | v1.2 | 0/? | Not started | - |
| 23. HITL + Domain Enrichment | v1.2 | 0/? | Not started | - |
| 12. PG Infrastructure & Schema | v2.0 | 0/? | Not started | - |
| 13. Test Infrastructure | v2.0 | 0/? | Not started | - |
| 14. Async Data Layer | v2.0 | 0/? | Not started | - |
| 15. PG-Native Features | v2.0 | 0/? | Not started | - |
| 16. Model Refactoring | v2.0 | 0/? | Not started | - |
| 17. Data Migration & CI | v2.0 | 0/? | Not started | - |
