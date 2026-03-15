---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Agent Orchestration Dashboard
status: verifying
stopped_at: Completed 10-01-PLAN.md
last_updated: "2026-03-15T04:10:04.279Z"
last_activity: 2026-03-15 — Completed 09-02-PLAN.md tasks 1-2 (three-panel agent dashboard UI)
progress:
  total_phases: 21
  completed_phases: 7
  total_plans: 16
  completed_plans: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-15)

**Core value:** Paper-specific capability layer surfaced as standard MCP tools + agent-agnostic dashboard
**Current focus:** v1.2 DeepCode Agent Dashboard -- roadmap created, ready for phase planning

## Current Position

Phase: 9 of 17 (Three-Panel Dashboard)
Plan: 2 completed
Status: Active — awaiting human verification checkpoint (Task 3)
Last activity: 2026-03-15 — Completed 09-02-PLAN.md tasks 1-2 (three-panel agent dashboard UI)

## Milestones

| Milestone | Phases | Status |
|-----------|--------|--------|
| v1.0 MCP Server | 1-6 | In progress (phases 3, 6 remaining) |
| v1.1 Agent Orchestration Dashboard | 7-11 | Planned (EventBus/SSE partially built) |
| v1.2 DeepCode Agent Dashboard | 18-23 | Roadmap created 2026-03-15 |
| v2.0 PostgreSQL Migration | 12-17 | Roadmap created 2026-03-14 |

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 6 min
- Total execution time: 0.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 2/3 | 12min | 6min |
| 03-remaining-mcp-tools P01 | 1 | 2min | 2min |
| 03-remaining-mcp-tools P02 | 1 | 2min | 2min |
| 03-remaining-mcp-tools P03 | 1 | 5min | 5min |
| 04-mcp-resources P01 | 1 | 3min | 3min |
| 04-mcp-resources P02 | 1 | 2min | 2min |
| 05-transport-entry-point P01 | 1 | 3min | 3min |

**Recent Trend:**
- Last 3 plans: 3min, 2min, 3min
- Trend: Stable
| Phase 06-agent-skills P01 | 3 | 2 tasks | 5 files |
| Phase 07-eventbus-sse-foundation P01 | 3 | 2 tasks | 3 files |
| Phase 07 P02 | 4 | 2 tasks | 3 files |
| Phase 08-agent-event-vocabulary P01 | 2min | 2 tasks | 4 files |
| Phase 08-agent-event-vocabulary P02 | 6min | 2 tasks | 10 files |
| Phase 09 P01 | 3 | 2 tasks | 8 files |
| Phase 09 P02 | 8min | 2 tasks | 6 files |
| Phase 09-three-panel-dashboard P02 | 8min | 3 tasks | 6 files |
| Phase 10-agent-board-codex-bridge P01 | 5min | 1 tasks | 6 files |

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v2.0 roadmap] Phase 12 (infra) before Phase 13 (tests) -- tests need real PG schema to verify against
- [v2.0 roadmap] Phase 13 (tests) before Phase 14 (async stores) -- CI greenlight meaningless without PG fixture
- [v2.0 roadmap] lazy="raise" applied at Phase 14 start (before first store conversion), not as separate phase
- [v2.0 roadmap] anyio.to_thread.run_sync removed per-store during Phase 14, not as final cleanup sweep
- [v2.0 roadmap] PGINFRA-03/04 (data migration tooling) deferred to Phase 17 -- schema must be final first
- [v1.1 init] EventBus as CompositeEventLog backend -- extends existing event system, not parallel
- [v1.1 init] Codex bridge is a .claude/agents/ file, not PaperBot server code
- [Phase 04-mcp-resources]: Track resources use anyio.to_thread.run_sync() because stores are sync (v2.0 removes this)
- [Phase 05-transport-entry-point]: default HTTP port 8001 avoids FastAPI conflict; serve.py redirects logging to stderr for stdio purity
- [Phase 06-agent-skills]: Skill tool names copied verbatim from @mcp.tool() source to prevent name mismatch bugs
- [Phase 06-agent-skills]: Degraded Mode section included in all four skills — LLM tools return degraded=True when API key missing
- [Phase 07-01]: EventBusEventLog uses drop-oldest backpressure (get_nowait+put_nowait) so append() never blocks the producer
- [Phase 07-01]: AgentEventEnvelope serialized once via .to_dict() in append(); fan-out distributes the dict (no re-serialization)
- [Phase 07-01]: stream() returns iter(()) — EventBusEventLog is a live delivery channel, not a historical store
- [Phase 07-02]: Late import of EventBusEventLog inside _get_bus() prevents circular import (events.py loaded at app creation before bus is wired)
- [Phase 07-02]: No wrap_generator() in events.py: events carry own AgentEventEnvelope fields; second envelope layer would confuse consumers
- [Phase 07-02]: asyncio.wait_for(q.get(), timeout=15.0) drives both delivery and idle heartbeat at single await point
- [Phase 08-agent-event-vocabulary]: EventType is a plain class with string annotations (not enum) — constants usable as str directly without .value unwrapping
- [Phase 08-agent-event-vocabulary]: make_tool_call_event auto-generates run_id/trace_id if not provided — callers can omit for standalone tool logging
- [Phase 08-agent-event-vocabulary P02]: Zustand 5 create() single-call form (no curry) for non-persisted stores — store test reset uses getInitialState() not plain setState
- [Phase 08-agent-event-vocabulary P02]: useAgentEvents hook mounted exactly once at page root — child components read Zustand store (no duplicate SSE connections)
- [Phase 08-agent-event-vocabulary P02]: tool_call type added to TOOL_TYPES in parsers.ts alongside tool_result and tool_error (handles pre-result events)
- [Phase 09]: [Phase 09-01] parseFileTouched handles two event shapes: explicit file_change type and tool_result with payload.tool=='write_file' (fallback path)
- [Phase 09]: [Phase 09-01] Store 20-run eviction uses Object.keys(updated)[0] deletion; path dedup within run_id is first-wins
- [Phase 09]: [Phase 09-02] TasksPanel derives run_ids from feed[].raw.run_id (ActivityFeedItem.raw has run_id; top-level item does not)
- [Phase 09]: [Phase 09-02] FileListPanel toggles in-place between file list and InlineDiffPanel via Zustand selectedFile (no URL/router change)
- [Phase 09]: [Phase 09-02] AgentStatusPanel compact=false default preserves backward compatibility with /agent-events page
- [Phase 09-three-panel-dashboard]: Human visual verification PASSED: dashboard layout, resizable panels, sidebar nav, and empty states confirmed working
- [Phase 10-agent-board-codex-bridge]: [Phase 10-01] _emit_codex_event uses _get_event_log_from_container() lazy helper for testability without live FastAPI app
- [Phase 10-agent-board-codex-bridge]: [Phase 10-01] _should_overflow_to_codex is a stub only — actual Codex overflow wiring in Orchestrator.run() deferred to a later plan

### Pending Todos

None.

- [v1.2 roadmap] Adapter layer (Phase 18) gates all v1.2 features; nothing ships without BaseAgentAdapter + ClaudeCodeAdapter
- [v1.2 roadmap] SSE id: field fix and run-scoped filtering land in Phase 19, not retrofitted later
- [v1.2 roadmap] CodexAdapter and OpenCodeAdapter deferred to Phase 22 until ClaudeCodeAdapter proven in production
- [v1.2 roadmap] CTRL-03 (HITL) and DOMAIN-01/02 grouped in Phase 23 -- both require stable panels + adapter state machine
- [v1.2 roadmap] CHAT_DELTA events excluded from ring buffer (live fan-out only) to prevent buffer saturation at 40 tokens/sec

### Blockers/Concerns

- v1.0 MCP server (phases 1-6) must be functional before v1.1 work begins
- v1.1 must complete before v2.0 work begins (Phase 12 depends on Phase 11)
- [v2.0] memory_store (Phase 14 Group 2) is highest-complexity store: FTS5 + sqlite-vec + hybrid search + MCP connections; warrants dedicated mini-plan before that group ships
- [v2.0] Phase 17 data migration FK violation profile in existing SQLite DBs is unknown -- run PRAGMA integrity_check on representative DB before Phase 17 planning is finalized

## Session Continuity

Last session: 2026-03-15T04:10:04.271Z
Stopped at: Completed 10-01-PLAN.md
Resume file: None
