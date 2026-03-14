# Research Summary: Agent Orchestration Dashboard + Codex Bridge

**Domain:** Agent orchestration dashboard and Codex subagent bridge for PaperBot
**Researched:** 2026-03-14 (updated with verified Codex CLI and Claude Code agent findings)
**Overall confidence:** HIGH

## Executive Summary

PaperBot's existing codebase already contains roughly 60% of the infrastructure needed for the v1.1 agent orchestration dashboard. The `AgentEventEnvelope` provides a flexible event schema with run_id/trace_id/span_id tracing. The `EventLogPort` abstraction with `CompositeEventLog` supports tee-ing events to multiple backends. SSE streaming via `wrap_generator` handles heartbeats, timeouts, and envelope wrapping. The `agent_board` routes implement a complete Kanban lifecycle with Claude/Codex dispatch, AI review, and human review. The studio page already has resizable panels, Monaco editor, file browser, and multiple view modes including `agent_board`.

The critical gap is real-time push. The current architecture is pull-only for event replay (`GET /runs/{id}/events`). Adding an `EventBus` (in-process `asyncio.Queue` per subscriber) as a third `CompositeEventLog` backend solves this with zero changes to existing event-emitting code. Every existing `event_log.append()` call automatically pushes to SSE subscribers.

The Codex subagent bridge is a `.claude/agents/codex-worker.md` file -- a Claude Code custom agent definition using YAML frontmatter (name, description, tools, model) and a Markdown system prompt. This is not PaperBot server code. Claude Code loads it at session start and invokes `codex exec --json --full-auto` via Bash tool for delegated tasks. The Codex CLI is now a Rust binary (v0.98+, 95.7% Rust rewrite) that streams JSONL events (thread.started, turn.completed, item.*, error) to stdout. The existing `CodexDispatcher` (OpenAI Chat API) stays for lightweight API-only tasks; the subagent bridge adds full agentic Codex CLI delegation with OS-level sandboxing.

No new packages are required. Zero new npm dependencies, zero new pip dependencies. The entire milestone builds on existing FastAPI SSE, SQLAlchemy event tables, Zustand stores, react-resizable-panels, @xyflow/react, Monaco, and XTerm.

## Key Findings

**Stack:** Zero new dependencies. All needs covered by existing packages. Codex CLI is a system-level tool, not a project dependency.
**Architecture:** EventBus as CompositeEventLog backend enables real-time push with zero changes to existing event emission code.
**Critical pitfall:** Building a second event system alongside AgentEventEnvelope. Extend types, not schema.
**Codex bridge:** Claude Code custom agent definition (`.claude/agents/codex-worker.md`) using `codex exec --json --full-auto`. NOT server-side subprocess management.
**Codex CLI status:** Rust binary v0.98+ with JSONL streaming, OS sandboxing. Multi-agent features are experimental -- do not depend on them.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **EventBus + SSE subscription endpoint** - Foundation for all real-time features
   - Addresses: Live event stream, agent activity visibility
   - Avoids: Polling anti-pattern, SSE connection exhaustion (bounded queues + cleanup)

2. **AgentEventEnvelope type extensions** - Richer event vocabulary for dashboard
   - Addresses: Task lifecycle events, tool call logging, file change tracking
   - Avoids: Second event system pitfall (extend types, not schema)

3. **Agent dashboard store + useEventStream hook** - Frontend state management
   - Addresses: Client-side event ingestion, state derivation for UI
   - Avoids: Re-implementing AgentBoard logic (embed existing component)

4. **Three-panel IDE layout** - User-facing dashboard experience
   - Addresses: Tasks panel + activity feed + file viewer
   - Avoids: Studio page regression (new view mode, not page replacement)

5. **Generalize agent_board routes** - Support non-paper workflows
   - Addresses: Make paper_id optional, add workflow_type
   - Avoids: Tight coupling to Paper2Code

6. **Codex worker agent definition** - Claude Code subagent bridge
   - Addresses: Codex CLI delegation via `codex exec`, overflow workflow
   - Avoids: Making PaperBot an agent runtime (agent definition is a file, not server code)

**Phase ordering rationale:**
- EventBus must come first because all dashboard components depend on real-time event push
- Event type extensions are additive and low-risk, but needed before dashboard can render meaningful content
- Frontend store before UI because components need state to render
- Three-panel layout depends on store being ready
- Agent board generalization is a backend refactor independent of frontend
- Codex bridge depends on MCP server (v1.0) and is parallel to dashboard work

**Research flags for phases:**
- Phase 1 (EventBus): Standard asyncio pattern. Unlikely to need deeper research.
- Phase 4 (Three-panel): react-resizable-panels v4.0.11+ supports nested PanelGroups for IDE layouts. Shadcn wraps this natively. No deeper research needed.
- Phase 6 (Codex bridge): RESOLVED -- Claude Code agent files use YAML frontmatter (name, description, tools, model) + Markdown body. Codex CLI `codex exec --json` streams JSONL with event types: thread.started, turn.started, turn.completed, turn.failed, item.*, error. No deeper research needed at build time beyond testing actual CLI output.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new deps. All verified in package.json and requirements.txt. |
| Features | HIGH | Table stakes verified against existing components. Existing asset inventory is thorough. |
| Architecture | HIGH | All integration points verified by reading source code. EventBus pattern is well-understood. |
| Pitfalls | HIGH | Updated with Codex CLI version mismatch risk, SSE reconnection gap, agent definition scope creep. |
| Codex bridge | MEDIUM | Claude Code agent format verified via official docs. Codex CLI JSONL schema verified. Multi-agent features are experimental -- avoid reliance. |

## Gaps to Address

- AgentBoard.tsx component API: need to verify props/state interface before embedding in three-panel layout (not read in this research pass due to context limits -- straightforward to inspect during implementation)
- studio-store.ts: need to understand existing Zustand store shape before designing agent-dashboard-store extension (implementation-time task)
- Codex CLI JSONL output: verified event types from docs, but actual output format should be tested with a real `codex exec --json` call before building parser

## Files Created/Updated

| File | Purpose |
|------|---------|
| `.planning/research/SUMMARY.md` | This file -- executive summary with roadmap implications |
| `.planning/research/STACK.md` | Technology recommendations (zero new deps, integration architecture, Codex CLI details) |
| `.planning/research/FEATURES.md` | Feature landscape (pre-existing, not updated this pass) |
| `.planning/research/ARCHITECTURE.md` | Architecture patterns, data flow, integration points (pre-existing, not updated this pass) |
| `.planning/research/PITFALLS.md` | Domain pitfalls with Codex CLI and Claude Code agent specifics |
