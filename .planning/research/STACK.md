# Technology Stack

**Project:** PaperBot v1.1 -- Agent Orchestration Dashboard + Codex Subagent Bridge
**Researched:** 2026-03-14 (updated with verified integration details)

## Principle: Minimal Additions

The existing stack is comprehensive. This milestone adds **zero new frameworks and zero new packages**. Every capability needed is either already installed or is a system-level tool (Codex CLI) that lives outside PaperBot. The biggest risk is adding unnecessary dependencies when the existing stack already covers the need.

## What You Already Have (DO NOT Add)

| Capability | Existing Solution | Version | Status |
|---|---|---|---|
| Resizable panels | `react-resizable-panels` | `^4.0.11` (latest: 4.7.2) | Installed. Use for three-panel IDE layout. |
| DAG visualization | `@xyflow/react` | `^12.10.0` | Installed. Use for agent task graph. |
| Code editor | `@monaco-editor/react` | `^4.7.0` | Installed. Reuse in file panel. |
| Terminal | `xterm` + `xterm-addon-fit` | `^5.3.0` / `^0.8.0` | Installed. Reuse in agent output panel. |
| SSE streaming | `sse-starlette` + `api/streaming.py` | `>=2.1.0` | Installed. Existing `wrap_generator` + `StreamEvent` envelope. |
| State management | `zustand` | `^5.0.9` | Installed. Use for agent event store. |
| Motion/animation | `framer-motion` | `^12.23.26` | Installed. Use for panel transitions, status indicators. |
| Icons | `lucide-react` | `^0.562.0` | Installed. Agent status icons. |
| Auth | `next-auth` | `5.0.0-beta.30` | Installed. Protect dashboard routes. |
| DB persistence | `SQLAlchemy` + `alembic` | `>=2.0.0` / `>=1.13.0` | Installed. Extend for agent event tables. |
| Task queue | `arq` + `redis` | `>=0.25.0` / `>=5.0.0` | Installed. Use for async Codex dispatch jobs if needed. |
| HTTP client | `httpx` | `>=0.27.0` | Installed. Webhook callbacks. |
| OpenAI SDK | `openai` | `>=1.0.0` | Installed. Used by existing `CodexDispatcher`. |
| Pub/sub (in-process) | `asyncio.Queue` | stdlib | No install. Per-subscriber SSE push channel. |

## New Additions Required

### Backend (Python): None

All backend needs are covered by existing deps. The Codex subagent bridge is a Claude Code custom agent file (`.claude/agents/codex-worker.md`), not Python server code. The dashboard backend extends existing SSE streaming and SQLAlchemy event logging.

### Frontend (web/): None

The three-panel layout uses `react-resizable-panels` (already at v4.0.11). Agent activity feed is a Zustand store consuming SSE via `EventSource` (native browser API). DAG view uses `@xyflow/react`. No new UI libraries needed.

### Claude Code Agent Definition (Configuration, NOT a dependency)

| File | Format | Purpose |
|---|---|---|
| `.claude/agents/codex-worker.md` | Markdown + YAML frontmatter | Custom subagent that Claude Code uses to delegate coding tasks to Codex CLI |

Claude Code custom agents are Markdown files with YAML frontmatter stored in `.claude/agents/`. They define a subagent's name, description, allowed tools, model, and system prompt. Claude Code loads them at session start (or via `/agents` reload). The codex-worker agent needs:

```yaml
---
name: codex-worker
description: Delegates coding tasks to OpenAI Codex CLI for parallel execution
tools: Bash, Read, Write, Glob
model: sonnet
---
```

The system prompt body instructs Claude Code how to:
1. Format `codex exec` invocations with appropriate flags
2. Parse JSONL output events from Codex
3. Report results back to the main Claude Code session
4. Log events to PaperBot via MCP `event_log` tool

### Codex CLI (System-level tool, NOT a project dependency)

| Tool | Install | Version | Purpose |
|---|---|---|---|
| `@openai/codex` | `npm i -g @openai/codex` | v0.98+ (Rust binary) | CLI that Claude Code invokes via Bash tool in codex-worker subagent |

**Key `codex exec` capabilities for subagent bridge:**

| Flag | Purpose | When to Use |
|---|---|---|
| `--json` | JSONL streaming output | Always. Parse structured events (thread.started, turn.completed, item.*, error). |
| `--full-auto` | No human approval for file writes | Always for delegated tasks. Codex runs in its own OS sandbox. |
| `--output-schema <file>` | Enforce JSON response shape | When structured results needed (e.g., generated file manifest). |
| `--model <model>` | Override model | Use gpt-5.2-codex for coding, gpt-5 for planning. |

Codex CLI v0.98+ is a Rust binary (95.7% Rust rewrite from the original Node.js). Starts in milliseconds with OS-level sandboxing (Landlock on Linux, Seatbelt on macOS).

## Integration Architecture

### Data Flow: Codex Subagent Bridge

```
Claude Code (host agent)
  |-- Loads .claude/agents/codex-worker.md
  |-- Detects Paper2Code overflow condition
  |-- Invokes codex-worker subagent (own context window)
  |
  +-- codex-worker subagent:
      |-- Runs: codex exec --json --full-auto "<prompt>"
      |-- Parses JSONL events from stdout
      |-- Reports results back to main Claude Code session
      |-- Logs events to PaperBot via MCP event_log tool
```

Note: The existing `CodexDispatcher` in `infrastructure/swarm/codex_dispatcher.py` uses the OpenAI Chat Completions API directly (single completion, no sandbox, no agentic loop). The subagent bridge via `codex exec` is fundamentally different: Codex gets its own sandbox, file system access, multi-turn agentic loop, and tool use. Keep both -- the dispatcher for lightweight API tasks, the subagent bridge for full agentic delegation.

### Data Flow: Agent Dashboard

```
Agent activity events arrive via two paths:

1. MCP tool calls (from Claude Code/Codex):
   POST /api/agent-events --> SQLAlchemy persistence
                          --> asyncio.Queue fan-out to SSE subscribers

2. Dashboard consumption:
   GET /api/agent-events/stream (SSE) --> wrap_generator from streaming.py
   GET /api/agent-events?run_id=X (REST) --> query persisted events

Frontend:
   Zustand useAgentEventStore
     |-- EventSource(/api/agent-events/stream)
     |-- Maintains: event buffer, run state map, task tree
     |
     +-- Three-panel layout (react-resizable-panels PanelGroup):
         |-- Left panel: Task/run list (run_id, agent_name, status, timestamps)
         |-- Center panel: Agent activity feed (chronological events, tool calls)
         |-- Right panel: File viewer (Monaco) + Terminal output (XTerm)
         |     (vertical nested PanelGroup for editor/terminal split)
```

### SSE Event Schema Extension

Extend existing `StandardEvent` enum in `api/streaming.py`:

```python
# Add to StandardEvent enum
AGENT_LIFECYCLE = "agent_lifecycle"     # spawn, complete, error, timeout
AGENT_TOOL_CALL = "agent_tool_call"     # tool invocations with args/result
AGENT_FILE_CHANGE = "agent_file_change" # file create/modify/delete
AGENT_TASK_STATUS = "agent_task_status" # queued, running, done, failed
```

These reuse the existing `StreamEvent.envelope` structure (`workflow`, `run_id`, `trace_id`, `seq`, `phase`, `ts`). The `run_id`/`trace_id`/`span_id` from `AgentEventEnvelope` in `application/collaboration/message_schema.py` provides correlation.

### Database Schema Extension

```sql
-- New table via Alembic migration. Reuses run_id/trace_id/span_id from AgentEventEnvelope.
CREATE TABLE agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    span_id TEXT,
    event_type TEXT NOT NULL,   -- lifecycle | tool_call | file_change | task_status
    agent_name TEXT NOT NULL,   -- "codex-worker", "claude-code", etc.
    data JSON NOT NULL,         -- event payload (flexible schema)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_agent_events_run_id ON agent_events(run_id);
CREATE INDEX ix_agent_events_trace_id ON agent_events(trace_id);
CREATE INDEX ix_agent_events_created ON agent_events(created_at);
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|---|---|---|---|
| Panel layout | `react-resizable-panels` (existing) | `allotment`, `react-split-pane` | Already installed at v4.0.11. Shadcn wraps it natively. Supports nested groups for IDE layout. |
| Real-time transport | SSE via `sse-starlette` (existing) | WebSockets (`socket.io`, `ws`) | SSE is simpler, already battle-tested in this codebase, sufficient for server-to-client push. No bidirectional need. |
| In-process pub/sub | `asyncio.Queue` (stdlib) | Redis pub/sub | Single-process is fine for v1.1. Redis adds operational complexity for zero benefit at current scale. |
| Codex invocation | `codex exec` via Claude Code subagent | Direct OpenAI API (`CodexDispatcher`) | `codex exec` gives Codex its own sandbox and agentic loop. API dispatch is a single completion, not an agent. |
| Agent event persistence | SQLAlchemy table (existing infra) | Separate event store (Kafka, EventStore) | Massive overkill. SQLAlchemy handles event volume at this scale. |
| State management | Zustand (existing) | Redux, Jotai, React Context | Already installed and used throughout the app. |
| Codex bridge location | `.claude/agents/codex-worker.md` | PaperBot server-side subprocess | Claude Code owns orchestration (project constraint). PaperBot is skill provider, not runtime. |
| SSE consumption | `EventSource` (native browser API) | `@microsoft/fetch-event-source` | Native API is sufficient. The MS library adds retry/POST support not needed here. |

## What NOT to Install

| Library | Why You Might Think You Need It | Why You Don't |
|---|---|---|
| `socket.io` / `ws` | "Real-time needs WebSockets" | SSE is sufficient for server-push. No client-to-server streaming needed for event display. |
| `@tanstack/react-query` | "Cache API data" | Zustand store handles event state. REST queries are simple fetch calls. |
| `langchain` / `langgraph` | "Agent orchestration framework" | Claude Code IS the orchestrator. PaperBot is skill provider. Adding an agent framework contradicts the architecture. |
| `bull` / `bullmq` | "Need job queue" | ARQ + Redis already installed and working. |
| `prisma` / `drizzle` | "Frontend needs ORM" | Next.js talks to FastAPI, not directly to DB. |
| `@openai/codex` as npm dep in web/ | "Need Codex SDK" | Codex is a CLI tool on the dev machine, invoked by Claude Code. PaperBot never calls Codex. |
| `claude-agent-sdk` | "Need Claude SDK" | Not available on PyPI (noted in requirements.txt). Claude Code is the host; PaperBot provides MCP tools. |
| `react-virtualized` / `react-window` | "Virtualize long event lists" | Premature optimization. Start with simple scroll. Add if event volume causes perf issues (unlikely at v1.1 scale). |

## Installation

```bash
# Backend -- no changes needed
pip install -e ".[dev]"

# Frontend -- no changes needed
cd web && npm install

# System-level: Codex CLI (on developer machines only, not project dep)
npm i -g @openai/codex
codex --version  # verify v0.98+
```

## Version Compatibility Notes

| Existing Dep | Current Pin | Latest | Action |
|---|---|---|---|
| `react-resizable-panels` | `^4.0.11` | 4.7.2 | Compatible. Semver minor. No action. |
| `@xyflow/react` | `^12.10.0` | Stable | No action. |
| `sse-starlette` | `>=2.1.0` | Stable | No action. |
| `SQLAlchemy` | `>=2.0.0` | Stable | No action. Alembic migration handles schema. |
| `openai` | `>=1.0.0` | Stable | Keep for API-mode CodexDispatcher. |
| `zustand` | `^5.0.9` | Stable | No action. |

## Sources

- [Claude Code Custom Subagents Docs](https://code.claude.com/docs/en/sub-agents) -- MEDIUM confidence, official Anthropic docs
- [Codex CLI Non-Interactive Mode](https://developers.openai.com/codex/noninteractive/) -- HIGH confidence, official OpenAI docs
- [Codex CLI Reference](https://developers.openai.com/codex/cli/reference) -- HIGH confidence, official OpenAI docs
- [Codex Multi-Agent Docs](https://developers.openai.com/codex/multi-agent/) -- MEDIUM confidence, experimental feature
- [react-resizable-panels npm](https://www.npmjs.com/package/react-resizable-panels) -- HIGH confidence, npm registry
- [Shadcn Resizable Component](https://www.shadcn.io/ui/resizable) -- HIGH confidence, wraps react-resizable-panels
- Codebase: `web/package.json` -- confirmed all frontend deps
- Codebase: `requirements.txt` -- confirmed all backend deps
- Codebase: `src/paperbot/api/streaming.py` -- confirmed SSE envelope structure
- Codebase: `src/paperbot/infrastructure/swarm/codex_dispatcher.py` -- confirmed existing Codex API integration
- Codebase: `src/paperbot/application/collaboration/message_schema.py` -- confirmed AgentEventEnvelope schema
