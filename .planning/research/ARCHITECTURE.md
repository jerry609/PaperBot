# Architecture Patterns

**Domain:** Agent orchestration dashboard + Codex subagent bridge for existing PaperBot app
**Researched:** 2026-03-14

## Current Architecture Snapshot

PaperBot already has significant agent infrastructure. Understanding what exists is critical to avoid duplication.

### What Already Exists

| Component | Location | What It Does |
|-----------|----------|-------------|
| `AgentEventEnvelope` | `application/collaboration/message_schema.py` | Unified event schema with run_id/trace_id/span_id, workflow/stage/attempt, agent_name/role, typed payload, metrics, tags |
| `EventLogPort` | `application/ports/event_log_port.py` | Protocol with `append()`, `stream(run_id)`, `close()` |
| `SqlAlchemyEventLog` | `infrastructure/event_log/sqlalchemy_event_log.py` | DB persistence with `list_runs()`, `list_events()` beyond port interface |
| `CompositeEventLog` | `infrastructure/event_log/composite_event_log.py` | Tee to multiple backends (logging + SQLAlchemy) |
| `AgentRunModel` / `AgentEventModel` | `infrastructure/stores/models.py` | SQLAlchemy models for run rows and event rows |
| `ClaudeCommander` | `infrastructure/swarm/claude_commander.py` | Task decomposition, prompt building, review, wisdom accumulation |
| `CodexDispatcher` | `infrastructure/swarm/codex_dispatcher.py` | OpenAI API dispatch, file extraction/persistence, review doc generation |
| `agent_board` routes | `api/routes/agent_board.py` | Full Kanban lifecycle: sessions, plan, run, execute, dispatch, review (SSE) |
| `runs` routes | `api/routes/runs.py` | Event replay: `GET /runs`, `GET /runs/{run_id}/events` |
| `StreamEvent` + `wrap_generator` | `api/streaming.py` | SSE envelope with workflow/run_id/trace_id/seq/phase/ts, heartbeat, timeout |
| Studio page | `web/src/app/studio/page.tsx` | Resizable panels, PaperGallery, ReproductionLog (with agent_board view mode), FilesPanel, ChatHistoryPanel |

### Key Insight: The Gap Is NOT Infrastructure

The existing codebase already has event logging, SSE streaming, agent dispatch, and a board route. The gap is:

1. **No real-time push** -- `runs` routes are pull-only (GET). No SSE subscription for live event tailing.
2. **No Codex CLI bridge** -- `CodexDispatcher` calls OpenAI chat API, not the Codex CLI. The v1.1 milestone needs a Claude Code custom agent definition that invokes Codex CLI as a subprocess.
3. **No three-panel IDE layout** -- Studio has resizable panels but the layout is paper-gallery/reproduction-log/files, not tasks/activity/files.
4. **No event type extensions** -- `AgentEventEnvelope.type` supports `tool_call/tool_result/insight/fact/score/error/stage_event` but not `task_started/task_completed/file_changed/agent_lifecycle`.
5. **Agent board is paper-specific** -- `BoardSession` requires `paper_id` and `context_pack_id`. The dashboard needs to work for any agent workflow.

## Recommended Architecture

### New Components (must create)

| Component | Location | Purpose |
|-----------|----------|---------|
| `codex-worker.md` | `.claude/agents/codex-worker.md` | Claude Code custom agent definition that bridges to Codex CLI |
| `EventBus` | `application/services/event_bus.py` | In-process pub/sub: `EventLogPort.append()` tees to subscribers for SSE push |
| `event_stream` route | `api/routes/event_stream.py` | `GET /api/events/stream?run_id=X` -- SSE endpoint that subscribes to EventBus |
| `AgentDashboardPage` | `web/src/app/studio/page.tsx` (replace) | Three-panel layout: TaskPanel, ActivityPanel, FilesPanel |
| `useEventStream` hook | `web/src/hooks/useEventStream.ts` | React hook consuming SSE event stream with reconnection |
| `agent-dashboard-store` | `web/src/lib/store/agent-dashboard-store.ts` | Zustand store: tasks, events, file tree, active run |

### Modified Components (extend existing)

| Component | Change | Why |
|-----------|--------|-----|
| `AgentEventEnvelope` | Add new type constants: `task_queued`, `task_started`, `task_completed`, `task_failed`, `file_created`, `file_modified`, `agent_spawned`, `agent_exited` | Dashboard needs richer event vocabulary |
| `EventLogPort` | Add optional `subscribe(run_id) -> AsyncGenerator` method | Enable push-based SSE streaming |
| `agent_board` routes | Generalize: make `paper_id` optional, add `workflow_type` field | Support non-paper agent workflows |
| `streaming.py` | Add `StandardEvent.AGENT` and `StandardEvent.TASK` event kinds | Dashboard SSE needs agent-specific canonical events |
| `CompositeEventLog` | Wire EventBus as a backend | Events flow to SSE subscribers automatically |

### Components to NOT Create

| Avoided Component | Why |
|-------------------|-----|
| Custom agent runtime | Host agents (Claude Code, Codex) own orchestration. PaperBot observes, does not orchestrate. |
| WebSocket server | SSE already works, is simpler, and matches existing infrastructure. No bidirectional need. |
| Separate event DB | Existing `AgentEventModel` + `AgentRunModel` tables are sufficient. Add columns, not tables. |
| Per-agent adapters | One MCP surface serves all agents. Dashboard reads events, does not care which agent wrote them. |

## Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `codex-worker.md` | Agent definition file consumed by Claude Code. Contains instructions for Codex CLI invocation, MCP tool usage, event reporting. | Claude Code runtime (reads file), PaperBot MCP server (calls tools) |
| `EventBus` | Distributes events to SSE subscribers in real-time. Wraps `asyncio.Queue` per subscriber. | `CompositeEventLog` (receives events), `event_stream` route (pushes to clients) |
| `event_stream` route | SSE endpoint. Client connects, receives real-time events for a run_id. | `EventBus` (subscribes), frontend `useEventStream` (consumes) |
| `AgentDashboardPage` | Three-panel UI. Left: task list/control. Center: live activity feed. Right: file tree + Monaco. | `agent-dashboard-store` (state), `useEventStream` (events), existing API routes (actions) |
| `agent-dashboard-store` | Client-side state: task list, event log, selected task, file tree, run status. | Dashboard components (read/write), SSE hook (event ingestion) |

## Data Flow

### Event Flow (write path)

```
Agent (Claude Code / Codex)
  |
  v
MCP Tool Call: log_event(type, payload)    <-- or direct API POST
  |
  v
FastAPI route handler
  |
  v
CompositeEventLog.append(AgentEventEnvelope)
  |
  +---> LoggingEventLog (stdout/file)
  +---> SqlAlchemyEventLog (DB persistence)
  +---> EventBus.publish(event)            <-- NEW
          |
          v
        Per-subscriber asyncio.Queue
          |
          v
        event_stream SSE route yields to browser
```

### Event Flow (read path -- dashboard)

```
Browser connects: GET /api/events/stream?run_id=X
  |
  v
event_stream route subscribes to EventBus for run_id
  |
  v
EventBus yields events as they arrive (asyncio.Queue)
  |
  v
SSE frames via wrap_generator() (existing infrastructure)
  |
  v
useEventStream hook receives, dispatches to Zustand store
  |
  v
React components re-render (TaskPanel, ActivityPanel, FilesPanel)
```

### Task Control Flow

```
User clicks "Start Task" in TaskPanel
  |
  v
POST /api/agent-board/tasks/{id}/execute   (existing endpoint)
  |
  v
_execute_task_stream yields StreamEvents    (existing)
  |
  v
Each StreamEvent also appended to EventLog  <-- NEW: bridge existing stream to EventLog
  |
  v
EventBus distributes to dashboard SSE subscribers
```

## Patterns to Follow

### Pattern 1: EventBus as CompositeEventLog Backend

The EventBus is not a separate system. It plugs into the existing `CompositeEventLog` as a third backend alongside `LoggingEventLog` and `SqlAlchemyEventLog`.

```python
# In DI container setup
event_bus = EventBus()
event_log = CompositeEventLog([
    LoggingEventLog(),
    SqlAlchemyEventLog(db_url),
    event_bus,  # NEW -- implements EventLogPort
])
```

This means every existing `event_log.append()` call automatically pushes to SSE subscribers. Zero changes to existing code that emits events.

### Pattern 2: Agent Definition as Markdown File

The Codex bridge is a `.claude/agents/codex-worker.md` file -- not server-side code. Claude Code reads this file and knows how to spawn Codex CLI for delegated tasks.

```markdown
# codex-worker.md structure
- Role description (coding worker)
- Available MCP tools (from PaperBot MCP server)
- Event reporting instructions (call log_event tool)
- File output conventions
- Error handling protocol
```

### Pattern 3: Extend AgentEventEnvelope Types, Not Schema

Do not add new fields to `AgentEventEnvelope`. Instead, use the existing `type` field with new constants and put structured data in `payload`. The envelope schema is already flexible enough.

```python
# Good: new type constant, structured payload
make_event(
    type="task_started",
    payload={"task_id": "...", "assignee": "codex-worker"},
    ...
)

# Bad: adding new fields to the dataclass
@dataclass
class AgentEventEnvelope:
    task_id: str = ""  # NO -- use payload dict
```

### Pattern 4: Studio Page Evolution, Not Replacement

The studio page already has:
- `ResizablePanelGroup` with horizontal orientation
- View modes including `agent_board`
- `PaperGallery` for paper selection
- `ReproductionLog` component with multiple view modes

Extend `ReproductionLog` to support a new `dashboard` view mode that renders the three-panel layout. Do not create a separate page.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Dual Event Systems
**What:** Creating a separate "dashboard event" model alongside `AgentEventEnvelope`.
**Why bad:** Two sources of truth. Events logged to one system not visible in the other.
**Instead:** Use `AgentEventEnvelope` for everything. Add type constants, not new models.

### Anti-Pattern 2: Polling for Real-Time Updates
**What:** Frontend polling `GET /runs/{id}/events` every N seconds.
**Why bad:** Latency, wasted requests, poor UX for "live" feel.
**Instead:** SSE subscription via EventBus. Fall back to polling only if SSE disconnects.

### Anti-Pattern 3: Server-Side Codex CLI Management
**What:** PaperBot server spawning/managing Codex CLI processes.
**Why bad:** PaperBot is a skill provider, not an agent runtime. Claude Code owns process management.
**Instead:** Agent definition file tells Claude Code how to use Codex. PaperBot only observes events.

### Anti-Pattern 4: Tightly Coupling Dashboard to Paper2Code
**What:** Dashboard only works when a paper_id and context_pack_id are provided.
**Why bad:** Limits reuse. Agent workflows may not involve papers.
**Instead:** Make paper_id optional in BoardSession. Dashboard shows any run with agent events.

## Integration Points Summary

| Integration Point | Existing Component | New Component | Connection Type |
|-------------------|-------------------|---------------|-----------------|
| Event emission | `CompositeEventLog.append()` | `EventBus` (as backend) | EventBus implements `EventLogPort` |
| SSE streaming | `wrap_generator()` + `sse_response()` | `event_stream` route | Route uses `wrap_generator` with EventBus subscription |
| Task management | `agent_board` routes | Dashboard TaskPanel | Frontend calls existing REST endpoints |
| File viewing | `runbook` routes + `FilesPanel` | Dashboard FilesPanel | Reuse existing components |
| Run history | `runs` routes (`list_runs`, `list_events`) | Dashboard history view | Frontend calls existing GET endpoints |
| Agent definition | (none) | `.claude/agents/codex-worker.md` | Claude Code reads at agent spawn time |
| MCP tools | MCP server (v1.0 prerequisite) | Agent event logging tools | Agents call `log_event` MCP tool |

## Build Order (Dependency-Driven)

Phase order matters because of dependencies:

1. **EventBus + event_stream route** -- Everything else needs real-time push. Build this first.
   - Depends on: existing `EventLogPort`, `CompositeEventLog`, `wrap_generator`
   - Blocks: dashboard live updates, agent event visibility

2. **AgentEventEnvelope type extensions** -- Dashboard needs richer event vocabulary.
   - Depends on: nothing (additive constants)
   - Blocks: meaningful dashboard rendering

3. **Agent dashboard Zustand store + useEventStream hook** -- Frontend state management before UI.
   - Depends on: event_stream route (to consume)
   - Blocks: all dashboard UI components

4. **Three-panel dashboard layout** -- UI shell with TaskPanel, ActivityPanel, FilesPanel.
   - Depends on: store + hook (step 3), existing ResizablePanelGroup
   - Blocks: user-facing experience

5. **Generalize agent_board routes** -- Make paper_id optional, add workflow_type.
   - Depends on: nothing (backend refactor)
   - Blocks: non-paper agent workflows

6. **Codex worker agent definition** -- Claude Code custom agent file.
   - Depends on: MCP server (v1.0 milestone), event type extensions (step 2)
   - Blocks: Paper2Code overflow delegation

## Sources

- Codebase inspection: `src/paperbot/application/collaboration/message_schema.py` (AgentEventEnvelope schema)
- Codebase inspection: `src/paperbot/api/streaming.py` (SSE infrastructure)
- Codebase inspection: `src/paperbot/infrastructure/swarm/` (ClaudeCommander, CodexDispatcher)
- Codebase inspection: `src/paperbot/api/routes/agent_board.py` (existing board lifecycle)
- Codebase inspection: `src/paperbot/api/routes/runs.py` (event replay endpoints)
- Codebase inspection: `src/paperbot/infrastructure/event_log/` (event log implementations)
- Codebase inspection: `web/src/app/studio/page.tsx` (existing studio layout)
- Project requirements: `.planning/PROJECT.md` (v1.1 milestone definition)
