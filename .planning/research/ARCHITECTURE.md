# Architecture Patterns: v1.2 Agent-Agnostic Dashboard

**Domain:** Agent-agnostic proxy/dashboard — multi-agent code agent integration
**Researched:** 2026-03-15
**Confidence:** HIGH

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER (external)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ Claude Code │  │  Codex CLI  │  │  OpenCode   │  ← CLI processes  │
│  │ (NDJSON via │  │ (JSONL via  │  │ (HTTP API + │                   │
│  │ stream-json)│  │   --json)   │  │  ACP stdio) │                   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         │ stdout/stdin   │                 │                         │
└─────────┼────────────────┼─────────────────┼─────────────────────────┘
          │                │                 │ subprocess / HTTP
┌─────────▼────────────────▼─────────────────▼─────────────────────────┐
│                     ADAPTER LAYER (new)                               │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                   AgentAdapterRegistry                          │   │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌────────────────┐  │   │
│  │  │ClaudeCodeAdapter │ │  CodexAdapter    │ │OpenCodeAdapter │  │   │
│  │  │ (subprocess +    │ │ (subprocess +    │ │ (HTTP client)  │  │   │
│  │  │  NDJSON parser)  │ │  JSONL parser)   │ │                │  │   │
│  │  └────────┬─────────┘ └────────┬─────────┘ └───────┬────────┘  │   │
│  └───────────┼────────────────────┼────────────────────┼───────────┘   │
│              └────────────────────┼────────────────────┘               │
│  ┌─────────────────────────────────▼────────────────────────────────┐  │
│  │             AgentAdapter (unified interface)                      │  │
│  │  send_message(msg) -> AsyncIterator[AgentEventEnvelope]           │  │
│  │  send_control(cmd: ControlCommand) -> None                        │  │
│  │  get_status() -> AgentStatus                                      │  │
│  │  stop() -> None                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
          │ normalized AgentEventEnvelope stream
┌─────────▼─────────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER (existing + extended)              │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │               AgentProxyService (new)                            │ │
│  │  - Owns active adapter instance per session                      │ │
│  │  - Routes normalized events to EventBusEventLog                  │ │
│  │  - Handles lifecycle (start, stop, crash-recover, reconnect)     │ │
│  │  - Persists session registry in DB                               │ │
│  └──────────────────────────┬───────────────────────────────────────┘ │
│                             │ event_log.append()                      │
│  ┌──────────────────────────▼───────────────────────────────────────┐ │
│  │    EventBusEventLog (Phase 7 - existing, unmodified)             │ │
│  │    CompositeEventLog + asyncio.Queue fan-out                     │ │
│  └──────────────────────────┬───────────────────────────────────────┘ │
└─────────────────────────────┼─────────────────────────────────────────┘
                              │ GET /api/events/stream (SSE)
┌─────────────────────────────▼─────────────────────────────────────────┐
│                     API LAYER (FastAPI - extended)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ ┌───────────┐ │
│  │POST /api/    │ │POST /api/    │ │GET /api/agents/ │ │GET /api/  │ │
│  │agent/chat    │ │agent/control │ │{id}/status      │ │events/    │ │
│  │(proxy chat   │ │(stop/restart │ │(current state)  │ │stream     │ │
│  │ to adapter)  │ │ /send-task)  │ │                 │ │(existing) │ │
│  └──────────────┘ └──────────────┘ └─────────────────┘ └───────────┘ │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │ SSE stream (text/event-stream)
┌─────────────────────────────▼─────────────────────────────────────────┐
│                     FRONTEND LAYER (Next.js - extended)               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │            useAgentEvents (Phase 8 - existing hook)              │  │
│  │            Zustand: useAgentEventStore (extended)                │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
│           ┌──────────────────┼──────────────────────┐                 │
│  ┌────────▼─────┐  ┌────────▼──────────┐  ┌────────▼────────────┐    │
│  │  AgentChat   │  │  TeamDAGPanel     │  │  FileChangePanel    │    │
│  │  Panel       │  │  (@xyflow/react)  │  │  (Monaco diff)      │    │
│  └──────────────┘  └───────────────────┘  └─────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `AgentAdapter` (abstract) | Unified interface: send message, stream events, control, status | `infrastructure/adapters/agent/base.py` (new) |
| `ClaudeCodeAdapter` | Spawn `claude -p --output-format stream-json`, parse NDJSON, normalize to `AgentEventEnvelope` | `infrastructure/adapters/agent/claude_code.py` (new) |
| `CodexAdapter` | Spawn `codex exec --json`, parse JSONL, normalize events | `infrastructure/adapters/agent/codex.py` (new) |
| `OpenCodeAdapter` | Connect to OpenCode HTTP API or ACP subprocess, normalize events | `infrastructure/adapters/agent/opencode.py` (new) |
| `AgentAdapterRegistry` | Resolve correct adapter type from user config; registered in DI container | `infrastructure/adapters/agent/registry.py` (new) |
| `AgentProxyService` | Manage adapter lifecycle per session; route normalized events to EventBusEventLog; handle reconnect | `application/services/agent_proxy_service.py` (new) |
| `EventBusEventLog` | Fan-out asyncio.Queue — Phase 7, zero changes required | `infrastructure/event_log/event_bus_event_log.py` (existing) |
| `/api/agent/chat` | Accept user chat message, forward to active adapter, emit events via EventBus | `api/routes/agent_proxy.py` (new) |
| `/api/agent/control` | Accept control commands (stop, restart, send-task), dispatch to `AgentProxyService` | `api/routes/agent_proxy.py` (new) |
| `/api/agents/{id}/status` | Return current agent status (connected/working/idle/crashed) | `api/routes/agent_proxy.py` (new) |
| `/api/events/stream` | Existing SSE bus — no changes; all events flow through it | `api/routes/events.py` (existing) |
| `useAgentEvents` | Existing SSE consumer hook — no changes; subscribes to event bus | `web/src/lib/agent-events/useAgentEvents.ts` (existing) |
| `AgentChatPanel` | Chat input + message history; posts to `/api/agent/chat` | `web/src/components/agent-events/AgentChatPanel.tsx` (new) |
| `TeamDAGPanel` | Renders agent-initiated team decomposition as interactive DAG using `@xyflow/react` | `web/src/components/agent-events/TeamDAGPanel.tsx` (new) |
| `FileChangePanel` | Shows file diffs from `FILE_CHANGED` events — Monaco diff view | `web/src/components/agent-events/FileChangePanel.tsx` (new) |

---

## Recommended Project Structure

```
src/paperbot/
├── infrastructure/
│   └── adapters/
│       └── agent/                        # NEW: agent adapter layer
│           ├── base.py                   # AgentAdapter ABC, ControlCommand, AgentStatus
│           ├── registry.py               # AgentAdapterRegistry (resolve by name)
│           ├── claude_code.py            # ClaudeCodeAdapter (subprocess + NDJSON)
│           ├── codex.py                  # CodexAdapter (subprocess + JSONL)
│           └── opencode.py               # OpenCodeAdapter (HTTP or ACP stdio)
├── application/
│   └── services/
│       └── agent_proxy_service.py        # NEW: lifecycle manager, event routing
├── api/
│   └── routes/
│       └── agent_proxy.py               # NEW: /api/agent/chat, /api/agent/control, /api/agents/{id}/status
└── ...

web/src/
├── lib/
│   ├── agent-events/                     # Phase 8 - extended (not replaced)
│   │   ├── types.ts                      # EXTENDED: add TEAM_UPDATE, FILE_CHANGED, TASK_UPDATE, CHAT_DELTA, CHAT_DONE
│   │   ├── store.ts                      # EXTENDED: add teamNodes, teamEdges, fileChanges, taskList state
│   │   ├── parsers.ts                    # EXTENDED: parseTeamUpdate, parseFileChange, parseTask, parseChatDelta
│   │   └── useAgentEvents.ts             # unchanged
│   └── store/
│       └── agent-proxy-store.ts          # NEW: chatHistory, selectedAgent, proxyStatus
├── components/
│   └── agent-events/
│       ├── ActivityFeed.tsx              # Phase 8 - unchanged
│       ├── AgentStatusPanel.tsx          # Phase 8 - unchanged
│       ├── ToolCallTimeline.tsx          # Phase 8 - unchanged
│       ├── TeamDAGPanel.tsx              # NEW: @xyflow/react DAG of agent teams
│       ├── FileChangePanel.tsx           # NEW: Monaco diff of recent file changes
│       └── AgentChatPanel.tsx            # NEW: chat input + message history
└── app/
    └── studio/
        └── page.tsx                      # MODIFIED: three-panel layout integrating above components
```

### Structure Rationale

- **`infrastructure/adapters/agent/`**: All agent-specific I/O and protocol differences isolated here. The application layer never sees Claude Code vs. Codex differences — it only speaks `AgentAdapter`.
- **`application/services/agent_proxy_service.py`**: Owns stateful concerns (process PID, session ID, reconnect timer) without touching transport. Keeps it testable.
- **`api/routes/agent_proxy.py`**: New routes consolidated in one file; easy to find and test independently from existing routes.
- **`web/src/lib/agent-events/`**: Existing types/store/parsers extended, not replaced. New event types follow the same `EventType` constant pattern from Phase 8.
- **`web/src/components/agent-events/`**: New components live alongside Phase 8 components in a coherent namespace.

---

## Architectural Patterns

### Pattern 1: Adapter Interface — Normalize Agent Differences at the Boundary

**What:** Each CLI agent has a different subprocess invocation, output format (NDJSON, JSONL, HTTP SSE), and event vocabulary. The `AgentAdapter` base class defines the contract all adapters satisfy. Everything above the adapter layer sees only `AgentEventEnvelope` objects.

**When to use:** Every time a new agent type is added, create a new adapter. Never leak agent-specific parsing into `AgentProxyService` or higher.

**Trade-offs:** One extra class per agent type. Worth it because the dashboard, proxy service, and API routes never change when a new agent is added.

**Interface:**

```python
# src/paperbot/infrastructure/adapters/agent/base.py
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

from paperbot.application.collaboration.message_schema import AgentEventEnvelope


class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    CONNECTED = "connected"
    CRASHED = "crashed"
    STOPPED = "stopped"


@dataclass
class ControlCommand:
    type: str          # "stop" | "restart" | "send_task" | "interrupt"
    payload: dict


class AgentAdapter(ABC):
    """
    Unified interface for all code agent types.

    Normalized event types (new EventType constants to add in message_schema.py):
      FILE_CHANGED  - agent wrote or modified a file
      TASK_UPDATE   - agent updated a task or subtask status
      TEAM_UPDATE   - agent spawned or described a subagent team
      CHAT_DELTA    - streaming assistant text token
      CHAT_DONE     - assistant turn complete (with cost_usd, duration_ms in metrics)
    """

    @abstractmethod
    async def send_message(
        self,
        message: str,
        *,
        session_id: str,
        run_id: str,
        trace_id: str,
    ) -> AsyncIterator[AgentEventEnvelope]:
        """Send a user message. Yields normalized events as the agent responds."""
        ...

    @abstractmethod
    async def send_control(self, command: ControlCommand) -> None:
        """Send a control command to the running agent."""
        ...

    @abstractmethod
    def get_status(self) -> AgentStatus:
        """Return current adapter status (non-blocking)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the agent process/connection."""
        ...
```

### Pattern 2: Subprocess Adapter — NDJSON/JSONL Process Bridge

**What:** `ClaudeCodeAdapter` and `CodexAdapter` both spawn a subprocess using `asyncio.create_subprocess_exec`, read stdout line-by-line as NDJSON/JSONL, and convert each line to `AgentEventEnvelope`. This is the same pattern already used in `studio_chat.py` (`stream_claude_cli`), generalized into an adapter.

**When to use:** Any CLI agent that supports machine-readable JSON output (`--output-format stream-json` for Claude Code, `--json` for Codex).

**Key implementation sketch:**

```python
# ClaudeCodeAdapter — the core subprocess + NDJSON loop
async def send_message(self, message, *, session_id, run_id, trace_id):
    cmd = [
        "claude", "-p", message,
        "--output-format", "stream-json",
        "--verbose",
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=self._working_dir,
    )
    async for line in self._read_lines(process.stdout):
        envelope = self._parse_ndjson_line(line, run_id=run_id, trace_id=trace_id)
        if envelope:
            yield envelope
    await process.wait()
```

**Claude Code NDJSON event mapping to `AgentEventEnvelope.type`:**

| CLI Event | Maps To |
|-----------|---------|
| `{"type": "assistant", "message": {"content": [{"type": "text"}]}}` | `CHAT_DELTA` |
| `{"type": "assistant", "message": {"content": [{"type": "tool_use"}]}}` | `TOOL_CALL` |
| `{"type": "tool_result"}` | `TOOL_RESULT` |
| `{"type": "result", "subtype": "success"}` | `CHAT_DONE` (cost_usd, duration_ms → metrics) |
| `{"type": "system"}` | ignored |

**Codex JSONL event mapping:**

| CLI Event | Maps To |
|-----------|---------|
| `{"type": "item.file_change"}` | `FILE_CHANGED` |
| `{"type": "item.plan_update"}` | `TASK_UPDATE` |
| `{"type": "turn.completed"}` | `CHAT_DONE` |
| `{"type": "thread.started"}` | `AGENT_STARTED` |
| `{"type": "error"}` | `AGENT_ERROR` |

**Trade-offs:** Subprocess output is unstructured if the agent does not support machine-readable mode — parsing ANSI escape sequences from raw terminal output is fragile. Only build adapters for agents with documented JSON output modes.

### Pattern 3: Event Routing Through Existing EventBusEventLog

**What:** `AgentProxyService` does not create its own fan-out mechanism. It calls `event_log.append(envelope)` on the existing `CompositeEventLog`, which fans out through `EventBusEventLog` to all SSE subscribers.

**Why:** The Phase 7 EventBus already delivers events to the dashboard via `/api/events/stream`. Routing agent proxy events through it means the existing `useAgentEvents` hook and Zustand store receive them automatically. No new SSE endpoint is needed.

**Full data flow:**

```
ClaudeCodeAdapter.send_message()
    yields AgentEventEnvelope
        AgentProxyService receives
            event_log.append(envelope)
                EventBusEventLog._fan_out()
                    asyncio.Queue per SSE client
                        GET /api/events/stream
                            useAgentEvents hook
                                Zustand store
                                    React components re-render
```

**Trade-offs:** High-frequency `CHAT_DELTA` events (40 tokens/sec from Claude Code) may saturate the ring buffer (`maxlen=200`). Consider filtering `CHAT_DELTA` events from ring buffer storage while still fanning them out live. Address in the phase design, not architecture.

### Pattern 4: Dashboard Control Surface — Bidirectional Command Flow

**What:** The dashboard sends commands back to running agents via `POST /api/agent/control`, which dispatches to `AgentProxyService.send_control()`.

**Reverse data flow:**

```
User clicks "Stop" in AgentChatPanel
    POST /api/agent/control {type: "stop", session_id: "..."}
        AgentProxyService.send_control(ControlCommand(type="stop"))
            adapter.stop()
                process.terminate()  (subprocess adapters)
            emits AgentEventEnvelope(type=AGENT_STOPPED)
                EventBusEventLog fan-out
                    dashboard status badge updates
```

**Why REST POST (not WebSocket):** Control commands are infrequent and do not need real-time streaming semantics. REST POST returns a synchronous acknowledgment; the actual status change appears asynchronously via the SSE event bus. This keeps the control channel simple.

### Pattern 5: Agent Lifecycle Management — Background Task per Session

**What:** `AgentProxyService` manages each adapter's subprocess in a background asyncio task. The API route returns immediately; events stream back through the EventBus asynchronously. Crash recovery uses exponential backoff (3s, 9s, 27s).

**Why:** Never block an async FastAPI handler waiting for an agent to complete (agent execution can take minutes). Blocking blocks the event loop and prevents other requests.

**Crash recovery flow:**

```
Subprocess exits unexpectedly (returncode != 0)
    ClaudeCodeAdapter detects via process.wait()
    Emits AgentEventEnvelope(type=AGENT_ERROR, payload={reason, returncode})
    AgentProxyService: status -> CRASHED
    Schedules reconnect after backoff (asyncio.create_task + asyncio.sleep)
    On reconnect: status -> CONNECTED, emits AGENT_STARTED
    Dashboard badge updates automatically via SSE
```

### Pattern 6: Hybrid Activity Discovery

**What:** Agent events arrive via two paths:
1. **Push path**: Adapter parses subprocess stdout, normalizes to `AgentEventEnvelope`, routes through EventBus.
2. **Pull/discovery path**: Optional file system watching (via `watchfiles`) for agents that do not emit reliable file change events.

**Why hybrid:** Claude Code and Codex both emit file change events in their JSON output, but these may be incomplete for bulk file operations. File system watching provides a fallback for `FileChangePanel` accuracy.

**Rule:** Push path is the default. File system watching is opt-in per adapter, defaults off. Never poll if the adapter reliably emits `FILE_CHANGED` events.

---

## Data Flow

### Chat Message Flow (User → Agent → Dashboard)

```
[User types in AgentChatPanel]
    POST /api/agent/chat {message, session_id}
        AgentProxyService.route_message()
            adapter.send_message() -> AsyncIterator[AgentEventEnvelope]
                for each envelope: event_log.append(envelope)
                    EventBusEventLog._fan_out(dict)
                        asyncio.Queue (one per SSE client)
                            GET /api/events/stream yields data: {...}
                                useAgentEvents reads, dispatches to Zustand
                                    addFeedItem, addToolCall, addChatDelta, updateAgentStatus
                                        React components re-render
```

### Control Command Flow (Dashboard → Agent)

```
[User clicks "Send Task" in dashboard]
    POST /api/agent/control {type: "send_task", payload: {task: "..."}}
        AgentProxyService.send_control(ControlCommand)
            adapter.send_control()
                writes to agent stdin (subprocess adapters)
                    or HTTP POST (OpenCode adapter)
                agent executes, emits TASK_UPDATE / AGENT_WORKING events
                    flows back through push path above
```

### Team Decomposition Flow (Agent → Dashboard)

```
[Agent spawns subagent, emits structured event]
    ClaudeCodeAdapter parses agent output containing subagent info
    Emits AgentEventEnvelope(type=TEAM_UPDATE, payload={nodes, edges})
        Zustand: updateTeamNodes(nodes, edges)
            TeamDAGPanel (@xyflow/react): re-renders DAG
```

Team decomposition is agent-initiated. PaperBot visualizes what the agent reports. The adapter extracts team structure from agent-specific output formats. PaperBot does not decide how to split tasks.

### Frontend State Management

```
Zustand: useAgentEventStore (Phase 8, extended)
    feed: ActivityFeedItem[]          (capped at 200, unchanged)
    agentStatuses: Map<name, entry>   (unchanged)
    toolCalls: ToolCallEntry[]        (capped at 100, unchanged)
    teamNodes: Node[]                 (NEW: @xyflow nodes)
    teamEdges: Edge[]                 (NEW: @xyflow edges)
    fileChanges: FileChangeEntry[]    (NEW: recent file diffs)
    taskList: TaskEntry[]             (NEW: agent task board)

Zustand: useAgentProxyStore (NEW)
    selectedAgent: "claude-code" | "codex" | "opencode" | null
    sessionId: string | null
    chatHistory: ChatMessage[]
    proxyStatus: "idle" | "connected" | "working" | "crashed"
```

---

## Integration Points with Existing PaperBot Architecture

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `AgentAdapter` to `EventBusEventLog` | `event_log.append(AgentEventEnvelope)` — synchronous | Existing pattern used by all producers; adapter calls via `AgentProxyService` |
| `AgentProxyService` to DI Container | `Container.instance().resolve(EventLogPort)` | Same pattern as `_audit.py`; service registered in DI at startup |
| `EventBusEventLog` to SSE clients | `asyncio.Queue` fan-out via `subscribe()` — Phase 7 | Existing `/api/events/stream` delivers all agent events; no changes |
| `useAgentEvents` to Zustand store | TypeScript Zustand actions — Phase 8, extended | New types follow same pattern: extend `EventType` constants, extend store state, extend parsers |
| `/api/agent/chat` to `AgentProxyService` | Direct Python call within FastAPI handler | In-process; no new infrastructure |
| `studio_chat.py` to new adapter | `studio_chat.py` pattern migrates into `ClaudeCodeAdapter` | Existing `StudioChatRequest` logic is superseded; studio_chat.py can be deprecated |

### Relationship with Existing `codex_dispatcher.py` and `claude_commander.py`

These files in `infrastructure/swarm/` implement Claude as a commander and Codex as a Popen API worker for the Paper2Code pipeline. The new adapter layer addresses the dashboard use case only:

- `ClaudeCodeAdapter` replaces the subprocess-spawn-and-stream pattern from `studio_chat.py` for the dashboard
- `CodexAdapter` is a new subprocess adapter; it does not replace `codex_dispatcher.py` for Paper2Code
- `codex_dispatcher.py` and `claude_commander.py` remain unchanged for the Paper2Code pipeline

The constraint from PROJECT.md is clear: swarm files stay for Paper2Code. The dashboard and Paper2Code are distinct consumers of different subsystems.

### Relationship with MCP Server

The MCP server is the tool-surface code agents consume. It is orthogonal to the dashboard adapter layer:

- Claude Code calls PaperBot MCP tools during its work
- Those MCP calls emit `TOOL_CALL` / `TOOL_RESULT` events via `_audit.py`
- These events flow through EventBus to the dashboard automatically
- `ToolCallTimeline` (Phase 8) already renders them

The MCP prerequisite: agents need a functional MCP server before meaningful tool calls appear in the dashboard. Dashboard infrastructure can be built without MCP live, but end-to-end tool call visualization requires it.

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Claude Code CLI | `asyncio.create_subprocess_exec` + NDJSON stdout | `find_claude_cli()` pattern from `studio_chat.py` reusable; `--output-format stream-json` required |
| Codex CLI | `asyncio.create_subprocess_exec` + JSONL stdout via `--json` flag | `codex exec --json` (Rust CLI, must be on PATH); emits `thread.started`, `item.*`, `turn.*` event types |
| OpenCode | HTTP API (`@opencode-ai/sdk`) or ACP stdin/stdout subprocess | HTTP mode via `opencode` local server is simpler; ACP is stdin/stdout nd-JSON with JSON-RPC 2.0 |

---

## Suggested Build Order

Dependencies determine sequencing. Build in this order:

1. **`AgentAdapter` base + `EventType` constants extension** — defines interface contract and new event type strings; no external dependencies. Extend `EventType` in `message_schema.py` with `FILE_CHANGED`, `TEAM_UPDATE`, `TASK_UPDATE`, `CHAT_DELTA`, `CHAT_DONE`.

2. **`ClaudeCodeAdapter`** — highest priority; migrates existing `studio_chat.py` logic into the adapter pattern. Proven subprocess + NDJSON parsing already works in production.

3. **`AgentProxyService`** — wires adapter to EventBus; depends on #1 and existing EventBusEventLog. No frontend dependency.

4. **`/api/agent/chat` + `/api/agent/control` routes** — depends on #3. Registers in `api/main.py`.

5. **Extend Zustand store + parsers + TypeScript types** — add new state slices and parse functions; no backend dependency. Can be done in parallel with #2-4.

6. **`AgentChatPanel` + `TeamDAGPanel` + `FileChangePanel`** — depends on #5; needs the extended store.

7. **Three-panel studio page layout** — integrates #6 into the page; depends on #4 for API calls. Dashboard is functional for Claude Code after this step.

8. **`CodexAdapter`** — adds second agent type; follows the same subprocess + JSONL pattern as `ClaudeCodeAdapter`.

9. **`OpenCodeAdapter`** — adds third agent type; HTTP variant differs from subprocess pattern. Lower priority; Claude Code coverage is sufficient for v1.2.

The dashboard delivers real value (Claude Code proxying) after step 7, without waiting for all three adapters.

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1 user, 1 agent | Current design sufficient; all in-process |
| 1 user, 3+ parallel agent sessions | `AgentProxyService` manages a map of `session_id → adapter`; one EventBus queue per SSE client is fine |
| Multiple users | EventBus has no user scoping — all events go to all connected SSE clients. For multi-user, add `session_id` filtering in the front-end Zustand store (filter by active session). Not needed for current single-user architecture. |
| High token throughput (streaming) | `CHAT_DELTA` at 40 tok/sec saturates the 200-item ring buffer in ~5 seconds. Fix: exclude `CHAT_DELTA` from ring buffer storage (fan-out live only, no catch-up); ring buffer should hold structural events (lifecycle, tool calls, file changes). |

### Scaling Priorities

1. **First bottleneck:** Ring buffer saturation by streaming tokens. Fix is a one-line filter in `EventBusEventLog.append()` or in the adapter itself — do not store `CHAT_DELTA` in the ring, only fan-out.
2. **Second bottleneck:** Multiple concurrent agent sessions in a multi-user scenario. Fix: add `session_id` tag to all proxy events; frontend filters on active session only.

---

## Anti-Patterns

### Anti-Pattern 1: Parsing Agent Output Above the Adapter Layer

**What people do:** Add Claude-Code-specific NDJSON parsing logic in `AgentProxyService` or an API route.
**Why it's wrong:** Adding a second agent (Codex) requires touching `AgentProxyService` and the route again. The adapter pattern collapses.
**Do this instead:** All parsing is encapsulated in the adapter. `AgentProxyService` only receives `AgentEventEnvelope` objects. The adapter is the only place that knows about agent-specific output formats.

### Anti-Pattern 2: Creating a Parallel Event Schema for Proxy Events

**What people do:** Define new Python dataclasses or TypeScript types specific to the proxy dashboard (`ProxyEvent`, `AgentMessage`).
**Why it's wrong:** Creates a second event vocabulary diverging from `AgentEventEnvelope`. The existing `useAgentEvents` hook, Zustand store, `ActivityFeed`, and `ToolCallTimeline` stop working for proxy events without modification.
**Do this instead:** All proxy events use `AgentEventEnvelope` with new `EventType` constants (`FILE_CHANGED`, `TEAM_UPDATE`, `TASK_UPDATE`, `CHAT_DELTA`, `CHAT_DONE`). Extend `EventType` in `message_schema.py`. Extend TypeScript types in `types.ts`. Parsers in `parsers.ts` handle the new types.

### Anti-Pattern 3: Adding Orchestration Logic to PaperBot

**What people do:** Have `AgentProxyService` decide how to split a task between Claude Code and Codex based on workload or complexity.
**Why it's wrong:** Violates the "no orchestration logic" constraint from PROJECT.md. PaperBot visualizes what the agent reports; it does not direct the agent's internal decisions.
**Do this instead:** Pass the user's task to the configured agent verbatim. Let the agent decompose and delegate. Visualize what the agent reports via `TEAM_UPDATE` events.

### Anti-Pattern 4: One SSE Connection Per Panel Component

**What people do:** `TeamDAGPanel`, `FileChangePanel`, and `AgentChatPanel` each mount their own `useAgentEvents` hook instance.
**Why it's wrong:** Three SSE connections create three `asyncio.Queue` instances in `EventBusEventLog`. Every event is triplicated. This is the "multiple mounts" pitfall documented in Phase 8 research.
**Do this instead:** Mount `useAgentEvents` exactly once at the page or layout root. All panels read from the shared Zustand store.

### Anti-Pattern 5: Blocking the Event Loop on Subprocess Management

**What people do:** `await process.wait()` directly in a FastAPI request handler before returning a response.
**Why it's wrong:** Blocks the uvicorn event loop for the entire duration of the agent run (potentially minutes). No other requests can be served.
**Do this instead:** `AgentProxyService` manages subprocess lifecycle in a background asyncio task. The API route returns immediately after handing off to the service. Events stream back through the EventBus asynchronously.

### Anti-Pattern 6: Storing High-Frequency CHAT_DELTA in the Ring Buffer

**What people do:** Route all agent events, including token-by-token `CHAT_DELTA` events, through the ring buffer.
**Why it's wrong:** At 40 tokens/sec, the 200-item ring buffer saturates in 5 seconds. Structural events (file changes, lifecycle, tool calls) are evicted from the catch-up buffer before a new SSE client can receive them.
**Do this instead:** Tag `CHAT_DELTA` events for live fan-out only (not ring buffer storage). Either filter in the adapter before calling `event_log.append()`, or extend `EventBusEventLog` with a `no_buffer` flag for high-frequency event types.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)

- `src/paperbot/api/routes/studio_chat.py` — existing Claude CLI subprocess pattern: `asyncio.create_subprocess_exec`, NDJSON parsing, `--output-format stream-json`, `find_claude_cli()`
- `src/paperbot/infrastructure/swarm/codex_dispatcher.py` — existing Codex API integration (to be superseded for dashboard path)
- `src/paperbot/infrastructure/swarm/claude_commander.py` — existing commander orchestration (Paper2Code, not dashboard)
- `src/paperbot/application/collaboration/message_schema.py` — `AgentEventEnvelope` schema, `EventType` constants, `make_event()`
- `src/paperbot/infrastructure/event_log/event_bus_event_log.py` — Phase 7 fan-out design, `subscribe()`/`unsubscribe()`/`_fan_out()`
- `src/paperbot/api/routes/events.py` — existing `/api/events/stream` SSE endpoint; confirmed working
- `src/paperbot/mcp/tools/_audit.py` — `log_tool_call()` pattern; demonstrates event routing via `event_log.append()`
- `web/src/lib/sse.ts` — `readSSE()` async generator; existing SSE consumption pattern
- `web/src/lib/agent-events/` — Phase 8 types, store, parsers, hook
- `.planning/phases/07-eventbus-sse-foundation/07-RESEARCH.md` — Phase 7 design decisions and constraints
- `.planning/phases/08-agent-event-vocabulary/08-RESEARCH.md` — Phase 8 event vocabulary, Zustand patterns, anti-patterns
- `.planning/PROJECT.md` — constraints: no orchestration logic, agent-agnostic, reuse EventBus, extend AgentEventEnvelope

### Primary (HIGH confidence — official documentation)

- [Claude Code headless docs](https://code.claude.com/docs/en/headless) — `--output-format stream-json` NDJSON format, `-p` flag, event types: `assistant`, `tool_result`, `result`
- [Codex CLI non-interactive mode](https://developers.openai.com/codex/noninteractive/) — `codex exec --json` JSONL format; event types: `thread.started`, `item.file_change`, `item.plan_update`, `turn.completed`, `error`

### Secondary (MEDIUM confidence)

- [OpenCode CLI docs](https://opencode.ai/docs/cli/) — `opencode -p --output-format json`, local HTTP API
- [OpenCode DeepWiki SDK](https://deepwiki.com/sst/opencode/7-command-line-interface-(cli)) — ACP stdin/stdout nd-JSON, HTTP API spec
- [Agent Client Protocol architecture](https://agentclientprotocol.com/overview/architecture) — JSON-RPC 2.0 over stdin/stdout as emerging standard for agent-agnostic CLI interfaces
- [Claude Code GitHub issue: Agent Hierarchy Dashboard](https://github.com/anthropics/claude-code/issues/24537) — confirms real-world demand for agent hierarchy + team visualization dashboards

---

*Architecture research for: agent-agnostic proxy/dashboard (v1.2 DeepCode Agent Dashboard)*
*Researched: 2026-03-15*
