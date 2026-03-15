# Feature Research

**Domain:** Agent-agnostic code agent dashboard/IDE
**Researched:** 2026-03-15
**Confidence:** HIGH — products in this space are publicly documented and actively evolving; specific UX patterns verified across multiple sources

---

## Scope Note

This file covers **v1.2 DeepCode Agent Dashboard** features only. It replaces the prior v2.0 PG migration
feature file for the current research cycle. The question: what features do users expect from a dashboard
that visualizes any code agent's activity?

Competitors analyzed: Cursor, Windsurf, Cline (agent-IDE hybrids); LangSmith, AgentOps, Claude Code
Agent Monitor (observability dashboards); OpenHands, SWE-agent (open-source agent UIs); GitHub Agent HQ,
VS Code Multi-Agent view, Datadog AI Agents Console (enterprise control planes).

---

## Table Stakes

Features users assume exist. Missing these means the product feels incomplete or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Real-time agent activity stream** | Every tool in this space (Cursor, Windsurf, LangSmith, AgentOps) shows live agent events. Users expect to see what the agent is doing *right now*, not after the fact. | MEDIUM | SSE infrastructure already exists in PaperBot. Needs a structured event feed component that auto-scrolls, with pause/resume control. Component: `ActivityFeed` consuming SSE stream. |
| **Tool call log with arguments and results** | Cursor/Windsurf show each tool invocation inline. LangSmith records every tool call with input/output. Users need this to debug agent behavior and understand decisions. | MEDIUM | Each `AgentEventEnvelope` already has `run_id`/`trace_id`/`span_id`. Render tool name, truncated args, result status, and duration per event row. |
| **File diff viewer for agent-modified files** | Cline requires diff approval before applying changes. Cursor shows diffs. Users expect to see exactly what files changed and how. This is the #1 safety primitive for code agents. | HIGH | Build on existing Monaco editor in studio page. Requires tracking which files an agent touched: `file_path`, `before`, `after` snapshots in event log. Render as Monaco diff editor (already available). |
| **Agent session list** | Every dashboard (Claude Code Agent Monitor, AgentOps, OpenSync) shows a list of sessions. Users need to switch between sessions, see which are active vs completed. | LOW | Table of sessions: `session_id`, start time, agent type, status, task summary. Click-through to session detail. Built on existing `AgentEventModel` records. |
| **Session detail view** | Clicking a session shows its full event timeline: all tool calls, file changes, LLM turns, errors. LangSmith waterfall view is the reference. | MEDIUM | Timeline component rendering ordered `AgentEventEnvelope` records for a `run_id`. Group by agent if sub-agents are present. |
| **Agent status indicator (active/idle/error)** | Windsurf, Cursor, Claude Code Teams all have status badges. Users need to know if the agent is currently working, waiting for input, or failed. | LOW | Badge component driven by latest event type per `run_id`. States: running, waiting, complete, error, idle. Already modeled in event vocabulary. |
| **Chat input → agent dispatch** | Cursor, Windsurf, OpenHands all have a chat box where you type a task and the agent executes it. This is the primary control surface. Without it, the dashboard is read-only (a monitor, not a controller). | HIGH | Proxy layer: chat input → HTTP/WebSocket/CLI to the configured agent. Agent-agnostic: send to Claude Code CLI, Codex API, or OpenCode depending on configured adapter. |
| **Token usage and cost display** | AgentOps, Datadog, Claude Code Agent Monitor, SigNoz — all show token counts and cost per session. Developers actively track this because agent costs accumulate rapidly (Agent Teams cost was cited at $7.80 per complex task). | MEDIUM | Show input tokens, output tokens, estimated cost per session and cumulative. Cost formula: configurable pricing per model. Pull from `AgentEventEnvelope` token fields if populated, or from LLM provider responses via adapter layer. |
| **Error and failure surfacing** | LangSmith highlights failed runs. Claude Code Agent Monitor tracks error states. Users need to know when an agent failed without reading a scroll of events. | LOW | Error badge + filter in session list. Error events rendered in red in the activity feed. Toast notification on agent failure. |
| **Connection status** | Claude Code Agent Monitor and CliDeck show live/offline indicator. Users need to know if the dashboard is receiving events or disconnected. | LOW | SSE connection heartbeat indicator: connected (green dot), reconnecting (yellow), disconnected (red). |

---

## Differentiators

Features that set this dashboard apart from pure monitoring tools and from agent-specific UIs.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Agent-agnostic proxy layer** | No other dashboard proxies chat to multiple heterogeneous agents behind a unified UI. GitHub Agent HQ approximates this but is GitHub-ecosystem-only. DeepCode targets any agent: Claude Code, Codex, OpenCode, custom. Users get one UI regardless of which agent they run. | HIGH | Adapter pattern: `BaseAgentAdapter` with implementations for `ClaudeCodeAdapter` (CLI subprocess/socket), `CodexAdapter` (REST API), `OpenCodeAdapter` (TBD). Config: user picks active agent in settings. Existing `codex_dispatcher.py` and `claude_commander.py` are the starting point. |
| **Hybrid activity discovery (push + pull)** | Most dashboards require agents to push events explicitly. Pure pull (polling) misses events. PaperBot's design — agent pushes structured events via MCP tool + dashboard can discover independently — is more robust than either alone. This is rare in the market. | HIGH | Two channels: (1) MCP tool `log_agent_event` → event store; (2) hook-based discovery (file system watcher for `~/.claude/` logs, or OS-level hooks). Neither channel alone is sufficient. |
| **Team decomposition visualization** | Claude Code Agent Teams launched Feb 2026. No existing dashboard renders agent-initiated team decomposition as a live graph. The Claude Code issue tracker feature request (#24537) confirmed this is an unmet need. xyflow/react is already in the web dashboard for DAG visualization. | HIGH | Render parent-child agent relationships as a live DAG. Nodes: agents. Edges: spawned-by relationships. Node state: running/waiting/complete/error. Update in real-time via SSE. Component: `AgentTeamGraph` built on `@xyflow/react` (already in codebase). |
| **Dashboard control surface (task dispatch)** | Most monitoring dashboards are read-only. The ability to send new tasks, interrupt, or redirect agents from the web UI is rare. GitHub Agent HQ's "mission control" is the only comparable product. | MEDIUM | `TaskDispatch` panel: text input + submit → sends task to active agent via adapter. Interrupt button: sends interrupt signal (SIGINT for CLI agents, API cancel for API agents). Requires bi-directional communication, not just SSE. |
| **Human-in-the-loop approval gate** | Cline already does this in the IDE (diff approval required before file writes). Surfacing this same approval UX in a web dashboard for any agent is novel. OpenAI Agents SDK, LangGraph, and Cloudflare Agents all support interrupt/resume patterns in code, but no web dashboard for code agents surfaces this cleanly. | HIGH | When agent emits `HUMAN_APPROVAL_REQUIRED` event: render approval modal with context (tool name, args, file diff). User approves/rejects → event sent back to agent via adapter. Agent resumes from saved state. Requires checkpoint/resume support in adapter layer. |
| **Paper2Code workflow integration** | Unique to PaperBot. When the active task is a Paper2Code reproduction run, the dashboard surfaces paper metadata, reproduction plan, and code generation progress in dedicated panels — not just a generic event stream. No competitor has domain-aware agent dashboards. | MEDIUM | Detect `run_type: paper2code` in event envelope. Render enriched view: paper title/abstract, current phase (Planning → Blueprint → Generation → Verification), code files being generated. Standard run shows generic activity stream. |
| **MCP tool surface visibility** | PaperBot exposes paper tools via MCP. When an agent calls a PaperBot MCP tool (e.g., `search_papers`, `analyze_paper`), the dashboard can show the tool call with paper-domain context (paper title, score, venue) rather than raw JSON. | LOW | Detect MCP tool call events where `tool_server: paperbot`. Render with PaperBot-specific formatting: paper card, score badge, venue tag. Fallback to raw JSON for non-PaperBot tool calls. |
| **Session replay** | Claude Code issue #24537 proposed replay mode explicitly. AgentOps has session replay. Being able to scrub through a completed agent session is valuable for debugging complex multi-agent runs. No current web-based code agent dashboard offers this with a proper timeline scrubber. | HIGH | Store complete event sequence per `run_id` (already going into event log). Replay UI: timeline scrubber, playback speed control, step-by-step navigation. Requires ordered, timestamped event storage. |

---

## Anti-Features

Features that seem natural to build but should be explicitly excluded from v1.2.

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| **Custom agent orchestration runtime** | Users want PaperBot to orchestrate agents automatically (e.g., "spin up 3 agents for this task"). | Violates the core architectural constraint: host agents own orchestration. Building a competing runtime creates maintenance burden and undermines the skill-provider positioning. This is explicitly out of scope in PROJECT.md. | Surface team decomposition *initiated by the agent*, not by PaperBot. The agent decides the team; DeepCode visualizes it. |
| **Per-agent custom UI skins** | Users of Claude Code may want a "Claude Code-branded" view; Codex users want different branding. | One UI per agent creates N maintenance paths. Defeats the agent-agnostic goal. Minor branding differences do not justify divergence. | Single unified dashboard. Agent type shown as a badge/label. Adapter handles protocol differences, not UI differences. |
| **Real-time streaming of every LLM token** | Cursor streams token-by-token during generation. Users find it visually engaging. | Token streaming requires per-agent streaming support in the adapter layer, doubles SSE event volume, and creates complex UI state (partial text, cancellation mid-stream). The value — watching tokens appear — is not meaningful for a monitoring dashboard. | Show full LLM turn as a single event when complete. For latency transparency, show "LLM thinking..." spinner with elapsed time. |
| **Full IDE replacement (built-in file editor)** | OpenHands and Cursor are full IDEs. Why not replace VS Code entirely? | PaperBot's studio page already has Monaco for editing, but a full IDE replacement requires language servers, extensions, debugging integration, and terminal multiplexing — months of scope. | Use Monaco for file viewing and diff display only. Terminal (XTerm) for command output. The user's actual IDE (VS Code, Cursor) remains the editing environment. |
| **Agent training / fine-tuning integration** | AgentOps tracks sessions for eval datasets. Users may want to fine-tune agents from dashboard data. | Training data curation and fine-tuning is a separate product domain. Adding it to a visualization dashboard creates feature bloat and distracts from the core control-surface value. | Export session data as JSONL for external fine-tuning pipelines. Keep the dashboard read/control only. |
| **Multi-user collaboration on agent sessions** | "Multiple developers watch the same agent session live" sounds useful for teams. | Requires real-time session sharing state, permission models, conflict resolution when two users send commands, and a WebSocket multiplexer. This is Tuple/Liveblocks territory, not a code agent dashboard. | One active user per session. Export/share session replays as read-only links. |
| **Autonomous agent scheduling (cron-style)** | "Run this agent task every night" is a natural feature request. ARQ already does cron for DailyPaper. | Scheduling arbitrary agent tasks creates a mini-orchestration runtime within PaperBot, which contradicts the skill-provider constraint. Also requires agent credential storage, retry logic, and error notifications at scale. | DailyPaper cron workflow (already built) handles scheduled research tasks. Agent task scheduling for code work is out of scope. |

---

## Feature Dependencies

```
Chat input / Task dispatch
    |
    +--requires--> Agent adapter layer (ClaudeCodeAdapter / CodexAdapter / OpenCodeAdapter)
    |                   |
    |                   +--requires--> Agent-agnostic proxy interface (BaseAgentAdapter)
    |
    +--enhances--> Human-in-the-loop approval gate (needs bi-directional adapter, not just receive)

Real-time activity stream
    |
    +--requires--> SSE infrastructure (already exists)
    +--requires--> AgentEventEnvelope flowing into event log (partially built in v1.1)
    |
    +--enhances--> Session detail view / timeline
    +--enhances--> Team decomposition graph
    +--enhances--> File diff viewer (on file_changed events)

File diff viewer
    |
    +--requires--> File snapshot capture in event log (file_path + before + after)
    +--requires--> Monaco diff editor (already in studio page)

Team decomposition graph (AgentTeamGraph)
    |
    +--requires--> Parent-child agent relationship in event envelope (agent_id + parent_agent_id)
    +--requires--> @xyflow/react (already in codebase)
    +--requires--> Real-time activity stream (updates graph state)

Session replay
    |
    +--requires--> Ordered, timestamped event storage per run_id (event log)
    +--requires--> Session detail view (shares timeline component)

Token usage / cost display
    |
    +--requires--> Token counts in event envelope or adapter response
    +--enhances--> Session list (cost-per-session column)

Paper2Code workflow integration
    |
    +--requires--> run_type field in AgentEventEnvelope
    +--requires--> Real-time activity stream
    +--enhances--> Session detail view (domain-specific rendering)

Human-in-the-loop approval gate
    |
    +--requires--> Agent adapter layer (needs bi-directional control, not SSE-only)
    +--requires--> HUMAN_APPROVAL_REQUIRED event type in vocabulary
    +--requires--> State persistence / checkpoint in adapter (agent must be resumable)
```

### Dependency Notes

- **Adapter layer is the critical dependency.** Chat dispatch, HITL approval, and interrupt control all require the adapter to be bidirectional — not just receive events but send commands back. Build the adapter interface early; it unblocks chat dispatch, control surface, and approval features simultaneously.
- **Activity stream unblocks three features.** Real-time stream is prerequisite for team graph updates, session detail, and file diff triggering. Ship it first.
- **Team graph requires agent_id in event envelope.** The Claude Code issue #24537 identified `agent_id` in hook payloads as the missing infrastructure prerequisite for any team visualization. This must be part of the event schema from the start; retrofitting it later requires re-logging all historical events.
- **Session replay is independent.** It only requires event storage (already planned) and a timeline UI component. Can be added after core monitoring is stable without blocking other features.
- **HITL approval conflicts with read-only SSE transport.** SSE is one-directional. Approval responses must go back via HTTP POST (or WebSocket). Design the adapter to accept both SSE (for receiving) and REST (for control) from the start.

---

## MVP Definition

### Launch With (v1.2 Core)

Minimum needed to validate the agent-agnostic dashboard concept.

- [ ] **Agent adapter layer** (BaseAgentAdapter + ClaudeCodeAdapter) — without this, there is no agent to visualize or control
- [ ] **Real-time activity stream** (SSE → ActivityFeed component) — live events are the core value; a static dashboard is a monitoring tool, not a control surface
- [ ] **Tool call log** (per-event rendering in activity feed) — users need to understand what the agent did
- [ ] **Chat input → task dispatch** (send task to configured agent via adapter) — control is what makes this a dashboard, not a log viewer
- [ ] **Session list + session detail** (timeline of events per run_id) — navigation between sessions
- [ ] **Agent status indicator** (running / waiting / complete / error) — basic situational awareness
- [ ] **Token usage / cost per session** — agents burn money; users need visibility immediately
- [ ] **Connection status indicator** — trust signal; users need to know the dashboard is live

### Add After Validation (v1.x)

Once core monitoring + dispatch works with one agent:

- [ ] **File diff viewer** — add when file_changed events are flowing; requires snapshot capture in event log
- [ ] **Team decomposition graph** — add when Claude Code Teams events are present; requires agent_id in envelope
- [ ] **CodexAdapter + OpenCodeAdapter** — second and third agent adapters; add after ClaudeCodeAdapter is stable
- [ ] **Human-in-the-loop approval gate** — add after bidirectional adapter is proven; requires checkpoint/resume in adapter
- [ ] **Paper2Code workflow integration** — enriched view for paper2code runs; add after generic activity stream is stable
- [ ] **Hybrid activity discovery** — MCP push + filesystem discovery; add after event push path is proven

### Future Consideration (v2+)

Features to defer until product-market fit is established:

- [ ] **Session replay** — high value but high complexity; requires replay UI with scrubber; defer until event storage is stable
- [ ] **MCP tool surface visibility** — paper-specific enrichment of tool calls; nice-to-have for PaperBot users
- [ ] **Export session data** (JSONL, CSV) — useful for eval pipelines; low priority vs core features

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Agent adapter layer (BaseAgentAdapter + ClaudeCodeAdapter) | HIGH | HIGH | P1 |
| Real-time activity stream (SSE → ActivityFeed) | HIGH | LOW (SSE exists) | P1 |
| Tool call log | HIGH | LOW | P1 |
| Chat input → task dispatch | HIGH | MEDIUM | P1 |
| Session list + detail view | HIGH | MEDIUM | P1 |
| Agent status indicator | HIGH | LOW | P1 |
| Token usage / cost display | HIGH | MEDIUM | P1 |
| Connection status indicator | MEDIUM | LOW | P1 |
| File diff viewer | HIGH | MEDIUM | P2 |
| Team decomposition graph | HIGH | HIGH | P2 |
| CodexAdapter / OpenCodeAdapter | HIGH (for multi-agent) | MEDIUM | P2 |
| Human-in-the-loop approval gate | HIGH (safety) | HIGH | P2 |
| Paper2Code workflow integration | MEDIUM (PaperBot-specific) | MEDIUM | P2 |
| Hybrid activity discovery | MEDIUM (robustness) | HIGH | P2 |
| Session replay | HIGH | HIGH | P3 |
| MCP tool surface visibility | MEDIUM | LOW | P3 |
| Session data export (JSONL/CSV) | LOW | LOW | P3 |

**Priority key:**
- P1: Must have for v1.2 launch — without these, the dashboard is not functional
- P2: Should have — add after P1 stable; these define the product's competitive position
- P3: Nice to have — defer until post-validation

---

## Competitor Feature Analysis

| Feature | Cursor/Windsurf/Cline | LangSmith/AgentOps | Claude Code Agent Monitor | GitHub Agent HQ | DeepCode (Our Approach) |
|---------|-----------------------|--------------------|---------------------------|-----------------|-------------------------|
| Real-time activity stream | IDE-embedded | Yes (web) | Yes (WebSocket) | Yes | Yes (SSE, existing infra) |
| Tool call log | Yes (inline) | Yes (trace waterfall) | Yes | Partial | Yes |
| File diff viewer | Yes (diff approval) | No | No | No | Yes (Monaco diff) |
| Session list | Partial (history) | Yes | Yes | Yes | Yes |
| Token cost tracking | Partial (Cursor: no; Cline: yes) | Yes | Yes (configurable pricing) | Yes (org-level) | Yes |
| Chat → agent dispatch | Yes (native) | No | No | Yes | Yes (proxy model) |
| Team decomposition graph | No | No | Partial (Kanban) | Partial | Yes (@xyflow/react) |
| Agent-agnostic (multiple agents) | No (vendor-locked) | Partial (SDK-based) | Claude Code only | GitHub-ecosystem | Yes (any agent) |
| Human-in-the-loop approval | Cline: Yes | Partial (SDK) | No | Partial | Planned |
| Session replay | No | Partial (trace replay) | No | No | Planned |
| Domain-aware enrichment | No | No | No | No | Yes (Paper2Code) |
| MCP tool visibility | Cline: Yes | No | No | No | Yes (planned) |

**Key insight:** No existing product combines agent-agnostic proxying with real-time team visualization and a web-based control surface. The closest is GitHub Agent HQ, but it is GitHub-ecosystem-only and not deployable as a self-hosted web UI. DeepCode's differentiation is: (1) any agent, (2) team graph via existing xyflow/react, (3) paper-domain enrichment, (4) self-hosted.

---

## Sources

- [GitHub Claude Code Issue #24537 — Agent Hierarchy Dashboard feature request](https://github.com/anthropics/claude-code/issues/24537) — comprehensive list of TUI/desktop dashboard features requested by community
- [Claude Code Agent Monitor (hoangsonww/Claude-Code-Agent-Monitor)](https://github.com/hoangsonww/Claude-Code-Agent-Monitor) — open-source reference implementation: Kanban, activity feed, token cost, session timeline
- [VS Code Multi-Agent Development Blog (Feb 2026)](https://code.visualstudio.com/blogs/2026/02/05/multi-agent-development) — VS Code Agent Sessions view, MCP Apps, agent session management patterns
- [GitHub Agent HQ announcement](https://visualstudiomagazine.com/articles/2025/10/28/github-introduces-agent-hq-to-orchestrate-any-agent-any-way-you-work.aspx) — "any agent, any way you work" mission control concept
- [LangSmith Observability](https://www.langchain.com/langsmith/observability) — waterfall trace view, custom dashboards, token/latency metrics
- [AgentOps Learning Path](https://www.analyticsvidhya.com/blog/2025/12/agentops-learning-path/) — session replay, cost tracking, multi-agent workflow monitoring
- [OpenHands Review and SDK](https://openhands.dev/) — chat panel + terminal + browser + VS Code integration reference UI
- [Cursor vs Windsurf vs Cline comparison (UI Bakery)](https://uibakery.io/blog/cursor-vs-windsurf-vs-cline) — agent mode features, diff approval, MCP integration
- [Claude Code Agent Teams (claude.fast)](https://claudefa.st/blog/guide/agents/agent-teams) — team architecture, task list, mailbox, context isolation
- [Human-in-the-Loop: OpenAI Agents SDK](https://openai.github.io/openai-agents-js/guides/human-in-the-loop/) — interrupt/approve/resume patterns
- [Cloudflare Agents HITL](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — durable approval gates, checkpoint patterns
- [CliDeck](https://github.com/rustykuntz/clideck) — multi-agent CLI session dashboard (Claude Code, Codex, Gemini CLI, OpenCode simultaneously)
- [OpenSync](https://github.com/waynesutton/opensync) — cloud-synced dashboards for OpenCode, Claude Code, Codex
- [Datadog Claude Code Monitoring](https://www.datadoghq.com/blog/claude-code-monitoring/) — enterprise AI Agents Console, adoption + reliability metrics
- [15 AI Agent Observability Tools 2026 (AIMultiple)](https://research.aimultiple.com/agentic-monitoring/) — ecosystem survey, tool comparison

---
*Feature research for: Agent-agnostic code agent dashboard/IDE (PaperBot v1.2 DeepCode)*
*Researched: 2026-03-15*
