# Project Research Summary

**Project:** PaperBot v1.2 — DeepCode Agent Dashboard
**Domain:** Agent-agnostic proxy dashboard / multi-agent IDE control surface
**Researched:** 2026-03-15
**Confidence:** HIGH

## Executive Summary

The v1.2 DeepCode Agent Dashboard is an agent-agnostic proxy and visualization layer that lets users observe, dispatch tasks to, and control any code agent (Claude Code, Codex, OpenCode) from a single web UI. Research across four domains converges on the same architectural prescription: build a thin `AgentAdapter` abstraction in Python that normalizes heterogeneous CLI/HTTP event streams into the existing `AgentEventEnvelope` format, route those events through the already-built `EventBusEventLog` fan-out, and extend the current studio page with three new panels (chat, team DAG, file diffs). The stack is almost entirely additive — no new Python packages are required; the Node.js side adds five npm packages (three agent SDKs, `node-pty`, and `ws`). The existing SSE infrastructure, `@xyflow/react`, Monaco, XTerm, Zustand, and Vercel AI SDK are all already in place and need extension, not replacement.

The recommended delivery sequence is: adapter interface and `ClaudeCodeAdapter` first (unblocks all downstream features), then the proxy service and API routes, then frontend panels, then additional adapters (Codex, OpenCode). The adapter layer is the critical dependency — chat dispatch, human-in-the-loop approval, and interrupt control all require a bidirectional adapter, not just an event receiver. Session replay, MCP tool surface enrichment, and full hybrid activity discovery are explicitly deferred to v2+ to keep v1.2 focused on the core monitoring and control surface.

The two most dangerous risks are (1) the stateless `claude -p` subprocess pattern already present in `studio_chat.py`, which burns approximately 50k tokens per conversation turn and must be replaced by a persistent REPL/stdin-mode session before any real usage, and (2) SSE reconnection delivering duplicate or missing events because the current `events.py` emits no `id:` field. Both have clear, low-effort fixes that must land in the first adapter phase, not retrofitted later.

---

## Key Findings

### Recommended Stack

The existing PaperBot stack already covers SSE streaming, agent event schema, DAG visualization, Monaco editing, XTerm terminal, chat streaming, Zustand state, and MCP tool surface. The v1.2 additions are precisely targeted: three Node.js agent SDKs to programmatically control Claude Code, Codex CLI, and OpenCode; `node-pty` for PTY process management in interactive terminal mode; `ws` for the WebSocket relay that `@xterm/addon-attach` expects; and a migration of the existing `xterm`/`xterm-addon-fit` imports to the current scoped `@xterm/*` package names (the unscoped packages are deprecated). The Python backend requires zero new packages — the adapter layer is built entirely from stdlib `asyncio` subprocess APIs and the existing `AgentEventEnvelope` schema.

**Core technologies (new additions only):**
- `@anthropic-ai/claude-agent-sdk ^0.2.47` (Node.js): programmatic Claude Code CLI control — official SDK handles process lifecycle, permission callbacks, and streaming; do not shell-exec `claude -p` manually
- `@openai/codex-sdk ^0.112.0` (Node.js): Codex CLI programmatic control — `Codex.startThread()` + `thread.runStreamed()` gives a structured async event generator; stateful thread management avoids token explosion
- `@opencode-ai/sdk latest` (Node.js): typed REST+SSE client for a running OpenCode server — generated from OpenAPI spec; MEDIUM confidence on full event schema, validate against live instance
- `node-pty ^1.1.0` (Node.js): PTY process management for interactive terminal relay — must run in a custom Next.js server, not a serverless API route
- `ws ^8.18.0` (Node.js): WebSocket server for PTY relay; `@xterm/addon-attach` expects raw WebSocket, not Socket.io
- `@xterm/addon-attach ^0.11.0` (frontend): replaces deprecated `xterm-addon-attach`; attaches XTerm terminal to a WebSocket for live PTY output

**What NOT to add:** Socket.io (overhead for PTY byte streaming), LangGraph/AutoGen/CrewAI (assume ownership of agent loop; PaperBot is a proxy, not a runtime), `puppeteer`/`playwright` (CLI tools need SDKs, not browser automation), the unscoped `xterm` packages (deprecated).

See `.planning/research/STACK.md` for full version compatibility matrix and installation commands.

### Expected Features

Research against Cursor, Windsurf, Cline, LangSmith, AgentOps, Claude Code Agent Monitor, OpenHands, GitHub Agent HQ, and VS Code Multi-Agent view produced a clear P1/P2/P3 split. No existing product combines agent-agnostic proxying with real-time team visualization and a web-based control surface — that gap is DeepCode's differentiation.

**Must have (table stakes — P1):**
- Real-time agent activity stream (SSE → `ActivityFeed` component) — every competing tool shows live events; SSE infrastructure already exists in PaperBot
- Tool call log with arguments, results, and duration — users need this to debug agent decisions
- Chat input → agent task dispatch — without bidirectional control, the dashboard is a read-only log viewer, not a control surface
- Session list and session detail view — navigation between sessions; LangSmith waterfall trace is the reference UI
- Agent status indicator (running / waiting / complete / error) — basic situational awareness
- Token usage and cost display per session — agents burn money; community data cites $7.80 per complex task for Claude Code Agent Teams
- Connection status indicator — trust signal that the dashboard is receiving events

**Should have (differentiators — P2):**
- File diff viewer (Monaco diff editor) — the #1 safety primitive for code agents; Cline already requires diff approval in-IDE
- Team decomposition DAG visualization (`@xyflow/react`) — Claude Code Agent Teams launched Feb 2026; no existing dashboard renders live team graphs; confirmed unmet need via GitHub issue #24537
- `CodexAdapter` and `OpenCodeAdapter` — second and third agent adapters; add after `ClaudeCodeAdapter` is stable
- Human-in-the-loop approval gate — render approval modal on `HUMAN_APPROVAL_REQUIRED` events; requires bidirectional adapter
- Paper2Code workflow enrichment — domain-aware enriched view for `run_type: paper2code` sessions
- Hybrid activity discovery (MCP push + filesystem watcher)

**Defer (v2+):**
- Session replay with timeline scrubber — high value, high complexity; requires stable event storage and replay UI
- MCP tool surface visibility (paper-card formatting for PaperBot tool calls)
- Session data export (JSONL/CSV) for eval pipelines

**Anti-features to avoid:** custom agent orchestration runtime, per-agent UI skins, real-time token-by-token streaming in the activity feed, full IDE replacement, agent scheduling/cron, multi-user session collaboration.

See `.planning/research/FEATURES.md` for full prioritization matrix and competitor feature comparison table.

### Architecture Approach

The architecture follows a strict layered separation: agent-specific I/O is fully encapsulated in `infrastructure/adapters/agent/` adapter classes; the application layer (`AgentProxyService`) sees only normalized `AgentEventEnvelope` objects; those events route through the existing `EventBusEventLog` fan-out (unchanged) to the existing SSE endpoint (unchanged) to the existing `useAgentEvents` hook (unchanged) to an extended Zustand store. Three new frontend panels (`AgentChatPanel`, `TeamDAGPanel`, `FileChangePanel`) read from the extended store. Control commands travel the reverse path: `POST /api/agent/control` → `AgentProxyService.send_control()` → `adapter.stop()` / `adapter.interrupt()`. Adding a new agent type requires only a new adapter class — no changes to the proxy service, API routes, or frontend.

**Major components:**
1. `AgentAdapter` ABC (`infrastructure/adapters/agent/base.py`) — unified interface with `send_message()`, `send_control()`, `get_status()`, `stop()`; plus a `capabilities: dict` for feature negotiation and a `raw()` escape hatch to prevent lowest-common-denominator collapse
2. `ClaudeCodeAdapter` / `CodexAdapter` / `OpenCodeAdapter` — concrete adapters; subprocess + NDJSON/JSONL for CLI agents; HTTP client for OpenCode; all normalize to `AgentEventEnvelope` with five new `EventType` constants: `FILE_CHANGED`, `TEAM_UPDATE`, `TASK_UPDATE`, `CHAT_DELTA`, `CHAT_DONE`
3. `AgentProxyService` (`application/services/agent_proxy_service.py`) — manages adapter lifecycle per session, routes events to `EventBusEventLog`, implements crash recovery with exponential backoff (3s → 9s → 27s), enforces `IDLE | PROCESSING | AWAITING_INPUT` state machine for command injection safety
4. `/api/agent/chat` + `/api/agent/control` + `/api/agents/{id}/status` (`api/routes/agent_proxy.py`) — new endpoints; chat and control route through `AgentProxyService`
5. Extended Zustand stores — `useAgentEventStore` gains `teamNodes`, `teamEdges`, `fileChanges`, `taskList` slices; new `useAgentProxyStore` holds `selectedAgent`, `sessionId`, `chatHistory`, `proxyStatus`
6. `AgentChatPanel`, `TeamDAGPanel`, `FileChangePanel` — new React components; mounted once at the studio page root and read from shared Zustand store (not separate SSE connections)

**Build order enforced by dependencies:** base ABC → `ClaudeCodeAdapter` → `AgentProxyService` → API routes → Zustand/types → frontend panels → studio layout → `CodexAdapter` → `OpenCodeAdapter`. The dashboard delivers real value for Claude Code after step 7 without waiting for all three adapters.

See `.planning/research/ARCHITECTURE.md` for the full system diagram, data flow sequences, anti-patterns, and scaling considerations.

### Critical Pitfalls

1. **Stateless `claude -p` subprocess per message (token explosion)** — the existing `studio_chat.py` spawns a fresh process for every message, reloading full context each time; community research confirms ~50k tokens per turn. Fix: switch to persistent REPL/stdin mode; stateful `AgentSession` must hold a process handle. Address in the adapter layer phase before any end-to-end testing.

2. **PTY absence causes block-buffered subprocess output** — `asyncio.create_subprocess_exec` with `stdout=PIPE` causes CLI tools to switch to 4–8 KB block buffering; events arrive in bursts seconds apart. Fix: for agents with `--output-format stream-json` mode (Claude Code, Codex), always use that mode — it bypasses libc buffering. For agents without structured output, use `pty.openpty()`. Verify in a non-TTY Docker container; local dev masks this bug.

3. **SSE reconnection delivers duplicate or missing events** — `events.py` currently emits no `id:` field; browser `EventSource` cannot send `Last-Event-ID` on reconnect. Fix: emit `id: {seq}` on every SSE frame; replay only events with `seq > last_id`. Address in the event stream phase.

4. **Adapter abstraction collapses to lowest-common-denominator** — designing the adapter around only the intersection of Claude Code and Codex means adding a third agent forces breaking changes. Fix: define three layers from day one — minimal core contract, capability flags dict, and `raw()` escape hatch. Dashboard checks capabilities before using advanced features.

5. **No run-scoped SSE filtering causes multi-session chaos** — the current `EventBusEventLog` fans out all events to all clients; two concurrent sessions interleave in the activity feed and DAG. Fix: add a `run_id` query parameter to `/api/events/stream` and filter the fan-out queue at subscription time. `AgentEventEnvelope.run_id` is already present.

See `.planning/research/PITFALLS.md` for the full pitfall list including security mistakes, UX pitfalls, performance traps, and recovery strategies.

---

## Implications for Roadmap

Based on research, the dependency graph is unambiguous. The adapter layer gates everything else. SSE reconnection and run-scoped filtering must be resolved before visualization panels are meaningful. The frontend panels are largely independent of each other once the Zustand store is extended.

### Phase 1: Proxy Adapter Layer Foundation

**Rationale:** The adapter layer is the critical dependency for every subsequent feature. Two critical pitfalls (stateless subprocess token explosion and adapter lowest-common-denominator collapse) must be fixed at this layer before any code builds on top of it. Configuration-driven agent selection must replace the existing binary-detection heuristics (`find_claude_cli()`) before multiple adapters are registered.

**Delivers:** `AgentAdapter` ABC with capability flags and `raw()` escape hatch; `ClaudeCodeAdapter` (subprocess + NDJSON, persistent REPL/stdin mode, PTY-safe structured output mode); `AgentAdapterRegistry` (config-driven via `.paperbot/agent.yaml`, not binary heuristics); `AgentProxyService` (session lifecycle, crash recovery with exponential backoff, `IDLE/PROCESSING/AWAITING_INPUT` state machine); `/api/agent/chat` + `/api/agent/control` + `/api/agents/{id}/status` routes; five new `EventType` constants in `message_schema.py`; deprecation of stateless `studio_chat.py` subprocess pattern.

**Addresses:** P1 features — chat input/task dispatch, agent status indicator (lifecycle events now flow)
**Avoids:** Pitfalls 1 (token explosion), 2 (PTY absence), 3 (abstraction collapse), 6 (hardcoded agent detection)

**Research flag:** STANDARD PATTERNS — asyncio subprocess management, abstract base classes, and session lifecycle are well-documented. Claude Code headless CLI flags are fully documented in official Anthropic docs. No phase-level research needed.

---

### Phase 2: Real-Time Event Stream and Session Management

**Rationale:** The activity stream is the prerequisite for team graph updates, session detail, and file diff triggering. SSE reconnection reliability (pitfall 4) and run-scoped filtering (pitfall 7) must be fixed before building visualization panels that depend on a clean event stream. Session list and session detail are table-stakes features users expect before any differentiating features are added.

**Delivers:** SSE `id:` field emission with `seq`-based reconnection recovery; `run_id` filter on `/api/events/stream`; `ActivityFeed` component with auto-scroll, pause/resume, and error highlighting; session list table (agent type, status, start time, cost); session detail timeline (ordered events per `run_id`); token usage and cost display per session; connection status indicator; `CHAT_DELTA` excluded from ring buffer (live fan-out only) to prevent buffer saturation at 40 tokens/sec.

**Addresses:** P1 features — real-time activity stream, tool call log, session list + detail, token/cost display, connection status indicator
**Avoids:** Pitfalls 4 (SSE reconnection duplicates/gaps), 7 (multi-session event chaos)

**Research flag:** NEEDS VALIDATION — `seq`-based ring-buffer replay and `run_id` filtering interact with the existing `EventBusEventLog._put_nowait_drop_oldest` and ring-buffer catch-up logic in non-obvious ways. Review `.planning/phases/07-eventbus-sse-foundation/07-RESEARCH.md` before writing the phase spec; verify behavior under concurrent reconnect + multi-session scenarios.

---

### Phase 3: Frontend Dashboard Panels and Studio Layout

**Rationale:** With the adapter layer delivering events and the SSE stream clean and session-scoped, the frontend panels can be built without architectural risk. The extended Zustand store slices can be built in parallel with phases 1–2 (no backend dependency). All three new panels must share one `useAgentEvents` hook instance mounted at the studio page root — separate SSE connections per panel hit the browser 6-connection limit and triplicate event delivery.

**Delivers:** Extended Zustand `useAgentEventStore` (teamNodes, teamEdges, fileChanges, taskList) and new `useAgentProxyStore`; extended TypeScript event types and parsers for `FILE_CHANGED`, `TEAM_UPDATE`, `TASK_UPDATE`, `CHAT_DELTA`, `CHAT_DONE`; `AgentChatPanel` (chat input + history, posts to `/api/agent/chat`); `FileChangePanel` (Monaco diff view triggered by `FILE_CHANGED` events); `TeamDAGPanel` (`@xyflow/react` live DAG of agent-reported team structure); three-panel studio page layout integrating all components; `@xterm/addon-attach` WebSocket addon and `node-pty` PTY relay in Next.js custom server.

**Addresses:** P1 — real-time stream visualization, chat dispatch UI; P2 — file diff viewer, team decomposition graph
**Avoids:** ARCHITECTURE.md anti-pattern 4 (multiple SSE mount points); Pitfall 7 (panels scoped to active `run_id` via store)

**Research flag:** STANDARD PATTERNS — `@xyflow/react` dynamic updates, Monaco diff editor, Zustand slices, and Next.js custom server are all well-documented with prior art. No phase-level research needed.

---

### Phase 4: Additional Agent Adapters (Codex + OpenCode)

**Rationale:** Once `ClaudeCodeAdapter` is stable and the end-to-end pipeline is validated, adding `CodexAdapter` and `OpenCodeAdapter` follows the same subprocess-adapter pattern. Adding a stub adapter with different event types validates that the capability-flag abstraction holds without modifying any shared code — this is the anti-pattern 1 and 3 compliance test.

**Delivers:** `CodexAdapter` (subprocess + JSONL, `codex exec --json`, stateful thread resumption via `--resume`); `OpenCodeAdapter` (HTTP REST+SSE via `@opencode-ai/sdk`, or ACP stdin/stdout as fallback); settings UI for agent selection (writes to `.paperbot/agent.yaml`); adapter validation: dashboard renders correctly for all three agents without adapter-specific code changes upstream.

**Addresses:** P2 — CodexAdapter/OpenCodeAdapter multi-agent support
**Avoids:** Pitfall 3 (capability flag compliance tested across three concrete adapters)

**Research flag:** NEEDS VALIDATION — OpenCode SDK event schema is only MEDIUM confidence; the full event type list must be validated against a running OpenCode instance before this phase begins. Verify `@opencode-ai/sdk` schema against the current server version before writing the `OpenCodeAdapter` parser.

---

### Phase 5: Human-in-the-Loop Approval Gate

**Rationale:** HITL approval is a high-value safety feature that requires the bidirectional adapter (phase 1), clean event stream (phase 2), and frontend panels (phase 3) to all be stable. It also requires SSE-only transport to be supplemented — approval responses must return via HTTP POST or WebSocket, not SSE. The `IDLE/PROCESSING/AWAITING_INPUT` state machine from phase 1 is the prerequisite that prevents command injection race conditions (pitfall 5).

**Delivers:** `HUMAN_APPROVAL_REQUIRED` event type and approval modal component; adapter checkpoint/resume support; `POST /api/agent/approve` and `POST /api/agent/reject` endpoints; control surface disable/enable tied to agent status; command injection race condition prevention (turn-boundary queuing for message injection; OS signals for lifecycle commands).

**Addresses:** P2 — human-in-the-loop approval gate
**Avoids:** Pitfall 5 (command injection race conditions during active tool calls)

**Research flag:** NEEDS RESEARCH — Claude Code's hook channel for lifecycle signals and Codex's approval flow for shell commands are partially documented; edge cases around mid-tool-call interrupt and per-agent checkpoint format differences need deeper research before the phase spec is written.

---

### Phase Ordering Rationale

- **Adapter first** because it is the single blocking dependency for all control-surface features; the event stream has nothing to deliver without an adapter feeding it.
- **SSE reliability second** because visualization panels built on a buggy event stream will need to be rebuilt; fix the foundation before building on top of it.
- **Frontend panels third** because they have no backend dependencies beyond the extended Zustand types, which can be built in parallel with phases 1–2.
- **Additional adapters fourth** rather than concurrently with `ClaudeCodeAdapter`, to avoid designing the abstraction around two agents simultaneously before the design is validated with one.
- **HITL last** because it requires all four preceding phases to be stable and has the most agent-specific edge cases that require per-agent validation.

### Research Flags

Phases needing deeper research during planning:
- **Phase 2:** `seq`-based ring buffer replay + `run_id` filtering interaction with `EventBusEventLog._put_nowait_drop_oldest` — verify under concurrent reconnect + multi-session scenarios before writing the phase spec
- **Phase 4:** OpenCode SDK event schema completeness — validate against a running OpenCode instance; the event type list in the SDK docs is incomplete (MEDIUM confidence only)
- **Phase 5:** Claude Code hook channel for lifecycle signals, Codex shell approval flow — documentation is sparse for edge cases; read official CLI docs against current installed versions before the phase spec

Phases with standard, well-documented patterns (skip research-phase):
- **Phase 1:** asyncio subprocess management, ABC patterns, Claude Code `--output-format stream-json` — fully documented in official Anthropic docs; existing `studio_chat.py` pattern to migrate is already in the codebase
- **Phase 3:** `@xyflow/react` dynamic updates, Monaco diff, Zustand slice patterns, Next.js custom server — all have rich documentation and prior art in the existing codebase

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All new packages verified against official npm registries and docs (2026-03-15); versions confirmed; compatibility matrix documented. Exception: `@opencode-ai/sdk` event schema is MEDIUM — generated from OpenAPI spec but full event type list unconfirmed against running server. |
| Features | HIGH | Competitor analysis across 10+ products; community demand for team visualization confirmed via GitHub issue #24537; P1/P2/P3 split grounded in user value and implementation cost analysis. |
| Architecture | HIGH | Primary confidence from direct codebase inspection of `studio_chat.py`, `codex_dispatcher.py`, `event_bus_event_log.py`, `message_schema.py`, `events.py`, and existing Phase 7/8 research. All patterns are proven in the existing codebase, not hypothetical. |
| Pitfalls | HIGH (core), MEDIUM (multi-agent) | Core infrastructure pitfalls (PTY absence, SSE reconnection, token explosion) verified against codebase with specific file locations + official docs + community primary sources. Multi-agent adapter edge cases grounded in community evidence and first-principles analysis. |

**Overall confidence:** HIGH

### Gaps to Address

- **OpenCode event schema:** `@opencode-ai/sdk` event type list is not fully documented. Mitigation: in phase 4, deploy a local OpenCode server and inspect its actual SSE event stream before writing the `OpenCodeAdapter` parser. Do not rely on docs alone.
- **Claude Code REPL/stdin session resumption format:** `--resume <session_id>` requires a UUID, not a human-readable name. The exact lifecycle for resuming persistent REPL sessions from a Python subprocess adapter (as opposed to print-mode) has limited documentation. Validate with the real CLI in a non-TTY Docker environment before the phase 1 spec is finalized.
- **Ring buffer size adequacy:** Current ring buffer `maxlen` is 200 events. With `CHAT_DELTA` filtered out (as recommended), structural events per typical agent session are estimated at 20–60. The 200-item ring should be sufficient, but validate against real agent runs and resize if sessions routinely exceed capacity.
- **Windows ConPTY compatibility:** `node-pty ^1.1.0` requires Windows 10 1809+ (ConPTY). If Windows deployment is required, validate the custom Next.js server + node-pty path in a Windows environment. The Python `pty.openpty()` fallback does not support Windows.

---

## Sources

### Primary (HIGH confidence — codebase + official documentation)

- `src/paperbot/api/routes/studio_chat.py` — existing Claude CLI subprocess pattern (the stateless `claude -p` per-message pattern to replace)
- `src/paperbot/infrastructure/swarm/codex_dispatcher.py` — existing Codex API integration (to be superseded for dashboard path)
- `src/paperbot/infrastructure/event_log/event_bus_event_log.py` — Phase 7 fan-out design, ring buffer behavior, `_put_nowait_drop_oldest`
- `src/paperbot/application/collaboration/message_schema.py` — `AgentEventEnvelope` schema, existing `EventType` constants, `make_event()`
- `src/paperbot/api/routes/events.py` — confirmed no `id:` field in emitted SSE frames
- `web/package.json` — confirmed existing deps: `@xyflow/react`, `xterm`, `ai`, `@ai-sdk/*`, `@modelcontextprotocol/sdk`
- [Claude Code headless docs](https://code.claude.com/docs/en/headless) — `--output-format stream-json`, event types, session resume flags
- [Claude Agent SDK TypeScript releases](https://github.com/anthropics/claude-agent-sdk-typescript/releases) — v0.2.47 confirmed
- [Codex SDK docs](https://developers.openai.com/codex/sdk/) + [npm](https://www.npmjs.com/package/@openai/codex-sdk) — v0.112.0 confirmed, JSONL event types verified
- [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive/) — `codex exec --json` event vocabulary
- [node-pty npm](https://www.npmjs.com/package/node-pty) — v1.1.0 confirmed; PTY behavior and Windows ConPTY requirements documented
- [@xterm/addon-attach npm](https://www.npmjs.com/package/@xterm/addon-attach) — v0.11.0 confirmed; scoped package migration documented
- `.planning/phases/07-eventbus-sse-foundation/07-RESEARCH.md` — Phase 7 EventBus design decisions
- `.planning/phases/08-agent-event-vocabulary/08-RESEARCH.md` — Phase 8 event vocabulary, Zustand patterns, anti-patterns

### Secondary (MEDIUM confidence — community and third-party)

- [GitHub Claude Code Issue #24537](https://github.com/anthropics/claude-code/issues/24537) — community demand for agent hierarchy dashboard; `agent_id` in hook payloads as missing infrastructure confirmed
- [Claude Code Agent Monitor reference implementation](https://github.com/hoangsonww/Claude-Code-Agent-Monitor) — open-source reference for Kanban + activity feed + token cost patterns
- [OpenCode SDK docs](https://opencode.ai/docs/sdk/) — REST+SSE patterns; event schema incomplete
- [Building a 24/7 Claude Code Wrapper — 50k token per turn analysis](https://dev.to/jungjaehoon/why-claude-code-subagents-waste-50k-tokens-per-turn-and-how-to-fix-it-41ma) — community validation of stateless subprocess pitfall
- [Human-in-the-Loop: OpenAI Agents SDK](https://openai.github.io/openai-agents-js/guides/human-in-the-loop/) — interrupt/approve/resume patterns
- [VS Code Multi-Agent Development Blog (Feb 2026)](https://code.visualstudio.com/blogs/2026/02/05/multi-agent-development) — agent session management and MCP Apps patterns
- [AgentOps learning path](https://www.analyticsvidhya.com/blog/2025/12/agentops-learning-path/) — session replay patterns, cost tracking design
- [FastAPI SSE vs WebSocket best practices 2025](https://potapov.me/en/make/websocket-sse-longpolling-realtime) — transport selection rationale

### Tertiary (LOW confidence — needs validation)

- [OpenCode DeepWiki SDK](https://deepwiki.com/sst/opencode/7-command-line-interface-(cli)) — ACP stdin/stdout nd-JSON protocol; unverified against current OpenCode version
- [Agent Client Protocol architecture](https://agentclientprotocol.com/overview/architecture) — JSON-RPC 2.0 over stdin/stdout as emerging standard for agent-agnostic CLI interfaces; specification is immature

---

*Research completed: 2026-03-15*
*Ready for roadmap: yes*
