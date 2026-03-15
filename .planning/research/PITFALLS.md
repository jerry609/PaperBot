# Pitfalls Research: v1.2 Agent Dashboard

**Domain:** Agent proxy/dashboard — CLI-based agent integration, real-time event visualization, agent-agnostic adapters
**Researched:** 2026-03-15
**Confidence:** HIGH (core infrastructure pitfalls verified against codebase + official docs), MEDIUM (multi-agent adapter pitfalls from community evidence)

> This file covers pitfalls specific to the v1.2 milestone: agent-agnostic proxy layer,
> multi-agent adapter abstraction, real-time event visualization, and dashboard control surface.
> The existing PITFALLS.md covers v2.0 PostgreSQL migration pitfalls.

---

## Critical Pitfalls

---

### Pitfall 1: Subprocess PTY Absence Strips ANSI and Causes Block-Buffered Output

**What goes wrong:**
When Claude Code CLI or any CLI agent is spawned via `asyncio.create_subprocess_exec` with `stdout=PIPE`, the child process detects it is not connected to a real TTY. Most CLI tools respond by disabling color output entirely and switching from line-buffering to block-buffering (typically 4–8 KB chunks). The dashboard receives silence for seconds, then a burst of events, then silence again. Users see a frozen stream that suddenly jumps forward.

PaperBot already has `studio_chat.py` using `create_subprocess_exec` with PIPE. This is the exact pattern that triggers the problem. The `FORCE_COLOR=0` env var in the current code makes this worse: some agents use color codes as status signals, and disabling color removes them entirely.

**Why it happens:**
Unix libc `stdio.h` checks `isatty(fileno(stdout))` before every write. If the check fails (pipe, not TTY), it activates full buffering. The subprocess has no way to know you want line-granularity delivery — it only knows about the fd type.

**How to avoid:**
For agents that support structured output modes (Claude Code CLI supports `--output-format stream-json`), use the structured mode. It emits reliable NDJSON regardless of TTY state because the output is explicitly controlled by the CLI, not by libc buffering heuristics.

For agents that do not support structured output, use `pty.openpty()` to create a pseudo-terminal pair, pass the slave fd as stdout/stderr, and read from the master fd. The child process believes it is running in a terminal.

```python
import pty, os, asyncio

master_fd, slave_fd = pty.openpty()
process = await asyncio.create_subprocess_exec(
    *cmd,
    stdin=asyncio.subprocess.DEVNULL,
    stdout=slave_fd,
    stderr=slave_fd,
)
os.close(slave_fd)
# Read from master_fd via loop.add_reader(), strip ANSI codes server-side
```

Strip ANSI escape codes before forwarding to the dashboard if you only need text content: `re.sub(r'\x1b\[[0-9;]*[mGKHFJ]', '', text)`.

**Warning signs:**
- Dashboard shows no output for several seconds, then a batch of events arrives together
- Tool-use events or cost summaries appear only after the process exits, not during execution
- Local dev works (your terminal is a TTY), Docker/CI is silent

**Phase to address:** Proxy adapter layer — the phase that builds the unified `AgentAdapter` abstraction for subprocess management

---

### Pitfall 2: Each Spawned Subprocess Reloads Full Context — Token and Latency Catastrophe

**What goes wrong:**
`studio_chat.py` spawns a fresh `claude -p` (print mode) subprocess for each user message. Every new process reloads the system prompt, tool definitions, and (if multi-turn) the full conversation history from scratch. A 2025 community investigation confirmed this pattern burns approximately 50k tokens per turn in the worst case. Costs multiply by the number of messages in a session.

For Codex API calls in `codex_dispatcher.py`, the same pattern applies: each `client.responses.create()` call re-sends the full tool list and any conversation history.

**Why it happens:**
`claude -p` (print mode) is designed for single-shot pipeline use, not interactive sessions. The stateful mode is `claude` (REPL mode), which holds conversation in memory and accepts new turns via stdin. Developers default to print mode because it is simpler to subprocess-manage; the token cost is not obvious until usage bills arrive.

**How to avoid:**
For Claude Code: use `claude` in REPL/interactive mode, writing new turns to stdin and reading events from stdout in stream-json mode. The process stays alive between turns; the context window is populated once. Use Anthropic's ephemeral cache control on the system prompt as a secondary defense.

For Codex API: maintain a `previous_response_id` field in the session object and pass it to continue a response chain, rather than rebuilding the full `messages` array each call.

For the agent-agnostic proxy: the `AgentSession` abstraction must be stateful. It must hold a process handle (for CLI agents) or a response-chain ID (for API agents). It must NOT be a stateless request transformer.

**Warning signs:**
- Each message takes as long as the first message regardless of conversation position
- Token usage logs show identical system-prompt token counts on every turn
- Session cost grows linearly with message count instead of incrementally

**Phase to address:** Proxy adapter layer — session lifecycle design must be the first decision, not an afterthought

---

### Pitfall 3: Agent-Agnostic Abstraction Collapses to Lowest Common Denominator

**What goes wrong:**
A unified `AgentAdapter` interface is designed around the intersection of what Claude Code and Codex share. A third agent (OpenCode, Gemini CLI) is later forced to fit that interface. Capabilities unique to each agent — Claude's extended thinking, Codex's shell approval flow, any future agent's structured event types — are either silently dropped or require dirty casting through the abstraction. The dashboard becomes simultaneously too rigid (no extension points) and too leaky (agent-specific logic bleeds into shared code via `isinstance` checks).

**Why it happens:**
The first design pass extracts the intersection of known agents. This is natural for two agents. The flaw is scaling: each new agent adds branches to the shared interface, violating the open/closed principle. The existing `find_claude_cli()` in `studio_chat.py` and the separate Codex-specific loop in `codex_dispatcher.py` demonstrate this pattern already forming in PaperBot.

**How to avoid:**
Design the adapter with three explicit layers:

1. **Minimal core contract** — what every agent must implement:
   - `send_message(text: str) -> AsyncIterable[AgentEvent]`
   - `interrupt() -> None`
   - `terminate() -> None`
   - `status() -> AgentStatus`

2. **Capability flags** — what the agent optionally supports:
   - `adapter.capabilities: dict` (e.g., `{"structured_events": True, "multi_turn_stdin": True, "tool_approval_flow": False}`)

3. **Escape hatch** — for agent-specific calls with no cross-agent equivalent:
   - `adapter.raw(command: str, **kwargs) -> Any`

The dashboard checks `adapter.capabilities` before using advanced features and degrades gracefully when a capability is absent. Provider-specific features go through the escape hatch, not through implicit behavioral differences.

**Warning signs:**
- Dashboard code contains `if isinstance(adapter, ClaudeAdapter)` or `if adapter.agent_type == "codex"` conditionals
- Adding a third agent requires modifying the base adapter interface
- Agent-specific event types are mapped to "closest equivalent" in the shared schema, losing information silently

**Phase to address:** Proxy adapter layer — define the interface and capability negotiation contract before writing any concrete adapter

---

### Pitfall 4: SSE Reconnection Delivers Duplicate or Missing Events

**What goes wrong:**
The `EventBusEventLog` ring buffer replay sends all buffered events to a new subscriber on connect. When the frontend EventSource disconnects and reconnects (network hiccup, browser tab resume, Nginx proxy timeout), it reconnects without a `Last-Event-ID`. The server replays the ring buffer again. The frontend renders duplicates for all events in the buffer. Alternatively, if the reconnect takes longer than the ring buffer drains, all events during the gap are silently lost.

PaperBot's current `events.py` sends raw JSON data frames with no SSE `id:` field:
```
data: {...}\n\n
```
Without the `id:` field, the browser EventSource API cannot send `Last-Event-ID` on reconnect. Every reconnection starts from the ring buffer beginning.

**Why it happens:**
The SSE specification's built-in reconnection recovery requires the server to send `id: <seq>` on every event and the client to echo it as `Last-Event-ID` on reconnect. Without sequence IDs, the browser has no reference point to resume from.

**How to avoid:**
Add a monotonic sequence number to every SSE event frame. The `AgentEventEnvelope` already has a `seq` field — use it:

```
data: {...}\n
id: 42\n
\n
```

On reconnect, read the `Last-Event-ID` header, replay events with `seq > last_id` from the ring buffer (if in range) or from the SQLAlchemy event log (if the event was evicted from the ring). The ring buffer size (currently 200) should be reviewed against the expected reconnect latency.

Also: configure Nginx with `proxy_buffering off` for SSE endpoints. Proxies that buffer responses will hold all events until the connection closes, making SSE effectively non-real-time.

**Warning signs:**
- Agent activity panel shows duplicate tool-use entries after a browser refresh
- Dashboard shows blank activity after tab switch (mobile browser backgrounding kills the connection)
- "Connected" indicator shows frequent reconnects during a single agent run

**Phase to address:** Real-time event stream phase (building on the v1.1 EventBus/SSE foundation)

---

### Pitfall 5: Sending Commands to a Running Agent Creates Undetected Race Conditions

**What goes wrong:**
The dashboard control surface sends a command (interrupt, inject task, change mode) to a running agent via a POST endpoint. The agent is mid-tool-call. The command arrives in the subprocess stdin buffer while the agent is blocked waiting for a tool result. Depending on timing: (a) the command is processed out of sequence, (b) it sits buffered until after the current tool call completes making control feel broken, or (c) the agent misinterprets the injected bytes as continuation of the tool result it was reading.

PaperBot's `agent_board.py` already has `_run_controls` with a `while ctrl.state == "paused": await asyncio.sleep(1.0)` polling pattern. This is correct for coarse lifecycle control (pause/cancel at turn boundaries), but breaks down for finer-grained command injection during active turns.

**Why it happens:**
CLI agents read stdin sequentially. There is no multiplexed command channel. A command injected while the agent is reading tool results is appended to whatever bytes the agent reads next — which may be JSON continuation, not a prompt boundary. The agent was not designed to receive out-of-band signals on its main stdin during an active tool execution.

**How to avoid:**
Distinguish between two command types and route them differently:

- **Turn-boundary commands** (inject next message, follow-up task): queue in the session manager, send only after the agent returns from its current turn. Use the `stream-json` result event (`{"type":"result"}`) as the signal that a turn is complete.

- **Lifecycle commands** (pause, cancel, terminate): use OS signals (`SIGTSTP` for pause, `SIGTERM` for terminate) or Claude Code hooks, not stdin writes. Claude Code exposes a hook channel specifically for this purpose.

Implement a minimal state machine in the session manager: `IDLE | PROCESSING | AWAITING_INPUT`. Only accept new message input in `IDLE` and `AWAITING_INPUT` states. Reject or queue commands received in `PROCESSING` state.

**Warning signs:**
- Commands sent during an active tool call appear to have no effect, then replay unexpectedly on the next turn
- Agent produces garbled output or malformed JSON events after a command is injected mid-turn
- "Interrupt" button requires double-clicking or appears to work but the agent continues for several more seconds

**Phase to address:** Dashboard control surface phase

---

### Pitfall 6: Hardcoding Agent Detection Logic Into the Application

**What goes wrong:**
The dashboard detects which agent is running based on runtime heuristics: presence of the `claude` binary (via `find_claude_cli()`), presence of `OPENAI_API_KEY`, or specific event field names. Over time, detection logic proliferates across multiple modules. When a third agent is added, detection breaks or requires changes in several places simultaneously. The "agent-agnostic" principle is violated through accumulated heuristics.

PaperBot already has two separate code paths: `find_claude_cli()` in `studio_chat.py` and Codex API calls in `codex_dispatcher.py` — the foundational split that the v1.2 milestone is explicitly trying to unify.

**Why it happens:**
Binary detection is a natural first step — it requires zero configuration from the user. The flaw is that it cannot scale past two agents, it creates invisible behavioral differences based on the user's environment, and it mixes agent selection concerns with agent execution concerns.

**How to avoid:**
Make agent selection explicit user configuration, not auto-detection. A settings page or a config file (`.paperbot/agent.yaml`) declares the active agent and its configuration. The backend creates the appropriate adapter from explicit configuration, not runtime detection.

Auto-detection can remain as a first-run convenience feature, but it must write the result to the config file and never persist as a live code path used in request handling.

**Warning signs:**
- `find_claude_cli()` or its equivalent is called in more than one module
- A new agent requires modifying existing adapter selection code, not just adding a new adapter class
- Tests need environment variable mocks to control which agent is selected

**Phase to address:** Proxy adapter layer — configuration-driven adapter selection must be established before building individual adapters

---

### Pitfall 7: Event Stream Contains No Run-Scoped Filtering — Multi-Session Chaos

**What goes wrong:**
The current `EventBusEventLog` delivers all events from all runs to all connected SSE clients without filtering. When two agent sessions run concurrently (or when the user opens two tabs), the activity stream interleaves events from both runs. The DAG visualization and file-change panel show a mix of activities from different agents, and the user cannot tell which event belongs to which task.

**Why it happens:**
Global fan-out is the simplest design for a single-user, single-session scenario. The `AgentEventEnvelope` already carries `run_id`, `trace_id`, and `workflow` fields — the information for filtering exists. It just is not used at the delivery layer.

**How to avoid:**
Add a `run_id` query parameter to the `/api/events/stream` endpoint. The `_event_generator` filters the fan-out queue to only deliver events matching the requested `run_id`. The ring buffer catch-up on subscribe should also filter by `run_id`.

```python
@router.get("/stream")
async def events_stream(request: Request, run_id: Optional[str] = None):
    bus = _get_bus(request)
    return StreamingResponse(
        _event_generator(request, bus, run_id=run_id),
        ...
    )
```

The frontend subscribes to a specific `run_id` per panel. A session-level overview panel subscribes without a filter to see all runs.

**Warning signs:**
- Opening two agent sessions in the same browser results in interleaved events in both panels
- The DAG visualization shows nodes from different agents mixed in the same graph
- File-change events from one agent appear in another agent's activity feed

**Phase to address:** Real-time event stream phase — add run_id filtering to the SSE fan-out before building the visualization panels

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `claude -p` per message (print mode, stateless) | Simple subprocess management, no session state | ~50k token overhead per turn; no session continuity; costs scale quadratically | Never — switch to stdin-based session mode before any production usage |
| Auto-detect active agent by binary presence | Zero config for first run | Detection logic spreads to multiple modules; third agent breaks heuristic | First-run wizard only; writes result to config file, never stays as live runtime code path |
| Global SSE fan-out with no run_id filter | Simple frontend subscription, single EventSource | Two concurrent sessions produce interleaved UI state | Acceptable until multi-session support is built in the same milestone |
| Map all agent events to five generic types | Unified UI without agent-specific handling | Rich agent behavior (extended thinking, approval flows) is invisible; debugging becomes impossible | Only for an early prototype demo; add typed events before any real usage |
| Ring buffer only, no persistent event replay | No database dependency for SSE reconnection | Gap after reconnect exceeds buffer; post-mortem debugging loses events | Acceptable for development; add replay from SQLAlchemy event log before shipping |
| Polling loop for agent state (`asyncio.sleep(1.0)`) | Simple pause/cancel implementation | 1-second latency on control commands; wrong for command injection | Acceptable for coarse lifecycle control (pause/cancel); never for fine-grained turn-level commands |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Claude Code CLI (`--output-format stream-json`) | Treating stream-json as the complete event vocabulary | Subscribe to all NDJSON line types; `system` events carry `session_id` needed for session resume; do not skip unknown types |
| Claude Code CLI session resume | Using `--session-id` with a human-readable string | `--session-id` requires a valid UUID; use `claude --resume <name>` after naming sessions with `/rename` inside the session |
| Codex API (Responses API) | Rebuilding `messages` array each turn from scratch | Pass `previous_response_id` to continue a stateful chain; only `input` changes each turn |
| PTY master fd reading | Using blocking `os.read()` on the master fd inside asyncio | Register the master fd with `loop.add_reader()` for non-blocking reads; do not wrap in `asyncio.to_thread` |
| SSE behind Nginx | Default proxy buffering swallows events until the stream closes | Add `proxy_buffering off; proxy_cache off; chunked_transfer_encoding on` in the Nginx location block for `/api/events/stream` |
| EventBusEventLog fan-out | Iterating `_queues` directly during fan-out | Use `list(self._queues)` snapshot — PaperBot already does this correctly; do not regress when adding filtering |
| Subprocess env in agent adapter | Inheriting full parent environment including all API keys | Build an explicit allowlist env dict; Codex subprocess must NOT receive `ANTHROPIC_API_KEY`; Claude Code subprocess must NOT receive `OPENAI_API_KEY` |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| One SSE connection per browser panel | Browser 6-connection limit hit when studio + events + research panels are all open | Upgrade to HTTP/2 (multiple SSE streams share one TCP connection); or multiplex manually over one EventSource | At 3–4 simultaneously open dashboard panels |
| Global fan-out to unbounded SSE subscriber list | Memory grows linearly with connected clients; fan-out loop takes milliseconds | Enforce max subscriber cap; drop oldest client connection when cap is reached | At ~500 concurrent SSE connections |
| Ring buffer flush to every new subscriber | Subscribe causes burst delivery of 200 events, overwhelming slow or mobile clients | Cap catch-up replay to 50 most-recent events; send the rest from persistent log on-demand | When clients reconnect frequently (mobile, flaky network) |
| Subprocess per message (print mode) | Each message takes as long as the first; latency grows with context length | Stateful session mode with persistent subprocess | From message 3+ in any session |
| Synchronous SQLAlchemy event log writes on `append()` | High-frequency tool calls block the asyncio event loop | Move to async writes in v2.0; for now, batch writes or write in a background thread | At >50 tool calls per minute in a session |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| `project_dir` path traversal in subprocess cwd | User-controlled path executes agent with access to sensitive directories (e.g., `/etc/`) | `studio_chat.py` already has `_resolve_cli_project_dir()` with an allowlist — do NOT remove this guard during refactoring into the unified adapter |
| Passing raw user message text directly to agent stdin without role-scoping | Prompt injection: attacker-crafted message causes agent to exfiltrate files or execute arbitrary tools | Wrap user messages in a structured role envelope; the agent's system prompt must define user-sourced content as untrusted |
| Leaking `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to the wrong agent subprocess | Agent can exfiltrate the key via file-write tools or network requests | Build explicit per-agent env allowlists; Codex subprocess receives only `OPENAI_API_KEY`; Claude CLI subprocess receives only `ANTHROPIC_API_KEY` |
| Unauthenticated SSE event stream exposing tool arguments | Internal codebase structure, file contents, and implementation details visible to unauthenticated observers | Require auth (JWT or session cookie) on `/api/events/stream`; strip or redact sensitive payload fields at the fan-out layer |
| Agent name or session ID used in filesystem paths without sanitization | Attacker sets `agent_name=../../etc/cron.d/backdoor` in an event payload | Always use `uuid.uuid4()` for filesystem-facing identifiers; validate/sanitize any agent-provided strings before path construction |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Showing raw NDJSON event stream in the activity panel | Users see `{"type":"tool_use","name":"read_file","input":{"path":"..."}}` — incomprehensible | Parse events into human-readable activity descriptions server-side before streaming to the dashboard |
| Binary "connected / disconnected" agent status indicator | Users cannot tell if the agent is idle, thinking, mid-tool-execution, or hung | Expose a 4-state indicator: `idle / thinking / executing-tool / awaiting-input`; derive state from the event stream, not a ping |
| No progress indication during long tool calls | Agent appears frozen for 30+ seconds during file analysis or web search | Show the tool name and elapsed time from the `tool_use` event until the corresponding `tool_result` arrives |
| Control surface buttons visible when agent is not running | "Interrupt" button exists when there is nothing to interrupt; user clicks it, nothing happens | Disable control surface commands when agent status is `idle`; only enable during active runs |
| Interleaved events from concurrent sessions in one panel | Multi-session runs produce incomprehensible event streams | Filter by `run_id` per panel; implement a session switcher; `AgentEventEnvelope.run_id` is already present |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Subprocess streaming:** Appears to work in dev (your terminal provides a TTY), silently block-buffers in Docker/production — verify with `docker run -i` without `-t` and confirm events arrive in real-time
- [ ] **SSE reconnection:** EventSource fires a reconnect event, but verify by simulating a network drop mid-run and checking that no events are duplicated and none are lost
- [ ] **Agent session persistence:** UI restores conversation history from local storage, but verify that the agent subprocess actually received the prior context (check token counts, not UI appearance)
- [ ] **Adapter abstraction:** Works with one agent, but verify by adding a stub second adapter with different event types and confirming the dashboard renders without any adapter-specific code changes
- [ ] **Control surface interrupt:** POST returns 200, but verify the agent actually stopped — no further `tool_use` events should arrive after the interrupt signal
- [ ] **Event deduplication on reconnect:** Events look correct after reconnect, but verify by counting `seq` values for duplicates across a deliberate reconnect cycle

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| PTY absence discovered in production | MEDIUM | Add `pty.openpty()` wrapper in the subprocess adapter; for agents with stream-json mode, prefer that over PTY; test in non-TTY container |
| Token explosion from stateless subprocess discovered after launch | MEDIUM | Switch to REPL/stdin mode for new sessions; existing sessions already have high cost but new sessions will be efficient |
| Adapter abstraction hardcoded to two agents discovered when adding third | HIGH | Extract capability flags retroactively; audit all `isinstance` and agent-type conditionals; typically 1–2 sprint effort |
| SSE duplicates after reconnect discovered | LOW | Add `id:` field to SSE frames; frontend deduplicates by `seq`; no schema changes required |
| Command injection race condition discovered | MEDIUM | Add `IDLE/PROCESSING/AWAITING_INPUT` state machine to session manager; queue commands at application layer; no agent changes needed |
| Agent detection heuristics proliferated across modules | HIGH | Introduce explicit config file; migrate all detection logic to a config reader; deprecate runtime binary detection |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| PTY absence / block-buffered subprocess output (#1) | Proxy adapter layer | Test against real Claude CLI in a non-TTY Docker container; events must arrive line-by-line in real-time |
| Stateless subprocess token explosion (#2) | Proxy adapter layer — session lifecycle design | Confirm token counts do not grow with conversation length after the second turn |
| Adapter lowest-common-denominator collapse (#3) | Proxy adapter layer — interface design phase | Add a stub third adapter with different event types; confirm dashboard renders without adapter code changes |
| SSE duplicate/missing events on reconnect (#4) | Real-time event stream phase | Simulate network drop mid-run; verify no duplicate `seq` values and no event gaps |
| Command injection race condition (#5) | Dashboard control surface phase | Send interrupt during active tool call; verify no garbled agent output and no missed subsequent events |
| Hardcoded agent detection (#6) | Proxy adapter layer — configuration design | Agent selection must come from config file; verify no `find_cli()` calls remain in request-handling paths |
| No run-scoped event filtering (#7) | Real-time event stream phase | Open two concurrent sessions; confirm each panel shows only its own events |
| SSE proxy buffering (Nginx) | Deployment / infrastructure phase | Deploy behind Nginx; confirm events arrive within 1 second of emission using SSE event timestamps |
| Subprocess env secret leakage | Proxy adapter layer — security | Run Codex subprocess; assert `ANTHROPIC_API_KEY` is not present in child process env |

---

## Sources

- Codebase: `studio_chat.py` — `find_claude_cli()`, `asyncio.create_subprocess_exec` with PIPE, `FORCE_COLOR=0`, stateless `claude -p` per message
- Codebase: `codex_dispatcher.py` — Codex API loop, separate code path from Claude CLI
- Codebase: `agent_board.py` — `_run_controls` dict, `asyncio.sleep(1.0)` pause polling, command state machine
- Codebase: `events.py` — SSE fan-out, ring buffer replay, no `id:` field in emitted frames
- Codebase: `event_bus_event_log.py` — `_put_nowait_drop_oldest`, global fan-out, `list(self._queues)` snapshot
- Codebase: `agent_events.py` — `AgentEventEnvelope` with `seq`, `run_id`, `trace_id` fields already present
- [Forcing Immediate Output from Subprocesses in Python: PTY vs Buffering Solutions](https://sqlpey.com/python/forcing-immediate-subprocess-output/)
- [Building a 24/7 Claude Code Wrapper? Each Subprocess Burns 50K Tokens](https://dev.to/jungjaehoon/why-claude-code-subagents-waste-50k-tokens-per-turn-and-how-to-fix-it-41ma)
- [Claude Code CLI Playbook: REPL, Pipes, Sessions & Permissions](https://www.vibesparking.com/en/blog/ai/claude-code/docs/cli/2025-08-28-claude-code-cli-playbook-repl-pipes-sessions-permissions/)
- [Inside the Claude Agent SDK: From stdin/stdout Communication to Production](https://buildwithaws.substack.com/p/inside-the-claude-agent-sdk-from)
- [Claude Code CLI Reference — Official Docs](https://code.claude.com/docs/en/cli-reference)
- [The Law of Leaky Abstractions & the Unexpected Slowdown](https://abaditya.com/2025/08/12/the-law-of-leaky-abstractions-the-unexpected-slowdown/)
- [Introducing Any-Agent: Abstraction Layer Between Code and Agentic Frameworks — Mozilla AI](https://blog.mozilla.ai/introducing-any-agent-an-abstraction-layer-between-your-code-and-the-many-agentic-frameworks/)
- [Agent Streams Are a Mess. Here's How We Got Ours to Make Sense](https://medium.com/@ranst91/agent-streams-are-a-mess-heres-how-we-got-ours-to-make-sense-10eb3523ed57)
- [Server-Sent Events Are Still Not Production Ready After a Decade — DEV Community](https://dev.to/miketalbot/server-sent-events-are-still-not-production-ready-after-a-decade-a-lesson-for-me-a-warning-for-you-2gie)
- [The Hidden Risks of SSE: What Developers Often Overlook](https://medium.com/@2957607810/the-hidden-risks-of-sse-server-sent-events-what-developers-often-overlook-14221a4b3bfe)
- [Weaponizing Real Time: WebSocket/SSE with FastAPI — Connection Management, Reconnection, Scale-Out](https://blog.greeden.me/en/2025/10/28/weaponizing-real-time-websocket-sse-notifications-with-fastapi-connection-management-rooms-reconnection-scale-out-and-observability/)
- [GPT-5.3-Codex Bug Reports: Sessions Stall, Terminals Hang, Safety Boundaries Desync](https://www.penligent.ai/hackinglabs/gpt-5-3-codex-bug-reports-verified-why-sessions-stall-terminals-hang-and-safety-boundaries-desync/)
- [Fix Codex CLI Reconnecting Loop](https://smartscope.blog/en/generative-ai/chatgpt/codex-cli-reconnecting-issue-2025/)
- [Why Multi-Agent LLM Systems Fail: Key Issues Explained — orq.ai](https://orq.ai/blog/why-do-multi-agent-llm-systems-fail)
- [10 Reasons Your Multi-Agent Workflows Fail — InfoQ](https://www.infoq.com/presentations/multi-agent-workflow/)
- [Prompt Injection to RCE in AI Agents — Trail of Bits Blog](https://blog.trailofbits.com/2025/10/22/prompt-injection-to-rce-in-ai-agents/)
- [Patterns That Work and Pitfalls to Avoid in AI Agent Deployment — HackerNoon](https://hackernoon.com/patterns-that-work-and-pitfalls-to-avoid-in-ai-agent-deployment)

---

*Pitfalls research for: v1.2 DeepCode Agent Dashboard — agent proxy/dashboard, CLI tool proxying, agent-agnostic adapters, real-time event visualization, control surface*
*Researched: 2026-03-15*
