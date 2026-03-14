# Domain Pitfalls

**Domain:** Agent orchestration dashboard + Codex subagent bridge
**Researched:** 2026-03-14 (updated with verified Codex CLI and Claude Code agent findings)

## Critical Pitfalls

Mistakes that cause rewrites or major architectural issues.

### Pitfall 1: Building a Second Event System

**What goes wrong:** Creating separate "dashboard events" or "agent activity" models alongside the existing `AgentEventEnvelope` and `EventLogPort`. Two event streams diverge, one falls behind, dashboard shows stale data.
**Why it happens:** It feels natural to create a "clean" event type for the dashboard rather than extending the existing one. The existing `AgentEventEnvelope` has 15+ fields and feels heavyweight.
**Consequences:** Two sources of truth. Events logged to one system not visible in the other. Maintenance burden doubles. Debugging becomes "which event log did this go to?"
**Prevention:** Extend `AgentEventEnvelope.type` with new string constants (`task_queued`, `agent_spawned`, etc.). Put structured data in the existing `payload` dict. The envelope is already flexible enough.
**Detection:** If you find yourself writing a new `@dataclass` for events, stop. Use `make_event()` with a new `type` value.

### Pitfall 2: Making PaperBot an Agent Runtime

**What goes wrong:** Adding process management, agent spawning, retry logic, or scheduling to PaperBot's server. This turns PaperBot into a competing orchestration runtime alongside Claude Code.
**Why it happens:** The `ClaudeCommander` and `CodexDispatcher` already hint at orchestration capabilities. It feels natural to extend them into a full runtime.
**Consequences:** Architectural conflict with the "PaperBot = skill provider" principle. Claude Code and PaperBot fight over who owns agent lifecycle. Duplicated retry/scheduling logic. Test complexity explodes.
**Prevention:** PaperBot observes and provides tools. The `.claude/agents/codex-worker.md` file lives in Claude Code's domain. PaperBot's API receives events about what happened, never sends commands about what to do next.
**Detection:** If the server is calling `subprocess.Popen` for Codex CLI, or implementing retry/backoff for agent tasks, it has crossed the line.

### Pitfall 3: SSE Connection Exhaustion

**What goes wrong:** Each browser tab opens an SSE connection. Each SSE connection holds an `asyncio.Queue` in the EventBus. With multiple tabs, multiple runs, or long-running dashboards, connections accumulate and consume memory.
**Why it happens:** SSE connections are long-lived. Browsers have per-domain connection limits (6 in HTTP/1.1). EventBus queues grow if the consumer is slower than the producer.
**Consequences:** Memory pressure on the server. Browser hits connection limit and cannot open new SSE streams. Dead connections linger.
**Prevention:** (1) Bounded queues with backpressure -- drop oldest events if queue exceeds limit. (2) Connection cleanup on client disconnect using `request.is_disconnected()`. (3) Heartbeat timeout to kill stale connections. (4) The existing `wrap_generator` already has 30-minute timeout -- keep this.
**Detection:** Monitor active SSE connection count. Alert if > 20 concurrent connections for a single-user app.

### Pitfall 4: Studio Page Regression

**What goes wrong:** Replacing the studio page with the agent dashboard breaks existing Paper2Code workflows that do not use agents. Users who launch "generate code" from the papers page find a Kanban board instead of the reproduction log.
**Why it happens:** PROJECT.md says "replaces studio page" but existing studio functionality (PaperGallery, ReproductionLog, context pack generation) must be preserved.
**Consequences:** Users cannot use Paper2Code without the agent board. Existing deep links (`/studio?paper_id=X&generate=true`) break.
**Prevention:** The three-panel layout should be a new view mode within the existing studio page, not a complete replacement. The studio page already has `viewMode` state (`log`, `context`, `agent_board`). Add a `dashboard` mode that shows the three-panel layout. Keep other modes working.
**Detection:** Existing URL params (`paper_id`, `context_pack_id`, `generate`) must still work after the change.

### Pitfall 5: Codex CLI Version Mismatch

**What goes wrong:** Agent definition assumes `codex exec` flags that exist in one version but not another. The Rust rewrite (v0.98+) changed the CLI surface significantly from the Node.js era.
**Why it happens:** Blog posts, tutorials, and training data reference old Node.js Codex CLI syntax. Flag behavior, JSONL event schemas, and sandbox mechanics differ between versions. The codebase is now 95.7% Rust as of v0.98.0 (February 2026).
**Consequences:** Codex subagent silently fails or produces unparseable output. Claude Code reports vague "command failed" errors.
**Prevention:** (1) Pin minimum version in agent definition: require `codex --version` >= 0.98. (2) Use only flags from [official CLI reference](https://developers.openai.com/codex/cli/reference): `--json`, `--full-auto`, `--output-schema`, `--model`. (3) JSONL event types to expect: `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.*`, `error`. (4) Test the agent definition with a simple `codex exec --json "echo hello"` before complex tasks.
**Detection:** JSONL parsing errors in codex-worker subagent output. Missing expected event types.

## Moderate Pitfalls

### Pitfall 6: Overloading AgentEventEnvelope with UI Concerns

**What goes wrong:** Adding UI-specific fields to the event envelope (color, icon, display_name, panel_position). The domain model becomes coupled to a specific frontend.
**Prevention:** Keep `AgentEventEnvelope` as a domain-level data structure. Map to UI concerns in the frontend Zustand store or rendering components. The `tags` dict can carry hints, but do not add UI fields to the dataclass.

### Pitfall 7: EventBus Memory Leak on Unsubscribe

**What goes wrong:** Subscribers are added to EventBus but never removed when SSE connections close. Queue references accumulate.
**Prevention:** Use `try/finally` in the SSE route handler to always unsubscribe. The existing `wrap_generator` finally block already closes the generator -- hook cleanup into that same pattern. Set max queue size per subscriber (e.g., 1000 events). Add a TTL sweep that removes stale subscribers.

### Pitfall 8: Codex Worker Agent Definition Scope Creep

**What goes wrong:** The `.claude/agents/codex-worker.md` file grows to include complex decision-making logic, branching workflows, or Paper2Code-specific pipeline knowledge.
**Why it happens:** Claude Code agent definitions support rich frontmatter fields (`tools`, `disallowedTools`, `model`, `permissionMode`, `mcpServers`, `hooks`, `maxTurns`, `skills`, `memory`). The temptation is to use all of them.
**Prevention:** Keep the agent definition under 100 lines. Minimal frontmatter: `name`, `description`, `tools` (Bash, Read, Write, Glob), `model`. The system prompt body describes: (1) role, (2) how to invoke `codex exec`, (3) event reporting protocol via MCP tools, (4) output conventions. Pipeline-specific logic stays in PaperBot's Python services. The agent definition is reloaded by Claude Code at session start or via `/agents` command.

### Pitfall 9: Ignoring Existing AgentBoard Component

**What goes wrong:** Building the dashboard from scratch instead of reusing `AgentBoard.tsx`, which already has Kanban columns, task cards, execution logs, human review, and SSE consumption.
**Prevention:** The three-panel layout should embed `AgentBoard` as the left (tasks) panel. The existing component handles task CRUD, status transitions, and review workflows. Wrap it, do not rewrite it.

### Pitfall 10: SSE Reconnection Gap

**What goes wrong:** Browser `EventSource` disconnects (network blip, laptop sleep) and misses events. No catch-up mechanism. Native `EventSource` auto-reconnects but starts from "now".
**Prevention:** Track `seq` (sequence number) from the SSE envelope. On reconnect, pass `Last-Event-Id` header or query param `?after_seq=N`. Server replays missed events from SQLAlchemy store before switching to live EventBus.

### Pitfall 11: Codex exec Timeout Handling

**What goes wrong:** `codex exec` hangs on a complex task. Claude Code subagent waits indefinitely.
**Prevention:** Set explicit timeout in agent definition: `timeout 300 codex exec ...`. Handle timeout as a structured error event. The existing `CodexDispatcher.dispatch_timeout_seconds` (default 180s) is a good reference pattern.

## Minor Pitfalls

### Pitfall 12: Timezone Inconsistencies in Event Display

**What goes wrong:** `AgentEventEnvelope.ts` uses UTC. `agent_board.py` uses `datetime.utcnow()` (naive UTC). Frontend receives ISO strings and must format for user's timezone. Mixed naive/aware timestamps cause sorting bugs.
**Prevention:** Always use `datetime.now(timezone.utc)` (aware UTC), never `datetime.utcnow()` (naive). The existing `utcnow()` helper in `message_schema.py` does this correctly -- use it consistently.

### Pitfall 13: Breaking the DI Container Singleton Pattern

**What goes wrong:** EventBus created outside the DI container, or multiple instances exist.
**Prevention:** Register EventBus in `Container.instance()`. Access via DI, not module-level globals.

### Pitfall 14: Codex Multi-Agent Feature Flag

**What goes wrong:** Assuming Codex multi-agent workflows are generally available. They are currently experimental and require explicit opt-in.
**Prevention:** The codex-worker agent definition does NOT use Codex's built-in multi-agent features. It uses single-task `codex exec` invocations. Codex multi-agent (`/experimental` or `multi_agent` config flag) is orthogonal and should not be relied upon.

### Pitfall 15: Zustand Store Size for Long Sessions

**What goes wrong:** Long-running agent sessions accumulate thousands of events in the Zustand store, causing React re-render performance issues.
**Prevention:** Ring buffer for events (keep last N, e.g., 500). Persist older events server-side only. Virtualize the activity feed list if performance degrades.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|---|---|---|
| EventBus implementation | Memory leak from unsubscribed queues (#7) | Bounded queues + cleanup in finally block |
| Event type extensions | Second event system temptation (#1) | Extend existing types only, no new models |
| Three-panel layout | Studio page regression (#4) | New view mode, not page replacement |
| Agent dashboard store | Re-implementing AgentBoard logic (#9) | Embed existing component, extend store |
| Codex worker definition | Scope creep (#8), version mismatch (#5) | Keep under 100 lines, pin CLI version, test with simple task first |
| Overflow delegation | Making PaperBot an agent runtime (#2) | Agent definition delegates, server observes |
| SSE streaming | Connection exhaustion (#3), reconnection gaps (#10) | Bounded queues, cleanup, seq-based catch-up |

## Sources

- Codebase: `agent_board.py` uses `datetime.utcnow()` (naive) vs `message_schema.py` uses `datetime.now(timezone.utc)` (aware)
- Codebase: `streaming.py` has 30-minute timeout and heartbeat -- good defaults to preserve
- Codebase: `studio/page.tsx` has `viewMode` state supporting multiple modes -- extend, do not replace
- Codebase: `codex_dispatcher.py` has `dispatch_timeout_seconds` defaulting to 180s
- Project: `.planning/PROJECT.md` constraints: "reuse existing", "PaperBot = skill provider"
- [Codex CLI Reference](https://developers.openai.com/codex/cli/reference) -- verified CLI flags and JSONL event types
- [Codex Non-Interactive Mode](https://developers.openai.com/codex/noninteractive/) -- `codex exec` behavior, --json output format
- [Claude Code Custom Subagents](https://code.claude.com/docs/en/sub-agents) -- agent definition format, frontmatter fields, hot reload behavior
- [Codex Multi-Agent Docs](https://developers.openai.com/codex/multi-agent/) -- experimental status of multi-agent feature
