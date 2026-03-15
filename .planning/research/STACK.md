# Technology Stack — v1.2 Agent Dashboard Additions

**Milestone:** v1.2 DeepCode Agent Dashboard (agent-agnostic proxy + multi-agent adapter layer)
**Researched:** 2026-03-15
**Confidence:** HIGH (agent SDK APIs verified via official docs and npm; transport patterns confirmed)

---

## Principle: Additive Only

The existing stack covers every concern except CLI agent proxying, subprocess/PTY management,
multi-agent adapter translation, and the control-path WebSocket needed for bidirectional
chat-to-agent communication. This document covers only the **new capabilities required for v1.2**
and exactly how they integrate with the existing FastAPI SSE, EventBus, and @xyflow/react stack.

Nothing is added for its own sake. Three areas need new libraries:
1. **Python backend** — subprocess management for CLI-based agents + agent adapter layer
2. **Node.js (Next.js web)** — PTY process management + WebSocket relay for terminal display
3. **Web frontend** — WebSocket upgrade for the XTerm terminal already on the studio page

---

## What Already Exists (Do NOT Re-add)

| Capability | Package | Status |
|---|---|---|
| SSE streaming | FastAPI `StreamingResponse` + `EventBus` | Ships in v1.1. All agent activity events go through this path. |
| Agent event schema | `AgentEventEnvelope` (run_id, trace_id, span_id) | Extend; never create a parallel schema. |
| DAG visualization | `@xyflow/react ^12.10.0` | In `web/package.json`. Reuse for team decomposition graph. |
| Monaco editor | `@monaco-editor/react ^4.7.0` | In `web/package.json`. Keep on studio page. |
| XTerm terminal | `xterm ^5.3.0` + `xterm-addon-fit ^0.8.0` | In `web/package.json`. Needs WebSocket addon added (see below). |
| Chat streaming | Vercel AI SDK `ai ^5.0.116` | In `web/package.json`. Use for proxy chat stream surface. |
| MCP tool surface | `@modelcontextprotocol/sdk ^1.25.1` | In `web/package.json`. PaperBot MCP server is v1.0 prerequisite. |
| Zustand state | `zustand ^5.0.9` | In `web/package.json`. Use for agent activity store. |
| React panel layout | `react-resizable-panels ^4.0.11` | In `web/package.json`. Use for three-panel IDE layout. |

---

## New Additions: Python Backend

### 1. Agent SDK — Claude Code (Claude Agent SDK)

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `@anthropic-ai/claude-agent-sdk` | `^0.2.47` (npm) | Programmatic control of Claude Code CLI | The official Anthropic SDK spawns the Claude Code CLI as a subprocess, communicating via JSON-lines over stdin/stdout. Provides `query()` for one-shot tasks and `ClaudeSDKClient` for stateful multi-turn sessions. Emits typed message objects: `SystemMessage`, `AssistantMessage`, `ResultMessage`, `CompactBoundaryMessage`, `StreamEvent` (with `includePartialMessages: true`). This is the correct integration surface — do NOT shell-exec `claude -p` manually; the SDK handles process lifecycle, permission callbacks, and streaming. **Node.js side only.** |

The Claude Agent SDK is a **Node.js / TypeScript library** — it must run in the Next.js API route layer (or a dedicated Node.js sidecar), not in the Python FastAPI backend. Python interacts with it via the `claude -p --output-format stream-json` CLI flag pattern if needed from Python, or delegates to the Node.js layer.

**Python alternative for Claude Code subprocess (when needed directly from FastAPI):**
Use `asyncio.create_subprocess_exec` with `claude -p <prompt> --output-format stream-json --include-partial-messages` and stream stdout line-by-line as JSONL into `AgentEventEnvelope`. This is the adapter implementation pattern (see Architecture section).

### 2. Agent SDK — Codex CLI

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `@openai/codex-sdk` | `^0.112.0` (npm) | Programmatic control of OpenAI Codex CLI | Official TypeScript SDK (`npm install @openai/codex-sdk`, Node.js 18+). API: `new Codex()` → `codex.startThread()` → `thread.run(prompt)` for stateful sessions; `thread.runStreamed(prompt)` returns an async generator of structured events (tool calls, streaming responses, file change notifications). Threads persist in `~/.codex/sessions` and can be resumed via `resumeThread(threadId)`. Internally wraps `codex exec --json` JSONL stream. **Node.js side only.** |

For Python-side Codex integration, use `codex exec --json <prompt>` subprocess and parse JSONL events. Event types emitted: `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.*` (agent messages, reasoning, command executions, file changes, MCP tool calls, web searches, plan updates). The existing `codex_dispatcher.py` uses OpenAI API directly — replace with CLI-subprocess approach for the unified adapter, or keep API path as a fallback when the CLI is not installed.

### 3. Agent SDK — OpenCode

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `@opencode-ai/sdk` | `latest` (npm) | Type-safe client for a running OpenCode server | OpenCode exposes a REST+SSE server. The TypeScript SDK wraps it: `createOpencodeClient()` connects to a running instance; `client.event.subscribe()` returns an SSE stream of typed events (`event.type`, `event.properties`). Session methods include `sessions.prompt()`, `sessions.command()`, `sessions.shell()`. OpenCode also supports non-interactive mode: `opencode -p "prompt" -f json` for scripted use. **Node.js side only.** |

### 4. No New Python Packages Required

The Python adapter layer is built using stdlib only:

- `asyncio.create_subprocess_exec` — spawn agent CLI processes (already used in `repro/` executors)
- `asyncio.StreamReader.readline()` — line-by-line JSONL parsing from agent stdout
- `abc.ABC` + `abc.abstractmethod` — abstract `AgentAdapter` base class
- `AgentEventEnvelope` — existing event schema (extend, not replace)

No new Python packages are needed for the adapter layer. The existing `codex_dispatcher.py` and `claude_commander.py` contain the patterns to extract; the new `AgentAdapter` ABC unifies them.

---

## New Additions: Web Frontend (Next.js)

### 5. XTerm WebSocket Addon (Migration + Addition)

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `@xterm/addon-attach` | `^0.11.0` | Attach XTerm terminal to a WebSocket for live PTY relay | The existing `xterm-addon-attach` (unscoped) is deprecated. The new `@xterm/addon-attach` is the official successor. Usage: `new AttachAddon(webSocket)` → `terminal.loadAddon(attachAddon)`. Required to relay PTY output from the backend to the in-browser XTerm terminal when an agent runs interactively. The existing `xterm ^5.3.0` and `xterm-addon-fit ^0.8.0` stay; only the attach addon changes. |

Also migrate the import: `from 'xterm'` → `from '@xterm/xterm'` and `xterm-addon-fit` → `@xterm/addon-fit` for consistency with the scoped package ecosystem (the unscoped packages are deprecated).

### 6. WebSocket Server (Backend — Next.js API Routes)

No new npm package needed for WebSocket on the Next.js side. Next.js 16 supports WebSocket upgrade in API routes. However, for the PTY relay specifically, use the `ws` package which is already a transitive dependency of many existing packages (confirm with `ls node_modules/ws`). If not present:

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `ws` | `^8.18.0` | WebSocket server in Node.js API route | Low-level WebSocket server for the PTY relay path. The `@xterm/addon-attach` client expects a raw WebSocket, not Socket.io. `ws` is the minimal dependency — no overhead, well-maintained (Microsoft's node-pty documentation recommends it). |

**Why WebSocket for PTY, SSE for agent events:** The PTY terminal requires bidirectional communication (user keystrokes → agent stdin; agent stdout → browser). SSE is unidirectional (server → client only). The agent activity event stream (tool calls, file changes, task status) uses the existing FastAPI SSE path. The terminal relay uses WebSocket. These are two separate channels.

---

## New Additions: Node.js Backend (PTY Management)

### 7. node-pty (PTY Process Management)

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `node-pty` | `^1.1.0` | Spawn CLI agents in a pseudo-terminal | node-pty provides `forkpty(3)` bindings for Node.js, allowing CLI tools to behave as if they're running in a real terminal (color output, cursor control, interactive prompts). The Claude Code CLI and Codex CLI both detect whether they're connected to a TTY and behave differently in non-TTY mode — node-pty ensures agents get a real terminal environment. `pty.spawn()` returns an object with `onData()` (stream output to WebSocket) and `write()` (send user input). Platform support: Linux, macOS, Windows (Windows 10 1809+ ConPTY). Note: not thread-safe; use one pty per session, managed by a session map keyed on `run_id`. |

**Where it runs:** In a Node.js sidecar or Next.js custom server (`server.js`), NOT in a Next.js API route handler (API routes are serverless-style and do not support long-lived processes). The sidecar pattern: Next.js custom server (`next start` with `server.js`) holds the pty sessions map; API routes communicate with it via in-process function calls or local IPC.

**Alternative if no custom server:** Delegate PTY management entirely to the Python FastAPI backend using `asyncio.create_subprocess_exec` with `pty` module from Python stdlib for Linux/macOS. Python's `pty.openpty()` gives a file descriptor pair; wrap with asyncio for async reads. This avoids the Node.js sidecar but requires Python ≥3.10 (already required). Use this path when deploying in Docker (single-container, avoids Node.js process management complexity).

---

## Architecture Integration Map

```
User Browser
    │
    ├── SSE stream (EventSource)     → FastAPI /api/agent/events  → EventBus fan-out
    │                                                                    │
    │                                                              AgentEventEnvelope
    │                                                                    │
    │                                                       Agent Adapter (Python ABC)
    │                                                      ┌──────────┴────────────┐
    │                                               ClaudeAdapter         CodexAdapter
    │                                              (subprocess:           (subprocess:
    │                                               claude -p             codex exec --json)
    │                                               --output-format
    │                                               stream-json)
    │
    ├── WebSocket (ws)               → Next.js custom server → node-pty PTY process
    │   (XTerm terminal I/O)              (pty session map)        (agent CLI)
    │
    └── HTTP fetch (chat proxy)      → Next.js API route   → @anthropic-ai/claude-agent-sdk
        (Vercel AI SDK useChat)            /api/agent/chat    OR @openai/codex-sdk
                                                               (SDK spawns CLI subprocess)
```

**Key boundary:** Python backend owns agent event observability (via adapter → EventBus → SSE).
Node.js layer owns chat proxy and interactive PTY terminal (agent SDK + node-pty).
Frontend owns visualization (@xyflow/react team graph, XTerm terminal, event feed panel).

---

## Recommended Stack Summary

### Python Backend — New

| Package | Version | Install | Notes |
|---|---|---|---|
| No new packages | — | — | Use stdlib asyncio subprocess + existing AgentEventEnvelope |

### Node.js / Next.js — New

| Package | Version | Install | Notes |
|---|---|---|---|
| `@anthropic-ai/claude-agent-sdk` | `^0.2.47` | `npm install` | Claude Code programmatic control |
| `@openai/codex-sdk` | `^0.112.0` | `npm install` | Codex programmatic control |
| `@opencode-ai/sdk` | `latest` | `npm install` | OpenCode REST+SSE client |
| `node-pty` | `^1.1.0` | `npm install` | PTY process management |
| `ws` | `^8.18.0` | `npm install` | WebSocket server (if not already transitive) |

### Web Frontend — Migrate + Add

| Package | Change | Version | Notes |
|---|---|---|---|
| `xterm` | Migrate to `@xterm/xterm` | `^5.3.0` | Scoped package, same API |
| `xterm-addon-fit` | Migrate to `@xterm/addon-fit` | `^0.10.0` | Scoped package |
| `@xterm/addon-attach` | **Add new** | `^0.11.0` | WebSocket attach for PTY relay |

---

## Installation

```bash
# Web dashboard — add new agent SDKs and node-pty
cd web
npm install @anthropic-ai/claude-agent-sdk @openai/codex-sdk @opencode-ai/sdk
npm install node-pty ws
npm install @xterm/xterm @xterm/addon-fit @xterm/addon-attach

# Remove deprecated unscoped xterm packages after migrating imports
npm uninstall xterm xterm-addon-fit
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|---|---|---|---|
| Claude Code proxying | `@anthropic-ai/claude-agent-sdk` npm | Raw `claude -p` subprocess with manual JSON parsing | The SDK handles process lifecycle, permission callbacks, streaming event objects, and session management. Manual parsing duplicates work already done in the SDK. |
| Codex proxying | `@openai/codex-sdk` npm | `codex exec --json` subprocess directly | Same rationale — the SDK adds structured events, thread management, and `runStreamed()` async generator. Use CLI-subprocess path only in Python adapter fallback. |
| PTY management | `node-pty` | Python `pty.openpty()` (stdlib) | node-pty integrates naturally with the Node.js WebSocket + xterm stack. Python pty module is viable in single-container deployments but requires asyncio wrapping and doesn't support Windows ConPTY. |
| Terminal I/O transport | WebSocket (`ws` + `@xterm/addon-attach`) | Socket.io | Socket.io adds overhead (polling fallback, namespaces) for a use case that is simply PTY byte streaming. Raw `ws` is the right choice — it's what `@xterm/addon-attach` expects natively. |
| Agent event transport | FastAPI SSE (existing) | WebSocket for events | Events are unidirectional server→client; SSE is the correct primitive and matches the existing EventBus infrastructure. Adding WebSocket for events would duplicate transport logic. |
| OpenCode proxying | `@opencode-ai/sdk` + REST/SSE | Direct HTTP fetch with manual parsing | The SDK is generated from the server's OpenAPI spec — it's the authoritative typed interface. The SSE `event.subscribe()` method is the right path for real-time activity. |
| Agent adapter in Python | Pure Python ABC + asyncio subprocess | LangChain agent abstraction | LangChain's agent abstractions assume it owns the agent loop. PaperBot's requirement is pure proxy/observation — it does NOT own the agent loop. A lightweight custom ABC costs zero dependencies and makes the contract explicit. |

---

## What NOT to Install

| Avoid | Why | Use Instead |
|---|---|---|
| `socket.io` | Heavyweight transport layer with polling fallback, namespaces, rooms — unnecessary for a single PTY relay channel | Raw `ws` WebSocket + `@xterm/addon-attach` |
| `blessed` / `ink` terminal output parsing | Designed for building TUI apps, not parsing other processes' terminal output | Parse JSONL events from agent SDKs directly; use xterm.js to render raw PTY bytes |
| Any "universal agent framework" (LangGraph, AutoGen, CrewAI) | These assume ownership of the agent loop and orchestration logic. PaperBot v1.2 is explicitly a proxy/observer — the host agent decides; PaperBot visualizes | Custom `AgentAdapter` ABC with thin subprocess/SDK wrappers |
| `xterm-addon-attach` (unscoped) | Deprecated; last published 2+ years ago | `@xterm/addon-attach ^0.11.0` |
| `xterm` (unscoped) | Being deprecated in favor of scoped `@xterm/xterm` | Migrate imports to `@xterm/xterm` |
| `puppeteer` / `playwright` for agent control | Browser automation overhead; agents are CLI tools, not web apps | Agent SDKs (`@anthropic-ai/claude-agent-sdk`, `@openai/codex-sdk`) |

---

## Agent Protocol Details

### Claude Code CLI

- **Headless mode:** `claude -p "<prompt>" --output-format stream-json --include-partial-messages`
- **Event stream:** JSONL on stdout. Message types: `stream_event` (type: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_stop`), `AssistantMessage`, `ResultMessage`, `SystemMessage`
- **Session continuity:** `--continue` (last session) or `--resume <session_id>`
- **SDK package:** `@anthropic-ai/claude-agent-sdk ^0.2.47` — wraps CLI subprocess; `query()` for one-shot, `ClaudeSDKClient` for multi-turn; `includePartialMessages: true` enables `StreamEvent` objects
- **Confidence:** HIGH — official Anthropic docs verified at code.claude.com/docs/en/headless

### OpenAI Codex CLI

- **Non-interactive mode:** `codex exec --json "<prompt>"` — emits JSONL to stdout
- **Event types:** `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.*` (covers agent messages, reasoning, command executions, file changes, MCP tool calls, web searches, plan updates)
- **Session continuity:** `codex exec --resume <thread-id>`
- **SDK package:** `@openai/codex-sdk ^0.112.0` — `Codex` class, `startThread()`, `thread.run()`, `thread.runStreamed()` (async generator of structured events)
- **Confidence:** HIGH — developers.openai.com/codex/sdk verified, npm version 0.112.0 confirmed

### OpenCode

- **Non-interactive mode:** `opencode -p "<prompt>" -f json`
- **Server mode:** `opencode serve` starts REST+SSE server; attach with `@opencode-ai/sdk`
- **Event stream:** `client.event.subscribe()` — SSE stream of `{type, properties}` events
- **Session methods:** `sessions.prompt()`, `sessions.command()`, `sessions.shell()`
- **Confidence:** MEDIUM — opencode.ai/docs/sdk verified; event type schema not fully documented; validate against a running instance in Phase 1

---

## Version Compatibility

| Package | Compatible With | Notes |
|---|---|---|
| `node-pty ^1.1.0` | Node.js 18+, Linux/macOS/Windows 10 1809+ | Not thread-safe; one PTY per session. Windows requires ConPTY (Win10 1809+). |
| `@xterm/addon-attach ^0.11.0` | `@xterm/xterm ^5.x` | Must use scoped `@xterm/xterm`, not unscoped `xterm` package. |
| `@anthropic-ai/claude-agent-sdk ^0.2.47` | Node.js 18+, requires `claude` CLI installed | SDK spawns `claude` CLI as subprocess — Claude Code must be installed and authenticated separately. |
| `@openai/codex-sdk ^0.112.0` | Node.js 18+, requires `codex` CLI installed | SDK wraps `codex` CLI; Codex must be installed and authenticated. |
| `ai ^5.0.116` (existing) | React 19, Next.js 16 | Already in package.json. Use `useChat` hook as the chat proxy surface — no upgrade needed for v1.2. |
| `@xyflow/react ^12.10.0` (existing) | React 19 | Already in package.json. Extend with dynamic `setNodes`/`setEdges` for team decomposition graph updates from agent events. |

---

## Sources

- [Claude Code headless/programmatic docs](https://code.claude.com/docs/en/headless) — HIGH confidence, verified 2026-03-15
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — HIGH confidence
- [Claude Agent SDK TypeScript releases](https://github.com/anthropics/claude-agent-sdk-typescript/releases) — HIGH confidence, v0.2.47 confirmed
- [Codex SDK docs](https://developers.openai.com/codex/sdk/) — HIGH confidence, verified 2026-03-15
- [@openai/codex-sdk npm](https://www.npmjs.com/package/@openai/codex-sdk) — HIGH confidence, v0.112.0 confirmed
- [Codex non-interactive mode docs](https://developers.openai.com/codex/noninteractive/) — HIGH confidence
- [OpenCode SDK docs](https://opencode.ai/docs/sdk/) — MEDIUM confidence (event type schema needs validation)
- [node-pty npm](https://www.npmjs.com/package/node-pty) — HIGH confidence, v1.1.0 confirmed
- [@xterm/addon-attach npm](https://www.npmjs.com/package/@xterm/addon-attach) — HIGH confidence, v0.11.0 confirmed
- [Vercel AI SDK 5 blog post](https://vercel.com/blog/ai-sdk-5) — HIGH confidence
- [FastAPI SSE vs WebSocket best practices 2025](https://potapov.me/en/make/websocket-sse-longpolling-realtime) — MEDIUM confidence
- Codebase: `web/package.json` — confirmed existing deps (@xyflow/react, xterm, ai, @ai-sdk/*, @modelcontextprotocol/sdk)
- Codebase: `src/paperbot/infrastructure/swarm/codex_dispatcher.py` — confirmed existing Codex API integration (to be replaced by adapter layer)
- Codebase: `src/paperbot/infrastructure/swarm/claude_commander.py` — confirmed existing Claude API integration (to be replaced by adapter layer)

---

*Stack research for: v1.2 DeepCode Agent Dashboard — agent-agnostic proxy, multi-agent adapter layer, real-time visualization*
*Researched: 2026-03-15*
