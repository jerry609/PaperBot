# Feature Landscape

**Domain:** Agent orchestration dashboard for AI coding agents (Claude Code / Codex subagent bridge)
**Researched:** 2026-03-14
**Confidence:** HIGH -- existing codebase already implements ~60% of the foundation; ecosystem patterns are well-established

## Table Stakes

Features users expect from an agent orchestration dashboard. Missing any of these makes the product feel broken or incomplete.

| Feature | Why Expected | Complexity | Dependencies on Existing | Notes |
|---------|--------------|------------|--------------------------|-------|
| **Task Kanban board** | Every orchestration dashboard (CrewAI, Conductor, Composio, VS Code Agent HQ) uses column-based task visualization. Users mentally model agent work as Planning -> Running -> Review -> Done. | Low | **Already built**: `AgentBoard.tsx` has 5-column Kanban (planning, in_progress, ai_review, human_review, done) with task cards, status badges, progress bars. | Extend, do not rebuild. Current implementation is solid. |
| **Real-time activity feed** | Users need to see what agents are doing *right now*. Every competing dashboard (Claude Code Agent Monitor, multi-agent-dashboard, CrewAI tracing) streams live events. Without this, the dashboard feels static/dead. | Medium | **Partially built**: SSE infrastructure exists (`streaming.py`, `readSSE` client). `AgentEventEnvelope` has run_id/trace_id/span_id/type/payload. `AgentBoard` already consumes SSE progress events. | Need new event types (tool_call, file_change, lifecycle) and a dedicated scrolling feed panel. |
| **Agent lifecycle status** | Users must see which agents are idle/working/completed/errored at a glance. Claude Code Agent Monitor tracks SessionStart -> working -> SubagentStop -> SessionEnd. | Low | **Partially built**: `AgentBoard` shows per-task status. `AgentEventEnvelope` has `role` (orchestrator/worker/evaluator) and `agent_name` fields. | Add agent-level status indicators (not just task-level). Show Claude Code vs Codex agent identity. |
| **Execution log viewer** | Terminal-style log output for each task is universal. VS Code renders read-only xterm.js inside conversations. Current studio has `ExecutionLog.tsx`. | Low | **Already built**: `TaskDetailDialog` in `AgentBoard.tsx` has a terminal-styled execution log tab with timestamp/phase/event/message formatting and level-based coloring. | Reuse existing implementation. Consider adding log-level filtering. |
| **Task detail modal/panel** | Click a task card, see full details (description, subtasks, logs, files, review actions). Standard pattern across all orchestration UIs. | Low | **Already built**: `TaskDetailDialog` with tabs for Overview, Subtasks, Logs, Files. Includes human review decision flow (approve/request_changes). | Already complete. |
| **Error surfacing and failure states** | When an agent fails, users need immediate visibility. `FailFastEvaluator` already exists in the core. | Low | **Partially built**: `AgentBoard` shows `lastError` in red, handles `task_failed` SSE events, displays run-level errors. | Extend to show Codex-specific failures (timeout, sandbox crash). |
| **Human review workflow** | Human-in-the-loop review is table stakes for code generation. Every serious tool (Conductor, Codex app) requires human approval before merging agent output. | Low | **Already built**: `AgentBoard` has full human review flow -- review notes textarea, approve/request_changes buttons, review doc generation, VS Code deep-link for file inspection. `codex_dispatcher.py` generates review docs. | Already complete and well-designed. |
| **File list per task** | Users need to see what files an agent created/modified. Codex app and Conductor both show per-agent file lists. | Low | **Already built**: `TaskDetailDialog` Files tab extracts files from Codex output. `CodexResult.files_generated` tracks written files. `codex_dispatcher._persist_output` generates user review docs with file/function inventories. | Extend to show diffs rather than just file paths (see Differentiators). |
| **SSE-based live updates** | WebSocket or SSE for push-based updates. Polling is unacceptable for real-time agent monitoring. | Low | **Already built**: FastAPI SSE streaming (`streaming.py`), client-side `readSSE` utility, `AgentBoard` already consumes SSE stream from `/api/agent-board/sessions/{id}/run`. | Extend existing SSE infrastructure to new event types. |

## Differentiators

Features that set PaperBot's agent dashboard apart. Not universally expected, but highly valued.

| Feature | Value Proposition | Complexity | Dependencies on Existing | Notes |
|---------|-------------------|------------|--------------------------|-------|
| **Three-panel IDE layout** | Most dashboards are either pure Kanban (AgentBoard) or pure IDE (studio). Combining task management + live activity feed + file changes in a unified three-panel layout (like VS Code's sidebar/editor/panel) is rare. PROJECT.md specifies: tasks panel | agent activity panel | files panel. | Medium | **Partially built**: Studio page has Monaco editor (`DeepCodeEditor.tsx`) and file browser (`FilesPanel.tsx`). AgentBoard is a standalone Kanban. Need to merge into single layout. | Key differentiator. Replaces studio page per project requirements. |
| **File change visualization with diffs** | Show what agents changed in files, not just file names. `DiffViewer.tsx` already exists in studio components. Competing tools like Conductor show "review their code as they progress." | Medium | **Already built**: `DiffViewer.tsx` exists. Runbook API has `/api/runbook/diff` and `/api/runbook/snapshot` endpoints for before/after comparison. | Wire DiffViewer into the files panel of the three-panel layout. Existing infrastructure covers this. |
| **Codex subagent bridge** | Claude Code delegates overflow tasks to Codex via `.claude/agents/codex-worker.md`. This is PaperBot-specific -- the agent definition understands paper reproduction context. Most Codex bridges are generic; PaperBot's knows about Paper2Code stages. | Medium | **Partially built**: `codex_dispatcher.py` sends tasks to OpenAI API. `claude_commander.py` exists in swarm/. `AgentBoard` handles `task_codex_done` events and distinguishes Claude vs Codex assignees (Bot icon vs Cpu icon). | The bridge is a `.claude/agents/` definition file, not server code. Server-side event logging captures delegation events. |
| **Paper2Code overflow delegation** | When Claude Code hits capacity during Paper2Code reproduction, specific pipeline stages (generation, verification) can overflow to Codex. Unique to PaperBot -- no generic orchestrator understands paper reproduction pipelines. | High | **Partially built**: Paper2Code has multi-stage pipeline (planning -> blueprint -> environment -> generation -> verification) in `repro/`. `AgentBoard` already handles Codex dispatch/completion lifecycle. | Need delegation logic: which stages overflow, how results merge back. This is the highest-complexity differentiator. |
| **Agent DAG visualization** | Show the dependency graph of agent tasks. @xyflow/react is already in the web stack for DAG visualization. CrewAI and Langflow offer visual workflow builders; PaperBot can show the live execution DAG. | Medium | **Already available**: @xyflow/react is a web dependency (used for workflow DAG visualization elsewhere). `AgentEventEnvelope` has parent_span_id for tree relationships. | Build a reactive DAG view where nodes represent agents/tasks, edges show delegation/dependency, and status colors update in real-time. |
| **Tool call logging visualization** | Show each tool invocation by agents (MCP tools, file operations, shell commands) in a structured timeline. Claude Code Agent Monitor and the hooks-based observability dashboard both track tool calls. CrewAI calls this "real-time tracing." | Medium | **Partially built**: `AgentEventEnvelope.type` supports `tool_call` and `tool_result`. Event log infrastructure (SQLAlchemy, memory, composite) persists events. | Need structured tool call rendering: tool name, arguments, result summary, duration. Timeline view rather than just log lines. |
| **Cross-agent context sharing** | `ScoreShareBus` enables cross-stage evaluation sharing. Unique to PaperBot -- share paper analysis scores between judge/summarizer/relevance agents. | Low | **Already built**: `ScoreShareBus` and `AgentCoordinator` in `core/collaboration/`. | Expose in dashboard: show which scores/insights flow between agents. |

## Anti-Features

Features to explicitly NOT build. Building these would violate the architecture or waste effort.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Custom agent orchestration runtime** | PROJECT.md explicitly says: "host agents (Claude Code) own orchestration." PaperBot is a skill provider, not a runtime. Building an orchestration engine duplicates what Claude Code already does. | Expose tools via MCP. Let Claude Code handle agent spawning, scheduling, retries. Dashboard is read-only observation + human review triggers. |
| **Per-host adapters** | PROJECT.md: "one MCP surface serves all." Do not build separate integrations for Claude Code vs Codex vs Cursor. | Single MCP tool surface. Event logging accepts any agent's events through the same envelope schema. |
| **Visual workflow builder** | Langflow-style drag-and-drop agent design is a massive scope expansion. PaperBot's workflows are code-defined (Paper2Code pipeline stages). Visual builders add complexity without value for this use case. | Show execution DAG read-only. Workflows are defined in code (`repro/orchestrator.py`, pipeline nodes). |
| **Business logic duplication in dashboard** | PROJECT.md: "tools must reuse existing services." Do not reimplement paper analysis, scholar tracking, or Paper2Code logic in the dashboard frontend. | Dashboard calls existing API endpoints. New features extend existing services, not parallel them. |
| **Agent chat interface** | The dashboard should not become another chat UI. Claude Code and Codex already have their own conversation interfaces. Adding chat duplicates their core UX. | Show agent *output* (logs, files, results) not agent *input* (prompts, conversations). Human review is via structured approve/reject, not freeform chat. |
| **Real-time code editing in dashboard** | Monaco editor exists in studio, but the agent orchestration dashboard should not be where users write code. That is the IDE's job. | Show file diffs (read-only DiffViewer). Deep-link to VS Code for actual editing (already implemented in AgentBoard). |
| **Codex CLI wrapper** | Do not wrap or proxy Codex CLI commands through PaperBot's server. The Codex subagent bridge is a Claude Code agent definition (`.claude/agents/codex-worker.md`), not server-side Codex management. | Agent definition file instructs Claude Code how to delegate. PaperBot server receives events about what happened, not commands about what to do. |

## Feature Dependencies

```
MCP Server (v1.0 prerequisite)
  |
  +-> Agent Event Logging (extends AgentEventEnvelope)
  |     |
  |     +-> Real-time Activity Feed (SSE consumer of event log)
  |     |     |
  |     |     +-> Tool Call Visualization (structured rendering of tool_call events)
  |     |
  |     +-> Agent Lifecycle Status (derived from lifecycle events)
  |     |
  |     +-> Agent DAG Visualization (built from parent_span_id relationships)
  |
  +-> Three-Panel IDE Layout (merges AgentBoard + FilesPanel + Activity Feed)
  |     |
  |     +-> File Change Visualization (DiffViewer in files panel)
  |
  +-> Codex Subagent Bridge (.claude/agents/codex-worker.md)
        |
        +-> Paper2Code Overflow Delegation (stage-level dispatch to Codex)
```

**Critical path**: MCP Server -> Agent Event Logging -> Three-Panel Layout is the minimum viable sequence. Codex bridge can develop in parallel since it is a file definition, not server code.

## MVP Recommendation

**Phase 1 -- Foundation (build on what exists):**
1. Agent event logging with new event types (tool_call, file_change, agent_lifecycle) extending AgentEventEnvelope
2. Three-panel layout replacing studio page (reuse AgentBoard Kanban + add activity feed + reuse FilesPanel/DiffViewer)
3. SSE streaming for new event types through existing infrastructure

**Phase 2 -- Codex Bridge:**
4. Codex subagent bridge (`.claude/agents/codex-worker.md` definition file)
5. Paper2Code overflow delegation workflow

**Phase 3 -- Polish (differentiators):**
6. Tool call timeline visualization
7. Agent DAG visualization with @xyflow/react
8. Cross-agent context sharing display

**Defer:**
- Visual workflow builder: massive scope, low value for code-defined pipelines
- Agent chat interface: duplicates Claude Code/Codex UX
- Per-host adapters: violates single-MCP-surface architecture

## Existing Asset Inventory

Assets that already exist and should be reused, not rebuilt:

| Asset | Location | Reuse Strategy |
|-------|----------|----------------|
| AgentBoard Kanban | `web/src/components/studio/AgentBoard.tsx` | Embed as left panel in three-panel layout |
| Task detail dialog | Inside `AgentBoard.tsx` | Keep as modal overlay |
| Execution log viewer | Inside `AgentBoard.tsx` | Reuse log formatting in activity feed |
| DiffViewer | `web/src/components/studio/DiffViewer.tsx` | Embed in files panel |
| FilesPanel | `web/src/components/studio/FilesPanel.tsx` | Embed as right panel |
| Monaco editor | `web/src/components/studio/DeepCodeEditor.tsx` | Read-only mode for file inspection |
| SSE utilities | `web/src/lib/sse.ts` (readSSE) | Extend for new event types |
| AgentEventEnvelope | `src/paperbot/application/collaboration/message_schema.py` | Add new type values, keep schema stable |
| Event log stores | `src/paperbot/infrastructure/event_log/` | SQLAlchemy store for persistence, memory for dev |
| CodexDispatcher | `src/paperbot/infrastructure/swarm/codex_dispatcher.py` | Backend for Codex task dispatch |
| ClaudeCommander | `src/paperbot/infrastructure/swarm/claude_commander.py` | Backend for Claude Code integration |
| ScoreShareBus | `src/paperbot/core/collaboration/score_bus.py` | Cross-agent evaluation sharing |
| FailFastEvaluator | `src/paperbot/core/collaboration/` | Early termination of low-quality agent work |
| Studio store | `web/src/lib/store/studio-store.ts` | Zustand store with AgentTask type, board session management |
| Runbook API | `/api/runbook/*` endpoints | File management, snapshots, diffs |
| Sandbox API | `/api/sandbox/*` endpoints | Execution queue, run logs, resource metrics |
| DAG visualization lib | @xyflow/react (npm dependency) | Agent task DAG rendering |

## Sources

- [Claude Code Agent Monitor](https://github.com/hoangsonww/Claude-Code-Agent-Monitor) -- real-time monitoring dashboard using hooks, Node.js/React/SQLite
- [Claude Code Hooks Multi-Agent Observability](https://github.com/disler/claude-code-hooks-multi-agent-observability) -- tool call tracing across agent swim lanes
- [Claude Code Subagents Official Docs](https://code.claude.com/docs/en/sub-agents) -- subagent definition, built-in agents (Explore, Plan, general-purpose)
- [Codex Subagent Management Skill](https://mcpmarket.com/tools/skills/codex-subagent-management) -- disk-based prompt/result handling, manifest batching
- [Multi-Agent Dashboard](https://github.com/TheAIuniversity/multi-agent-dashboard) -- Claude Code sub-agent observability with 68+ agent tracking
- [VS Code 1.107 Multi-Agent Orchestration](https://visualstudiomagazine.com/articles/2025/12/12/vs-code-1-107-november-2025-update-expands-multi-agent-orchestration-model-management.aspx) -- Agent HQ, background agents, inline xterm.js
- [ComposioHQ Agent Orchestrator](https://github.com/ComposioHQ/agent-orchestrator) -- parallel agent task planning, CI fix handling
- [CrewAI Platform](https://crewai.com/) -- Crews/Flows model, real-time tracing of agent steps
- [OpenAI Codex App Guide](https://intuitionlabs.ai/articles/openai-codex-app-ai-coding-agents) -- multi-agent command center, parallel workflow management
- [Claude Code Sub-Agent Delegation Setup](https://gist.github.com/tomas-rampas/a79213bb4cf59722e45eab7aa45f155c) -- delegation rules and agent definitions
