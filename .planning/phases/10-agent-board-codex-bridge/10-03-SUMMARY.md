---
plan: "10-03"
phase: "10-agent-board-codex-bridge"
status: done
subsystem: ui+agent-config
tags: [codex-worker, agent-dashboard, kanban, view-toggle, claude-code-agent]
dependency_graph:
  requires: [10-01, 10-02]
  provides: [codex-worker-agent-definition, dashboard-view-toggle]
  affects: [agent-dashboard page, claude-code-subagent-delegation]
tech_stack:
  added: []
  patterns: [claude-code-subagent-definition, zustand-store-selector, conditional-render-view-toggle]
key_files:
  created:
    - .claude/agents/codex-worker.md
  modified:
    - web/src/app/agent-dashboard/page.tsx
decisions:
  - "KanbanBoard not placed inside SplitPanels — renders full-width at page level to avoid horizontal scroll conflict (Pitfall 6)"
  - "Task source: studioTasks (primary, from useStudioStore) merged with eventKanbanTasks (fallback) — prefer canonical studio store when non-empty"
  - "useAgentEvents() SSE hook stays mounted regardless of view mode — SSE connection never drops on toggle"
  - "codex-worker.md tools list: only Bash and Read — always available in Claude Code without extra config"
metrics:
  duration: "3 min"
  completed_date: "2026-03-15"
  tasks_completed: 2
  files_changed: 2
requirements: [CDX-01, DASH-02]
---

# Phase 10 Plan 03: codex-worker Sub-Agent + Dashboard View Toggle Summary

**codex-worker.md Claude Code sub-agent definition with 4-step curl delegation protocol, and agent-dashboard page with Panels/Kanban toggle rendering full-width KanbanBoard from merged studio+event store tasks.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-15
- **Completed:** 2026-03-15
- **Tasks:** 2 (1 auto + 1 human-verify checkpoint)
- **Files modified:** 2

## What Was Built

### .claude/agents/codex-worker.md
Claude Code custom sub-agent definition with valid YAML frontmatter (`name: codex-worker`, `tools: [Bash, Read]`) and a markdown body covering:
- **Purpose:** Delegates self-contained coding tasks to the PaperBot Codex worker via the agent-board API
- **When to use:** 3 criteria (self-contained task, clear acceptance criteria, high workload)
- **Delegation Protocol:** 4-step curl sequence:
  1. Confirm task exists — `GET /api/agent-board/sessions/{session_id}`
  2. Dispatch to Codex — `POST /api/agent-board/tasks/{task_id}/dispatch`
  3. Stream execution — `GET /api/agent-board/tasks/{task_id}/execute`
  4. Report result (success + failure templates)
- **Error Handling:** OPENAI_API_KEY missing, timeout, sandbox crash

### web/src/app/agent-dashboard/page.tsx
Updated with view-mode toggle:
- `useState<"panels" | "kanban">("panels")` for active view
- Lucide icons: `Columns3` (panels) and `LayoutGrid` (kanban) in header as icon buttons
- Active button gets `bg-muted`, inactive gets `hover:bg-muted/50`
- Task source merger: `studioTasks.length > 0 ? studioTasks : eventKanbanTasks`
- Conditional render: "panels" -> unchanged `<SplitPanels .../>`, "kanban" -> full-width `<KanbanBoard tasks={kanbanTasks} />`
- `useAgentEvents()` hook remains mounted at page root regardless of view (SSE stays connected)

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 9717a3c | feat | feat(10-03): add codex-worker sub-agent and dashboard Panels/Kanban toggle |

## Decisions Made

- KanbanBoard rendered at page level (not inside SplitPanels) — avoids horizontal scroll conflict documented in research as Pitfall 6
- Task source uses studio store (useStudioStore.agentTasks) as primary and event store (useAgentEventStore.kanbanTasks) as fallback — studio store is the canonical source for tasks created via API
- SSE hook (`useAgentEvents()`) stays mounted regardless of active view — prevents SSE reconnect on each toggle
- codex-worker.md tools limited to Bash and Read — always available in Claude Code without additional configuration

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Human Verification

Task 2 checkpoint: Human verified the agent dashboard at http://localhost:3000/agent-dashboard.
- Toggle icons (Columns3/LayoutGrid) visible in header
- Kanban toggle shows full-width KanbanBoard with 5 column headers
- Panels toggle restores three-panel SplitPanels layout
- .claude/agents/codex-worker.md confirmed present with valid frontmatter

**Verification result: APPROVED**

## Self-Check

**Status: PASSED**

- FOUND: .claude/agents/codex-worker.md
- FOUND: web/src/app/agent-dashboard/page.tsx
- FOUND: commit 9717a3c (feat(10-03))
- Build: Next.js build passed (verified during Task 1)
- Requirements CDX-01 and DASH-02 satisfied
