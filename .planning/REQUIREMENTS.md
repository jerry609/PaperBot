# Requirements: PaperBot

**Defined:** 2026-03-14
**Core Value:** Paper-specific capability layer surfaced as standard MCP tools + agent orchestration dashboard

## v1.1 Requirements

Requirements for Agent Orchestration Dashboard milestone. Each maps to roadmap phases.

### Event System

- [ ] **EVNT-01**: User can view a real-time scrolling activity feed showing agent events as they happen
- [ ] **EVNT-02**: User can see each agent's lifecycle status (idle, working, completed, errored) at a glance
- [ ] **EVNT-03**: User can view a structured tool call timeline showing tool name, arguments, result summary, and duration
- [ ] **EVNT-04**: Agent events are pushed to connected dashboard clients in real-time via SSE (no polling)

### Dashboard

- [ ] **DASH-01**: User can view agent orchestration in a three-panel IDE layout (tasks | activity | files)
- [ ] **DASH-02**: User can manage agent tasks via Kanban board showing Claude Code and Codex agent identity
- [ ] **DASH-03**: User can see Codex-specific error states (timeout, sandbox crash) surfaced prominently
- [ ] **DASH-04**: User can resize panels in the three-panel layout to customize workspace

### File Visualization

- [ ] **FILE-01**: User can view inline diffs showing what agents changed in each file
- [ ] **FILE-02**: User can see a per-task file list showing created/modified files with status indicators

### Codex Bridge

- [ ] **CDX-01**: Claude Code can delegate tasks to Codex via custom agent definition (codex-worker.md)
- [ ] **CDX-02**: Paper2Code pipeline stages can overflow from Claude Code to Codex when workload is high
- [ ] **CDX-03**: User can observe Codex delegation events in the agent activity feed

### Visualization

- [ ] **VIZ-01**: User can view an agent task dependency DAG with real-time status color updates
- [ ] **VIZ-02**: User can see cross-agent context sharing (ScoreShareBus data flow) in the dashboard

## Future Requirements

Deferred to future milestone. Tracked but not in current roadmap.

(None -- all proposed features included in v1.1)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Custom agent orchestration runtime | Host agents (Claude Code) own orchestration; PaperBot is a skill provider |
| Per-host adapters | One MCP surface serves all agents; no Claude Code vs Codex vs Cursor adapters |
| Visual workflow builder | Massive scope, low value for code-defined pipelines (Paper2Code stages are in code) |
| Agent chat interface | Duplicates Claude Code/Codex conversation UX; dashboard shows output, not input |
| Real-time code editing in dashboard | IDE's job; dashboard shows diffs read-only and deep-links to VS Code |
| Codex CLI wrapper | Agent definition is a file, not server-side Codex management |
| Business logic duplication | Dashboard calls existing API endpoints; no reimplementation of analysis/tracking |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVNT-01 | Phase 8 | Pending |
| EVNT-02 | Phase 8 | Pending |
| EVNT-03 | Phase 8 | Pending |
| EVNT-04 | Phase 7 | Pending |
| DASH-01 | Phase 9 | Pending |
| DASH-02 | Phase 10 | Pending |
| DASH-03 | Phase 10 | Pending |
| DASH-04 | Phase 9 | Pending |
| FILE-01 | Phase 9 | Pending |
| FILE-02 | Phase 9 | Pending |
| CDX-01 | Phase 10 | Pending |
| CDX-02 | Phase 10 | Pending |
| CDX-03 | Phase 10 | Pending |
| VIZ-01 | Phase 11 | Pending |
| VIZ-02 | Phase 11 | Pending |

**Coverage:**
- v1.1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after roadmap creation (phases 7-11)*
