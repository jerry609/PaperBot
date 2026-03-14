# Requirements: PaperBot

**Defined:** 2026-03-14
**Core Value:** Paper-specific capability layer surfaced as standard MCP tools + agent orchestration dashboard

## v1.0 Requirements

Requirements for MCP Server milestone. Phases 1-2 shipped; phases 3-6 remain.

### MCP Tools (shipped)

- [x] **MCP-S1**: Agent can search papers via `paper_search` MCP tool (Phase 2)
- [x] **MCP-S2**: Agent can judge paper quality via `paper_judge` MCP tool (Phase 2)
- [x] **MCP-S3**: Agent can summarize papers via `paper_summarize` MCP tool (Phase 2)
- [x] **MCP-S4**: Agent can assess paper relevance via `relevance_assess` MCP tool (Phase 2)

### MCP Tools (remaining)

- [x] **MCP-01**: Agent can analyze trends across a set of papers via `analyze_trends` MCP tool
- [x] **MCP-02**: Agent can check a scholar's recent publications and activity via `check_scholar` MCP tool
- [x] **MCP-03**: Agent can retrieve research context for a track via `get_research_context` MCP tool
- [x] **MCP-04**: Agent can save research findings to memory via `save_to_memory` MCP tool
- [x] **MCP-05**: Agent can export papers/notes to Obsidian vault format via `export_to_obsidian` MCP tool

### MCP Resources

- [x] **MCP-06**: Agent can read track metadata via `paperbot://track/{id}` resource
- [x] **MCP-07**: Agent can read track paper list via `paperbot://track/{id}/papers` resource
- [x] **MCP-08**: Agent can read track memory via `paperbot://track/{id}/memory` resource
- [x] **MCP-09**: Agent can read scholar subscriptions via `paperbot://scholars` resource

### Transport & Infrastructure

- [ ] **MCP-10**: MCP server runs via stdio transport for local agent integration
- [ ] **MCP-11**: MCP server runs via Streamable HTTP transport for remote agent integration
- [ ] **MCP-12**: User can start MCP server via `paperbot mcp serve` CLI command

### Agent Skills

- [ ] **MCP-13**: Agent can discover and load PaperBot workflow skills via `.claude/skills/` SKILL.md files (literature-review, paper-reproduction, trend-analysis, scholar-monitoring)

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
| MCP-S1 | Phase 2 | Complete |
| MCP-S2 | Phase 2 | Complete |
| MCP-S3 | Phase 2 | Complete |
| MCP-S4 | Phase 2 | Complete |
| MCP-01 | Phase 3 | Complete |
| MCP-02 | Phase 3 | Complete |
| MCP-03 | Phase 3 | Complete |
| MCP-04 | Phase 3 | Complete |
| MCP-05 | Phase 3 | Complete |
| MCP-06 | Phase 4 | Complete |
| MCP-07 | Phase 4 | Complete |
| MCP-08 | Phase 4 | Complete |
| MCP-09 | Phase 4 | Complete |
| MCP-10 | Phase 5 | Pending |
| MCP-11 | Phase 5 | Pending |
| MCP-12 | Phase 5 | Pending |
| MCP-13 | Phase 6 | Pending |
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
- v1.0 requirements: 17 total (4 shipped, 13 remaining)
- Mapped to phases: 17
- Unmapped: 0
- v1.1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after v1.0 phases 3-6 requirements defined*
