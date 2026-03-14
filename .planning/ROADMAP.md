# Roadmap: PaperBot

## Milestones

- 🚧 **v1.0 MCP Server** - Phases 1-6 (in progress)
- 📋 **v1.1 Agent Orchestration Dashboard** - Phases 7-11 (planned)

## Phases

<details>
<summary>v1.0 MCP Server (Phases 1-6) - In Progress</summary>

**Milestone Goal:** Complete the MCP server with all 9 tools, 4 resources, transport configuration, and Agent Skills — making PaperBot's full capability surface available to any MCP-compatible agent.

- [x] **Phase 1: MCP Server Setup** - FastMCP instance, package structure (shipped)
- [x] **Phase 2: Core Paper Tools** - paper_search, paper_judge, paper_summarize, relevance_assess + audit helper (shipped)
- [ ] **Phase 3: Remaining MCP Tools** - analyze_trends, check_scholar, get_research_context, save_to_memory, export_to_obsidian
- [ ] **Phase 4: MCP Resources** - 4 resource URIs for track/scholar data access
- [ ] **Phase 5: Transport & Entry Point** - stdio + Streamable HTTP transports, CLI command
- [ ] **Phase 6: Agent Skills** - SKILL.md files for core workflows

### Phase 3: Remaining MCP Tools
**Goal**: All 9 MCP tools are registered and callable, completing the tool surface
**Depends on**: Phase 2 (audit helper and registration pattern)
**Requirements**: MCP-01, MCP-02, MCP-03, MCP-04, MCP-05
**Success Criteria** (what must be TRUE):
  1. Agent can call `analyze_trends` and receive trend analysis for a set of papers
  2. Agent can call `check_scholar` and receive a scholar's recent publications
  3. Agent can call `get_research_context` and receive context for a research track
  4. Agent can call `save_to_memory` and persist research findings retrievable later
  5. Agent can call `export_to_obsidian` and receive Obsidian-formatted markdown
  6. All 9 tools appear in MCP tools/list
  7. All tools log calls via audit helper
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — analyze_trends + check_scholar tools with TDD
- [ ] 03-02-PLAN.md — get_research_context + save_to_memory + export_to_obsidian tools with TDD
- [ ] 03-03-PLAN.md — Server registration + integration tests for all 9 tools

### Phase 4: MCP Resources
**Goal**: Agents can read PaperBot data via MCP resource URIs without tool calls
**Depends on**: Phase 2 (server instance)
**Requirements**: MCP-06, MCP-07, MCP-08, MCP-09
**Success Criteria** (what must be TRUE):
  1. Agent can read `paperbot://track/{id}` and receive track metadata
  2. Agent can read `paperbot://track/{id}/papers` and receive paper list for a track
  3. Agent can read `paperbot://track/{id}/memory` and receive saved research memory
  4. Agent can read `paperbot://scholars` and receive scholar subscription list
  5. All 4 resources appear in MCP resources/list
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: Transport & Entry Point
**Goal**: MCP server is runnable via stdio (local) and Streamable HTTP (remote) with a CLI command
**Depends on**: Phase 3, Phase 4 (all tools and resources registered)
**Requirements**: MCP-10, MCP-11, MCP-12
**Success Criteria** (what must be TRUE):
  1. `paperbot mcp serve --stdio` starts MCP server on stdio transport
  2. `paperbot mcp serve --http` starts MCP server on Streamable HTTP transport
  3. Claude Code can connect to PaperBot MCP server via stdio in `claude_desktop_config.json`
  4. Remote agent can connect via HTTP and call tools
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: Agent Skills
**Goal**: Core PaperBot workflows are available as SKILL.md files for agent discovery
**Depends on**: Phase 3 (tools must exist for skills to reference)
**Requirements**: MCP-13
**Success Criteria** (what must be TRUE):
  1. `.claude/skills/` directory contains SKILL.md files for literature-review, paper-reproduction, trend-analysis, scholar-monitoring
  2. Each SKILL.md has valid YAML frontmatter (name, description, tools) and markdown instructions
  3. Skills reference MCP tools by name and provide multi-step workflow guidance
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

</details>

### v1.1 Agent Orchestration Dashboard

**Milestone Goal:** Build a real-time agent orchestration dashboard and Codex subagent bridge, enabling users to observe and manage Claude Code and Codex agent activity from PaperBot's web UI.

**Phase Numbering:**
- Integer phases (7, 8, 9...): Planned milestone work
- Decimal phases (7.1, 7.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 7: EventBus + SSE Foundation** - In-process event bus with SSE subscription endpoint for real-time push
- [ ] **Phase 8: Agent Event Vocabulary** - Extend AgentEventEnvelope types and build activity feed, lifecycle indicators, and tool call timeline
- [ ] **Phase 9: Three-Panel Dashboard** - Frontend store, SSE hook, and three-panel IDE layout with file visualization
- [ ] **Phase 10: Agent Board + Codex Bridge** - Kanban board generalization, Codex worker agent definition, and overflow delegation
- [ ] **Phase 11: DAG Visualization** - Task dependency DAG and cross-agent context sharing visualization

## Phase Details

### Phase 7: EventBus + SSE Foundation
**Goal**: Agent events are pushed to connected clients in real-time without polling
**Depends on**: Phase 6 (v1.0 MCP server provides tool surface)
**Requirements**: EVNT-04
**Success Criteria** (what must be TRUE):
  1. A dashboard client connected via SSE receives agent events within 1 second of emission
  2. Multiple simultaneous SSE clients each receive all events independently
  3. Existing event_log.append() calls automatically push to SSE subscribers with zero changes to calling code
  4. SSE connections clean up gracefully on client disconnect (no leaked queues or background tasks)
**Plans**: 2 plans

Plans:
- [ ] 07-01-PLAN.md — EventBusEventLog TDD (ring buffer + fan-out + backpressure)
- [ ] 07-02-PLAN.md — SSE endpoint + main.py wiring + integration tests

### Phase 8: Agent Event Vocabulary
**Goal**: Users can see meaningful, structured agent activity as it happens
**Depends on**: Phase 7 (events must be pushable before they can be rendered)
**Requirements**: EVNT-01, EVNT-02, EVNT-03
**Success Criteria** (what must be TRUE):
  1. User sees a scrolling activity feed that updates in real-time as agents emit events
  2. User can see at a glance whether each agent is idle, working, completed, or errored
  3. User can view a structured tool call timeline showing tool name, arguments, result summary, and duration for each call
  4. New event types extend AgentEventEnvelope (no parallel event schema created)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

### Phase 9: Three-Panel Dashboard
**Goal**: Users can observe agent work in a three-panel IDE layout with file-level detail
**Depends on**: Phase 8 (activity feed and lifecycle events must exist before dashboard renders them)
**Requirements**: DASH-01, DASH-04, FILE-01, FILE-02
**Success Criteria** (what must be TRUE):
  1. User sees a three-panel layout (tasks | agent activity | files) when opening the agent dashboard
  2. User can drag panel dividers to resize each panel and the layout persists across page navigation
  3. User can view inline diffs showing exactly what an agent changed in each file
  4. User can see a per-task file list with created/modified indicators for every file an agent touched
  5. Dashboard state is managed by a Zustand store fed by the SSE event stream (no polling)
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD
- [ ] 09-03: TBD

### Phase 10: Agent Board + Codex Bridge
**Goal**: Users can manage agent tasks on a Kanban board and Claude Code can delegate work to Codex
**Depends on**: Phase 9 (dashboard layout must exist for board embedding; Phase 7 SSE for delegation events)
**Requirements**: DASH-02, DASH-03, CDX-01, CDX-02, CDX-03
**Success Criteria** (what must be TRUE):
  1. User can view and manage agent tasks on a Kanban board that shows which tasks belong to Claude Code vs Codex
  2. User sees Codex-specific error states (timeout, sandbox crash) surfaced prominently on failed tasks
  3. Claude Code can delegate tasks to Codex via the codex-worker.md custom agent definition
  4. Paper2Code pipeline stages can overflow from Claude Code to Codex when workload exceeds capacity
  5. User can observe Codex delegation events (dispatched, accepted, completed, failed) in the activity feed
**Plans**: TBD

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD
- [ ] 10-03: TBD

### Phase 11: DAG Visualization
**Goal**: Users can see task dependencies and cross-agent data flow visually
**Depends on**: Phase 10 (tasks and agents must exist before visualizing their relationships)
**Requirements**: VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. User can view an interactive task dependency DAG where node colors update in real-time to reflect task status
  2. User can see ScoreShareBus data flow edges in the DAG showing which agents shared evaluation context with which other agents
  3. DAG renders using existing @xyflow/react (no new visualization dependencies)
**Plans**: TBD

Plans:
- [ ] 11-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 3 -> 4 -> 5 -> 6 (v1.0) -> 7 -> 8 -> ... -> 11 (v1.1)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. MCP Server Setup | v1.0 | — | Complete | 2026-03-13 |
| 2. Core Paper Tools | v1.0 | 3/3 | Complete | 2026-03-14 |
| 3. Remaining MCP Tools | 2/3 | In Progress|  | - |
| 4. MCP Resources | v1.0 | 0/? | Not started | - |
| 5. Transport & Entry Point | v1.0 | 0/? | Not started | - |
| 6. Agent Skills | v1.0 | 0/? | Not started | - |
| 7. EventBus + SSE Foundation | v1.1 | 0/2 | Planning complete | - |
| 8. Agent Event Vocabulary | v1.1 | 0/? | Not started | - |
| 9. Three-Panel Dashboard | v1.1 | 0/? | Not started | - |
| 10. Agent Board + Codex Bridge | v1.1 | 0/? | Not started | - |
| 11. DAG Visualization | v1.1 | 0/? | Not started | - |
