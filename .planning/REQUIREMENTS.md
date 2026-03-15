# Requirements: PaperBot

**Defined:** 2026-03-14
**Core Value:** Paper-specific capability layer surfaced as standard MCP tools + agent-agnostic dashboard

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

- [x] **MCP-10**: MCP server runs via stdio transport for local agent integration
- [x] **MCP-11**: MCP server runs via Streamable HTTP transport for remote agent integration
- [x] **MCP-12**: User can start MCP server via `paperbot mcp serve` CLI command

### Agent Skills

- [x] **MCP-13**: Agent can discover and load PaperBot workflow skills via `.claude/skills/` SKILL.md files (literature-review, paper-reproduction, trend-analysis, scholar-monitoring)

## v1.1 Requirements

Requirements for Agent Orchestration Dashboard milestone. Each maps to roadmap phases.

### Event System

- [x] **EVNT-01**: User can view a real-time scrolling activity feed showing agent events as they happen
- [x] **EVNT-02**: User can see each agent's lifecycle status (idle, working, completed, errored) at a glance
- [x] **EVNT-03**: User can view a structured tool call timeline showing tool name, arguments, result summary, and duration
- [x] **EVNT-04**: Agent events are pushed to connected dashboard clients in real-time via SSE (no polling)

### Dashboard

- [x] **DASH-01**: User can view agent orchestration in a three-panel IDE layout (tasks | activity | files)
- [ ] **DASH-02**: User can manage agent tasks via Kanban board showing Claude Code and Codex agent identity
- [ ] **DASH-03**: User can see Codex-specific error states (timeout, sandbox crash) surfaced prominently
- [x] **DASH-04**: User can resize panels in the three-panel layout to customize workspace

### File Visualization

- [x] **FILE-01**: User can view inline diffs showing what agents changed in each file
- [x] **FILE-02**: User can see a per-task file list showing created/modified files with status indicators

### Codex Bridge

- [ ] **CDX-01**: Claude Code can delegate tasks to Codex via custom agent definition (codex-worker.md)
- [ ] **CDX-02**: Paper2Code pipeline stages can overflow from Claude Code to Codex when workload is high
- [ ] **CDX-03**: User can observe Codex delegation events in the agent activity feed

### Visualization

- [ ] **VIZ-01**: User can view an agent task dependency DAG with real-time status color updates
- [ ] **VIZ-02**: User can see cross-agent context sharing (ScoreShareBus data flow) in the dashboard

## v1.2 Requirements

Requirements for DeepCode Agent Dashboard milestone. Agent-agnostic proxy + visualization + control.

### Agent Adapter Layer

- [ ] **ADAPT-01**: User can configure which code agent to use (Claude Code, Codex, OpenCode) in settings
- [ ] **ADAPT-02**: System provides a BaseAgentAdapter interface abstracting agent-specific protocols
- [ ] **ADAPT-03**: User can interact with Claude Code via ClaudeCodeAdapter (subprocess + stream-json, stateful sessions)
- [ ] **ADAPT-04**: User can interact with Codex via CodexAdapter (REST API + JSONL events)
- [ ] **ADAPT-05**: User can interact with OpenCode via OpenCodeAdapter (HTTP/ACP protocol)
- [ ] **ADAPT-06**: Dashboard discovers agent activity through hybrid channels (agent push + independent discovery)

### Activity Monitoring

- [ ] **MONIT-01**: User sees real-time agent activity stream (SSE → ActivityFeed component with auto-scroll and pause)
- [ ] **MONIT-02**: User sees tool call log with tool name, arguments, result status, and duration per event
- [ ] **MONIT-03**: User sees agent status indicator (running/waiting/complete/error/idle)
- [ ] **MONIT-04**: User sees connection status indicator (connected/reconnecting/disconnected)
- [ ] **MONIT-05**: User sees errors surfaced prominently (error badge, red rendering, toast notification on failure)

### Chat & Control

- [ ] **CTRL-01**: User can send tasks to the configured agent via chat input in the web UI
- [ ] **CTRL-02**: User can interrupt or cancel a running agent from the dashboard
- [ ] **CTRL-03**: User can approve or reject agent actions via human-in-the-loop approval modal

### Session Management

- [ ] **SESS-01**: User can see a list of sessions (active/completed) with status, agent type, and cost
- [ ] **SESS-02**: User can view session detail with full event timeline for a run_id
- [ ] **SESS-03**: User can see token usage and estimated cost per session (input/output tokens, model pricing)
- [ ] **SESS-04**: User can replay completed sessions with timeline scrubber and step-by-step navigation
- [ ] **SESS-05**: User can checkpoint and restore agent sessions, branching like git

### Visualization

- [ ] **VIS-01**: User sees agent-initiated team decomposition as a live DAG (@xyflow/react)
- [ ] **VIS-02**: User sees file diffs for agent-modified files via Monaco diff editor
- [ ] **VIS-03**: User sees agent card grid (per-agent: cost, context %, status, latest action, color-graded context bar)
- [ ] **VIS-04**: User sees agent swim lanes (each agent gets a vertical lane showing events chronologically)

### Domain Integration

- [ ] **DOMAIN-01**: User sees enriched Paper2Code view when run_type is paper2code (paper metadata, stage progress)
- [ ] **DOMAIN-02**: User sees paper-specific rendering for PaperBot MCP tool calls (paper card, score badge)

## v2.0 Requirements

Requirements for PostgreSQL Migration & Data Layer Refactoring milestone.

### PG Infrastructure

- [ ] **PGINFRA-01**: 开发者可以通过 docker-compose up 一键启动 PostgreSQL + pgvector 本地开发环境
- [ ] **PGINFRA-02**: Alembic env.py 支持 async 执行路径和 PG/SQLite 双路径检测
- [ ] **PGINFRA-03**: 用户可以通过 pgloader 脚本将现有 SQLite 数据无损迁移到 PostgreSQL
- [ ] **PGINFRA-04**: 嵌入向量数据可以从 SQLite LargeBinary 迁移到 pgvector 列

### Async Data Layer

- [ ] **ASYNC-01**: 全局共享单个 AsyncEngine，替代 17+ 个独立 SessionProvider 的连接池
- [ ] **ASYNC-02**: AsyncSessionProvider 提供统一的 async session 工厂，所有 store 通过 DI 注入
- [ ] **ASYNC-03**: 全部 17 个 Store 类的 ~170 个方法完成 sync→async 转换
- [ ] **ASYNC-04**: ARQ worker 通过 on_job_start/on_job_complete 管理 async session 生命周期
- [ ] **ASYNC-05**: MCP 工具层移除 anyio.to_thread.run_sync 包装，直接 await async store 方法

### PG-Native Features

- [ ] **PGNAT-01**: memory_items 和 document_chunks 的全文搜索从 FTS5 迁移到 tsvector + GIN 索引
- [ ] **PGNAT-02**: 84 个 Text JSON 列迁移到 JSONB 类型，支持原生查询和 GIN 索引
- [ ] **PGNAT-03**: 向量搜索从 sqlite-vec LargeBinary 迁移到 pgvector Vector(N) + HNSW 索引

### Model Refactoring

- [ ] **MODEL-01**: 所有 relationship 设置 lazy="raise"，逐 store 审计并改为显式 selectinload/joinedload
- [ ] **MODEL-02**: 添加 NOT NULL、CHECK、UNIQUE 约束，修复 is_active int→bool 等类型问题
- [ ] **MODEL-03**: Author 去重、冗余列清理、表合并/拆分等 schema 规范化

### Test Infrastructure

- [ ] **TEST-01**: 建立 testcontainers[postgres] CI fixture，替代 SQLite in-memory 测试数据库
- [ ] **TEST-02**: 现有测试套件在 PostgreSQL 上全部通过
- [ ] **TEST-03**: async 测试基础设施：pytest-asyncio async fixture + 每测试事务回滚隔离
- [ ] **TEST-04**: 关键查询路径的性能基准测试（全文搜索、向量搜索、JSONB 查询）

### CI Integration

- [ ] **CI-01**: GitHub Actions 配置 PostgreSQL service container，所有测试在 PG 上运行
- [ ] **CI-02**: CI 流水线包含 Alembic 迁移验证（fresh DB upgrade head + downgrade 回退测试）
- [ ] **CI-03**: CI 中运行 SQLite→PG 数据迁移冒烟测试，验证 pgloader 脚本正确性

### Monitoring

- [ ] **MON-01**: AsyncEngine 连接池指标暴露（pool_size、checkedout、overflow）可通过 API 查询
- [ ] **MON-02**: 慢查询日志记录（超过阈值的 SQL 自动 warning 级别记录）
- [ ] **MON-03**: 数据库健康检查端点（/api/health/db），验证连接可用性和迁移版本

## Future Requirements

Deferred to future milestone. Tracked but not in current roadmap.

(None)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Custom agent orchestration runtime | Host agents own orchestration; PaperBot visualizes, not decomposes tasks |
| Per-agent custom UI skins | One unified dashboard for all agents; agent type shown as badge/label |
| Real-time token-by-token LLM streaming | Doubles SSE volume; show full turn on completion with "thinking..." spinner |
| Full IDE replacement (language servers, extensions, debugging) | Monaco for diffs only; user's IDE remains the editing environment |
| Agent training / fine-tuning integration | Separate product domain; export session data as JSONL instead |
| Multi-user collaboration on sessions | Requires real-time sharing, permissions, conflict resolution — too complex |
| Autonomous agent scheduling (cron-style) | Creates mini-orchestration runtime, contradicts skill-provider constraint |
| Visual workflow builder | Massive scope, low value for code-defined pipelines |
| Business logic duplication | Dashboard calls existing API endpoints; no reimplementation |

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
| MCP-10 | Phase 5 | Complete |
| MCP-11 | Phase 5 | Complete |
| MCP-12 | Phase 5 | Complete |
| MCP-13 | Phase 6 | Complete |
| EVNT-01 | Phase 8 | Complete |
| EVNT-02 | Phase 8 | Complete |
| EVNT-03 | Phase 8 | Complete |
| EVNT-04 | Phase 7 | Complete |
| DASH-01 | Phase 9 | Complete |
| DASH-02 | Phase 10 | Pending |
| DASH-03 | Phase 10 | Pending |
| DASH-04 | Phase 9 | Complete |
| FILE-01 | Phase 9 | Complete |
| FILE-02 | Phase 9 | Complete |
| CDX-01 | Phase 10 | Pending |
| CDX-02 | Phase 10 | Pending |
| CDX-03 | Phase 10 | Pending |
| VIZ-01 | Phase 11 | Pending |
| VIZ-02 | Phase 11 | Pending |
| PGINFRA-01 | Phase 12 | Pending |
| PGINFRA-02 | Phase 12 | Pending |
| PGINFRA-03 | Phase 17 | Pending |
| PGINFRA-04 | Phase 17 | Pending |
| ASYNC-01 | Phase 14 | Pending |
| ASYNC-02 | Phase 14 | Pending |
| ASYNC-03 | Phase 14 | Pending |
| ASYNC-04 | Phase 14 | Pending |
| ASYNC-05 | Phase 14 | Pending |
| PGNAT-01 | Phase 15 | Pending |
| PGNAT-02 | Phase 15 | Pending |
| PGNAT-03 | Phase 15 | Pending |
| MODEL-01 | Phase 16 | Pending |
| MODEL-02 | Phase 16 | Pending |
| MODEL-03 | Phase 16 | Pending |
| TEST-01 | Phase 13 | Pending |
| TEST-02 | Phase 13 | Pending |
| TEST-03 | Phase 13 | Pending |
| TEST-04 | Phase 13 | Pending |
| CI-01 | Phase 17 | Pending |
| CI-02 | Phase 17 | Pending |
| CI-03 | Phase 17 | Pending |
| MON-01 | Phase 17 | Pending |
| MON-02 | Phase 17 | Pending |
| MON-03 | Phase 17 | Pending |

**Coverage:**
- v1.0 requirements: 17 total (4 shipped, 13 remaining)
- Mapped to phases: 17
- Unmapped: 0
- v1.1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0
- v1.2 requirements: 24 total (ADAPT x6, MONIT x5, CTRL x3, SESS x5, VIS x4, DOMAIN x2)
- Mapped to phases: 0 ⚠️ (awaiting roadmap)
- Unmapped: 24
- v2.0 requirements: 25 total (counted: PGINFRA x4, ASYNC x5, PGNAT x3, MODEL x3, TEST x4, CI x3, MON x3)
- Mapped to phases: 25
- Unmapped: 0

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-15 after v1.2 milestone requirements added*
