# Project Issue Backlog (From ROADMAP_TODO)

This file splits unfinished items in `docs/ROADMAP_TODO.md` into issue-ready cards for GitHub Projects.

## Delivery Rule

- One issue -> one commit (`1 issue = 1 commit`)
- Commit message include issue id, e.g. `feat: agent runtime contract (#41)`

## Suggested Labels

- `roadmap`
- `phase-1`, `phase-2`, `phase-3`, `phase-4`
- `backend`, `frontend`, `data-model`, `workflow`, `infra`
- `priority-p0`, `priority-p1`, `priority-p2`

## Issue 1 - Saved Papers Frontend Page

- Title: `[Feature] Saved papers list page and actions`
- Labels: `roadmap`, `phase-1`, `frontend`, `priority-p1`
- Source TODO:
  - `前端：收藏列表页面组件`
- Scope:
  - Implement `web/src/components/research/SavedPapersList.tsx`
  - Consume `GET /api/research/papers/saved`
  - Support sort `judge_score | saved_at | published_at`
  - Add open-detail and status action entry points
- Acceptance:
  - Saved papers render with pagination/limit
  - Sorting works and is testable
  - Empty and loading states are handled

## Issue 2 - Repo Persistence Model and Per-Paper Repo API

- Title: `[Feature] Persist repo enrichment and expose paper repo detail API`
- Labels: `roadmap`, `phase-1`, `backend`, `data-model`, `priority-p1`
- Source TODO:
  - `新增 PaperRepoModel 表`
  - `DailyPaper 生成后异步调用 repo enrichment`
  - `API：GET /api/research/papers/{paper_id}/repos`
- Scope:
  - Add `PaperRepoModel` + migration
  - Persist enrichment output from `/paperscool/repos`
  - Add read API for single paper repos
  - Add async hook after daily generation
- Acceptance:
  - Repo metadata is queryable by paper id
  - Async job does not block daily response latency

## Issue 3 - Paper Detail Page (UI)

- Title: `[Feature] Paper detail page UI aggregation`
- Labels: `roadmap`, `phase-2`, `frontend`, `priority-p1`
- Source TODO:
  - `前端页面：web/src/app/papers/[id]/page.tsx`
- Scope:
  - Render paper base info, judge radar, feedback, repo panel
  - Render trend-related context and related papers placeholder
- Acceptance:
  - Page works from deep link `/papers/{id}`
  - Works with empty fields and fallback states

## Issue 4 - Paper Detail Aggregation API Hardening

- Title: `[Feature] Harden GET /api/research/papers/{paper_id} aggregation contract`
- Labels: `roadmap`, `phase-2`, `backend`, `priority-p1`
- Source TODO:
  - `API：GET /api/research/papers/{paper_id} 聚合返回上述所有数据`
- Scope:
  - Finalize response schema for detail page
  - Include stable sections: paper, reading_status, judge_scores, feedback_summary, repos
  - Add API tests for not-found, partial-data, complete-data
- Acceptance:
  - Frontend can render without shape ambiguity
  - Backward-compatible fields documented

## Issue 5 - Chat with Paper API + Dialog

- Title: `[Feature] Chat with Paper end-to-end`
- Labels: `roadmap`, `phase-2`, `backend`, `frontend`, `priority-p1`
- Source TODO:
  - `API：POST /api/research/papers/{paper_id}/ask`
  - `前端 Ask AI 按钮 + 对话弹窗`
- Scope:
  - Implement ask API using title+abstract+judge context
  - Add dialog UI from list/detail pages
  - Add error budget and timeout handling
- Acceptance:
  - Chat responses are grounded in paper context
  - UI supports retries and failure messaging

## Issue 6 - Full-Text Context for Ask (Optional Advanced)

- Title: `[Enhancement] Add PDF full-text context path for Ask API`
- Labels: `roadmap`, `phase-2`, `backend`, `priority-p2`
- Source TODO:
  - `可选增强：如果有 PDF URL，抽取全文作为上下文`
- Scope:
  - PDF fetch and extraction pipeline
  - Chunking + retrieval for ask endpoint
- Acceptance:
  - Ask quality improves for methods/experiment questions
  - Falls back cleanly when PDF unavailable

## Issue 7 - Rich Push Channels (Email/Slack/DingTalk)

- Title: `[Feature] Rich push formatting for daily digest`
- Labels: `roadmap`, `phase-2`, `backend`, `priority-p1`
- Source TODO:
  - `HTML 邮件模板`
  - `Slack Block Kit 消息`
  - `钉钉 Markdown 优化`
- Scope:
  - Implement HTML email template
  - Implement Slack blocks payload
  - Improve DingTalk markdown layout
- Acceptance:
  - Push output is structured and readable across channels

## Issue 8 - RSS/Atom Feed Export

- Title: `[Feature] RSS and Atom feed endpoints for DailyPaper`
- Labels: `roadmap`, `phase-2`, `backend`, `priority-p1`
- Source TODO:
  - `GET /api/feed/rss`
  - `GET /api/feed/atom`
  - `输出静态 RSS 文件`
- Scope:
  - Feed generation service + endpoints
  - Optional static `reports/feed.xml`
- Acceptance:
  - Valid RSS 2.0 and Atom 1.0 output
  - Supports recent-N and optional filters

## Issue 9 - Scholar Graph/Trend Frontend Visualization

- Title: `[Feature] Scholar network and trend visualization UI`
- Labels: `roadmap`, `phase-3`, `frontend`, `priority-p1`
- Source TODO:
  - `前端：学者关系图可视化`
  - `前端：学者趋势图表`
- Scope:
  - `ScholarNetworkGraph.tsx`
  - `ScholarTrendsChart.tsx`
  - Integrate with existing `/scholar/network|trends` APIs
- Acceptance:
  - Interactive graph + trend charts are stable with 10-200 nodes

## Issue 10 - Personalized Recommendation Pipeline

- Title: `[Feature] Personalized re-rank and recommendation endpoint`
- Labels: `roadmap`, `phase-3`, `backend`, `workflow`, `priority-p1`
- Source TODO:
  - `兴趣向量提取`
  - `DailyPaper re-rank`
  - `GET /api/research/recommended`
  - `前端 为你推荐 区块`
- Scope:
  - Interest profile service from feedback + track
  - Re-rank formula in daily workflow
  - Recommendation API + minimal UI block
- Acceptance:
  - Recommendations differ by user behavior
  - Score composition is observable/debuggable

## Issue 11 - Trending Papers Feature

- Title: `[Feature] Trending papers API and page`
- Labels: `roadmap`, `phase-3`, `backend`, `frontend`, `priority-p1`
- Source TODO:
  - `Trending 评分公式`
  - `GET /api/research/trending`
  - `前端 Trending 页面`
- Scope:
  - Implement trending score blend and endpoint
  - Add frontend page with filters
- Acceptance:
  - Trending list updates over time and is explainable

## Issue 12 - Similar Papers Retrieval

- Title: `[Feature] Similar paper retrieval via embeddings`
- Labels: `roadmap`, `phase-3`, `backend`, `priority-p1`
- Source TODO:
  - `论文 Embedding 存储`
  - `相似度检索`
  - `GET /api/research/papers/{paper_id}/similar`
  - `详情页 Related Papers`
- Scope:
  - Embedding model/table + async generation
  - Similarity service and endpoint
  - UI integration in paper detail page
- Acceptance:
  - Related papers are semantically relevant and deterministic

## Issue 13 - Subscription System and Scheduled Delivery

- Title: `[Feature] Per-topic/per-author subscriptions with scheduled digests`
- Labels: `roadmap`, `phase-3`, `backend`, `workflow`, `priority-p1`
- Source TODO:
  - `SubscriptionModel`
  - CRUD APIs for subscriptions
  - ARQ cron generation
  - Incremental new-paper detection
- Scope:
  - Model + migration + APIs
  - Scheduler and job execution path
  - Dedup/new-match logic
- Acceptance:
  - User receives only new matching results per subscription

## Issue 14 - OpenClaw Integration Package

- Title: `[Feature] OpenClaw skill integration for PaperBot workflows`
- Labels: `roadmap`, `opclaw`, `integration`, `priority-p2`
- Source TODO:
  - `OpenClaw Skill 包装层`
  - `Skill 调用 PaperBot API`
  - `推送渠道对接`
  - `OpenClaw Cron`
- Scope:
  - Provide OpenClaw-compatible wrappers
  - Connect existing PaperBot APIs and push system
- Acceptance:
  - Daily flow runnable from OpenClaw with end-to-end logs

## Issue 15 - Multi-Agent Framework Upgrade Evaluation

- Title: `[Research] Orchestration framework upgrade plan`
- Labels: `roadmap`, `research`, `priority-p2`
- Source TODO:
  - `评估 LangGraph / CrewAI / AutoGen`
  - `统一 Agent 抽象层`
  - `可观测性增强`
- Scope:
  - Benchmark matrix and migration recommendation
  - RFC for unified agent interface
- Acceptance:
  - Decision doc with measurable tradeoffs and migration steps

## Issue 16 - Platformization Foundation (Multi-user + Community + Extension + PDF)

- Title: `[Epic] Platformization foundation`
- Labels: `roadmap`, `phase-4`, `epic`, `priority-p2`
- Source TODO:
  - Multi-user auth/API key/isolation
  - Community interaction (comment/upvote/claim)
  - Browser extension + URL rewrite
  - PDF full-text indexing/annotation
- Scope:
  - Break into child issues before implementation
- Acceptance:
  - Child issue tree created and scheduled by quarter

---

## Project Board Setup Suggestion

Columns:

1. `Backlog`
2. `Ready`
3. `In Progress`
4. `Review`
5. `Done`

Custom fields:

- `Phase` (P1/P2/P3/P4)
- `Priority` (P0/P1/P2)
- `Area` (Backend/Frontend/Data/Workflow)
- `Size` (S/M/L/XL)


## Issue 15 - Agent Browser Source Runner

- Title: `[Feature] Add agent-browser source runner with fallback connectors`
- Labels: `roadmap`, `phase-4`, `backend`, `integration`, `priority-p1`
- Source TODO:
  - `新增 Browser Source Runner`
  - `失败后 fallback 到 API connector`
  - `DOM 抽取模板化`
- Scope:
  - Integrate `vercel-labs/agent-browser` as optional source collector
  - Support structured extraction for HF Papers / arXiv / OpenReview
  - Persist capture traces (steps/screenshots) for debugging
- Acceptance:
  - Browser collector can produce normalized paper candidates
  - Fallback path is observable and does not block workflow

## Issue 16 - Browser-Driven Workflow E2E in CI

- Title: `[Feature] Add browser-agent E2E for workflow streaming UX`
- Labels: `roadmap`, `phase-4`, `frontend`, `infra`, `priority-p1`
- Source TODO:
  - `SSE 增量渲染 E2E`
  - `关键截图和性能指标`
  - `CI artifacts`
- Scope:
  - Run end-to-end test for Search → DailyPaper → Analyze
  - Assert DAG restore, non-blank loading state, incremental judge/trend render
  - Upload run traces and screenshots in CI
- Acceptance:
  - Failing UX regressions are detectable in CI
  - Artifacts are attached for triage

## Issue 17 - Platform Benchmark Monitor Agent

- Title: `[Feature] Add benchmark monitor agent for HF/AlphaXiv parity tracking`
- Labels: `roadmap`, `phase-4`, `product`, `automation`, `priority-p2`
- Source TODO:
  - `对标监测 Agent`
  - `能力差距报告 docs/benchmark/`
- Scope:
  - Periodically crawl public product pages and extract capability signals
  - Generate versioned markdown reports under `docs/benchmark/`
- Acceptance:
  - Weekly benchmark report updates automatically
  - Capability diff is auditable over time

## Issue 18 - Push Preview Validation Agent

- Title: `[Feature] Add browser-agent push preview validation`
- Labels: `roadmap`, `phase-4`, `backend`, `qa`, `priority-p2`
- Source TODO:
  - `邮件/Slack/钉钉渲染预览`
  - `多端一致性截图`
- Scope:
  - Open rendered previews via browser agent and capture screenshots
  - Check mandatory sections and formatting constraints
- Acceptance:
  - Template breakages are detected before daily dispatch

## Issue 19 - Browser Extension Smoke Automation

- Title: `[Feature] Add browser extension smoke tests on arXiv pages`
- Labels: `roadmap`, `phase-4`, `frontend`, `qa`, `priority-p2`
- Source TODO:
  - `Browser Extension smoke test`
- Scope:
  - Validate content script injection and action flows on arXiv paper pages
  - Cover CTA button, detail sheet, and API roundtrip
- Acceptance:
  - Core extension flow is tested in CI/nightly

## Issue 20 - Browser Agent Security & Rate Control

- Title: `[Feature] Harden browser-agent secrets, auditing, and rate control`
- Labels: `roadmap`, `phase-4`, `security`, `infra`, `priority-p1`
- Source TODO:
  - `session 密钥管理`
  - `Agent 审计日志`
  - `速率限制/并发隔离`
- Scope:
  - Implement secure secret injection for browser sessions
  - Add auditable logs for domain/action/latency/error
  - Add source-level concurrency and throttling policies
- Acceptance:
  - No plaintext secrets in repo/runtime logs
  - Browser automation runs are traceable and throttled

## Issue 21 - Agent Inventory and Boundary Map

- Title: `[Chore] Build agent inventory and responsibility boundary map`
- Labels: `roadmap`, `phase-3`, `architecture`, `priority-p1`
- Source TODO:
  - `盘点现有 Agent 入口与责任边界`
  - `标记必须 Agent vs 普通 service`
- Scope:
  - Inventory all agent entrypoints (routes/workflows)
  - Document MUST-use-agent vs service-only decision rules
  - Output `docs/agent_inventory.md`
- Acceptance:
  - Every current agent entry has owner, input/output, SLA and fallback

## Issue 22 - AgentRuntime Contract and SourceCollector Port

- Title: `[Feature] Introduce unified AgentRuntime and SourceCollector contracts`
- Labels: `roadmap`, `phase-3`, `backend`, `architecture`, `priority-p1`
- Source TODO:
  - `定义 AgentRuntime 接口`
  - `定义 AgentMessage/AgentResult/AgentError schema`
  - `定义 SourceCollector 接口`
- Scope:
  - Add `core/abstractions/agent_runtime.py`
  - Add `application/ports/source_collector.py`
  - Add compatibility adapters for current agent classes
- Acceptance:
  - Existing agent flows compile and run through adapter layer

## Issue 23 - Unified Event Envelope for SSE + Trace Pipeline

- Title: `[Feature] Unify SSE/event-log envelope across workflows and agents`
- Labels: `roadmap`, `phase-3`, `backend`, `frontend`, `observability`, `priority-p1`
- Source TODO:
  - `SSE 事件统一`
  - `trace_id 全链路贯穿`
  - `前端统一 event parser`
- Scope:
  - Define shared event envelope schema
  - Map workflow events to the schema
  - Add frontend parser utility and migrate key pages
- Acceptance:
  - One parser handles Search/Daily/Analyze streams consistently

## Issue 24 - AgentRuntime Migration Wave 1 (Analyze/Review)

- Title: `[Refactor] Migrate analyze/review routes to AgentRuntime`
- Labels: `roadmap`, `phase-3`, `backend`, `priority-p1`
- Source TODO:
  - `Step 1: analyze + review`
- Scope:
  - Route-level migration for `analyze` and `review`
  - Keep API contract backward-compatible
  - Add regression tests for lifecycle events
- Acceptance:
  - No behavior regressions in analyze/review, with unified runtime events

