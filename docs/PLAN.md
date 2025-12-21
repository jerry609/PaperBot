# PaperBot 计划文档（Plan）

本文件承载“爬虫/信息源/信息聚合/企业级化”的**详细评估**与**可执行计划**。根 `README.md` 只保留对外展示所需的摘要与入口链接，以保持风格统一与阅读体验。

## 背景与目标

- **目标**：将 PaperBot 从“研究工作流工具链”逐步演进为“可长期维护、可运营、可扩展”的平台化项目（支持企业级治理能力）。
- **非目标**：短期内直接达到多租户 SaaS、SLA、高可用等完整企业级要求（需分阶段推进）。

## 现状评估（基于当前代码）

### 1) 爬虫/下载（会议抓取与 PDF 下载）

结论：**功能可用，但工程化与可扩展性不足**，主要风险在解析脆弱、网络策略不统一、并发模型不一致。

- **会议抓取覆盖**：`ConferenceResearchAgent` 支持 `ccs/sp/ndss/usenix`，抓取以 HTML 解析为主，受站点改版影响较大。
- **实现分裂**：存在多份 downloader/变体文件并存（维护成本高、行为不一致风险）。
- **并发模型不一致**：外层存在并发控制，但底层下载器可能强制串行，导致“名义并发、实际串行”。
- **反爬/限流策略偏弱**：缺少统一请求层（指数退避、熔断、站点级策略、代理/UA 池等）。
- **可观测性不足**：大量 `print`，缺少结构化日志与指标（错误码分布、成功率、延迟分布、源站健康度）。

### 2) 信息源管理（Source）与信息聚合（Aggregation）

结论：**当前为 MVP 抽象**：有数据源接口雏形与信息流/调度器原型，但缺少统一的“源注册、去重合并、可查询持久化”体系。

- **数据源抽象不足**：`BaseDataSource` 目前主要覆盖本地文件，DB 数据源为占位；`api/hybrid` 多依赖上层回退逻辑。
- **聚合链路偏原型**：信息流与调度器更多是“内存 + 本地 JSON 状态”，缺少统一数据层（DB）带来的可查询、可追溯、可回放能力。
- **去重规则分散**：标题/ID 去重在多处出现，缺少统一的 `PaperIdentity` 归一化策略（DOI/arXiv/S2/GitHub 等）。

### 3) 企业级化差距（架构与治理）

结论：现有架构适合研发迭代与研究使用，但企业化需要补齐平台治理能力。

- **边界与一致性**：重复实现、配置口径分裂（多处读取 env/配置）、模块边界需要收敛。
- **数据层缺失**：缺少统一主数据与运行记录（Run）体系，导致无法审计、回放、统计与运营。
- **任务编排缺失**：缺少作业队列、幂等、死信、回放机制，难以规模化运行。
- **API 治理不足**：缺鉴权/权限/配额/限流/审计日志/TraceID。
- **可观测性不足**：缺 metrics/tracing/logging 规范与落地。
- **安全与合规**：密钥管理、容器执行安全、数据许可追踪需要制度化。

### 4) 多智能体：记忆模块（Memory）与交互/编排（Orchestration）

结论：PaperBot 已具备“多智能体雏形”，但当前更多停留在**单次运行（run）内的共享上下文**与**轻量消息/评分协作**，距离“可长期演进的 Agent 系统”还缺少四类关键能力：**长期记忆、统一交互契约、可恢复编排、可观测与评测闭环**。

#### 已有能力（基于当前代码）

- **Paper2Code 侧（repro）**
  - `src/paperbot/repro/orchestrator.py`：多阶段流水线编排（Planning/Coding/Verification/Debugging），有 repair loop 与进度跟踪。
  - `src/paperbot/repro/memory/code_memory.py`：`CodeMemory` + `SymbolIndex` 做“跨文件上下文注入”（依赖预测 + 符号检索 + 接口摘要），但主要服务于**本次生成**。
- **学术工作流侧（core/collaboration + workflow_coordinator）**
  - `src/paperbot/core/collaboration/coordinator.py`：`AgentCoordinator` 提供注册、广播、收集结果与简单综合（synthesis）。
  - `src/paperbot/core/collaboration/score_bus.py`：`ScoreShareBus` 作为跨阶段的评分共享总线（含 history + 阈值判断），驱动 Fail-Fast。

#### 主要缺口（影响可扩展/可运营）

- **长期记忆缺失（跨 run / 跨任务）**
  - `CodeMemory` 更像“工作记忆（working memory）”：生命周期跟随一次生成；缺少**持久化、回放、跨 run 检索**。
  - `AgentCoordinator` 的 messages/results 只在内存；缺少“可查询的事件流（event log）”、长期知识沉淀与权限边界。
- **记忆写入没有“可追溯证据”**
  - 缺少事实（fact）/洞察（insight）与证据（citations/artifacts）的结构化存储：例如某个结论来自哪篇论文/哪条日志/哪次验证结果。
- **交互协议缺少统一“消息契约（contract）”**
  - 编排侧（`repro`）与协作侧（`core/collaboration`）各自为政：消息类型、payload 结构、TraceID/RunID 口径不统一。
- **可恢复编排（checkpoint/resume）不足**
  - 当前 orchestrator 有阶段与 repair loop，但缺少“中断恢复/幂等重跑”的状态持久化：运行中断后无法从某个 stage 恢复。
- **评测闭环不足（agent quality loop）**
  - 缺少面向 agent 的回归集、对比基线、以及“记忆策略/编排策略” A/B 的度量体系（成功率、成本、延迟、工具失败率、幻觉率等）。

#### 建议的改造方向（结合业界与论文经验）

- **记忆分层：从“工作记忆”升级到“可运营的长期记忆”**
  - **Run/Episodic Memory（可回放）**：把一次 run 的关键事件（输入、工具调用、产物、评分、验证报告、错误与修复）落到事件流，支持审计与回放。
  - **Semantic Memory（可检索）**：将“可复用的知识单元”（paper facts、实现坑点、稳定 prompt、解析规则、source 可靠性）写入向量/图索引，并带**来源与置信度**。
  - **Policy（写入/遗忘/冲突）**：定义写入门槛（高置信/经验证）、TTL/版本、冲突合并（同一 paper 多来源）与“可撤销”机制。
  - 参考：MemGPT 的“把 LLM 当作 OS，显式管理外部记忆”的范式；Generative Agents 的“存储-检索-反思（reflection）”记忆循环。  
- **交互契约：统一消息 schema + 可插拔工具协议**
  - 统一 `AgentMessage` 的 envelope（`run_id/trace_id/stage/agent_name/type/payload/evidence`），并把 `ScoreShareBus` 事件化（score 也是一种 message）。
  - 引入工具互操作层：优先兼容 **MCP（Model Context Protocol）**，让数据源/检索/执行工具以标准协议挂载，降低 agent 与工具的耦合。
- **编排工程化：可恢复、可追踪、可调度**
  - 引入 checkpoint（阶段完成即落库）与 resume（从 stage 恢复），并把 repair loop 的每次尝试作为独立 run-attempt 记录。
  - 参考：LangGraph 的“持久化/检查点（checkpointing）”设计，用于长流程可恢复执行。
- **评测与治理：从“能跑”到“可持续变强”**
  - 建立 agent regression suite：固定一批论文/仓库/站点快照/失败样例，持续评测“记忆策略 + 编排策略”的变化。
  - 对外部记忆引入“污染防护”：提示注入检测、来源白名单、敏感信息脱敏与权限控制（企业化必须项）。
- **（特别参考 Anthropic）工作流优先 + 多智能体分工 + 上下文治理**
  - **Workflows vs Agents**：优先用可控的 workflow（预定义代码路径）解决“明确且可验证”的子问题，仅在需要动态决策/探索时使用 agent（降低不可控性与成本）。
  - **多智能体 research 的组织方式**：采用“主控（orchestrator）+ 专家（workers）+ 汇总（synthesis）”，并把 *prompt engineering + evaluations + reliability* 作为同等重要的一等公民（不是事后补丁）。
  - **长任务上下文管理**：把“上下文压缩/编辑/裁剪 + 记忆落库 + 检索回填”作为编排层能力，而非分散在各 agent 的 prompt 里（否则会形成不可维护的隐性状态机）。

#### 可引用材料（厂商/框架/论文）

- **论文**
  - MemGPT（长期记忆代理）：`https://arxiv.org/abs/2310.08560`
  - Generative Agents（记忆：存储-检索-反思）：`https://arxiv.org/abs/2304.03442`
  - SWE-agent（软件工程代理实践）：`https://arxiv.org/abs/2405.15793`
- **Anthropic（重点）**
  - Building Effective AI Agents（agent 设计模式：workflow vs agent、组合式编排等）：`https://www.anthropic.com/research/building-effective-agents`
  - How we built our multi-agent research system（多智能体 research 的工程化、评测与可靠性）：`https://www.anthropic.com/engineering/multi-agent-research-system`
  - Managing context on the Claude Developer Platform（长任务上下文治理/性能）：`https://www.anthropic.com/news/context-management`
  - Introducing the Model Context Protocol（MCP 官方发布/生态入口）：`https://www.anthropic.com/news/model-context-protocol`
- **协议/框架**
  - MCP（规范站点）：`https://modelcontextprotocol.io`
  - OpenAI（开源多智能体编排样例）：`https://github.com/openai/swarm`
  - Microsoft AutoGen：`https://github.com/microsoft/autogen`
  - Microsoft GraphRAG：`https://github.com/microsoft/graphrag`
  - LangGraph（持久化/检查点概念入口）：`https://langchain-ai.github.io/langgraph/concepts/persistence/`

## 计划（Plan）

### Phase 0：统一与稳定（P0）

目标：把系统变成“稳定可跑、可回归测试、可定位问题”的工程形态。

- **下载/抓取收敛**
  - 统一 downloader 实现与配置入口（移除重复变体）
  - 抽象出站点适配器（会议/出版社为单位）
- **统一网络请求层**
  - 指数退避 + jitter
  - token bucket 限速
  - 站点级 header/cookie 策略
  - 熔断与降级（站点不可用时跳过/回退）
- **解析契约测试**
  - 为每个会议源加入“页面快照测试/解析契约测试”
  - 站点改版可快速发现并回滚/修复
- **日志标准化**
  - 从 `print` 迁移到结构化日志（JSON），统一字段（source、paper_id、trace_id、latency_ms、status_code）
- **多智能体“最小工程化”**
  - 统一 `run_id/trace_id`：让 `repro/Orchestrator` 与 `core/workflow_coordinator` 全链路贯通
  - 统一消息 envelope：为 `AgentCoordinator`/`ScoreShareBus` 定义稳定 schema（可落库、可回放）

### Phase 1：数据与运营（P1）

目标：让系统具备“可查询、可追溯、可运营”的数据与作业能力。

- **数据持久化**
  - 引入 DB（原型：SQLite；生产：PostgreSQL）
  - 统一主数据模型：Paper/Scholar/CodeRepo/Event/Run/Report
  - 统一 ID 归一化与合并策略（DOI/arXiv/S2/GitHub URL）
- **作业队列与调度**
  - 选择 Celery/Arq/Temporal 之一
  - 幂等与重试、死信队列、任务回放
- **指标与告警**
  - 指标：抓取成功率、解析空结果率、下载耗时分布、LLM 失败率/超时率、成本
- **长期记忆（MVP）**
  - Run/Episodic：把每次 pipeline 的输入/产物/错误/验证报告/评分落库（可查询、可回放）
  - Semantic：把“稳定可复用”的知识单元（解析规则、失败样例、修复补丁、paper facts）写入向量库，并带来源/置信度

### Phase 2：平台化扩展（P2）

目标：可插拔信息源与可配置工作流，支持多团队扩展。

- **Source Registry**
  - 信息源声明：字段覆盖、可靠性评级、速率限制、授权方式、抓取方式
- **Merge Engine**
  - 多源合并：字段置信度、冲突解决策略、合并可解释性
- **插件系统**
  - Source/Agent/Workflow 插件化（注册表 + 配置驱动）
- **工具互操作与编排升级**
  - 兼容 MCP：让 data source / retrieval / execution 工具标准化接入
  - 引入 checkpoint/resume：长流程可恢复（对齐 LangGraph 思路），并对 repair loop 做 attempt 级审计

### Phase 3：企业级治理（P3）

目标：满足对外服务/企业内部平台的治理要求。

- **API 治理**
  - 鉴权（API Key/JWT/OAuth2）
  - RBAC/ABAC、多租户隔离
  - 配额、限流、审计日志
  - OpenAPI 固化契约 + typed client
- **可观测性**
  - OpenTelemetry trace + Prometheus metrics + JSON logs
- **安全与合规**
  - 密钥管理（Vault/KMS）
  - 容器执行安全（网络隔离、资源限额、镜像白名单）
  - 数据许可与来源元数据记录（合规审计）
  - 记忆治理：权限边界（per-user/per-team）、敏感信息脱敏、可删除（Right-to-be-forgotten）与审计

## 问题清单（按优先级）

- **高优先级（正确性/稳定性）**
  - 站点解析易碎：缺少契约测试与回退链路
  - 并发/限流策略不统一：吞吐不可控、失败率不可控
  - 去重/合并规则分散：重复数据、指标不一致风险
  - 多智能体缺少统一 run_id/trace_id 与消息契约：难以观测、难以回放、难以复现
- **中优先级（可维护性/扩展性）**
  - 重复实现需要收敛（downloader/crawler 变体）
  - 配置口径统一（单一 truth source，避免多处读取 env）
  - API/CLI/Web 契约固定（OpenAPI + typed client）
  - 长期记忆缺失：知识无法沉淀复用（跨 run 复用能力弱）
- **长期（平台化/企业化）**
  - DB + 任务队列 + 多租户 + 可观测性 + 合规治理
  - 多智能体评测闭环：回归集、指标体系、A/B 与安全防护（提示注入/数据污染）


