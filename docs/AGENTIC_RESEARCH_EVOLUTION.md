# PaperBot → Agentic Research 演进方案

> **版本**: v1.0 · **日期**: 2026-03-02
> **关联文档**: [OpenClaw 源码综述](./OPENCLAW_RESEARCH_REPORT.md) · [P2C 记忆优化](./P2C_MEMORY_AND_CONTEXT_OPTIMIZATION.md) · [架构总览](./architecture_overview.md)
>
> **核心目标**: 将 PaperBot 从"论文搜索推荐 + Paper2Code"系统演进为 **Agentic Research** 平台，并与 OpenClaw 深度集成。

---

## 目录

1. [Agentic Research 行业调研](#1-agentic-research-行业调研)
2. [PaperBot 现状 Gap Analysis](#2-paperbot-现状-gap-analysis)
3. [OpenClaw 集成架构](#3-openclaw-集成架构)
4. [演进方案设计](#4-演进方案设计)
5. [实施路线图](#5-实施路线图)

---

## 1. Agentic Research 行业调研

### 1.1 主要项目对比

| 项目 | 核心架构 | 关键创新 | 与 PaperBot 关系 |
|------|----------|----------|------------------|
| **GPT Researcher** | Plan-and-Solve + RAG, LangGraph 多 Agent | 递归深度/广度树探索, 5-6 页报告 | 研究报告生成模式可借鉴 |
| **STORM** (Stanford) | 两阶段: 预写作(多视角对话) → 写作 | 多视角模拟对话, Wikipedia 级文章 | 多视角论文评估可借鉴 |
| **PaperQA2** (Future-House) | Agentic RAG, 模块化三组件 | 动态工具选择, 迭代式 RAG | 论文问答直接可集成 |
| **AI Scientist v2** (Sakana) | 端到端: 假设→实验→写作→审稿 | 渐进式树搜索实验管理 | Paper2Code + 实验管理参考 |
| **AI-Researcher** (HKUDS) | Idea Generator + Writer Agent | 分析高影响力论文→生成新方向 | 研究方向推荐可借鉴 |
| **PaperCoder** | 三阶段: Planning→Analysis→Generation | DAG 依赖排序, 验证+精炼 Agent | 直接竞品, 提升 Paper2Code |
| **Agent Reach** | AI Agent 信息采集层 | 零 API 费, 6 平台社交媒体访问 | 作为信息采集插件集成 |

### 1.2 核心架构模式

#### 模式 1: Orchestrator-Worker（Anthropic 设计）
```
                 ┌─────────────────────┐
                 │   LeadResearcher    │
                 │   (协调/分解/记忆)   │
                 └──┬──┬──┬──┬──┬─────┘
                    │  │  │  │  │
              ┌─────┘  │  │  │  └─────┐
              ▼        ▼  ▼  ▼        ▼
          ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
          │Search│ │Search│ │Cite  │ │Synth │
          │Sub-1 │ │Sub-2 │ │Agent │ │Agent │
          └──────┘ └──────┘ └──────┘ └──────┘
```
- Lead + Subagents 比单 agent 性能提升 90.2%
- Token 使用量解释 80% 性能差异
- 成本约 15x 标准对话

#### 模式 2: DAG 研究计划遍历（Egnyte）
```
   研究问题
       │
       ▼
   [问题分解] ──→ DAG 图
       │
       ▼
   拓扑排序遍历 ──→ 依赖满足的问题并行分配
       │
       ▼
   Map/Reduce ──→ 并行 Researcher → 同步收集 → 识别 Gap
       │
       ▼
   迭代精炼 ──→ Gap → 新问题 → 新 Researcher
```

#### 模式 3: 迭代搜索精炼循环
```
   ┌──→ Search ──→ Evaluate ──→ 识别 Gap ──→ 精炼 Query ──┐
   │                                                        │
   └────────────────────────────────────────────────────────┘
                     (直到 Gap 收敛或预算耗尽)
```
这是 "Agentic Deep Research" 的核心范式。

#### 模式 4: 多视角模拟对话（STORM）
```
   Topic
     │
     ├──→ Perspective 1 (方法论专家) ──→ 提问 → 搜索 → 回答
     ├──→ Perspective 2 (应用专家)   ──→ 提问 → 搜索 → 回答
     ├──→ Perspective 3 (批判者)     ──→ 提问 → 搜索 → 回答
     │
     ▼
   合并多视角 → 生成全面大纲 → 撰写报告
```

### 1.3 传统搜索 vs Agentic Research 对比

| 维度 | 传统论文搜索 (PaperBot 现状) | Agentic Research (目标) |
|------|------------------------------|--------------------------|
| 搜索 | 单次查询, 静态结果 | 多轮迭代, 自主精炼 |
| 评估 | 用户手动评估 | Agent 自动评估、过滤、重排 |
| 综合 | 用户自行阅读总结 | Agent 跨论文综合分析 |
| 引用遍历 | 手动追踪参考文献 | 自动多跳引用图遍历 |
| 广度 | 受限于用户查询表述 | Agent 发现用户未考虑的视角 |
| 输出 | 论文列表 | 结构化研究报告 + 引用 |
| 适应性 | 固定算法 | 基于发现动态调整策略 |

---

## 2. PaperBot 现状 Gap Analysis

### 2.1 Agent 架构 Gap

| 现有能力 | 缺失能力 | 关键文件 |
|----------|----------|----------|
| `BaseAgent` 模板方法 (validate→execute→post_process) | **无 tool-use/function-calling**: `ask_claude()` 是单轮调用, 不支持 ReAct 循环 | `agents/base.py:122-144` |
| 8 种 Agent + AgentCoordinator | **无自主搜索循环**: Agent 不能自主决定搜索→阅读→精炼 | `agents/base.py` 需新增 `run_with_tools()` |
| ScoreShareBus + FailFast | **无动态 Agent 生成**: Coordinator 是 fire-and-forget, 无反馈/重规划 | `core/collaboration/coordinator.py:32-258` |
| DeepResearchAgent 有反思循环 | **3/4 搜索类型标记 TODO**: code search/citation analysis/collaborator search 未实现 | `agents/scholar_tracking/deep_research_agent.py:371-419` |
| CollaborationBus 轮次消息 | **无综合 Agent**: `synthesize()` 只是字符串拼接, 非 LLM 跨论文综合 | `core/collaboration/coordinator.py` |

### 2.2 搜索/爬取 Gap

| 现有能力 | 缺失能力 | 关键文件 |
|----------|----------|----------|
| 5 数据源 (S2, OpenAlex, arXiv, PapersCool, HF) | **无全文检索**: 搜索只返回元数据, 无 PDF 下载+文本提取 | `infrastructure/api_clients/` |
| RRF 多源融合 + QueryRewriter | **无引用图遍历**: S2 有 references/citations API 但无 agent 级抽象 | `application/services/paper_search_service.py` |
| 异步 HTTP + 限流 + 重试 | **无 Web 搜索**: 无 Google Scholar, 无博客/教程搜索 | 需新增 client |
| title hash 去重 | **无跨源 DOI/arXiv ID 匹配** | `paper_search_service.py:_paper_key()` |
| - | **无 LLM 驱动查询生成**: 查询来自用户或模板, 非基于中间发现自动生成 | 需新增 |

### 2.3 工作流 Gap

| 现有能力 | 缺失能力 | 关键文件 |
|----------|----------|----------|
| Pipeline 顺序执行 | **无 DAG 执行**: 不支持分支、并行、条件扇出 | `core/pipeline/pipeline.py:88-111` |
| 5 阶段分析 Pipeline | **无动态阶段注入**: Pipeline 静态定义, 不能基于中间结果添加阶段 | `core/workflow_coordinator.py` |
| Paper2Code 4 阶段 + 修复循环 | **无 human-in-the-loop 检查点**: `CHECKPOINT` 消息类型存在但未实现 | `core/pipeline/` |
| Scheduler asyncio 循环 | **无工作流持久化/恢复**: Pipeline 状态临时, 中断无法续跑 | `workflows/scheduler.py` |
| 批量处理 | **批量处理串行**: `run_batch_pipeline()` 逐篇处理, 无并发 | `core/workflow_coordinator.py:415-444` |

### 2.4 Context Engine Gap

| 现有能力 | 缺失能力 | 关键文件 |
|----------|----------|----------|
| 研究阶段检测 + 个性化推荐 | **无对话式上下文**: context pack 每次重建, 不维护对话历史 | `context_engine/engine.py` |
| Track 路由 + 锚点作者 | **无工作记忆**: 无法记住"论文 X 与论文 Y 矛盾"并持续携带 | `context_engine/engine.py` |
| 关键词匹配论文 | **无语义搜索**: 论文匹配只有关键词重叠, 无 embedding 相似度 | `engine.py:_paper_keyword_match_count` |
| 平铺列表上下文 | **无知识图谱**: 无结构化表示(主张、证据、论文关系) | 需新建模块 |

### 2.5 核心能力缺失总结

```
                     PaperBot 当前                    Agentic Research 需要
                 ┌──────────────────┐            ┌──────────────────────────┐
  Agent 层       │ 单轮 ask_claude() │    →→→     │ ReAct 循环 + Tool-Use    │
                 │ 静态 Coordinator  │    →→→     │ 动态生成 + 反馈重规划     │
                 └──────────────────┘            └──────────────────────────┘
  搜索层         ┌──────────────────┐            ┌──────────────────────────┐
                 │ 单次查询, 元数据   │    →→→     │ 迭代搜索 + 全文 + 引用图  │
                 │ 5 数据源, 无 Web   │    →→→     │ N 数据源 + Web + 社交媒体  │
                 └──────────────────┘            └──────────────────────────┘
  工作流层       ┌──────────────────┐            ┌──────────────────────────┐
                 │ 顺序 Pipeline     │    →→→     │ DAG + 动态阶段 + 检查点   │
                 │ 串行批处理        │    →→→     │ 并行 + 持久化 + 可恢复    │
                 └──────────────────┘            └──────────────────────────┘
  记忆层         ┌──────────────────┐            ┌──────────────────────────┐
                 │ 关键词匹配, 无衰减 │    →→→     │ Hybrid Search + 衰减     │
                 │ 两个孤立记忆系统   │    →→→     │ 统一分层记忆 + 知识图谱   │
                 └──────────────────┘            └──────────────────────────┘
```

---

## 3. OpenClaw 集成架构

### 3.1 集成方式选择

经过对比分析，推荐 **Hybrid Plugin + Bridge** 架构：

| 方案 | 优点 | 缺点 | 评分 |
|------|------|------|------|
| **纯嵌入** (PaperBot 重写为 TS) | 最紧密集成 | 重写成本巨大, 丧失 Python 生态 | 3/10 |
| **纯桥接** (完全独立) | 零耦合 | 无法利用 OpenClaw 记忆/调度/渠道 | 5/10 |
| **Hybrid Plugin + Bridge** ✓ | 保留独立性 + 利用基础设施 | 需维护 TS shim (~200-400 行) | **8/10** |

### 3.2 整体架构

```
┌───────────────────────────────────────────────────────────────────┐
│                    OpenClaw Gateway (TypeScript)                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Plugin: paperbot-openclaw                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │ registerTool  │  │ registerHook │  │ registerCli      │  │  │
│  │  │ • paper_search│  │ • msg_recv   │  │ • openclaw paper │  │  │
│  │  │ • paper_analyze│ │ • before_    │  │   search|analyze │  │  │
│  │  │ • paper_track │  │   prompt     │  │   track|gen-code │  │  │
│  │  │ • gen_code    │  │ • gateway_   │  └──────────────────┘  │  │
│  │  │ • review      │  │   start      │  ┌──────────────────┐  │  │
│  │  │ • research    │  └──────────────┘  │ Cron Jobs (4)    │  │  │
│  │  └──────┬───────┘                     │ • paper-monitor  │  │  │
│  │         │ HTTP                        │ • weekly-digest  │  │  │
│  └─────────┼─────────────────────────────│ • conf-deadline  │──┘  │
│            │                             │ • cite-monitor   │     │
│  ┌─────────┼─────────────────────────────└──────────────────┘     │
│  │ OpenClaw 基础设施                                              │
│  │  ├─ Memory (用户层): MEMORY.md + FTS5 + sqlite-vec             │
│  │  ├─ Channels: Telegram / Discord / Slack / ...                │
│  │  ├─ Subagents: 隔离 session, 深度/并发守卫                     │
│  │  ├─ ACP: IDE 集成 (VS Code / Cursor)                         │
│  │  └─ Compaction: 上下文压缩 + 记忆保全                          │
│  └────────────────────────────────────────────────────────────────┘
│                             │
└─────────────────────────────┼─────────────────────────────────────┘
                              │ HTTP/gRPC
┌─────────────────────────────▼─────────────────────────────────────┐
│                PaperBot Python Backend (FastAPI)                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Agentic Research Core (新增)                                │   │
│  │  ├─ ResearchLoopAgent: 迭代搜索-分析-精炼循环               │   │
│  │  ├─ CitationGraphAgent: 多跳引用图遍历                      │   │
│  │  ├─ SynthesisAgent: LLM 跨论文综合分析                     │   │
│  │  └─ PerspectiveAgent: 多视角论文评估                        │   │
│  ├────────────────────────────────────────────────────────────┤   │
│  │ 现有模块 (保留 + 增强)                                      │   │
│  │  ├─ agents/ (8 Agent + BaseAgent 增加 tool-use)            │   │
│  │  ├─ repro/ (Paper2Code, CodeMemory 持久化)                 │   │
│  │  ├─ context_engine/ (增加语义搜索 + 工作记忆)               │   │
│  │  └─ infrastructure/ (增加全文检索 + Web 搜索 + Agent Reach) │   │
│  ├────────────────────────────────────────────────────────────┤   │
│  │ Domain Storage (PaperBot 专属)                              │   │
│  │  ├─ papers, tracks, code_experience (SQLAlchemy)           │   │
│  │  ├─ paper-scoped memory (per-paper 分析记忆)                │   │
│  │  └─ research_sessions (研究会话持久化)                      │   │
│  └────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.3 记忆层集成策略

```
┌────────────────────── 记忆分层 ──────────────────────┐
│                                                       │
│  Layer 0: Always-on (~200 tokens)        [OpenClaw]   │
│  → 用户 Profile: 姓名/研究方向/活跃 Track             │
│  → 来源: OpenClaw MEMORY.md                          │
│                                                       │
│  Layer 1: Track-relevant (~500 tokens)   [OpenClaw]   │
│  → 当前 Track 目标/关键词/近期决策                     │
│  → 来源: OpenClaw memory/YYYY-MM-DD.md               │
│                                                       │
│  Layer 2: Query-relevant (~1000 tokens)  [桥接]       │
│  → Hybrid Search 命中的记忆                           │
│  → 来源: OpenClaw sqlite-vec + PaperBot paper store   │
│                                                       │
│  Layer 3: On-demand (按需加载)           [PaperBot]   │
│  → paper-scoped 记忆 / CodeMemory / 完整分析结果       │
│  → 来源: PaperBot SQLAlchemy stores                   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**原则**: OpenClaw 管 **用户级非结构化知识**, PaperBot 管 **领域级结构化数据**。ContextEngineBridge 统一读取两者。

### 3.4 调度集成: OpenClaw Cron 替代 PaperBot Scheduler

| PaperBot 当前 | OpenClaw Cron 替代 |
|---|---|
| `asyncio.sleep(60)` 循环 | `{ kind: "cron", expression: "0 6 * * *" }` |
| JSON 文件持久化 | OpenClaw 持久化 job state |
| console-only 通知 | 8 渠道投递 (Telegram/Discord/Slack/...) |
| 无隔离 | 隔离 session, 不污染主上下文 |
| 无错误恢复 | MAX_SCHEDULE_ERRORS=3 自动禁用 + 卡死检测 |
| 无 stagger | SHA-256(jobId) % window 防雷群 |

**4 个 Cron Job**:
1. `paper-monitor` — 每日学者追踪 (替代 `Scheduler._run_loop`)
2. `weekly-digest` — 每周研究摘要 (替代 `NotificationType.WEEKLY_DIGEST`)
3. `conference-deadlines` — 每日截稿提醒 (替代 `ConferenceTracker`)
4. `citation-monitor` — 每小时引用里程碑检测

### 3.5 Subagent 映射

| PaperBot Agent | OpenClaw Subagent 模式 | 说明 |
|---|---|---|
| PaperSummarizer | `"run"` (一次性) | 单篇摘要 |
| PaperJudge | `"run"` | 评分 |
| RelevanceAssessor | `"run"` | 个性化相关性 |
| TrendAnalyzer | `"session"` (持久) | 需多轮探索 |
| ResearchLoopAgent (新) | `"session"` | 迭代搜索循环 |
| Paper2Code PlanningAgent | `"session"` | 迭代规划精炼 |
| Paper2Code CodingAgent | `"session"` | 代码生成+反馈 |
| Paper2Code DebuggingAgent | `"session"` | 交互式调试 |

**关键优势**: OpenClaw subagent 拥有独立 session 和 context window, 解决多 Agent 运行时上下文污染问题。

### 3.6 渠道投递升级

```
当前:  NotificationHandler → console only

未来:  OpenClaw Channels
       ├─ Telegram: 论文卡片 + 内联按钮 [分析] [追踪] [生成代码]
       ├─ Discord: Rich Embed + 反应式反馈 (👍=相关 👎=不相关)
       ├─ Slack: 频道通知 + 线程讨论
       └─ IDE (ACP): 编辑器内论文上下文查询
```

### 3.7 集成风险与缓解

| 风险 | 严重度 | 缓解策略 |
|------|--------|----------|
| OpenClaw 依赖 | 高 | PaperBot 保留独立 FastAPI, 插件是可选增强层 |
| 迁移复杂度 | 中 | 分 4 阶段, 每阶段独立交付价值 |
| HTTP 桥接延迟 | 低 | PaperBot 操作本身 I/O-bound (LLM ~1-30s), 桥接 ~1-5ms 可忽略 |
| 记忆一致性 | 中 | ContextEngineBridge 统一读取两个记忆系统 + hook 同步关键事件 |
| TS shim 维护 | 低 | ~200-400 行, PaperBot API 是稳定契约 |
| 多租户 | 中 | 需同步修复 PaperBot G7 (workspace_id 贯通) |

---

## 4. 演进方案设计

### 4.1 BaseAgent 升级: 从单轮到 ReAct

```python
# agents/base.py — 新增 tool-use 支持

class BaseAgent:
    def __init__(self):
        self.tools: list[Tool] = []  # 工具注册表

    async def run_with_tools(self, task: str, max_iterations: int = 10) -> ExecutionResult:
        """ReAct 循环: Reason → Act → Observe → Repeat"""
        messages = [{"role": "user", "content": task}]
        for i in range(max_iterations):
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                tools=[t.schema for t in self.tools],
            )
            if response.stop_reason == "end_turn":
                return ExecutionResult(success=True, data=response.content)
            # 执行 tool calls
            tool_results = await self._execute_tools(response.tool_calls)
            messages.extend([response, tool_results])
        return ExecutionResult(success=False, error="Max iterations reached")
```

### 4.2 ResearchLoopAgent: 迭代搜索核心

```python
# agents/research_loop/agent.py — 新增

class ResearchLoopAgent(BaseAgent):
    """自主研究循环: 搜索 → 评估 → 识别 Gap → 精炼查询 → 再搜索"""

    tools = [
        PaperSearchTool,        # 多源论文搜索
        CitationTraversalTool,  # 引用图遍历
        FullTextRetrievalTool,  # PDF 全文提取
        WebSearchTool,          # Web 搜索 (Agent Reach)
        MemoryStoreTool,        # 存储发现到工作记忆
        MemoryRecallTool,       # 检索已有发现
        SynthesisTool,          # 跨论文综合
    ]

    async def research(self, question: str, depth: int = 3) -> ResearchReport:
        plan = await self._decompose_question(question)  # DAG 分解
        for level in range(depth):
            results = await self._parallel_search(plan.open_questions)
            findings = await self._evaluate_and_extract(results)
            gaps = await self._identify_gaps(findings, plan)
            if not gaps:
                break
            plan = await self._refine_plan(plan, gaps)
        return await self._synthesize_report(plan.all_findings)
```

### 4.3 引用图遍历

```python
# infrastructure/api_clients/citation_graph.py — 新增

class CitationGraphClient:
    """多跳引用图遍历"""

    async def traverse(
        self,
        seed_paper_id: str,
        direction: Literal["references", "citations", "both"],
        max_hops: int = 2,
        max_papers_per_hop: int = 10,
        relevance_filter: Callable[[Paper], float] | None = None,
    ) -> CitationGraph:
        visited = set()
        frontier = [seed_paper_id]
        graph = CitationGraph()

        for hop in range(max_hops):
            next_frontier = []
            tasks = [self.s2_client.get_paper(pid) for pid in frontier if pid not in visited]
            papers = await asyncio.gather(*tasks)
            for paper in papers:
                visited.add(paper.id)
                related = paper.references if direction != "citations" else paper.citations
                scored = [(p, relevance_filter(p)) for p in related] if relevance_filter else [(p, 1.0) for p in related]
                top_k = sorted(scored, key=lambda x: -x[1])[:max_papers_per_hop]
                for p, score in top_k:
                    graph.add_edge(paper.id, p.id, score)
                    next_frontier.append(p.id)
            frontier = next_frontier
        return graph
```

### 4.4 DAG Pipeline 升级

```python
# core/pipeline/dag_pipeline.py — 新增

class DAGPipeline:
    """支持分支、并行、条件扇出的 DAG 流水线"""

    def __init__(self):
        self.stages: dict[str, PipelineStage] = {}
        self.edges: list[tuple[str, str, Callable | None]] = []  # (from, to, condition)

    def add_stage(self, stage: PipelineStage) -> "DAGPipeline":
        self.stages[stage.name] = stage
        return self

    def add_edge(self, from_: str, to: str, condition: Callable | None = None):
        self.edges.append((from_, to, condition))

    async def execute(self, context: dict) -> dict:
        """拓扑排序 + 并行执行就绪阶段"""
        completed = set()
        while len(completed) < len(self.stages):
            ready = self._get_ready_stages(completed, context)
            if not ready:
                break
            results = await asyncio.gather(*[s.execute(context) for s in ready])
            for stage, result in zip(ready, results):
                context[stage.name] = result
                completed.add(stage.name)
        return context
```

### 4.5 Agent Reach 集成

```python
# infrastructure/api_clients/agent_reach.py — 新增

class AgentReachClient:
    """通过 Agent Reach 获取社交媒体论文讨论"""

    async def get_paper_discussions(self, paper_title: str) -> SocialContext:
        twitter = await self._search_twitter(f"{paper_title} paper")
        reddit = await self._search_reddit(f"{paper_title}", subreddits=["MachineLearning", "deeplearning"])
        youtube = await self._search_youtube(f"{paper_title} explained")
        github = await self._search_github(paper_title)
        return SocialContext(
            twitter_threads=twitter,
            reddit_posts=reddit,
            youtube_videos=youtube,
            github_repos=github,
            community_sentiment=self._analyze_sentiment(twitter + reddit),
        )
```

---

## 5. 实施路线图

### Phase 1: 基础能力 (2-3 周)
```
Week 1-2:
  ├── BaseAgent 增加 tool-use / ReAct 循环
  ├── 实现 ContextEngineBridge (打通 G1)
  ├── 激活 paper scope 记忆 (修复 G2)
  └── CodeMemory 持久化 (修复 G3)

Week 3:
  ├── FTS5 索引替代 SQL LIKE (修复 G4 Phase A)
  └── 记忆衰减机制 (修复 G6)
```

### Phase 2: Agentic Research 核心 (3-4 周)
```
Week 4-5:
  ├── 实现 ResearchLoopAgent (迭代搜索循环)
  ├── 实现 CitationGraphClient (多跳引用遍历)
  ├── 实现 SynthesisAgent (跨论文综合)
  └── DAGPipeline 替代顺序 Pipeline

Week 6-7:
  ├── embedding + sqlite-vec hybrid search (G4 Phase B/C)
  ├── 全文检索 pipeline (PDF download + extraction)
  ├── Web 搜索集成
  └── Agent Reach 集成 (社交媒体上下文)
```

### Phase 3: OpenClaw 集成 (3-4 周)
```
Week 8-9:
  ├── 开发 paperbot-openclaw TypeScript 插件
  │   ├── registerTool: 6 个核心工具
  │   ├── registerHook: message_received / before_prompt
  │   └── registerCli: openclaw paper 子命令
  └── Cron 迁移: 4 个定时任务

Week 10-11:
  ├── 渠道投递: Telegram + Discord 论文通知
  ├── 记忆桥接: OpenClaw memory ↔ PaperBot domain store
  ├── Subagent 映射: PaperBot agents → OpenClaw subagents
  └── ACP: IDE 内论文查询原型
```

### Phase 4: 高级功能 (持续)
```
  ├── 多视角论文评估 (STORM 模式)
  ├── 研究报告自动生成 (GPT Researcher 模式)
  ├── Human-in-the-loop 检查点
  ├── 研究工作流持久化/恢复
  ├── Context 分层加载优化
  └── 跨 Track 批量搜索 (修复 G5)
```

### 里程碑与验收标准

| 里程碑 | 验收标准 |
|--------|----------|
| M1: ReAct Agent | BaseAgent 能自主调用 3+ 工具完成一个研究任务 |
| M2: 迭代搜索 | ResearchLoopAgent 对一个 topic 自主搜索 3 轮, 找到 20+ 相关论文 |
| M3: OpenClaw 插件 | 通过 OpenClaw 对话调用 `paper_search` 工具并返回结果 |
| M4: 渠道投递 | 每日论文推荐通过 Telegram 自动推送, 支持内联按钮交互 |
| M5: 完整 Agentic | 用户输入研究问题 → 系统自主搜索/分析/综合 → 输出结构化报告 |

---

## 附录: 参考项目

| 项目 | GitHub | 关键参考 |
|------|--------|----------|
| GPT Researcher | `assafelovic/gpt-researcher` | 递归研究树, 报告生成 |
| STORM | `stanford-oval/storm` | 多视角对话, Co-STORM |
| PaperQA2 | `Future-House/paper-qa` | Agentic RAG, 动态工具 |
| AI Scientist v2 | `SakanaAI/AI-Scientist-v2` | 端到端科研, 树搜索实验 |
| Agent Reach | `Panniantong/Agent-Reach` | 社交媒体信息采集 |
| PaperCoder | `going-doer/Paper2Code` | DAG 代码生成 |
| DeepCode | `HKUDS/DeepCode` | 论文复现 (73.5% 准确率) |
| Awesome Deep Research | `DavidZWZ/Awesome-Deep-Research` | 综述合集 |
