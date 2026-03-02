# PaperBot Context/Memory 优化分析

> 基于 OpenClaw 记忆机制调研 + PaperBot 现有系统诊断，提出 7 项优化建议。
>
> 关联文档：[P2C Overview](./P2C_OVERVIEW.md) · [Module 1 Core Engine](./P2C_MODULE_1_CORE_ENGINE.md) · [Module 2 API & Storage](./P2C_MODULE_2_API_STORAGE.md)

---

## Part 1: OpenClaw 记忆机制

OpenClaw 采用**三层记忆架构 + Markdown 文件为 source of truth**：

### 三层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Context（短期 / 工作记忆）                             │
│  当前对话上下文，受模型 context window 限制                       │
│  （Claude 200K / GPT-4 128K）                                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Session History（中期记忆）                             │
│  memory/YYYY-MM-DD.md 日志，自动加载当天 + 前一天                │
│  超过 2 天的日志不自动加载，通过 hybrid search 按需检索           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: User Profile（长期记忆）                               │
│  MEMORY.md 策划型长期知识（偏好 / 决策 / 项目上下文）             │
│  每次对话前自动注入，使 Agent "记住你是谁"                       │
└─────────────────────────────────────────────────────────────────┘
```

### 关键机制

| 机制 | 说明 |
|------|------|
| **Hybrid Search** | BM25 关键词 + sqlite-vec 向量，加权融合（`finalScore = vectorWeight × vectorScore + textWeight × textScore`） |
| **Temporal Decay** | 旧记忆分数衰减（148 天前的 0.91 match → 衰减到 0.03） |
| **Pre-Compaction Flush** | context window 快满时，自动触发静默 turn 把关键记忆写入文件，再压缩 |
| **MMR 去重** | 检索结果做最大边际相关性去重，避免注入重复信息 |
| **Provider Fallback** | 嵌入模型 local → OpenAI → Gemini 级联降级 |

### 性能优化手段

| 策略 | 效果 |
|------|------|
| `/compact` 命令 | 压缩对话历史为高密度摘要，保留核心逻辑 |
| `contextTokens` 限制 | 配置最大 context 50K，避免重载全部历史 |
| Model tiering | Heartbeat/简单任务用 Haiku，复杂任务用 Opus |
| Cron 隔离 session | 定时任务独立 context，不污染主会话 |
| Skill 封装 | 把指令从 personality.md 移入 Skill，按需加载而非每次注入 |
| `cache_control` | Anthropic API 的 prompt caching，静态 context 缓存读取成本降 90% |
| MEMORY.md 路由索引 | 保持 50 行以内，指向详细文件按需加载，每消息开销 ~200 tokens |

### 参考来源

- [OpenClaw Memory Docs](https://docs.openclaw.ai/concepts/memory)
- [OpenClaw Memory System Deep Dive](https://snowan.gitbook.io/study-notes/ai-blogs/openclaw-memory-system-deep-dive)
- [OpenClaw Token Cost Optimization Guide](https://help.apiyi.com/en/openclaw-token-cost-optimization-guide-en.html)
- [OpenClaw Runbook: Memory Configuration](https://moltfounders.com/openclaw-runbook/memory-configuration)
- [MemSearch (extracted from OpenClaw)](https://milvus.io/blog/we-extracted-openclaws-memory-system-and-opensourced-it-memsearch.md)
- [Mem0 Memory for OpenClaw](https://mem0.ai/blog/mem0-memory-for-openclaw)

---

## Part 2: PaperBot 当前 Context 系统诊断

通过代码分析，发现 PaperBot 存在**两个完全隔离的记忆系统**，以及 **10 个具体问题**。

### 现状：两个孤立的记忆世界

```
记忆系统 A: ContextEngine (推荐用)          记忆系统 B: CodeMemory (复现用)
══════════════════════════════              ══════════════════════════════

位置: context_engine/ + memory/            位置: repro/memory/
持久化: SQLite (跨会话)                    持久化: 纯内存 (单次运行后丢失)
作用域: global + track                     作用域: 无 (平铺 Dict)
检索: SQL LIKE + token overlap             检索: AST 符号索引 + 关键词 RAG
用途: 论文推荐 + 个性化                    用途: 代码生成上下文

          ╳ 两个系统之间没有任何桥接 ╳
```

### 10 个具体问题

#### G1 — 推荐上下文与执行上下文完全断裂 `P0`

- **位置**: `engine.py` → `orchestrator.py`
- **描述**: ContextEngine 构建的丰富用户画像（偏好/目标/track记忆/论文评分）从未传递给 Orchestrator。`PaperContext` 只有 title/abstract/method，用户的研究积累被浪费。

#### G2 — `project` 和 `paper` scope 声明了但从未使用 `P0`

- **位置**: `schema.py:46`, `engine.py:579-630`
- **描述**: `MemoryCandidate.scope_type` 注释写了 `global/track/project/paper` 四种 scope，但只有 `global` 和 `track` 有实际代码路径。paper-scoped 记忆（"这篇论文上次分析发现了什么"）无法存取。

#### G3 — CodeMemory 每次运行后丢失 `P1`

- **位置**: `code_memory.py`
- **描述**: 代码生成的所有经验（成功模式、失败原因、验证过的结构）不持久化。同一篇论文重新跑，从零开始。

#### G4 — 记忆检索只有关键词匹配，无语义搜索 `P1`

- **位置**: `memory_store.py`
- **描述**: `search_memories()` 用 SQL LIKE 粗过滤 + token overlap 打分。TrackRouter 用了 embedding，但记忆检索本身没有。语义相关但措辞不同的记忆会被漏掉。

#### G5 — 跨 Track 搜索是 N+1 查询 `P2`

- **位置**: `engine.py:615-635`
- **描述**: `include_cross_track=True` 时，遍历所有非归档 track（最多 50 个），每个发一次 `search_memories()`。用户 track 多时延迟线性增长。

#### G6 — 无记忆衰减/过期机制 `P2`

- **位置**: `memory_store.py`
- **描述**: `expires_at` 和 `last_used_at` 字段存在但从未赋值。`use_count` 被追踪但从未用于排序。旧记忆永远和新记忆同等权重。

#### G7 — `workspace_id` 未贯通 `P2`

- **位置**: `engine.py`, `memory_store.py`
- **描述**: 列存在、Store 方法接受参数，但 `build_context_pack()` 从未传入。多租户/团队隔离无法生效。

#### G8 — 评估数据只写不读 `P2`

- **位置**: `engine.py`
- **描述**: `ResearchContextRunModel` + `PaperImpressionModel` 持久化了，但从未反馈回推荐循环做在线学习。

#### G9 — 推荐策略不持久化/不可覆盖 `P3`

- **位置**: `engine.py`
- **描述**: 研究阶段策略（survey/writing/rebuttal）自动推断但不持久化，用户无法手动设置。

#### G10 — 记忆提取仅支持中文 `P3`

- **位置**: `extractor.py:49-119`
- **描述**: heuristic 提取全是中文正则，LLM 提取 system prompt 也是中文。英文用户效果会降级。

---

## Part 3: 优化建议

### 优化 1：打通推荐 → 执行的上下文桥（修复 G1）`P0`

这是最高优先级的修复。P2C 设计文档中已经规划了 `ContextEngineBridge`，但现有 Orchestrator 也需要修。

```
当前：
  ContextEngine.build_context_pack()
       ↓ (返回丰富上下文，但没人消费)
       ╳
  Orchestrator.run(paper_context=PaperContext(title, abstract))
       ↓ (只有最基础信息)

优化后：
  ContextEngine.build_context_pack()
       ↓
  ContextEngineBridge.enrich(normalized_input, request)
       ↓ (注入 user_memory, project_context, track_goals)
  ExtractionOrchestrator.run(normalized_input)
       ↓ (Stage A 拿到 user_memory → "为什么这篇论文和你的研究相关")
       ↓ (Stage E 拿到 track_goals → "路线图按你的项目目标调整")
```

**具体动作**：在 `NormalizedInput` 中注入 `user_memory` 和 `project_context`（P2C Module 1 已设计），并确保 Stage A 和 Stage E 的 prompt 使用这些上下文。

### 优化 2：激活 paper scope（修复 G2）`P0`

基础设施已就绪，只需在关键路径上写入和读取 paper-scoped 记忆。

**写入时机**：

1. P2C 生成完成 → 将 Observation 摘要写入 paper scope
   ```python
   memory_store.add_memories([MemoryCandidate(
       kind="fact",
       content="此论文使用 Transformer encoder-decoder, lr=1e-4, AdamW",
       scope_type="paper",
       scope_id=paper_id,
   )])
   ```
2. 用户在 Studio 手动编辑 → 写入 paper scope
3. 复现成功/失败 → 将结果写入 paper scope

**读取时机**：

4. 再次对同一论文生成 P2C → 先查 paper scope，复用已有记忆
5. `build_context_pack()` 推荐论文时 → 查 paper scope 判断是否已分析过

### 优化 3：给记忆检索加语义搜索（修复 G4，借鉴 OpenClaw）`P1`

借鉴 OpenClaw 的 hybrid search 模式：BM25 + 向量，加权融合。

```
当前：
  search_memories(query) → SQL LIKE → token overlap 打分

优化后（三阶段）：
  Phase A: 加 FTS5 索引（已在 P2C Module 2 DDL 中设计）
     → 替代 SQL LIKE，提升关键词搜索质量

  Phase B: 加 embedding 列 + sqlite-vec
     → 复用 TrackRouter 已有的 OpenAI embedding provider
     → 对 memory content 生成 embedding，存入 memory_items 表

  Phase C: Hybrid fusion
     → final_score = 0.6 × vector_score + 0.4 × bm25_score
     → 借鉴 OpenClaw 的 union 策略：任一通道命中即保留
```

### 优化 4：引入记忆衰减（修复 G6，借鉴 OpenClaw）`P2`

```
OpenClaw 做法：
  decayed_score = raw_score × decay_factor(age_days)
  decay_factor = exp(-age_days / half_life)

PaperBot 适配：
  ① 在 search_memories() 结果排序中引入 recency 权重
     final_score = relevance_score × 0.7 + recency_score × 0.2 + usage_score × 0.1

  ② recency_score = exp(-(now - last_used_at).days / 90)
     90 天半衰期，适合学术研究的长周期

  ③ usage_score = min(use_count / 10, 1.0)
     常用记忆权重更高

  ④ 给 add_memories() 设置默认 expires_at = created_at + 365 天
     一年未使用的记忆自动过期
```

### 优化 5：跨 Track 批量搜索（修复 G5）`P2`

```
当前（N+1 查询）：
  for track in user_tracks[:50]:
      hits = search_memories(query, scope_type="track", scope_id=track.id)

优化后（单次查询）：
  hits = search_memories_batch(
      query=query,
      scope_type="track",
      scope_ids=[t.id for t in user_tracks],
  )
  # SQL: WHERE scope_type='track' AND scope_id IN (:ids)
  # 一次查询，结果按 scope_id 分组
```

### 优化 6：CodeMemory 持久化（修复 G3）`P1`

借鉴 OpenClaw 的 `memory/YYYY-MM-DD.md` 日志模式，但落到 DB：

```
新增表：repro_code_experience
  pack_id       → 关联 P2C pack
  paper_id      → 关联论文
  pattern_type  → success_pattern / failure_reason / verified_structure
  content       → 具体经验描述
  code_snippet  → 关键代码片段
  created_at
```

**写入时机**：
1. 代码生成成功 → 记录成功模式
2. 验证通过 → 记录验证结构
3. Debug 修复 → 记录失败原因和修复方法

**读取时机**：
4. 同一论文重新生成 → 优先加载已有经验
5. 类似论文生成 → 通过 `paper_type` + 关键词匹配相关经验

### 优化 7：Context 分层加载（借鉴 OpenClaw 的 routing index 模式）`P2`

OpenClaw 社区的最佳实践：MEMORY.md 保持 50 行路由索引，指向详情文件按需加载。PaperBot 也应该分层：

```
当前：build_context_pack() 一次性拉取所有记忆注入 context

优化后：
  Layer 0: Always-on（每次都注入，~200 tokens）
     → 用户 profile 摘要：名字 / 研究方向 / 活跃 track 列表

  Layer 1: Track-relevant（按当前 track 注入，~500 tokens）
     → 当前 track 的 goals / keywords / recent_decisions

  Layer 2: Query-relevant（按查询按需检索，~1000 tokens）
     → semantic search 命中的记忆

  Layer 3: On-demand（仅在展开详情时加载）
     → paper-scoped 记忆 / 完整 observation structured_data
```

这样 context 注入从 "全量平铺" 变为 "分层按需"，token 消耗从 `O(全部记忆)` 降为 `O(相关记忆)`。

---

## 优先级总览

| 优先级 | 优化项 | 修复的问题 | 与 P2C 的关系 |
|--------|--------|------------|---------------|
| **P0** | 打通推荐→执行桥 | G1 | P2C Module 1 ContextEngineBridge |
| **P0** | 激活 paper scope | G2 | P2C Observation 写入 paper scope |
| **P1** | 加 FTS5 语义搜索 | G4 | P2C Module 2 已设计 FTS5 |
| **P1** | CodeMemory 持久化 | G3 | P2C feedback 闭环 |
| **P2** | 记忆衰减 | G6 | 长期运行后的质量保障 |
| **P2** | 跨 Track 批量搜索 | G5 | 性能 |
| **P2** | Context 分层加载 | — | 性能 + token 成本 |

---

## Part 4: Agentic Research 演进与 OpenClaw 集成 (2026-03-02 更新)

> 本节为后续调研补充。完整方案见 [AGENTIC_RESEARCH_EVOLUTION.md](./AGENTIC_RESEARCH_EVOLUTION.md)。

### 关键发现

1. **记忆系统与 OpenClaw 集成**: 推荐 Hybrid 方案 — OpenClaw 管理用户级非结构化记忆 (MEMORY.md + FTS5 + sqlite-vec), PaperBot 保留领域级结构化数据 (paper-scoped, CodeMemory)。ContextEngineBridge 统一读取两者。

2. **Agentic Research 对记忆的新需求**:
   - **工作记忆 (Working Memory)**: 研究会话期间的临时发现 (如"论文 X 与论文 Y 矛盾") — 映射到 OpenClaw Layer 1 (Session History)
   - **研究图谱**: 论文间的结构化关系 (引用、方法对比、数据集共享) — 需在 PaperBot domain store 新增 knowledge_graph 表
   - **跨会话记忆**: 用户的研究偏好和历史发现在 Agent 循环中持续可用 — 映射到 OpenClaw Layer 3 (User Profile)

3. **BaseAgent 需升级为 ReAct 循环**: 当前 `ask_claude()` 是单轮调用, 无法支持 agentic research 的"搜索→评估→精炼→再搜索"模式。新增 `run_with_tools()` 方法后, 记忆系统需同步支持工具级别的 memory_store / memory_recall。

4. **优化项与 Agentic Research 关系更新**:

| 原优化项 | Agentic Research 新重要性 | 说明 |
|----------|---------------------------|------|
| G1 推荐→执行桥 | **升级为 P0+** | Agentic research loop 必须统一上下文 |
| G4 语义搜索 | **升级为 P0** | 迭代搜索精炼依赖语义检索, 关键词匹配不够 |
| G3 CodeMemory 持久化 | 保持 P1 | AI Scientist 模式需要实验经验持久化 |
| G6 记忆衰减 | 保持 P2 | 长期研究会话的质量保障 |
| 新增: 工作记忆 | **P0** | 研究循环中的临时发现必须可追溯 |
| 新增: 知识图谱 | **P1** | 引用图遍历和跨论文综合的基础 |
