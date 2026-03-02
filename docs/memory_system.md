# PaperBot 记忆系统

> 跨平台记忆中间件 + 完整架构设计提案。基于 Manus 上下文工程、EverMemOS/Mem0/Zep/Letta 等主流实现、以及近期顶会论文的综合调研。

---

## 第一部分：已实现能力 — 跨平台记忆中间件

目标：把来自不同大模型平台（ChatGPT / Gemini / Claude / OpenRouter 等）的聊天记录统一解析、提炼为"长期记忆"，并在后续对话里作为上下文注入，避免各平台记忆割裂。

### 核心概念

- **Source（来源）**：一次导入的聊天记录文件（会记录 `sha256`、平台、文件名、解析统计）。
- **Memory Item（记忆条目）**：从对话中提炼出的稳定信息，例如偏好、目标、约束、长期项目背景等。
- **Context（上下文块）**：把若干记忆条目格式化成可直接塞进 system prompt 的片段。

### 解析导入

- ChatGPT 导出（`conversations.json` 结构，尽量容忍分支/缺字段）
- Gemini / 各类 API 日志（宽松 JSON：支持多种常见结构）
- 纯文本（`User:`/`Assistant:` 或中文 `我:`/`助手:` 前缀的松散格式）
- 其它 JSON：尝试 `{"messages": [{"role": "...", "content": "..."}]}` 通用结构

### 记忆提炼

- **默认：启发式规则**（离线可用，不依赖 API）
- 可选：**LLM 抽取**（需配置 API Key，失败自动回退规则）

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/memory/ingest` | POST | 导入聊天记录并抽取记忆 |
| `/api/memory/context` | POST | 获取可注入的记忆上下文块 |
| `/api/memory/list` | GET | 列出记忆条目 |
| `/api/memory/items/{item_id}` | PATCH | 编辑记忆条目 |
| `/api/memory/items/{item_id}` | DELETE | 删除记忆条目 |
| `/api/memory/metrics` | GET | 记忆指标概览 |
| `/api/memory/metrics/{metric_name}` | GET | 指标历史 |

Chat API 中启用记忆：`POST /api/chat` body 增加 `user_id` + `use_memory: true`。

### 数据库

默认 SQLite：`sqlite:///data/paperbot.db`（可用 `PAPERBOT_DB_URL` 覆盖）。表：`memory_sources`、`memory_items`。

---

## 第二部分：调研综述

### 外部系统调研

| 系统 | 架构 | LoCoMo | 核心思想 |
|------|------|--------|---------|
| **EverMemOS** | 4 层仿脑架构 | 92.3% | 前额叶皮层+大脑皮层网络类比，当前 SOTA |
| **Zep/Graphiti** | 时序知识图谱（Neo4j） | 85.2% | 双时态模型，P95 延迟 300ms |
| **Letta** | 文件系统即记忆 | 74.0% | 迭代文件搜索优于专用记忆工具 |
| **Mem0** | 向量+图双存储 | 64.2% | 生产级 SaaS，自动记忆提取管线 |
| **memU** | 基于文件的 Agent 记忆 | 66.7% | 面向 24/7 主动式 Agent |

### Manus 上下文工程核心原则

1. **KV-Cache 命中率是第一指标** — 缓存 vs 非缓存 token 成本差 10x
2. **上下文即 RAM** — LLM 是 CPU，上下文窗口是 RAM
3. **Raw > Compaction > Summarization** — 可逆压缩优先
4. **文件系统是无限记忆** — 上下文只保留引用
5. **渐进式披露（Skills）** — 三级加载：元数据 → 指令 → 资源

### 关键论文

| 论文 | 会议 | 关键贡献 |
|------|------|---------|
| A-MEM | NeurIPS 2025 | Zettelkasten 式自组织互联笔记网络 |
| HiMem | arXiv 2026.01 | Episode + Note 两层层级 + 冲突感知重整合 |
| Agent Workflow Memory | ICML 2025 | 从历史轨迹归纳可复用工作流模板 |
| RMM (Reflective Memory) | ACL 2025 | 前瞻/回顾双向反思 + RL 精化检索 |
| Memoria | arXiv 2025.12 | SQL + KG + 向量三存储混合 |

---

## 第三部分：现状分析

### 现有架构

```
src/paperbot/memory/
├── schema.py           # NormalizedMessage, MemoryCandidate, MemoryKind (11种)
├── extractor.py        # 双策略提取：LLM + 启发式 (中文正则)
├── eval/collector.py   # 5 个 P0 指标
└── parsers/            # 多格式聊天记录解析

src/paperbot/context_engine/
├── engine.py           # ContextEngine — build_context_pack()
├── track_router.py     # TrackRouter — 多特征 track 评分
└── embeddings.py       # EmbeddingProvider (OpenAI)

src/paperbot/infrastructure/stores/
├── memory_store.py     # SqlAlchemyMemoryStore (CRUD + 粗粒度搜索)
└── models.py           # MemoryItemModel, MemorySourceModel, MemoryAuditLogModel
```

### 现有问题

| 问题 | 严重度 | 说明 |
|------|--------|------|
| **无向量检索** | 高 | `search_memories()` 使用 SQL CONTAINS + 内存 token 评分，无语义匹配 |
| **无时间感知** | 高 | 记忆无衰减机制，无时序推理能力 |
| **无记忆整合** | 中 | 记忆只有 CRUD，无 consolidation/forgetting |
| **层级耦合** | 中 | ContextEngine 直接依赖 SqlAlchemyMemoryStore |
| **无跨记忆关联** | 中 | 记忆项之间无链接关系 |

### 现有优势（可复用）

- 完整的 schema 设计（MemoryKind 11 种、scope、confidence、status lifecycle）
- 审计日志（全量变更记录）
- PII 检测与脱敏
- 基于 confidence 的自动审核
- 使用量追踪（last_used_at, use_count）
- 评估指标框架

---

## 第四部分：架构设计

### 设计原则

1. **记忆即基础设施** — 独立 infra 层服务，不依赖业务模块
2. **混合存储** — 向量（语义）+ 结构化（关系/时间）+ 文件（全文）
3. **层级记忆** — Episode Memory（具体事件）+ Note Memory（抽象知识）
4. **时间感知** — 双时态模型（事件时间 + 录入时间）
5. **渐进式上下文** — 三级加载控制 token 消耗
6. **自组织链接** — Zettelkasten 式记忆项双向关联

### 分层架构

```
┌── Application Layer（业务消费者） ────────────────────────┐
│  DailyPaper · Judge · TopicSearch · ScholarPipeline      │
│        ↓                                                  │
│  Context Assembly Service（上下文装配）                     │
└──────────────────────────┬────────────────────────────────┘
                           │ MemoryService Protocol
┌──────────────────────────│────────────────────────────────┐
│  Memory Infrastructure Layer                              │
│                          │                                │
│  MemoryService (Facade)                                   │
│  - write / recall / forget / consolidate / link           │
│           │            │              │                    │
│     Extractor     Retriever     Consolidator              │
│      (Write)       (Read)       (Maintain)                │
│           │            │              │                    │
│  ┌── Storage Backends ────────────────────────────────┐   │
│  │  SQLite (结构化) · Vector (语义) · File (全文)      │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

### 记忆类型体系

```
Memory Types
├── Episode Memory（具体事件）
│   ├── paper_read / search_query / feedback
│   ├── workflow_run / interaction
│
└── Note Memory（抽象知识）
    ├── profile / preference / interest / fact
    ├── goal / constraint / decision / insight
```

Episode → Note 整合规则：多次同领域 paper_read → interest Note；多次 like 同 venue → preference Note 等。

### 存储层设计

新增 SQLite 表：`memory_links`（记忆关联）、`memory_embeddings`（记忆向量）。扩展 `memory_items` 增加 `memory_layer`、`event_at`、`embedding_id`、`decay_factor` 字段。

向量检索策略：Phase 1 SQLite + numpy cosine（<5000 条延迟 <50ms），Phase 2 sqlite-vec，Phase 3 可选 Qdrant/FAISS。

### 检索管线（Hybrid Recall）

```
Query → ┌── 向量检索（语义匹配） ─── weight: 0.50
        ├── 关键词匹配（BM25/token）── weight: 0.25
        └── scope/tag 精确过滤    ── weight: 0.25
                    │
             Merge & Re-rank → Time Decay → Token Budget Trim
```

### 渐进式上下文管理（三级加载）

| 级别 | 何时加载 | 内容 | Token 消耗 |
|------|---------|------|-----------|
| **L0: 元数据** | 每次 LLM 调用 | 用户名 + track 名 | ~50 tokens |
| **L1: 画像** | task 开始时 | profile + preferences | ~300 tokens |
| **L2: 任务记忆** | query 确定后 | recall(query) top-k | ~1200 tokens |
| **L3: 深度上下文** | 仅在需要时 | insights + linked items | 按需分配 |

---

## 第五部分：迁移计划

### Phase 0: 接口定义 + 向量化（无破坏性变更）

- [ ] `memory/protocol.py` — MemoryService Protocol
- [ ] `memory/retriever.py` — 向量检索 + 混合检索
- [ ] `memory_embeddings` + `memory_links` 表迁移
- [ ] 扩展 `MemoryItemModel` 新字段
- [ ] 在 `add_memories()` 中异步计算 embedding
- [ ] 在 `search_memories()` 中加入向量检索分支

### Phase 1: 分离 Facade + Consolidator

- [ ] `memory/service.py` — MemoryServiceImpl (Facade)
- [ ] `memory/consolidator.py` — 记忆整合器
- [ ] Episode/Note 双层记忆类型支持
- [ ] `recall()` 混合检索管线
- [ ] `link()` 记忆关联
- [ ] 迁移 `ContextEngine` 记忆逻辑到 `ContextAssemblyService`

### Phase 2: 业务集成 + 自动记忆生成

- [ ] DailyPaper 自动写入 Episode
- [ ] Judge 高分论文洞察写入 Note
- [ ] feedback 路由写入 Episode
- [ ] Judge prompt 注入用户画像和研究偏好
- [ ] Track Router 使用向量化记忆
- [ ] Consolidator 注册到 ARQ Worker

### Phase 3: 高级功能

- [ ] 时间衰减调度 / 冲突检测 / 自动链接发现
- [ ] 记忆导出/快照
- [ ] 可选升级到 sqlite-vec 或 Qdrant

---

## 文件清单

| 文件 | 类型 | Phase | 说明 |
|------|------|-------|------|
| `src/paperbot/memory/protocol.py` | 新建 | 0 | MemoryService Protocol |
| `src/paperbot/memory/retriever.py` | 新建 | 0 | 向量检索 + 混合检索 |
| `src/paperbot/memory/service.py` | 新建 | 1 | MemoryServiceImpl (Facade) |
| `src/paperbot/memory/consolidator.py` | 新建 | 1 | 记忆整合器 |
| `src/paperbot/memory/types.py` | 新建 | 0 | 数据类定义 |
| `src/paperbot/infrastructure/stores/models.py` | 修改 | 0 | 扩展模型 |
| `src/paperbot/infrastructure/stores/memory_store.py` | 修改 | 0-1 | 向量检索/链接 CRUD |
| `src/paperbot/application/services/context_assembly.py` | 新建 | 1 | 上下文装配服务 |

---

## 参考文献

### 系统与框架

- [Manus Context Engineering](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [EverMemOS](https://github.com/EverMind-AI/EverMemOS) — 92.3% LoCoMo
- [Zep/Graphiti](https://github.com/getzep/graphiti) — 时序知识图谱
- [Mem0](https://github.com/mem0ai/mem0) — 生产级记忆层
- [Letta](https://www.letta.com/blog/benchmarking-ai-agent-memory) — 文件系统即记忆

### 学术论文

1. A-MEM: Agentic Memory for LLM Agents — NeurIPS 2025
2. HiMem: Hierarchical Long-Term Memory — arXiv 2026
3. Agent Workflow Memory — ICML 2025
4. RMM: Reflective Memory Management — ACL 2025
5. Memoria: Scalable Agentic Memory — arXiv 2025
6. TiMem: Temporal-Hierarchical Memory — arXiv 2026
7. Collaborative Memory — ICML 2025
8. Survey of Context Engineering — arXiv 2025

### Benchmarks

- [LoCoMo](https://snap-research.github.io/locomo/) — 300-turn 长对话记忆评估
- [LongMemEval](https://arxiv.org/abs/2410.10813) — 500 问题，5 核心记忆能力 (ICLR 2025)
