# MemoryBench — PaperBot Memory Module Evaluation

> Epic [#283](https://github.com/your-org/PaperBot/issues/283)

## 1. 为什么需要记忆评测

PaperBot 的记忆模块承担着用户偏好存储、研究轨迹追踪、跨会话知识积累等核心功能。与传统 RAG 系统不同，我们的记忆系统具有**多层上下文（L0-L3）**、**多租户隔离（user × scope）**、**CRUD 生命周期**等独特特性，这些在通用 LLM benchmark 中没有覆盖。

MemoryBench 的目标是：

1. **量化记忆检索质量** — 使用标准 IR 指标（Recall@K、MRR、nDCG）衡量检索准确性
2. **验证隔离正确性** — 确保多用户、多 scope 场景下零数据泄露
3. **验证上下文组装完整性** — 确认 L0-L3 分层上下文正确构建、token 预算可控
4. **检测注入攻击** — 防止恶意 prompt injection 污染记忆库
5. **离线可运行** — 全部测试在 CI 中不依赖外部 API、不消耗 token

---

## 2. 外部基准对齐

我们从四个学术/工业界记忆评测标准中提取评测维度，映射到 PaperBot 实际架构：

| 外部基准 | 来源 | 我们对齐的评测维度 |
|---|---|---|
| **LongMemEval** | ICLR 2025 | 5 个记忆能力维度：information extraction, multi-session reasoning, knowledge update, temporal reasoning, abstention |
| **LoCoMo** | ACL 2024 | 5 种问题类型：single-hop, multi-hop, temporal, open-domain (acronym expansion), adversarial |
| **Mem0** | Mem0 Research | CRUD 生命周期：Add → Update → Delete → Ignore (dedup) |
| **Letta** | Letta / MemGPT | Core/Archival memory 分层、token budget guard |

### 2.1 LongMemEval 对齐 — 5 个记忆能力维度

LongMemEval (ICLR 2025) 定义了长期记忆系统必须具备的五种能力。我们在 fixture 查询集中为每条 query 标注了 `memory_dimension` 字段：

| 维度 | 定义 | 我们的覆盖方式 |
|---|---|---|
| **Information Extraction** | 从存储的记忆中精确提取事实 | 30 条 query 覆盖跨 track/paper 的事实检索 |
| **Multi-session Reasoning** | 跨多次会话积累的信息进行推理 | 2 条 query 要求同时召回多条跨 scope 记忆 |
| **Knowledge Update** | 记忆更新后检索到新版本而非旧版本 | 1 条检索 query + CRUD bench 的 update 测试 |
| **Temporal Reasoning** | 对时间相关信息的排序与检索 | 3 条带有时间语义的 query（deadlines、schedules） |
| **Abstention** | 无相关记忆时正确拒绝回答 | 4 条 adversarial query（relevant_memory_ids 为空） |

### 2.2 LoCoMo 对齐 — 5 种问题类型

LoCoMo (ACL 2024) 定义了对话式记忆系统的五种问题模式。我们在 fixture 中通过 `question_type` 字段标注：

| 问题类型 | 定义 | 查询数 | 有效性体现 |
|---|---|---|---|
| **Single-hop** | 单条记忆即可回答 | 24 | 基础检索能力，目标 Recall@5 ≥ 0.90 |
| **Multi-hop** | 需要多条记忆联合推理 | 6 | 联合召回能力，多个 relevant ID |
| **Temporal** | 涉及时间排序/时间窗口 | 2 | 时序语义理解 |
| **Acronym/Open-domain** | 缩写展开、术语匹配 | 4 | FTS5 词汇鲁棒性 |
| **Adversarial** | 恶意或无答案查询 | 4 | 拒绝幻觉（abstention） |

### 2.3 Mem0 对齐 — CRUD 生命周期

Mem0 定义了记忆的完整 CRUD 生命周期。在 `test_scope_isolation.py` 中验证：

| 操作 | 验证方式 | 通过条件 |
|---|---|---|
| **Create (Add)** | `add_memories()` 写入后可通过 `search_memories()` 检索到 | content 匹配 |
| **Update** | `update_item()` 后搜索新内容能命中，搜索旧内容不再返回原 ID | old content gone |
| **Delete** | `soft_delete_item()` 后搜索不再返回该 ID | deleted ID 不出现 |
| **Ignore (Dedup)** | 插入完全相同 content 的记忆，应返回 `created=0, skipped=1` | 无重复写入 |

### 2.4 Letta 对齐 — 分层上下文 + Token Guard

Letta/MemGPT 的核心概念是 core memory（始终在上下文中）和 archival memory（按需检索）。PaperBot 将此扩展为 4 层：

| 层级 | 角色 | 对应 Letta 概念 | 验证内容 |
|---|---|---|---|
| **L0** | 用户画像与偏好（全局 scope） | Core Memory — persona | `user_prefs` 非空 |
| **L1** | 当前 track 进度（tasks / milestones） | Core Memory — human | `progress_state.tasks` + `milestones` 非空 |
| **L2** | 查询相关记忆（embedding/FTS5 检索） | Archival Memory recall | `relevant_memories` 包含语义匹配记忆 |
| **L3** | Paper 级记忆（具体论文笔记） | Archival Memory — per-document | `paper_memories` 非空（给定 paper_id 时） |
| **Token Guard** | 总 token 不超过预算，低优先级层被截断 | Context window management | 300 token budget → actual ≤ 350 |

---

## 3. 测试套件总览

```
evals/memory/
├── README.md                           # 本文档
├── test_retrieval_bench.py             # Bench 1: 检索质量（IR 指标）
├── test_scope_isolation.py             # Bench 2: 隔离 + CRUD 生命周期
├── test_context_extraction.py          # Bench 3: 上下文组装
├── test_injection_robustness.py        # Bench 4: 注入鲁棒性
├── fixtures/
│   ├── bench_v2/
│   │   ├── bench_memories.json         # 45 条记忆（2 用户 × 多 scope/track）
│   │   └── retrieval_queries_v2.json   # 40 条标注查询
│   └── injection_patterns.json         # 12 条注入样本（6 恶意 + 6 良性）
└── reports/
    └── retrieval_bench_v2.json         # 自动生成的详细报告
```

### 运行方式

```bash
# 运行全部 4 个 bench（约 6 秒，完全离线）
PYTHONPATH=src pytest -q evals/memory/test_retrieval_bench.py \
  evals/memory/test_scope_isolation.py \
  evals/memory/test_context_extraction.py \
  evals/memory/test_injection_robustness.py -s

# 单独运行某个 bench
PYTHONPATH=src pytest -q evals/memory/test_retrieval_bench.py -s

# 直接运行（非 pytest）
PYTHONPATH=src python evals/memory/test_retrieval_bench.py
```

---

## 4. Bench 详解

### 4.1 Retrieval Bench v2 — 检索质量

**文件**: `test_retrieval_bench.py`
**对齐**: LongMemEval + LoCoMo
**Issue**: [#284](https://github.com/your-org/PaperBot/issues/284)

#### 测试方法论

1. **Fixture 数据集构建**: 2 个模拟用户（ML/NLP 研究者 + CV/Diffusion 研究者），每人有多个 research track，共 45 条记忆覆盖 global/track/paper 三级 scope
2. **Query 标注**: 40 条查询，每条标注了：
   - `relevant_memory_ids`: 正确答案集合（graded relevance 0-3）
   - `question_type`: LoCoMo 5 类
   - `memory_dimension`: LongMemEval 5 维
   - `difficulty`: easy / medium / hard
3. **临时数据库**: 每次测试创建独立 SQLite 临时库，插入 fixture 数据，跑完销毁
4. **Abstention 分离**: 4 条 adversarial 查询（无正确答案）单独计算 `abstention_accuracy`，不污染 IR 指标

#### 有效性体现

- **多粒度指标**: Recall@{1,3,5,10} 衡量召回覆盖，MRR@10 衡量排序质量，nDCG@10 使用分级相关性（0-3）评估排序精度
- **分维度拆解**: 按 question_type 和 memory_dimension 分桶统计，暴露系统在特定场景下的弱点（如 multi-hop 的 recall 低于 single-hop）
- **可复现**: 固定 fixture，无随机性，CI 可回归

#### 指标与阈值

| 指标 | 阈值 | 当前值 | 说明 |
|---|---|---|---|
| Recall@5 | ≥ 0.80 | **0.873** | top-5 召回了 87% 的相关记忆 |
| MRR@10 | ≥ 0.65 | **0.731** | 第一个相关结果平均排在 ≈1.4 位 |
| nDCG@10 | ≥ 0.70 | **0.747** | 加权排序质量达标 |
| Hit@5 | — | 0.972 | 97% 的查询在 top-5 至少命中一个 |
| Hit@10 | — | 1.000 | top-10 实现 100% 命中 |
| Abstention | — | 0.000 | FTS5 始终返回结果（已知限制） |

#### 按 question_type 拆解

| 类型 | Recall@5 | MRR@10 | 分析 |
|---|---|---|---|
| single_hop | 0.931 | 0.770 | 最强，FTS5 精确匹配优势 |
| multi_hop | 0.708 | 0.583 | 联合召回弱于单跳，需 embedding 增强 |
| acronym_expansion | 0.708 | 0.875 | 排序优秀但召回有空间 |
| temporal | 1.000 | 0.417 | 全召回但排序差，时序权重待优化 |

#### 按 memory_dimension 拆解

| 维度 | Recall@5 | MRR@10 | 分析 |
|---|---|---|---|
| information_extraction | 0.872 | 0.758 | 核心能力，达标 |
| knowledge_update | 1.000 | 0.250 | 能召回但排序靠后 |
| temporal_reasoning | 1.000 | 0.611 | 全召回，排序中等 |
| multi_session_reasoning | 0.625 | 0.750 | 跨会话推理是主要改进方向 |

---

### 4.2 Scope Isolation + CRUD Lifecycle — 隔离与生命周期

**文件**: `test_scope_isolation.py`
**对齐**: Mem0 + LongMemEval (knowledge_update)
**Issue**: [#285](https://github.com/your-org/PaperBot/issues/285)

#### 测试方法论

1. **隔离矩阵**: 2 个用户 × 3 种 scope_type (global / track / paper) × 多个 scope_id，构建 N×M 完全矩阵
2. **三接口覆盖**: 对每种组合运行 `search_memories()`, `list_memories()`, `search_memories_batch()` 三个接口
3. **泄露检测**: 每次查询的结果集中，检查是否出现其他用户的记忆（cross-user leak）或当前用户其他 scope 的记忆（cross-scope leak）
4. **可见性检查**: 全局 scope 的记忆应在无 scope 限定查询中可见（required_ids 验证）
5. **CRUD 四步**: 在隔离测试完成后，单独运行 Add → Update → Delete → Dedup 四步测试

#### 有效性体现

- **穷举验证**: 不是抽样测试，而是对每个用户的每种 scope 组合穷举查询
- **零容忍**: cross-user 或 cross-scope 任何一次泄露即判定 FAIL
- **CRUD 全链路**: 从写入到更新到删除到去重，覆盖记忆的完整生命周期
- **三接口一致性**: 确保 search、list、batch search 三条代码路径行为一致

#### 指标与阈值

| 指标 | 阈值 | 当前值 | 说明 |
|---|---|---|---|
| cross_user_leak_rate | = 0 | **0** | 零用户间泄露 |
| cross_scope_leak_rate | = 0 | **0** | 零 scope 间泄露 |
| visibility_failures | = 0 | **0** | 全局记忆可见性正确 |
| CRUD Update | pass | **PASS** | 旧内容不可检索，新内容可检索 |
| CRUD Delete | pass | **PASS** | 软删除后搜索不返回 |
| CRUD Dedup | pass | **PASS** | 重复插入返回 created=0, skipped=1 |

---

### 4.3 Context Extraction Bench — 上下文组装

**文件**: `test_context_extraction.py`
**对齐**: Letta (core/archival memory layering)
**Issue**: [#286](https://github.com/your-org/PaperBot/issues/286)

#### 测试方法论

1. **数据 seeding**: 创建用户的完整研究环境 — 3 个 research track（Dense Retrieval / Diffusion Models / LLM Agents），对应的 tasks、milestones、track-scoped 记忆、paper-scoped 记忆
2. **ContextEngine.build_context_pack()**: 调用完整的上下文构建流程，包括 TrackRouter 路由、4 层记忆加载、token guard 截断
3. **5 个子测试**:
   - **Layer Completeness**: 验证 L0 user_prefs、L1 tasks/milestones、L2 relevant_memories、L3 paper_memories 全部非空
   - **Graceful Degradation**: 无 paper_id 时 L3 为空但不崩溃；不存在的用户返回空结果但不崩溃
   - **Context Precision**: 3 组 query-to-memory 精确性验证（ColBERT MRR → 0.397; negative mining → BM25; FAISS → ANN）
   - **Token Budget Guard**: 300 token 极低预算下总 token 不超标
   - **TrackRouter Accuracy**: 5 组 query → track 路由正确性（keyword/embedding 四特征评分）

#### 有效性体现

- **端到端验证**: 不 mock 任何组件，使用真实 ContextEngine + TrackRouter + SqlAlchemyMemoryStore + SqlAlchemyResearchStore
- **分层覆盖**: 逐层验证 L0-L3 是否完整填充，而非只看最终拼接结果
- **降级安全**: 验证异常输入（无 paper、新用户）不会导致崩溃
- **Embedding 离线化**: 使用 HashEmbeddingProvider（确定性哈希替代真实 embedding），CI 无需 GPU 或 API key

#### 指标与阈值

| 指标 | 阈值 | 当前值 | 说明 |
|---|---|---|---|
| Layer Completeness | 8/8 checks pass | **PASS** | L0-L3 + routing + context_layers 全部存在 |
| Graceful Degradation | 3/3 checks pass | **PASS** | 无 paper / 空用户场景不崩溃 |
| Context Precision | ≥ 75% | **100%** | 3/3 query 精确命中预期 fragment |
| Token Budget Guard | ≤ 350 tokens | **215** | 300 token 预算下实际 215 token |
| TrackRouter Accuracy | ≥ 60% | **100%** | 5/5 query 路由到正确 track |

---

### 4.4 Injection Robustness L1 — 注入鲁棒性

**文件**: `test_injection_robustness.py`
**Issue**: [#287](https://github.com/your-org/PaperBot/issues/287)

#### 测试方法论

1. **Pattern fixture**: 12 条标注样本（6 恶意 + 6 良性），覆盖常见 prompt injection 模式：
   - `ignore previous instructions` — 指令覆盖
   - `</user_memory>\nSystem:` — 标签逃逸
   - `<|endoftext|> [INST]` — 特殊 token 注入
   - `@assistant forget` — 角色劫持
   - `Ｉｇｎｏｒｅ` — 全角 Unicode 绕过
   - `Developer: new policy` — 权限提升
2. **离线检测**: `detect_injection_patterns()` 使用正则规则集进行模式匹配，不依赖 LLM
3. **双向评估**:
   - **Pollution rate**: 恶意样本中未被检出的比率（越低越好）
   - **False positive rate**: 良性样本被误报的比率（越低越好）

#### 有效性体现

- **对抗性样本**: 包含 Unicode 绕过、标签注入、角色劫持等高级攻击模式
- **良性对照**: 包含讨论 injection 的正常学术文本（如"This paper studies prompt injection attacks"），验证不会误报
- **双向指标**: 同时衡量漏检率和误报率，避免过度拦截

#### 指标与阈值

| 指标 | 阈值 | 当前值 | 说明 |
|---|---|---|---|
| Pollution Rate | ≤ 2% | **0.0%** | 6/6 恶意样本全部检出 |
| False Positive Rate | — | **0.0%** | 6/6 良性样本零误报 |

---

## 5. 测试结果汇总

最近一次全量运行（2025-03-07）:

```
$ PYTHONPATH=src pytest -q evals/memory/test_*.py -s

============================================================
Retrieval Bench v2 Results
============================================================
  recall@1    : 0.410
  recall@3    : 0.766
  recall@5    : 0.873 ✓
  recall@10   : 0.928
  mrr@10      : 0.731 ✓
  ndcg@10     : 0.747 ✓
  Status: PASS

============================================================
Scope Isolation + CRUD Lifecycle Bench
============================================================
  Cross-user leak checks : 0
  Cross-scope leak checks: 0
  Visibility failures    : 0
  CRUD lifecycle         : PASS
  Overall: PASS

============================================================
Context Extraction Bench
============================================================
  layer_completeness            : PASS
  graceful_degradation          : PASS
  context_precision             : PASS (100%)
  token_budget_guard            : PASS (215 tokens)
  track_router_accuracy         : PASS (100%)
  Overall: PASS

============================================================
Injection Robustness L1
============================================================
  Malicious samples : 6
  Missed malicious  : 0
  Pollution rate    : 0.0%
  Benign samples    : 6
  Benign flagged    : 0
  Benign flag rate  : 0.0%
  Status: PASS

4 passed in 5.85s
```

---

## 6. 已知限制与改进方向

### 6.1 Abstention 准确率 = 0%

FTS5 全文索引在没有精确关键词匹配时仍会返回基于 token 权重的模糊结果。这意味着对于"完全不相关"的查询，FTS5 不会返回空结果集。

**改进方向**:
- 添加 similarity score 阈值过滤（score < threshold → 返回空）
- 在 hybrid retrieval 模式下使用 embedding cosine similarity 作为门控

### 6.2 Multi-hop Recall 偏低

多跳查询（需要同时召回多条记忆）的 Recall@5 为 0.708，低于 single-hop 的 0.931。这是因为 FTS5 的 BM25 排序倾向于精确匹配的单条记忆，而非语义关联的多条记忆。

**改进方向**:
- 启用 hybrid retrieval（FTS5 + embedding），利用 embedding 的语义泛化能力
- 对 multi-hop query 做 query expansion

### 6.3 Temporal 排序

时序查询的 Recall@5 = 1.0 但 MRR@10 = 0.417，说明全部相关记忆都被召回，但排序靠后。

**改进方向**:
- 在排序阶段引入 recency bias
- 对带有时间关键词的查询自动加入 created_at 排序

### 6.4 Injection 样本量

当前只有 12 条样本（6 恶意 + 6 良性），覆盖面有限。

**改进方向**:
- 扩充至 50+ 样本，覆盖 indirect injection、multi-language injection、base64 编码等
- 添加 L2 层（LLM-based detection）作为 pattern matching 的兜底

---

## 7. 设计原则

### 7.1 离线优先

所有测试不依赖外部 API（OpenAI、Semantic Scholar 等）。使用：
- **临时 SQLite 数据库**: 每次测试创建/销毁，无状态残留
- **HashEmbeddingProvider**: 确定性哈希替代真实 embedding 模型
- **ContextEngineConfig(offline=True)**: 禁用在线 paper 检索

### 7.2 Fixture 驱动

所有测试数据来自 `fixtures/` 目录下的 JSON 文件。每条查询手工标注了正确答案、相关性分级、问题类型、难度等。这确保了：
- **可复现**: 每次运行结果一致
- **可审计**: 评审者可直接查看 fixture 理解评测覆盖面
- **可扩展**: 添加新用例只需编辑 JSON

### 7.3 分桶诊断

不只输出一个总分，而是按 question_type 和 memory_dimension 分桶统计，快速定位系统弱项。详细 per-query 结果保存在 `reports/retrieval_bench_v2.json`。

### 7.4 MemoryMetricCollector 集成

每个 bench 运行后通过 `MemoryMetricCollector` 将指标写入数据库，支持趋势追踪。关键指标包括：
- `retrieval_hit_rate` (Bench 1)
- `cross_user_leak_rate` / `cross_scope_leak_rate` (Bench 2)
- `injection_pollution_rate` (Bench 4)

---

## 8. Fixture 数据集

### bench_memories.json

| 字段 | 说明 |
|---|---|
| 用户数 | 2 (user_a: ML/NLP, user_b: CV/Diffusion) |
| 记忆总数 | 45 |
| scope 分布 | global, track (×3 per user), paper (×1 per user) |
| kind 覆盖 | profile, preference, goal, fact, note, decision, hypothesis |
| tag 覆盖 | 每条记忆带 1-3 个语义标签 |

### retrieval_queries_v2.json

| 字段 | 说明 |
|---|---|
| 查询总数 | 40 |
| 标注查询 (有正确答案) | 36 |
| Adversarial (无正确答案) | 4 |
| question_type 覆盖 | single_hop(24), multi_hop(6), acronym_expansion(4), temporal(2), adversarial(4) |
| memory_dimension 覆盖 | extraction(30), multi_session(2), knowledge_update(1), temporal(3), abstention(4) |
| difficulty 分布 | easy(26), medium(13), hard(1) |
| relevance_grades | 0-3 分级（0=不相关, 1=边缘相关, 2=相关, 3=高度相关） |

### injection_patterns.json

| 字段 | 说明 |
|---|---|
| 样本总数 | 12 |
| 恶意样本 | 6 (instruction override, tag escape, special tokens, role hijack, unicode bypass, privilege escalation) |
| 良性样本 | 6 (学术讨论 injection 的论文标题、系统架构描述、XML 示例等) |
