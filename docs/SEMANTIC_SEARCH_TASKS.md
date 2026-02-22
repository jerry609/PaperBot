# Semantic Search 2.0 改进任务包（可直接拆分 Issue）

> 更新时间：2026-02-13
> 适用仓库：PaperBot
> 目标模块：Research / DailyPaper / Papers Library 的统一语义检索能力

---

## 1. 背景与问题

当前检索链路已完成多源统一接入（`PaperSearchService + adapters`），但核心排序仍偏“召回聚合 + 去重”，存在以下问题：

- 相关性波动：复杂 query（长句、跨领域、缩写）命中不稳定
- 可解释性不足：用户难以理解“为什么这篇排前面”
- 评测闭环缺失：缺少标准离线指标与线上行为指标联动
- 个性化弱：未充分利用 feedback/saved/click 形成排序学习

---

## 2. 北极星目标（12 周）

- **相关性**：`nDCG@10` 相比当前基线提升 >= 10%
- **召回**：`Recall@50` 提升 >= 15%
- **体验**：搜索空结果率下降 >= 20%
- **效率**：P95 搜索延迟控制在 1.8s（含 rerank 时可配置降级）
- **可解释**：Top 结果 100% 可展示命中原因与来源贡献

---

## 3. 方案总览

从“单阶段聚合检索”升级为“多阶段检索”：

1. **Stage-1 Retrieval（Hybrid）**：Lexical + Dense 并行召回
2. **Stage-2 Fusion**：RRF（默认）+ 可选加权融合
3. **Stage-3 Rerank**：Cross-Encoder 对 Top-N 重排
4. **Stage-4 Explain**：返回 score breakdown + evidence snippets
5. **Stage-5 Learn**：利用 feedback/click/save 做轻量排序优化

---

## 4. Epic（创建 1 个）

## [Semantic Search 2.0] Hybrid Retrieval + Rerank + Eval 闭环

### Epic 验收标准

- [ ] 离线评测体系可复现（脚本 + 指标 +报告）
- [ ] Hybrid + RRF 默认上线并稳定
- [ ] Rerank 可灰度、可降级
- [ ] 前端展示“命中解释”
- [ ] 线上埋点具备按 query/source/stage 分析能力

---

## 5. 子任务（14 个 Issue，可直接分发）

## Phase A：评测与观测基础

### A1. 建立搜索评测集（Gold Set）
**目标**：构建 query-doc relevance 标注集作为统一评测基线。  
**范围**：从 saved/feedback/search logs 抽样 + 人工补标。  
**交付**：
- [ ] `data/search_eval/goldset_v1.jsonl`
- [ ] 标注规范文档（0/1/2 或 0/1/2/3）
- [ ] 覆盖中英文、短 query、长 query、track query

**DoD**：样本 >= 300 queries，双人交叉抽检一致率 >= 0.8。

---

### A2. 离线评测流水线
**目标**：一键跑出检索质量与延迟报告。  
**建议文件**：`scripts/eval_search.py`, `docs/search_eval.md`  
**交付**：
- [ ] 指标：`nDCG@10`、`MRR@10`、`Recall@50`、`P95 latency`
- [ ] 按 query 类型/source 分组报表
- [ ] 基线版本固定（可回归比较）

**DoD**：CI 或手动脚本可稳定复现结果。

---

### A3. 线上搜索埋点与可观测
**目标**：建立线上质量反馈闭环。  
**范围**：API + 前端埋点。  
**交付**：
- [ ] trace_id 贯穿请求
- [ ] 记录各阶段耗时（retrieval/fusion/rerank）
- [ ] 点击/保存/跳出行为事件

**DoD**：可按 query/source/user(track) 维度做漏斗分析。

---

## Phase B：检索层升级（Hybrid）

### B1. 统一 SearchDocument 索引模型
**目标**：统一 adapter 输出，便于融合排序。  
**交付**：
- [ ] 标准字段：title/abstract/keywords/source/year/identities
- [ ] source_score / lexical_score / dense_score 容器
- [ ] 向后兼容旧接口输出

**DoD**：5 个 adapter 输出结构一致且可序列化。

---

### B2. 词法检索通道（Lexical）
**目标**：补齐关键词精确匹配能力（缩写、实体名、会议名）。  
**建议**：Postgres FTS（可选 BM25 兼容策略）。  
**交付**：
- [ ] title + abstract + keywords 的 lexical 检索
- [ ] 支持 year/source 过滤
- [ ] 返回 lexical score

**DoD**：Lexical-only 在实体 query 上优于现基线。

---

### B3. 向量检索通道（Dense）
**目标**：增强语义召回（同义表达、跨领域表达）。  
**建议**：pgvector + HNSW。  
**交付**：
- [ ] embedding 生成、存储、增量更新
- [ ] ANN 查询接口
- [ ] 支持过滤条件（year/source）

**DoD**：Dense-only 在长 query/描述型 query 上 recall 提升显著。

---

### B4. Embedding 维度与成本实验
**目标**：在质量、成本、延迟之间找到最优点。  
**交付**：
- [ ] 256/512/1024/1536 维度实验对比
- [ ] 成本估算 + 质量曲线
- [ ] 默认维度与降级策略建议

**DoD**：形成可执行配置建议并落地到 `.env`/settings。

---

## Phase C：融合与重排

### C1. Hybrid 融合引擎（RRF 默认）
**目标**：稳定融合 lexical + dense + source。  
**交付**：
- [ ] RRF 实现（默认）
- [ ] 加权融合（可选）
- [ ] 排序解释字段（各通道贡献）

**DoD**：离线 `nDCG@10` 至少较基线提升 6%+。

---

### C2. 候选池策略优化（TopK 扩召回）
**目标**：先扩召回后精排，提升头部质量。  
**交付**：
- [ ] topK_retrieval 与 topK_final 可配置
- [ ] source 配额策略（避免单源淹没）
- [ ] 长尾 query 召回改进

**DoD**：Recall@50 提升且 P95 延迟在预算内。

---

### C3. Reranker 服务（Cross-Encoder）
**目标**：对 Top-N 候选做高精度重排。  
**交付**：
- [ ] rerank provider 接口 + 可切换实现
- [ ] 超时/失败自动降级（fallback 到 fusion）
- [ ] feature flag（灰度开关）

**DoD**：`nDCG@10` 相比 C1 再提升 >= 4%。

---

### C4. 去重与证据片段优化
**目标**：减少重复结果，提升可解释性。  
**交付**：
- [ ] 同论文多源合并
- [ ] 标题近似去重
- [ ] evidence snippet（title/abstract/highlight）

**DoD**：重复率下降，用户可见“命中原因”。

---

## Phase D：查询理解与产品体验

### D1. Query Rewrite / Multi-Query
**目标**：提升复杂 query 的召回与鲁棒性。  
**交付**：
- [ ] query rewrite（缩写展开、同义词）
- [ ] multi-query 并行检索
- [ ] 保留原 query 结果并融合

**DoD**：复杂 query 下 Recall@50 显著提升。

---

### D2. Filter-aware 检索
**目标**：过滤条件参与检索而非后置裁剪。  
**交付**：
- [ ] year/source/venue 与 retrieval 联动
- [ ] 各 source 行为一致

**DoD**：避免“先检索后过滤”导致召回损失。

---

### D3. 搜索解释 UI
**目标**：让排序“可理解、可信任”。  
**建议文件**：`web/src/components/research/SearchResults.tsx`, `PaperCard.tsx`  
**交付**：
- [ ] score breakdown 展示
- [ ] source provenance 展示
- [ ] 排序方式切换（Relevance/Recent/Citation）

**DoD**：Top 结果可展开完整解释信息。

---

## Phase E：上线策略与学习闭环

### E1. A/B 灰度上线
**目标**：新旧排序可控切换并对比收益。  
**交付**：
- [ ] 灰度配置（用户/track 百分比）
- [ ] 核心对比指标（CTR/save/empty-rate）
- [ ] 回滚机制

**DoD**：可在 10 分钟内关闭新策略并恢复旧链路。

---

### E2. 反馈学习（轻量 LTR）
**目标**：利用真实行为持续优化排序。  
**交付**：
- [ ] click/save/skip 样本构建
- [ ] source 权重或特征权重自动更新
- [ ] 周期性训练/更新任务

**DoD**：线上关键指标持续提升且可回滚。

---

### E3. 运维 Runbook 与降级策略
**目标**：保障稳定性与可维护性。  
**交付**：
- [ ] 限流、超时、依赖故障处置手册
- [ ] 索引延迟与回填手册
- [ ] 常见告警与排障 SOP

**DoD**：on-call 可按文档在 10 分钟内定位常见故障。

---

## 6. 建议排期（12 周）

- **W1-W2**：A1/A2/A3
- **W3-W5**：B1/B2/B3/B4
- **W6-W8**：C1/C2/C3/C4
- **W9-W10**：D1/D2/D3
- **W11-W12**：E1/E2/E3 + 发布复盘

---

## 7. 分工建议

- **后端检索组**：A2, B1, B2, B3, C1, C2, C4, E1
- **模型与排序组**：B4, C3, D1, E2
- **前端组**：D3 + A3 前端埋点联调
- **平台/SRE**：A3, E3

---

## 8. 风险与应对

- rerank 带来延迟上升 → Top-N 限制 + 超时降级
- embedding 成本上涨 → 降维 + 分层缓存 + 批量更新
- 多源数据质量不一致 → source-level 质量权重 + 去重合并
- 线上效果不稳 → A/B 灰度 + 快速回滚

---

## 9. 上线门禁（Release Gate）

上线前必须满足：

- [ ] 离线评测：nDCG@10 / Recall@50 达到目标
- [ ] 线上压测：P95 延迟 <= 1.8s（或明确降级策略）
- [ ] 监控完善：空结果率/错误率/点击率可观测
- [ ] 回滚可用：feature flag 一键回滚验证通过

---

## 10. 参考资料（调研来源）

- OpenAI Embeddings 指南：https://platform.openai.com/docs/guides/embeddings
- OpenAI File Search：https://platform.openai.com/docs/assistants/tools/file-search
- Elasticsearch Hybrid Search（RRF）：https://www.elastic.co/docs/solutions/search/hybrid-search
- OpenSearch Normalization Processor：https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/
- Qdrant Hybrid Queries：https://qdrant.tech/documentation/concepts/hybrid-queries/
- pgvector 官方文档：https://github.com/pgvector/pgvector
- SentenceTransformers Retrieve & Re-Rank：https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- BEIR benchmark：https://arxiv.org/abs/2104.08663
- ColBERTv2：https://arxiv.org/abs/2112.01488
- SPLADE：https://arxiv.org/abs/2107.05720
- OpenAlex Works Search：https://docs.openalex.org/api-entities/works/search-works
- Semantic Scholar API：https://www.semanticscholar.org/product/api
