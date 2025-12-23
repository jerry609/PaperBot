# Memory 中间件改造计划（TODO）

目标：把“多平台聊天记录 → 统一记忆 → 可检索注入/可审计可删除”的能力产品化，重点覆盖 GPT/ChatGPT、Claude(Anthropic)、Gemini、Augment Code（IDE），并为后续接入更多平台留出扩展点。

---

## P0：范围与验收（1–2 天）

- [ ] 定义 **记忆类型边界**（必须说清楚）
  - [ ] `User Memory`（profile/preference/goal/constraint/project）
  - [ ] `Episodic Memory`（会话摘要/关键事件）
  - [ ] `Workspace/Project Memory`（项目级背景、文档、代码索引）
  - [ ] 明确哪些属于“记忆”，哪些属于“上下文检索/缓存”
- [ ] 定义 **命名空间与隔离策略**
  - [ ] `user_id`（个人）
  - [ ] `workspace_id`（团队/项目/IDE 工作区）
  - [ ] `provider`（ChatGPT/Gemini/Claude/…）
- [ ] 定义验收指标（至少 4 个）
  - [ ] 抽取准确率（人工抽样评审：precision）
  - [ ] 误记/脏记率（false positive）
  - [ ] 检索命中率（“该用的记忆是否被拿到”）
  - [ ] 注入污染率（“错记忆影响回答”）
  - [ ] 数据控制闭环（删除后不可再被检索/注入）

---

## P1：数据模型与存储增强（2–5 天）

### 1.1 表结构补齐（在现有 `memory_sources` / `memory_items` 上扩展）
- [ ] `memory_items` 增加字段
  - [ ] `workspace_id`（可空，默认 user 全局）
  - [ ] `status`：`pending/approved/rejected/superseded`
  - [ ] `supersedes_id`（修订链）
  - [ ] `expires_at`（可选）
  - [ ] `last_used_at` / `use_count`（用于排序与衰减）
  - [ ] `pii_risk`（0/1/2 或枚举）
- [ ] 新增 `memory_audit_log`（审计轨迹）
  - [ ] who/when/what（创建、编辑、删除、审批、回滚）
- [ ] 新增（可选）`memory_embeddings`
  - [ ] `content_hash` → `embedding`（BLOB/JSON）、`model`, `dim`, `created_at`

### 1.2 数据生命周期/合规
- [ ] 明确保留期（sources 与 items 分开）
- [ ] 支持“硬删除”与“软删除”（软删用于审计，硬删用于合规）
- [ ] 可选：静态加密（SQLite 文件级或字段级）

---

## P2：导入与标准化（2–6 天）

### 2.1 导入器扩展（Parser）
- [ ] ChatGPT：conversations.json + 单会话导出 + 可能的附件/工具消息
- [ ] Gemini：Google Takeout/网页导出/SDK 日志（补齐更多真实样例）
- [ ] Claude：导出格式（如有）；Projects 相关内容（文档/指令）落到 workspace memory
- [ ] Perplexity：若官方导出/分享格式可获取则支持（否则先标注不支持）
- [ ] Augment Code：IDE 侧“上下文引擎”更多是 code retrieval，不是 user memory
  - [ ] 定义 IDE 侧可导出的最小事件：prompt、选区、文件引用、diff、回答

### 2.2 标准化协议（NormalizedMessage）
- [ ] 统一字段：role/content/ts/platform/conversation_id/message_id/metadata
- [ ] 统一时区与排序规则
- [ ] 统一附件/引用表示（例如 `citations: [path#line]`）
- [ ] 增加“来源可信度”（export/复制粘贴/抓包日志）

---

## P3：抽取（Extraction）升级：从“能用”到“可靠”（4–10 天）

### 3.1 抽取策略分层
- [ ] 规则抽取（高 precision）
  - [ ] 称呼/语言偏好/输出格式偏好/硬约束/项目名
- [ ] LLM 抽取（覆盖复杂文本）
  - [ ] 强 schema（JSON schema + repair + 类型校验）
  - [ ] 证据字段必须包含“原句摘要/位置指针”
  - [ ] 低置信度默认 `pending`（需要审批）

### 3.2 冲突与演化
- [ ] 冲突检测：同 kind 下互斥内容（例：喜欢/不喜欢）
- [ ] 合并策略：新条目 supersede 旧条目（保留链）
- [ ] 衰减策略：长期未被使用的记忆降权/过期

### 3.3 PII 与敏感信息
- [ ] 扩展脱敏：邮箱/电话/身份证/地址/银行卡/公司机密关键字
- [ ] “敏感信息默认不自动记忆”策略（除非显式确认）
- [ ] 抽取前后都做 PII 扫描（防止 LLM 复写出来）

---

## P4：治理（Governance）与数据控制（3–8 天）

- [ ] 记忆条目 CRUD API
  - [ ] 列表/搜索
  - [ ] 创建/编辑（人工修正）
  - [ ] 审批/驳回（pending → approved/rejected）
  - [ ] 删除（软删/硬删）
- [ ] “训练使用/共享”元数据字段（不是替代厂商开关，但用于你自己的系统）
  - [ ] `allow_training`（默认 false）
  - [ ] `allow_team_share`（默认 false）
- [ ] 访问控制（至少做到）
  - [ ] user_id token 或 session
  - [ ] workspace owner/admin 权限
  - [ ] 审计日志可追踪

---

## P5：检索与注入（Retrieval/Injection）（4–10 天）

### 5.1 检索：从 keyword → hybrid
- [ ] 基线：SQLite `LIKE` + tags（已做）
- [ ] 升级 1：SQLite FTS5（BM25）
  - [ ] `content`、`tags`、`project` 字段入索引
- [ ] 升级 2：Embeddings（可选）
  - [ ] 向量召回 topK + FTS topK 合并
  - [ ] rerank（轻量 cross-encoder/LLM 可选）
- [ ] 排序公式（可解释）
  - [ ] relevance + recency + confidence + use_count + kind_boost + workspace_boost

### 5.2 注入：严格预算与防污染
- [ ] 注入预算策略（按 TaskType）：chat/coding/review/analysis
- [ ] 注入格式稳定化（1 行 1 条，带 kind/confidence/tags）
- [ ] “可疑记忆”不上下文：低置信度、pending、pii_risk 高
- [ ] 引用与可追溯：注入块附上 memory_item ids（便于 debug）

---

## P6：IDE 场景（Augment/Copilot/Cursor 类）专项（5–15 天）

> IDE 更关键的是“代码库上下文检索”，不是“用户偏好记忆”。

- [ ] 新增 `code_index` 子系统（Repo Map 思路）
  - [ ] 文件树 + 符号摘要（类/函数签名）
  - [ ] 依赖/引用图（基础版：静态解析；增强版：LSP）
  - [ ] `git diff` 与最近文件优先
- [ ] 检索策略：先 repo map，再按需展开文件片段
- [ ] 安全：明确“不得把私有代码用于训练”（与 Augment 的安全承诺一致）
- [ ] 增加 IDE 插件/CLI 导出协议（最小可行：JSON lines）

---

## P7：产品化与验证（持续）

- [ ] Web 面板：记忆管理（列表/搜索/编辑/审批/删除/导出）
- [ ] 回归测试
  - [ ] parser 覆盖：每个平台至少 3 个真实样例（脱敏后）
  - [ ] extract 覆盖：冲突、去重、PII、pending/approve
  - [ ] retrieval 覆盖：FTS/embedding 混合、过滤策略
- [ ] 评测集与评审流程
  - [ ] 每周抽样 N 条：误记率/漏记率/污染案例
  - [ ] 记录“被用户纠正”的条目作为 hard-negative
- [ ] 观测与告警
  - [ ] 注入的 memory ids 日志化（便于追责/回放）
  - [ ] 删除后仍被检索到：硬告警

