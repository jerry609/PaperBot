# 记忆模块设计调研：大模型助手与智能 IDE

这份调研把“记忆（memory）/检索（retrieval）/上下文（context）”拆成可复用的工程模块，结合公开资料总结当前主流做法，并给出对 PaperBot 这类“跨平台记忆中间件”的落地建议。

> 重要区分：很多产品把“记忆”当作 UX 功能（个性化、偏好、长期事实），而很多开发者文档讨论的是“上下文构建/检索增强（RAG）”或“上下文缓存”。它们常被混用，但设计目标与风险边界不同。

---

## 1. 术语与目标拆解

### 1.1 三类“记忆”经常被混在一起

1) **持久化用户记忆（User Profile Memory）**
- 目标：跨会话稳定使用（偏好、背景、长期项目、约束）
- 典型形态：结构化 KV + 标签（“用户偏好简洁回答”“项目叫 PaperBot”）
- 关键问题：隐私、可解释、可编辑、可删除、过期策略

2) **会话记忆（Episodic / Conversation Memory）**
- 目标：在有限 context window 内维持连续对话
- 典型形态：对话摘要、最近 N 轮对话、关键事件列表
- 关键问题：摘要漂移、幻觉、信息丢失、策略可控性

3) **外部知识记忆（Knowledge / Tool Memory）**
- 目标：通过检索把“事实/文档/代码库”送入提示词（RAG）
- 典型形态：向量库/关键词索引 + rerank + 证据引用
- 关键问题：召回质量、去噪、权限控制、引用追踪

> “上下文缓存（prompt/context caching）”通常属于成本/延迟优化，不等同于“长期记忆”。

---

## 2. 公开资料中最常见的架构范式

### 2.1 事件日志 → 记忆抽取 → 存储 → 检索 → 注入

一个可扩展的长期记忆系统，通常分成 5 个步骤：

1) **Capture**：记录用户/助手/工具的事件流（messages + metadata）。
2) **Extract**：从事件中抽取“可长期复用”的条目（可规则、可 LLM）。
3) **Store**：把条目写入持久化存储（带类型、置信度、证据、版本）。
4) **Retrieve**：按当前问题 query 检索最相关条目（检索 + rerank + 过滤）。
5) **Inject**：以可控格式注入 system/context（尽量短、可解释、可审计）。

PaperBot 现有实现就是这个骨架：`parse -> extract -> store -> search -> inject`，并保留了 “source（导入批次）→ memory items（去重条目）” 的可追溯关系。

---

## 3. 大模型厂商/助手类产品：公开可观察到的“记忆设计”

> 说明：厂商 UI 功能细节常不完全公开，以下以公开文档/论文/可观测行为为主，避免猜测实现细节。

### 3.1 GitHub Copilot Chat（IDE 场景）

GitHub 文档对 Copilot Chat 的“上下文构建”描述非常明确：
- 用户提示会与**上下文信息**一起发送给模型，例如“当前仓库名、用户打开的文件”等。
- 允许使用可选的 `.github/copilot-instructions.md` 作为“隐式附加指令”。
- 模型侧可能“获取额外上下文”，如 GitHub 上的仓库数据；`@github` 参与者可利用 GitHub 上的代码上下文，并可选地结合 Bing 搜索。

参考：
- https://docs.github.com/api/article/body?pathname=/en/copilot/responsible-use/chat-in-your-ide

这类设计本质是“工作区上下文检索”，不是“用户长期记忆”，但它定义了 IDE 的主战场：**如何稳定拿到“对当前改动最有用的代码片段”**。

### 3.2 其它助手（OpenAI/Google/Anthropic 等）

多数厂商的消费级聊天产品都在做“个性化/长期偏好”，但工程层往往会落到本调研第 2 节的通用管线：
- 写入：显式触发（“记住这个”）+ 隐式抽取（偏好/背景）
- 存储：结构化属性（profile/preference/goal）优先
- 检索：按当前 query 选取少量条目注入，避免污染提示词
- 控制：可关闭、可查看/删除、可选择是否用于训练（产品层）

如果你的中间件要“跨平台合并记忆”，建议你把“厂商差异”收敛到统一的数据模型（见第 6 节），而不是试图完全复刻厂商策略。

---

## 4. 智能 IDE：代码场景的“记忆/检索”主流做法

代码场景最关键的不是“记住用户喜欢什么语气”，而是**在很小的 token 预算里给模型喂对上下文**。这里有两条主路线：

### 4.1 Repo Map / 结构化索引（符号级）

代表：aider 的 Repository map。

aider 的思路是：给模型提供一个“全仓库的压缩地图”，包含文件列表、类/方法/函数签名；如果需要更多代码，再按地图精确拉取文件。

aider 公开说明了：
- repo map 用于把 git 仓库结构提供给 LLM
- 大仓库时会做 map 的优化，并受 `--map-tokens`（默认约 1k tokens）影响

参考：
- https://aider.chat/docs/repomap.html

这类方法的本质：**先给“索引摘要”，再按需展开细节**，非常适合“代码库 + 限 token”场景，也比纯向量检索更可控（低幻觉、可解释）。

### 4.2 向量检索 / 混合检索（语义级）

常见做法：
- Chunk：按函数/类/文件块切分代码或文档
- Index：向量索引 +（可选）关键词/BM25 索引
- Retrieve：embedding 相似度召回 topK
- Rerank：用交叉编码器/LLM 对 topK 重排
- Inject：严格预算 + 引用路径（file:line）

学术参考（经典 RAG）：
- https://arxiv.org/abs/2005.11401

在 IDE 中通常还会混入强规则信号：
- 编辑器选择区、当前文件、最近改动（git diff）、报错栈、LSP 符号引用等

---

## 5. 研究界对“记忆系统”的典型设计

这些论文为“长期记忆 + 检索策略”提供了可复用的架构组件。

### 5.1 Generative Agents：记忆流 + 反思 + 计划

核心思想：
- 保存完整的“记忆流（experience stream）”
- 从记忆中定期生成更高层的“反思（reflection）”
- 检索时结合多个信号（常见是相关性/新近性/重要性）来决定注入哪些记忆

参考：
- https://arxiv.org/abs/2304.03442

### 5.2 MemGPT：分层记忆（像操作系统的虚拟内存）

核心思想：
- LLM context window 像“RAM”，需要把信息在不同记忆层之间搬运
- 用“中断/控制流”决定何时写入/何时召回/何时压缩
- 目标是把“超出上下文窗口的信息”以可控方式纳入推理

参考：
- https://arxiv.org/abs/2310.08560

### 5.3 Reflexion：把“反馈→反思文本”写进 episodic buffer

核心思想：
- 不做权重更新，而是把任务反馈转成可复用的“反思文本”
- 反思文本作为“episodic memory buffer”，下一轮决策时注入

参考：
- https://arxiv.org/abs/2303.11366

### 5.4 Voyager：技能库（可执行代码）作为长期记忆

核心思想：
- 长期记忆不仅是文本；“技能（可执行代码）”同样是可检索复用的记忆单元
- 用技能库做存储与检索，减少遗忘，提高可组合性

参考：
- https://arxiv.org/abs/2305.16291

### 5.5 ReAct：检索/工具调用作为“外部记忆访问机制”

ReAct 把“推理轨迹”和“行动（检索/调用工具）”交织在一起，降低幻觉并把信息获取外包给工具/知识库。

参考：
- https://arxiv.org/abs/2210.03629

---

## 6. 对“跨平台记忆中间件”的落地建议（设计与检索）

### 6.1 数据模型：把差异收敛成统一的 MemoryItem

建议拆成三张逻辑表（PaperBot 已实现前两张）：
- `memory_sources`：导入批次（平台、文件名、sha256、统计信息）
- `memory_items`：稳定条目（kind、content、tags、confidence、evidence、hash 去重）
- （可选）`memory_embeddings`：语义检索向量（content_hash → embedding）

条目 `kind` 最少覆盖：
- `profile` / `preference` / `goal` / `project` / `constraint` / `todo` / `fact`

### 6.2 写入策略：规则优先 + LLM 兜底 + 人工可编辑

1) **规则抽取**适合高精度字段：称呼、偏好、硬约束、明确目标。
2) **LLM 抽取**适合低结构、长文本，但必须有：
   - JSON schema 约束
   - PII 脱敏
   - 去重与置信度
3) **人工编辑/确认**是长期质量的关键（尤其是个人信息与偏好变化）。

### 6.3 检索策略：混合评分 + 强约束过滤

一个可解释的检索排序可以用“加权信号”：
- `relevance`：语义/关键词匹配
- `recency`：最近更新/最近提及
- `confidence`：抽取置信度
- `type_boost`：某些 kind 在某些任务更重要（例如 coding 任务优先 `project/constraint`）
- `source_boost`：同一平台/同一项目的 source 更相关

落地上建议先做“便宜可靠”的方案：
- SQL contains + tags（PaperBot 当前实现）
- 再逐步加：BM25（Whoosh/Lucene/SQLite FTS5）→ embeddings → rerank

### 6.4 注入策略：短、少、可追溯

建议把注入块做成固定格式（PaperBot 已提供 `build_memory_context`）：
- 每条一行，包含 `kind`、`confidence`、`tags`
- 严格限制数量（例如 5~10 条）
- evidence 只用于审计/溯源，不要全部塞进 prompt

### 6.5 IDE 场景的建议：把“Repo Map 思路”带进中间件

如果你要做“中间件 + 智能 IDE”：
- 把“代码库记忆”单独建一种 memory：`code_index`
- 优先做 **结构化索引（符号/接口摘要）**，再加语义检索
- 对每个项目维护：
  - 文件树摘要
  - 接口摘要（类/函数签名）
  - 关键入口与依赖图

这和 aider 的 repo map 设计（先地图，再按需展开）高度一致：`https://aider.chat/docs/repomap.html`

---

## 7. PaperBot 当前实现与下一步增强路线

PaperBot 当前已经具备：
- 多平台聊天记录导入解析（ChatGPT export / 宽松 JSON / plaintext）
- 规则抽取 + 可选 LLM 抽取（失败回退）
- SQLite 持久化、来源追踪、按 user_id 去重
- 关键词检索 + Chat 注入开关

下一步如果要更接近“主流产品级记忆”：
1) 加 `memory_embeddings` 表 + embedding 召回（可选）
2) 增加“人审/确认”状态（pending/approved/rejected）
3) 加“过期/修订”机制（updated_at + supersedes_id）
4) 加“项目维度/工作区维度”命名空间（user_id + workspace_id）
5) 给记忆条目提供可视化编辑（Web/CLI）

## 8. 各产品 UI 控制项对照

见 `docs/memory_ui_controls_matrix.md`（重点覆盖：ChatGPT/GPT、Anthropic/Claude、Augment Code，并包含 Gemini/Grok/Copilot/Perplexity）。
