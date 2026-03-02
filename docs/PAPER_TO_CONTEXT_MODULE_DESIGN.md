# Paper-to-Context (P2C) 模块设计文档

- 日期：2026-02-21
- 状态：Draft（未提交）
- 目标仓库：PaperBot
- 关联方向：论文采集/收藏闭环、OneContext 上下文层、Claude Scholar skills/agents、Paper2Code 复现执行

---

## 1. 背景与目标

你当前产品方向是：

1. 每日个性化推荐论文；
2. LLM 解释“为什么值得复现”；
3. 用户在 PaperBot 内直接创建 task/job 去复现；
4. 后续对接 Claude Code / Codex 执行。

当前缺口不是“执行器能力”，而是**把论文转换成高质量可执行上下文**的中间层。  
P2C（Paper-to-Context）模块的职责就是：

- 将论文（以及用户项目上下文）拆解为结构化复现包；
- 在执行前完成“关键信息提取 + 任务拆解 + 约束注入”；
- 向后兼容现有 Paper2Code 管线；
- 为后续 Claude Code / Codex 适配层提供统一输入。

一句话：**P2C 是你的“论文理解与执行计划编译器”**。

---

## 2. 设计原则

1. **执行器无关（Executor-Agnostic）**  
   P2C 产物不绑定 CC/Codex；通过 adapter 层再转成各端会话格式。

2. **结构化优先（Structured First）**  
   先产出 JSON schema（可校验、可追踪），再渲染 Markdown（人可读）。

3. **多阶段提取（Multi-pass Extraction）**  
   不做“一次 Prompt 生成全部”；分阶段抽取与校验，失败可降级。

4. **证据可追溯（Evidence Traceability）**  
   每条关键结论必须带 evidence span/source，避免幻觉扩散。

5. **本地优先 + 外接可选（Local-first + Optional External Context）**  
   默认写入本地 memory/research store；OneContext 作为可选同步层。

---

## 3. 参考现状（仓库内）

### 3.1 论文来源与收集（已具备）

- Harvest 与检索：
  - `src/paperbot/application/workflows/harvest_pipeline.py`
  - `src/paperbot/application/services/paper_search_service.py`
- Seed discovery（related/cited/citing/coauthor）：
  - `POST /api/research/discovery/seed`（`src/paperbot/api/routes/research.py`）
- Collections / Saved Papers / BibTeX / Zotero：
  - `src/paperbot/api/routes/research.py`

### 3.2 记忆与个性化（已具备）

- `ContextEngine` + `TrackRouter`：
  - `src/paperbot/context_engine/engine.py`
  - `src/paperbot/context_engine/track_router.py`
- Memory schema：
  - `src/paperbot/memory/schema.py`

### 3.3 Paper2Code 关键信息提取（已具备核心能力）

- `PaperContext` / `Blueprint` / `ReproductionPlan` / `ImplementationSpec`：
  - `src/paperbot/repro/models.py`
- 关键提取节点：
  - Blueprint 蒸馏（LLM + heuristic fallback）：`src/paperbot/repro/nodes/blueprint_node.py`
  - Hyperparameter 分析（regex + LLM merge）：`src/paperbot/repro/nodes/analysis_node.py`
  - 环境推断（year/code/LLM 三策略）：`src/paperbot/repro/nodes/environment_node.py`
- 多代理编排与修复循环：
  - `src/paperbot/repro/orchestrator.py`

这意味着：**P2C 不需要从 0 到 1，而是把现有提取能力产品化、结构化、可复用化**。

---

## 4. 外部参考与可借鉴点

### 4.1 OneContext（外接上下文层）

可借鉴能力（以其公开 README/Documentation 为准）：

- 统一 context 管理（多 session 归属同一 context）；
- 记录 agent trajectory；
- context 分享（链接/Slack）；
- 共享会话导入与恢复；
- CLI 入口与会话操作（`onecontext`/`oc`）。

对 PaperBot 的价值：

- 适合做“**上下文协作与共享层**”，不替代你的推荐/复现领域逻辑；
- 可用于跨设备、跨会话续跑；
- 可作为外接 provider 做双写。

### 4.2 Claude Scholar（skills/agents 编排资产）

可借鉴能力：

- `agents/literature-reviewer.md`：文献检索与 Zotero 工作流规范；
- `agents/architect.md`：架构分解模板；
- `agents/dev-planner.md`：任务拆解与交付计划；
- `skills/results-analysis/SKILL.md`：实验结果分析与统计报告规范；
- `skills/planning-with-files/SKILL.md`：plan/notes/deliverable 的“文件化工作记忆”。

对 PaperBot 的价值：

- 可作为 P2C 各阶段的 prompt/规则资产来源；
- 不建议直接耦合全部 command/hook 体系，建议抽取“可复用的 skill prompt 片段”。

### 4.3 与现有 Paper2Code 的衔接

- 你已具备论文 -> Blueprint/Spec 的 extraction 能力；
- P2C 重点是补齐“面向产品上下文包”的 schema、质量门禁与版本化。

---

## 5. 模块边界与职责

### 5.1 模块名称

`paper_to_context`（P2C Engine）

### 5.2 In Scope

- 论文输入标准化（paper metadata + text + user/project memory）；
- 多阶段提取与校验（Blueprint/Spec/Plan/Metrics）；
- 输出标准化 `ReproContextPack`；
- 写入本地 store，并可选同步到外接 context provider；
- 提供 API 给 Research UI / Studio UI 调用。

### 5.3 Out of Scope

- 直接执行代码（由 ReproAgent/Orchestrator + executor adapter 负责）；
- 替代推荐引擎排序；
- 替代 OneContext 的产品 UI。

---

## 6. 端到端架构

```text
┌───────────────────────────────────────────────────────────────┐
│                      Paper Sources Layer                      │
│  Discovery Seed / Saved Papers / Collections / BibTeX/Zotero │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                v
┌───────────────────────────────────────────────────────────────┐
│                 P2C Input Normalizer                         │
│  normalize paper identity + full text + user/project memory  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                v
┌───────────────────────────────────────────────────────────────┐
│                 Skill Orchestration Pipeline                  │
│ Stage A: Literature Distill   (claude-scholar literature)    │
│ Stage B: Blueprint Extract     (Paper2Code blueprint node)    │
│ Stage C: Environment Extract   (env node)                     │
│ Stage D: Spec/Hyperparams      (analysis node)                │
│ Stage E: Dev Plan/Roadmap      (planner skill + planning node)│
│ Stage F: Result Criteria       (results-analysis style)       │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                v
┌───────────────────────────────────────────────────────────────┐
│                 Context Assembly + Validation                │
│  JSON schema validate + evidence links + confidence scoring  │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
                v                               v
┌──────────────────────────────┐   ┌────────────────────────────┐
│ Local Context Store (primary)│   │ External Context Provider  │
│ SQLAlchemyMemory/ResearchStore│   │ OneContext (optional sync) │
└───────────────┬──────────────┘   └──────────────┬─────────────┘
                │                                 │
                └───────────────┬─────────────────┘
                                v
┌───────────────────────────────────────────────────────────────┐
│                   Studio/Research UI Seed                     │
│ Generate Reproduction Session -> CC/Codex Adapter (next step)│
└───────────────────────────────────────────────────────────────┘
```

---

## 7. 核心数据模型（建议）

### 7.1 `ReproContextPack`

```json
{
  "context_pack_id": "ctxp_xxx",
  "paper": {
    "paper_id": "...",
    "title": "...",
    "year": 2026,
    "identifiers": {"doi": "...", "arxiv": "...", "s2": "..."}
  },
  "objective": "复现论文的核心方法并验证主要指标",
  "blueprint": {"architecture_type": "transformer", "module_hierarchy": {}, "data_flow": []},
  "environment": {"python_version": "3.10", "framework": "pytorch", "cuda": "11.8"},
  "implementation_spec": {
    "optimizer": "adamw",
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 100
  },
  "task_roadmap": [
    {"id": "T1", "title": "数据预处理", "acceptance": ["可加载训练集"], "depends_on": []},
    {"id": "T2", "title": "模型实现", "acceptance": ["forward 正确"], "depends_on": ["T1"]}
  ],
  "success_criteria": [
    {"metric": "Top-1", "target": ">= 93.0", "source": "paper_table_2"}
  ],
  "evidence_links": [
    {"type": "paper_span", "ref": "method_section#L120-L140", "supports": ["optimizer", "lr"]}
  ],
  "confidence": {
    "overall": 0.81,
    "blueprint": 0.84,
    "env": 0.78,
    "metrics": 0.73
  },
  "version": "v1",
  "created_at": "2026-02-21T00:00:00Z"
}
```

### 7.2 存储建议

新增实体（可在 `research_store` 侧落地）：

- `repro_context_pack`
- `repro_context_stage_result`
- `repro_context_evidence`
- `repro_context_feedback`

关键索引：

- `(user_id, project_id, paper_id, created_at desc)`
- `(context_pack_id)`
- `(paper_id, version)`

---

## 8. 技能编排设计（结合 claude-scholar + Paper2Code）

### Stage A: 文献机制抽取（Literature Distill）

- 输入：paper meta + abstract + method section + related citations
- 参考资产：`agents/literature-reviewer.md`
- 输出：
  - 论文问题定义
  - 方法核心创新点
  - 与用户项目的关联点（why now）

### Stage B: Blueprint 抽取

- 复用：`BlueprintDistillationNode`
- 机制：LLM JSON 抽取失败时 fallback heuristic
- 输出：`Blueprint`

### Stage C: 环境推断

- 复用：`EnvironmentInferenceNode`
- 三路推断：year mapping + code pattern + LLM inference
- 输出：`EnvironmentSpec`

### Stage D: 实现规格提取

- 复用：`AnalysisNode`
- 机制：regex hyperparams + LLM hyperparams merge
- 输出：`ImplementationSpec` + `config_yaml`

### Stage E: 开发任务拆解（Roadmap）

- 参考资产：
  - `agents/dev-planner.md`
  - `skills/planning-with-files/SKILL.md`
- 输出：
  - checkpoint 列表
  - 依赖关系 DAG
  - 每步验收标准

### Stage F: 结果与复现成功标准抽取

- 参考资产：`skills/results-analysis/SKILL.md`
- 输出：
  - 指标定义（metric name / split / aggregation）
  - 统计显著性要求（可选）
  - 最小可接受复现标准

---

## 9. 模块内部组件设计

### 9.1 `SkillLoader`

职责：

- 从本地 skill/agent markdown 读取模板；
- 提取 frontmatter + instruction body；
- 提供可参数化 prompt 片段。

接口建议：

```python
class SkillLoader:
    def load_skill(self, key: str) -> SkillTemplate: ...
    def render(self, key: str, variables: dict) -> str: ...
```

### 9.2 `ExtractionOrchestrator`

职责：

- 串联 Stage A-F；
- 阶段失败降级；
- 汇总 stage confidence。

接口建议：

```python
class ExtractionOrchestrator:
    async def run(self, request: GenerateContextRequest) -> ReproContextPack: ...
```

### 9.3 `EvidenceLinker`

职责：

- 为每个关键字段绑定证据来源（文本 span / 表格 / meta）；
- 对“无证据高风险字段”打低置信度并标记人工确认。

### 9.4 `ContextAssembler`

职责：

- 将 Blueprint/Env/Spec/Roadmap 合成为 `ReproContextPack`；
- 输出 JSON + Markdown（`REPRODUCTION_PLAN.md`）。

### 9.5 `ContextProviderBridge`

职责：

- local store 为主写；
- OneContext 可选双写；
- 外接 provider 异常不阻断主流程。

---

## 10. API 设计（MVP）

### 10.1 生成上下文包

`POST /api/research/repro/context/generate`

请求：

```json
{
  "user_id": "default",
  "project_id": "proj_001",
  "paper_id": "paper_xxx",
  "track_id": 30,
  "depth": "standard",
  "executor_preference": "auto"
}
```

响应：

```json
{
  "context_pack_id": "ctxp_xxx",
  "status": "completed",
  "summary": "...",
  "confidence": {"overall": 0.81},
  "next_action": "create_repro_session"
}
```

### 10.2 获取上下文包详情

`GET /api/research/repro/context/{context_pack_id}`

返回完整 `ReproContextPack`。

### 10.3 由上下文包创建复现会话

`POST /api/research/repro/context/{context_pack_id}/session`

响应：

```json
{
  "session_id": "sess_xxx",
  "runbook_steps": [...],
  "initial_prompt": "..."
}
```

> 备注：后续 CC/Codex adapter 对接时，可直接消费该接口输出。

---

## 11. 与现有 UI 的对接方案

### 11.1 Research 页面入口

在以下页面增加按钮：

- discovery 卡片
- collections item
- saved papers item

按钮：`Generate Reproduction Session`

动作：

1. 调 `/api/research/repro/context/generate`；
2. 轮询或 SSE 获取阶段进度；
3. 成功后跳转 `/studio` 并注入 context pack。

### 11.2 Studio 页面注入

复用 `web/src/lib/store/studio-store.ts` 的 `paperDraft` / task timeline 能力：

- 将 `objective/blueprint/task_roadmap` 注入 runbook panel；
- 在 `BlueprintPanel` 展示 extraction 产物；
- 允许用户“编辑后再执行”（human-in-the-loop）。

---

## 12. 质量门禁与评估

### 12.1 抽取质量门禁

- JSON schema 校验必须通过；
- 必填字段空值率 < 5%；
- `success_criteria` 至少 1 项；
- 关键字段证据覆盖率 >= 80%。

### 12.2 离线评测集

构建 50 篇带“人工标准答案”的论文集，评估：

- 架构类型识别准确率；
- 超参数抽取准确率；
- 指标目标抽取准确率；
- 路线图可执行率（人工评分）。

### 12.3 线上业务指标

- context 生成成功率
- context -> job 创建转化率
- job 成功率提升幅度（对比无 P2C）
- 失败任务中“缺少关键信息”占比下降

---

## 13. 风险与缓解

1. **技能提示词漂移（claude-scholar 上游更新）**  
   - 缓解：skill snapshot 固化到本仓库；版本号管理；差异审查。

2. **外接 provider 可用性风险（OneContext）**  
   - 缓解：local-first；双写异步；失败自动回退。

3. **抽取幻觉导致错误执行**  
   - 缓解：evidence-link 强制 + 低置信度人工确认。

4. **上下文过长导致执行退化**  
   - 缓解：Blueprint 压缩优先；分层注入（必要字段优先）。

5. **许可与合规风险（第三方 prompt 资产）**  
   - 缓解：仅引用 MIT/明确许可资产；保留归因；禁止受限内容直拷。

---

## 14. 目录与实现建议（代码组织）

```text
src/paperbot/
  application/services/p2c/
    __init__.py
    models.py                 # ReproContextPack / stage outputs
    skill_loader.py           # load/parse skill templates
    extraction_orchestrator.py
    evidence_linker.py
    assembler.py
    provider_bridge.py        # local + optional onecontext
  api/routes/
    repro_context.py          # new endpoints
  infrastructure/connectors/
    onecontext_connector.py   # optional, behind feature flag

web/src/
  app/api/research/repro/context/
    route.ts
  components/research/
    GenerateReproSessionButton.tsx
  components/studio/
    ContextPackPanel.tsx
```

---

## 15. 分阶段落地计划（4 周）

### Week 1: 基础骨架

- 定义 `ReproContextPack` schema；
- 打通 `generate/get` API（先 mock stage）；
- UI 加入口按钮与状态反馈。

### Week 2: 复用 Paper2Code 提取能力

- 集成 Blueprint/Environment/Analysis 节点；
- 产出第一版 context pack；
- 加 schema + evidence 校验。

### Week 3: skill 编排增强

- 接入 claude-scholar 参考模板（literature/dev-planner/results）；
- 生成 roadmap + success criteria；
- 引入置信度评分。

### Week 4: Provider 与运营化

- local-first 持久化稳定；
- OneContext 可选双写（feature flag）；
- 指标埋点与 A/B（有无 P2C）上线。

---

## 16. 最小可用 DoD

- 可从任一论文卡片触发“Generate Reproduction Session”；
- 生成结构化 `ReproContextPack` 并可在 Studio 可视化；
- 可一键创建复现会话并带 roadmap；
- 失败可回退并提供可读错误原因；
- 对比基线，job 成功率或启动效率有可量化提升。

---

## 17. 实施分工（建议）

> 目标：并行推进，减少跨团队阻塞。以下是建议 owner，可按你团队实际调整。

### Workstream A：P2C Core（后端 + 算法）

- **Owner**：Backend/ML（1-2 人）
- **范围**：
  - `application/services/p2c/*` 核心 pipeline
  - `ReproContextPack` schema 与 stage 结果持久化
  - evidence-link + confidence 评分
- **里程碑交付**：
  - `POST /api/research/repro/context/generate` 可用
  - 50 篇离线评测脚本可跑通

### Workstream B：Skill 资产与提示词治理（PromptOps）

- **Owner**：Prompt/Research（1 人）
- **范围**：
  - claude-scholar skill/agent 的可复用片段抽取
  - skill snapshot 版本管理（避免上游漂移）
  - 失败降级策略与质量规则
- **里程碑交付**：
  - `skills_manifest.yaml`
  - 每个 stage 的 prompt 模板与回归样例

### Workstream C：Provider Bridge（OneContext 对接）

- **Owner**：Backend Platform（1 人）
- **范围**：
  - `onecontext_connector.py`
  - feature flag / 双写 / 重试 / dead-letter queue
  - provider 健康检查与观测指标
- **里程碑交付**：
  - 本地写成功不受外接失败影响（local-first）
  - 同步延迟和失败率可观测

### Workstream D：Studio/Research UI（前端）

- **Owner**：Frontend（1-2 人）
- **范围**：
  - Research 页触发入口（Generate Reproduction Session）
  - Studio `ContextPackPanel` 展示与确认编辑
  - 执行器适配输入可视化（CC/Codex 统一入口）
- **里程碑交付**：
  - 用户 3 步内完成 “论文 -> context -> 任务启动”
  - 关键链路 e2e 用例稳定

### Workstream E：Infra / SRE / 合规

- **Owner**：Infra（1 人）
- **范围**：
  - 成本与限流策略（LLM 调用预算）
  - 数据合规（license、来源归因、删除策略）
  - 线上回滚预案（feature flag + canary）
- **里程碑交付**：
  - SLA dashboard
  - 合规审计 checklist

---

## 18. Issue 拆分建议（可直接建单）

> 对应“一个 issue 一个可验收产物”，避免大而全任务。

1. `P2C-01`：定义 `ReproContextPack` schema + 校验器 + 示例数据
2. `P2C-02`：实现 Stage A/B（literature + blueprint）并落库
3. `P2C-03`：实现 Stage C/D（environment + hyperparams）并落库
4. `P2C-04`：实现 Stage E/F（roadmap + success criteria）
5. `P2C-05`：evidence-link + confidence 评分
6. `P2C-06`：`/repro/context/generate` + `/repro/context/{id}` API
7. `P2C-07`：OneContext bridge（feature flag + async 双写）
8. `P2C-08`：Research 页面生成入口 + 进度反馈
9. `P2C-09`：Studio `ContextPackPanel` + 人工确认 UI
10. `P2C-10`：离线评测集 + 线上指标埋点 + A/B 报告

每个 issue 验收建议至少包含：

- API contract（请求/响应示例）；
- 单元测试或 e2e 证据；
- 指标或日志截图（成功率/耗时/失败原因分布）。

---

## 19. 与现有 Paper Collection Pipeline 的对接细节

### 输入拼装（from collection/discovery）

- 主数据：`paper_id/title/abstract/year/identifiers`
- 扩展数据：discovery graph 邻居（related/cited/citing/coauthor）
- 用户上下文：track、收藏、最近任务失败原因、项目目标

### 输出回流（to memory/execution）

- 写入 `repro_context_pack`（主表）+ `stage_result`（诊断）
- 更新 track 记忆：该论文的复现优先级、风险标签、建议 action
- 触发 Studio 会话创建：将 roadmap 作为默认 task backlog

### 闭环信号（execution feedback -> recommendation）

- 执行成功/失败原因写回 `repro_context_feedback`
- 失败类型（环境不匹配/关键参数缺失/数据不可得）用于修正下次推荐
- 对高价值失败样本触发“二次 context 生成”（补充提取）

---

## 20. 关键参考链接

### 外部资料

- OneContext 仓库：<https://github.com/TheAgentContextLab/OneContext>
- OneContext 使用文档：<https://github.com/TheAgentContextLab/OneContext/blob/main/Documentation.md>
- Claude Scholar 仓库：<https://github.com/Galaxy-Dawn/claude-scholar>
- Claude Scholar literature-reviewer：<https://github.com/Galaxy-Dawn/claude-scholar/blob/main/agents/literature-reviewer.md>
- Claude Scholar architect：<https://github.com/Galaxy-Dawn/claude-scholar/blob/main/agents/architect.md>
- Claude Scholar dev-planner：<https://github.com/Galaxy-Dawn/claude-scholar/blob/main/agents/dev-planner.md>
- Claude Scholar planning-with-files：<https://github.com/Galaxy-Dawn/claude-scholar/blob/main/skills/planning-with-files/SKILL.md>
- Claude Scholar results-analysis：<https://github.com/Galaxy-Dawn/claude-scholar/blob/main/skills/results-analysis/SKILL.md>

### 本仓库实现参考

- `src/paperbot/repro/models.py`
- `src/paperbot/repro/nodes/blueprint_node.py`
- `src/paperbot/repro/nodes/analysis_node.py`
- `src/paperbot/repro/nodes/environment_node.py`
- `src/paperbot/repro/orchestrator.py`
- `src/paperbot/api/routes/research.py`
- `src/paperbot/api/routes/gen_code.py`
- `web/src/components/studio/RunbookPanel.tsx`
- `web/src/lib/store/studio-store.ts`
