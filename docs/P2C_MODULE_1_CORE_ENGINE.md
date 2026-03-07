# P2C Module 1: Core Engine — 提取管线与数据模型

- 日期：2026-02-22
- 状态：Draft
- 负责范围：提取编排、数据模型定义、质量门禁
- 上游依赖：论文源数据（paper metadata + full text）
- 下游消费方：Module 2（API & Storage）、Module 3（Frontend）

---

## 1. 模块职责

Core Engine 是 P2C 的计算核心，职责是：

1. 接收论文原始数据 + 用户/项目上下文，执行多阶段信息提取；
2. 产出结构化的 `ReproContextPack`（唯一输出契约）；
3. 对每个阶段输出执行质量校验，低质量时降级或标记人工确认；
4. 不负责持久化、不负责 HTTP 路由、不负责 UI 渲染。

**一句话定位：论文理解与执行计划的编译器。**

---

## 2. 与现有代码的关系

本模块 **复用** 以下现有能力，不重复实现：

| 现有组件 | 路径 | P2C 中的角色 |
|---|---|---|
| `Blueprint` dataclass | `src/paperbot/repro/models.py` | Stage B 输出类型 |
| `EnvironmentSpec` dataclass | `src/paperbot/repro/models.py` | Stage C 输出类型 |
| `ImplementationSpec` dataclass | `src/paperbot/repro/models.py` | Stage D 输出类型 |
| `BlueprintDistillationNode` | `src/paperbot/repro/nodes/blueprint_node.py` | Stage B 执行节点 |
| `EnvironmentInferenceNode` | `src/paperbot/repro/nodes/environment_node.py` | Stage C 执行节点 |
| `AnalysisNode` | `src/paperbot/repro/nodes/analysis_node.py` | Stage D 执行节点 |
| `PaperContext` | `src/paperbot/repro/models.py` | 输入归一化参考 |
| `ContextEngine` | `src/paperbot/context_engine/engine.py` | 用户记忆注入 |

**新增** 的部分：Stage A（文献蒸馏）、Stage E（任务拆解）、Stage F（成功标准）、编排器、证据链接器、组装器。

---

## 3. 核心数据模型

### 3.1 输入：`GenerateContextRequest`

```python
@dataclass
class GenerateContextRequest:
    """P2C 管线的唯一入口参数。"""
    paper_id: str                          # Semantic Scholar ID / arXiv ID / DOI
    user_id: str = "default"
    project_id: Optional[str] = None       # 关联项目（用于注入项目上下文）
    track_id: Optional[int] = None         # 关联 track（用于注入个性化记忆）
    depth: Literal["fast", "standard", "deep"] = "standard"
    # fast: 仅 Stage B+C，跳过 A/E/F
    # standard: 全部 Stage A-F
    # deep: 全部 Stage + 交叉验证 + 多模型投票
```

### 3.2 输出：`ReproContextPack`

这是 P2C 的 **唯一输出契约**，Module 2/3 均通过此结构消费数据。

```python
@dataclass
class ReproContextPack:
    context_pack_id: str                   # "ctxp_{uuid}"
    version: str = "v1"
    created_at: str = ""                   # ISO 8601

    # ── 论文身份 ──
    paper: PaperIdentity = field(default_factory=PaperIdentity)

    # ── 复现目标 ──
    objective: str = ""                    # 一句话复现目标

    # ── Stage 产出（每个阶段一个字段） ──
    literature_digest: Optional[LiteratureDigest] = None   # Stage A
    blueprint: Optional[Blueprint] = None                   # Stage B（复用现有）
    environment: Optional[EnvironmentSpec] = None           # Stage C（复用现有）
    implementation_spec: Optional[ImplementationSpec] = None # Stage D（复用现有）
    task_roadmap: List[TaskCheckpoint] = field(default_factory=list)  # Stage E
    success_criteria: List[SuccessCriterion] = field(default_factory=list)  # Stage F

    # ── 质量元数据 ──
    evidence_links: List[EvidenceLink] = field(default_factory=list)
    confidence: ConfidenceScores = field(default_factory=ConfidenceScores)
    warnings: List[str] = field(default_factory=list)      # 人工确认提示
```

### 3.3 子类型定义

```python
@dataclass
class PaperIdentity:
    paper_id: str = ""
    title: str = ""
    year: int = 0
    authors: List[str] = field(default_factory=list)
    identifiers: Dict[str, str] = field(default_factory=dict)  # doi/arxiv/s2

@dataclass
class LiteratureDigest:
    """Stage A 输出：文献机制蒸馏。"""
    problem_definition: str = ""           # 论文解决什么问题
    core_innovation: str = ""              # 方法核心创新点
    relation_to_user: str = ""             # 与用户项目/兴趣的关联
    key_references: List[str] = field(default_factory=list)  # 关键引用论文 ID

@dataclass
class TaskCheckpoint:
    """Stage E 输出：开发路线图单步。"""
    id: str                                # "T1", "T2", ...
    title: str
    description: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # 依赖的 checkpoint id
    estimated_difficulty: Literal["low", "medium", "high"] = "medium"

@dataclass
class SuccessCriterion:
    """Stage F 输出：复现成功标准。"""
    metric_name: str                       # e.g. "Top-1 Accuracy"
    target_value: str                      # e.g. ">= 93.0"
    dataset_split: str = "test"
    aggregation: str = "mean"
    source: str = ""                       # 论文中的出处 (e.g. "Table 2")
    tolerance: Optional[str] = None        # 允许偏差 (e.g. "±1.0")

@dataclass
class EvidenceLink:
    """证据溯源记录。"""
    type: Literal["paper_span", "table", "figure", "code_snippet", "metadata"]
    ref: str                               # e.g. "method_section#L120-L140"
    supports: List[str]                    # 支持的字段名列表
    confidence: float = 0.0

@dataclass
class ConfidenceScores:
    overall: float = 0.0
    literature: float = 0.0
    blueprint: float = 0.0
    environment: float = 0.0
    spec: float = 0.0
    roadmap: float = 0.0
    metrics: float = 0.0
```

---

## 4. 提取管线设计（6 阶段）

### 总体流程

```
GenerateContextRequest
    │
    ▼
┌─────────────────────────────────────┐
│          InputNormalizer             │
│  paper metadata + full text + memory│
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Stage A: Literature │  ← 新增
    │  Distill             │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Stage B: Blueprint  │  ← 复用 BlueprintDistillationNode
    │  Extract             │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Stage C: Environment│  ← 复用 EnvironmentInferenceNode
    │  Infer               │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Stage D: Spec &     │  ← 复用 AnalysisNode
    │  Hyperparams         │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Stage E: Task       │  ← 新增
    │  Roadmap             │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Stage F: Success    │  ← 新增
    │  Criteria            │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  EvidenceLinker      │
    │  + ContextAssembler  │
    └──────────┬──────────┘
               │
               ▼
       ReproContextPack
```

### 4.1 InputNormalizer

**职责**：将异构论文来源统一为管线可消费的内部格式。

```python
class InputNormalizer:
    async def normalize(self, request: GenerateContextRequest) -> NormalizedInput:
        """
        1. 通过 paper_id 从 ResearchStore 或 SemanticScholar 获取 metadata
        2. 获取论文全文（PDF parse / preprint text）
        3. 从 ContextEngine 注入用户/项目记忆
        4. 输出统一的 NormalizedInput
        """
        ...

@dataclass
class NormalizedInput:
    paper: PaperIdentity
    abstract: str
    method_text: str                       # 方法章节文本
    full_text: Optional[str] = None        # 全文（deep 模式用）
    algorithm_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    user_memory: Optional[str] = None      # 用户个性化上下文（格式化文本）
    project_context: Optional[str] = None  # 项目上下文（格式化文本）
```

### 4.2 Stage A: 文献机制蒸馏

**新增实现。** 参考 claude-scholar `literature-reviewer.md` 的 prompt 结构。

- 输入：`NormalizedInput.abstract` + `method_text` + `user_memory`
- 输出：`LiteratureDigest`
- 失败降级：跳过，`confidence.literature = 0`，`warnings` 追加提示

### 4.3 Stage B: Blueprint 抽取

**复用** `BlueprintDistillationNode`，增加包装层对接 P2C 接口。

- 输入：`NormalizedInput` → 构建 `PaperContext`
- 输出：`Blueprint`（现有类型）
- 降级：heuristic fallback（已内置于现有节点）

### 4.4 Stage C: 环境推断

**复用** `EnvironmentInferenceNode`。

- 输入：`Blueprint.paper_year` + `Blueprint.framework_hints` + 论文代码片段
- 输出：`EnvironmentSpec`（现有类型）
- 三路推断策略已内置：year mapping → code pattern → LLM inference

### 4.5 Stage D: 实现规格提取

**复用** `AnalysisNode`。

- 输入：`NormalizedInput.method_text` + `Blueprint`
- 输出：`ImplementationSpec`（现有类型）
- regex + LLM merge 策略已内置

### 4.6 Stage E: 任务拆解（Roadmap）

**新增实现。** 参考 claude-scholar `dev-planner.md` + `planning-with-files` skill。

- 输入：`Blueprint` + `ImplementationSpec` + `LiteratureDigest`
- 输出：`List[TaskCheckpoint]`，带依赖 DAG
- 约束：
  - 每个 checkpoint 必须有 `acceptance_criteria`
  - 总步数控制在 3-12 步
  - 第一步必须是数据/环境准备

### 4.7 Stage F: 成功标准抽取

**新增实现。** 参考 claude-scholar `results-analysis` skill。

- 输入：`NormalizedInput.tables` + `method_text`
- 输出：`List[SuccessCriterion]`
- 约束：至少产出 1 项可量化指标；无指标时 `confidence.metrics = 0`

---

## 5. 核心组件接口

### 5.1 ExtractionOrchestrator（编排器）

```python
class ExtractionOrchestrator:
    """串联 Stage A-F，管理阶段依赖与降级。"""

    def __init__(
        self,
        skill_loader: SkillLoader,
        llm_client: LLMClient,
        blueprint_node: BlueprintDistillationNode,
        environment_node: EnvironmentInferenceNode,
        analysis_node: AnalysisNode,
    ):
        ...

    async def run(
        self,
        normalized_input: NormalizedInput,
        depth: Literal["fast", "standard", "deep"] = "standard",
        on_stage_complete: Optional[Callable[[str, float], None]] = None,
    ) -> ReproContextPack:
        """
        执行提取管线。

        参数:
            normalized_input: 归一化后的论文输入
            depth: 提取深度
            on_stage_complete: 阶段完成回调 (stage_name, progress_pct)

        返回:
            完整的 ReproContextPack

        depth 行为:
            fast     → Stage B + C only
            standard → Stage A-F
            deep     → Stage A-F + cross-validation + multi-model voting
        """
        ...
```

**阶段依赖关系：**

```
A (独立) ──┐
B (独立) ──┼──→ E (依赖 A, B, D)
C (依赖 B) │
D (依赖 B) ┘
F (独立) ──────→ ContextAssembler (依赖全部)
```

注意：A、B、F 互相独立，可并行执行。

### 5.2 SkillLoader（技能模板加载器）

```python
class SkillLoader:
    """从本地 skill/agent markdown 加载 prompt 模板。"""

    def __init__(self, skills_dir: Path):
        ...

    def load_skill(self, key: str) -> SkillTemplate:
        """加载技能模板。key 示例: 'literature_distill', 'dev_planner'"""
        ...

    def render(self, key: str, variables: Dict[str, Any]) -> str:
        """渲染带变量的 prompt 模板。"""
        ...

@dataclass
class SkillTemplate:
    key: str
    instruction: str                       # prompt body
    output_schema: Optional[Dict] = None   # 期望的 JSON schema
    metadata: Dict[str, str] = field(default_factory=dict)  # frontmatter
```

### 5.3 EvidenceLinker（证据链接器）

```python
class EvidenceLinker:
    """为 context pack 中的关键字段绑定论文证据。"""

    async def link(
        self,
        pack: ReproContextPack,
        normalized_input: NormalizedInput,
    ) -> Tuple[List[EvidenceLink], ConfidenceScores]:
        """
        遍历 pack 中的关键字段，在原文中查找支撑证据。
        无证据的高风险字段会降低 confidence 并追加 warning。
        """
        ...
```

### 5.4 ContextAssembler（组装器）

```python
class ContextAssembler:
    """将各阶段产出组装为 ReproContextPack。"""

    def assemble(
        self,
        request: GenerateContextRequest,
        normalized_input: NormalizedInput,
        literature: Optional[LiteratureDigest],
        blueprint: Optional[Blueprint],
        environment: Optional[EnvironmentSpec],
        spec: Optional[ImplementationSpec],
        roadmap: List[TaskCheckpoint],
        criteria: List[SuccessCriterion],
        evidence: List[EvidenceLink],
        confidence: ConfidenceScores,
    ) -> ReproContextPack:
        ...

    def render_markdown(self, pack: ReproContextPack) -> str:
        """渲染为人可读的 REPRODUCTION_PLAN.md。"""
        ...
```

---

## 6. 质量门禁

### 6.1 阶段级校验

每个 Stage 完成后立即校验：

| 阶段 | 校验规则 | 失败策略 |
|---|---|---|
| A | `problem_definition` 非空 | 跳过，confidence=0 |
| B | `architecture_type` 非 "unknown"，`module_hierarchy` 至少 1 项 | heuristic fallback |
| C | `python_version` 合法，框架版本可解析 | 使用默认值 |
| D | `optimizer` 非空 | 使用论文常见默认值 |
| E | checkpoint 数量 3-12，DAG 无环 | 重试一次后降级为线性列表 |
| F | 至少 1 项 `SuccessCriterion` | 标记 warning，不阻断 |

### 6.2 Pack 级校验

组装完成后：

- JSON schema 校验必须通过
- 必填字段（`paper.title`, `blueprint`, `environment`）不能为 None
- `success_criteria` 至少 1 项（否则 warning）
- `evidence_links` 覆盖关键字段 >= 80%（否则降低 `confidence.overall`）
- `confidence.overall < 0.5` 时自动标记 `warnings: ["建议人工审查后再执行"]`

### 6.3 离线评测集

构建 50 篇带人工标注的论文集，评估维度：

| 维度 | 指标 |
|---|---|
| 架构识别 | `architecture_type` 准确率 |
| 超参数抽取 | precision / recall |
| 指标目标 | target value 精确匹配率 |
| 路线图可执行性 | 人工 1-5 分评分 |
| 证据覆盖 | evidence link 覆盖率 |

---

## 7. 代码组织

```
src/paperbot/p2c/
    __init__.py
    models.py                     # ReproContextPack 及所有子类型
    input_normalizer.py           # InputNormalizer
    skill_loader.py               # SkillLoader + SkillTemplate
    orchestrator.py               # ExtractionOrchestrator
    evidence_linker.py            # EvidenceLinker
    assembler.py                  # ContextAssembler
    validators.py                 # 阶段校验 + Pack 校验
    stages/
        __init__.py
        literature_distill.py     # Stage A (新增)
        blueprint_adapter.py      # Stage B (包装现有 BlueprintDistillationNode)
        environment_adapter.py    # Stage C (包装现有 EnvironmentInferenceNode)
        spec_adapter.py           # Stage D (包装现有 AnalysisNode)
        task_roadmap.py           # Stage E (新增)
        success_criteria.py       # Stage F (新增)
    skills/                       # prompt 模板资产
        literature_distill.md
        task_roadmap.md
        success_criteria.md
```

---

## 8. 与 Module 2 / Module 3 的接口契约

### 对 Module 2（API & Storage）的契约

Module 2 通过以下方式调用 Core Engine：

```python
# Module 2 调用 Core Engine 的唯一入口
orchestrator = ExtractionOrchestrator(...)
pack = await orchestrator.run(normalized_input, depth="standard", on_stage_complete=callback)
# pack: ReproContextPack — 唯一输出
```

- `on_stage_complete` 回调用于 SSE 进度推送
- Core Engine 不做持久化，由 Module 2 负责存储 `ReproContextPack`

### 对 Module 3（Frontend）的契约

Module 3 不直接调用 Core Engine，通过 Module 2 的 API 间接消费 `ReproContextPack` 的 JSON 序列化。

`ReproContextPack` 的 JSON schema 即为前后端契约。

---

## 9. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| Skill prompt 漂移（claude-scholar 上游更新） | 固化 snapshot 到 `p2c/skills/`，版本号管理 |
| 抽取幻觉导致错误执行 | evidence-link 强制 + 低置信度标记人工确认 |
| 上下文过长导致 LLM 退化 | Blueprint 压缩优先，分层注入（必要字段优先） |
| 阶段级联失败 | 每阶段独立降级，不阻断后续阶段 |
