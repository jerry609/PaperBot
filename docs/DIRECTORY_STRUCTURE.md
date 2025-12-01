# PaperBot 学者追踪 MVP 目录结构设计

## 📁 完整目录结构

```
PaperBot/
├── main.py                          # CLI 入口（新增 track_scholars 命令）
├── requirements.txt                 # 项目依赖
├── README.md                        # 项目说明
├── __init__.py
│
├── config/                          # 配置管理层
│   ├── __init__.py
│   ├── config.yaml                  # 现有全局配置
│   ├── settings.py                  # 现有配置解析
│   ├── scholar_subscriptions.yaml   # 🆕 学者订阅配置
│   └── top_venues.yaml              # 🆕 顶会/顶刊名单（用于 PIS 评分）
│
├── agents/                          # 现有 Agent 模块
│   ├── __init__.py
│   ├── base_agent.py                # 基础 Agent 类
│   ├── research_agent.py            # 论文研究 Agent
│   ├── code_analysis_agent.py       # 代码分析 Agent
│   ├── quality_agent.py             # 质量评估 Agent
│   └── documentation_agent.py       # 文档生成 Agent
│
├── scholar_tracking/                # 🆕 学者追踪子系统（核心新增模块）
│   ├── __init__.py
│   ├── agents/                      # 追踪相关 Agents
│   │   ├── __init__.py
│   │   ├── scholar_profile_agent.py # 学者画像管理 Agent
│   │   ├── semantic_scholar_agent.py# Semantic Scholar API 封装
│   │   └── paper_tracker_agent.py   # 新论文检测 Agent
│   │
│   ├── models/                      # 数据模型定义
│   │   ├── __init__.py
│   │   ├── scholar.py               # 学者数据模型
│   │   ├── paper.py                 # 论文元数据模型
│   │   └── influence.py             # 影响力评分模型
│   │
│   ├── services/                    # 业务服务层
│   │   ├── __init__.py
│   │   ├── subscription_service.py  # 订阅管理服务
│   │   ├── cache_service.py         # 缓存管理服务
│   │   └── api_client.py            # 外部 API 客户端封装
│   │
│   └── config/                      # 追踪模块专属配置
│       ├── __init__.py
│       └── schema.py                # 配置校验 Schema
│
├── influence/                       # 🆕 影响力计算子系统
│   ├── __init__.py
│   ├── calculator.py                # 影响力计算器（PIS 评分）
│   ├── metrics/                     # 各维度指标计算
│   │   ├── __init__.py
│   │   ├── academic_metrics.py      # 学术影响力指标 (I_a)
│   │   └── engineering_metrics.py   # 工程影响力指标 (I_e)
│   └── weights.py                   # 权重配置
│
├── core/                            # 核心工作流层
│   ├── __init__.py
│   ├── context.py                   # 现有上下文管理
│   ├── workflow.py                  # 现有工作流
│   └── workflow_coordinator.py      # 🆕 MVP 版工作流协调器
│
├── reports/                         # 🆕 报告生成与输出
│   ├── __init__.py
│   ├── templates/                   # 报告模板
│   │   ├── __init__.py
│   │   └── paper_report.md.j2       # 论文报告 Jinja2 模板
│   ├── writer.py                    # 报告写入器（ReportWriter）
│   └── notifier.py                  # 通知器（占位，MVP 不实现）
│
├── cache/                           # 🆕 本地缓存目录
│   └── scholar_papers/              # 学者论文缓存
│       └── {scholar_id}.json        # 每个学者的已处理论文 ID 列表
│
├── output/                          # 🆕 报告输出目录
│   └── reports/                     # 生成的报告存放位置
│       └── {scholar_name}/          # 按学者分组
│           └── {date}_{paper_id}.md # 单篇论文报告
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   ├── logger.py                    # 日志工具
│   ├── downloader.py                # 下载工具
│   ├── analyzer.py                  # 分析工具
│   └── github_client.py             # 🆕 GitHub API 客户端（可选）
│
├── prompts/                         # 🆕 Prompt 模板集中管理
│   ├── __init__.py
│   ├── research_prompts.py          # 论文研究相关 Prompt
│   ├── code_analysis_prompts.py     # 代码分析相关 Prompt
│   ├── quality_prompts.py           # 质量评估相关 Prompt
│   └── report_prompts.py            # 报告生成相关 Prompt
│
├── tests/                           # 🆕 测试目录
│   ├── __init__.py
│   ├── test_scholar_tracking/       # 学者追踪测试
│   │   ├── test_semantic_scholar.py
│   │   ├── test_paper_tracker.py
│   │   └── test_influence.py
│   └── fixtures/                    # 测试数据
│       └── sample_papers.json
│
└── docs/                            # 文档
    ├── MVP_DESIGN.md                # MVP 设计文档
    ├── SCHOLAR_TRACKING_DESIGN.md   # 学者追踪设计
    ├── BETTAFISH_RESEARCH.md        # BettaFish 调研
    ├── DIRECTORY_STRUCTURE.md       # 本文档
    └── USAGE.md                     # 使用指南
```

---

## 🏗️ 核心模块说明

### 1. 配置层 (`config/`)

#### `scholar_subscriptions.yaml` - 学者订阅配置
```yaml
# 学者订阅配置文件
subscriptions:
  scholars:
    - name: "Dawn Song"
      semantic_scholar_id: "1741101"
      keywords: ["Adversarial ML", "Blockchain Security"]
    - name: "Nicolas Papernot"
      semantic_scholar_id: "2810933"
      keywords: ["Machine Learning Security"]

  settings:
    check_interval: "weekly"        # daily / weekly
    papers_per_scholar: 20          # 每次拉取的论文数量
    min_influence_score: 50         # 最低影响力分数阈值
    output_dir: "output/reports"    # 报告输出目录
    cache_dir: "cache/scholar_papers"
```

#### `top_venues.yaml` - 顶会名单
```yaml
# 用于 PIS 学术影响力评分
top_conferences:
  security:
    - "CCS"
    - "S&P"
    - "USENIX Security"
    - "NDSS"
  ml:
    - "NeurIPS"
    - "ICML"
    - "ICLR"
  systems:
    - "OSDI"
    - "SOSP"
```

---

### 2. 学者追踪子系统 (`scholar_tracking/`)

```
scholar_tracking/
├── agents/
│   ├── scholar_profile_agent.py   # 管理学者画像与缓存
│   ├── semantic_scholar_agent.py  # 封装 Semantic Scholar API
│   └── paper_tracker_agent.py     # 检测新论文并触发分析
├── models/
│   ├── scholar.py                 # Scholar 数据类
│   ├── paper.py                   # PaperMeta 数据类
│   └── influence.py               # InfluenceResult 数据类
└── services/
    ├── subscription_service.py    # 解析订阅配置
    ├── cache_service.py           # 缓存读写操作
    └── api_client.py              # HTTP 请求封装
```

**核心数据模型示例：**

```python
# models/paper.py
@dataclass
class PaperMeta:
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    citation_count: int
    abstract: str
    url: str
    arxiv_id: Optional[str] = None
    github_url: Optional[str] = None
```

---

### 3. 影响力计算子系统 (`influence/`)

```
influence/
├── calculator.py           # InfluenceCalculator 主类
├── metrics/
│   ├── academic_metrics.py  # 计算 I_a (引用数、顶会匹配)
│   └── engineering_metrics.py # 计算 I_e (代码可用性、Stars)
└── weights.py               # 权重配置 (w1=0.6, w2=0.4)
```

**评分公式实现：**
```python
# calculator.py
class InfluenceCalculator:
    def calculate(self, paper: PaperMeta, code_meta: CodeMeta) -> InfluenceResult:
        i_a = self.academic_metrics.compute(paper)
        i_e = self.engineering_metrics.compute(code_meta)
        total = self.w1 * i_a + self.w2 * i_e
        return InfluenceResult(
            total_score=total,
            academic_score=i_a,
            engineering_score=i_e,
            explanation=self._generate_explanation(...)
        )
```

---

### 4. 工作流协调层 (`core/workflow_coordinator.py`)

```python
# MVP 版顺序流水线协调器
class WorkflowCoordinator:
    def __init__(self, config):
        self.research_agent = ResearchAgent(config)
        self.code_agent = CodeAnalysisAgent(config)
        self.quality_agent = QualityAgent(config)
        self.doc_agent = DocumentationAgent(config)
        self.calculator = InfluenceCalculator(config)
    
    async def run_paper_pipeline(self, paper: PaperMeta) -> Tuple[str, InfluenceResult]:
        """
        顺序执行：
        1. ResearchAgent → 扩展摘要 + 代码仓库链接
        2. CodeAnalysisAgent → 代码质量分析
        3. QualityAgent → 综合质量评价
        4. InfluenceCalculator → PIS 评分
        5. DocumentationAgent → 生成 Markdown 报告
        """
        # ... 实现细节
```

---

### 5. 报告生成 (`reports/`)

```
reports/
├── templates/
│   └── paper_report.md.j2   # Jinja2 模板
└── writer.py                # 文件写入逻辑
```

**报告模板结构：**
```markdown
# {{ paper.title }}

## 📋 元信息
| 属性 | 值 |
|------|-----|
| 作者 | {{ paper.authors | join(", ") }} |
| 年份 | {{ paper.year }} |
| 发表于 | {{ paper.venue }} |
| 引用数 | {{ paper.citation_count }} |

## 📝 执行摘要
{{ executive_summary }}

## 💻 代码与可复现性
{{ code_analysis }}

## 📊 影响力评分 (PIS)
- **总分**: {{ influence.total_score }}/100
- **学术影响力**: {{ influence.academic_score }}
- **工程影响力**: {{ influence.engineering_score }}

{{ influence.explanation }}

## 🎯 推荐级别
{{ recommendation }}
```

---

## 🔄 数据流向图

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                             │
│                  python main.py track_scholars                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              config/scholar_subscriptions.yaml                   │
│                    (订阅配置解析)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               scholar_tracking/agents/                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ ScholarProfile  │→ │ SemanticScholar  │→ │  PaperTracker  │  │
│  │     Agent       │  │     Agent        │  │     Agent      │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ new_papers
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              core/workflow_coordinator.py                        │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐ │
│  │ Research │→│ CodeAnal │→│ Quality │→│Influence │→│  Doc   │ │
│  │  Agent   │ │   Agent  │ │  Agent  │ │Calculator│ │ Agent  │ │
│  └──────────┘ └──────────┘ └─────────┘ └──────────┘ └────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ report_markdown + InfluenceResult
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    reports/writer.py                             │
│              output/reports/{scholar}/{date}_{id}.md             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 新增文件清单

### 必须创建的文件（按优先级排序）

| 优先级 | 文件路径 | 说明 |
|--------|----------|------|
| P0 | `config/scholar_subscriptions.yaml` | 学者订阅配置 |
| P0 | `scholar_tracking/__init__.py` | 模块初始化 |
| P0 | `scholar_tracking/agents/__init__.py` | Agent 子模块 |
| P0 | `scholar_tracking/agents/semantic_scholar_agent.py` | API 封装 |
| P0 | `scholar_tracking/agents/paper_tracker_agent.py` | 新论文检测 |
| P0 | `scholar_tracking/models/__init__.py` | 数据模型 |
| P0 | `scholar_tracking/models/paper.py` | 论文数据模型 |
| P1 | `scholar_tracking/agents/scholar_profile_agent.py` | 学者画像管理 |
| P1 | `scholar_tracking/services/cache_service.py` | 缓存服务 |
| P1 | `scholar_tracking/models/scholar.py` | 学者数据模型 |
| P1 | `influence/__init__.py` | 影响力模块 |
| P1 | `influence/calculator.py` | PIS 计算器 |
| P1 | `influence/metrics/academic_metrics.py` | 学术指标 |
| P1 | `influence/metrics/engineering_metrics.py` | 工程指标 |
| P2 | `core/workflow_coordinator.py` | 工作流协调器 |
| P2 | `reports/__init__.py` | 报告模块 |
| P2 | `reports/writer.py` | 报告写入器 |
| P2 | `reports/templates/paper_report.md.j2` | 报告模板 |
| P3 | `config/top_venues.yaml` | 顶会名单 |
| P3 | `prompts/__init__.py` | Prompt 模块 |
| P3 | `tests/test_scholar_tracking/` | 测试用例 |

---

## 🎯 与现有代码的集成点

1. **`main.py`**：新增 `track_scholars` CLI 命令
2. **`agents/base_agent.py`**：新 Agent 继承此基类
3. **`config/settings.py`**：扩展以支持 `scholar_subscriptions.yaml` 解析
4. **`utils/logger.py`**：复用现有日志工具
5. **现有 Agents**：ResearchAgent、CodeAnalysisAgent、QualityAgent、DocumentationAgent 被 WorkflowCoordinator 调用

---

## 🚀 下一步

1. 确认此目录结构符合预期
2. 开始创建骨架文件和基础类
3. 按阶段实现各模块功能
