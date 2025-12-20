# PaperBot: 顶会论文分析与学者追踪框架

## 📚 概述

PaperBot 是一个专为计算机领域设计的智能论文分析框架。它不仅支持从四大安全顶会（IEEE S&P、NDSS、ACM CCS、USENIX Security）自动获取论文，还新增了**学者追踪系统**，能够自动监测指定学者的最新发表，进行多 Agent 深度分析，并生成包含影响力评分（PIS）的详细报告。

此外，PaperBot 维护了一个 **AI for Science & LLM Papers Collection**，收录了相关的顶会论文与代码实现：[AI4S Repository](https://github.com/jerry609/AI4S)。

## ✨ 核心功能

### 1. 学者追踪与智能分析
- **全自动追踪**: 定期监测指定学者的最新论文（基于 Semantic Scholar）。
- **Deep Research 模式**: 引入迭代式反思循环（Reflection Loop），对发现的论文和研究方向进行多轮检索与验证，构建更完整的学者画像。
- **多 Agent 协作**:
  - **Research Agent**: 提取论文核心贡献与摘要。
  - **Code Analysis Agent**: 自动发现并分析关联 GitHub 仓库，评估代码质量与可复现性。
  - **Quality Agent**: 综合评估论文质量。
- **影响力评分 (PIS)**: 
  - **静态指标**: 基于引用数、顶会等级、GitHub Stars 计算。
  - **动态指标 (New)**: 引入 **Citation Velocity** (引用增速) 和 **Momentum Score** (动量评分)，识别上升期论文。
  - **情感分析 (New)**: 基于 LLM 分析引用语境，区分正面/负面引用。
- **自动化报告**: 生成包含关键指标、代码要点及推荐评级的 Markdown 报告。

### 2. 顶会论文获取
- 支持四大顶会论文自动下载：
  - IEEE Symposium on Security and Privacy (IEEE S&P)
  - Network and Distributed System Security Symposium (NDSS)
  - ACM Conference on Computer and Communications Security (ACM CCS)
  - USENIX Security Symposium
- 智能并发下载与元数据提取。

### 3. 代码深度分析
- 自动提取论文中的代码仓库链接。
- 代码质量、结构与安全性分析。
- **深度健康检查 (New)**:
  - **空壳检测**: 自动识别仅含有 README 的"占位"仓库。
  - **文档覆盖率**: 评估 README 质量、API 文档及示例代码完整性。
  - **依赖风险扫描**: 检查过时或高风险依赖包。

### 4. 深度评审 (ReviewerAgent)
- **DeepReview 模式**: 模拟人工同行评审流程（初筛 → 深度批评 → 决策）。
- 输出结构化评审报告：Summary、Strengths、Weaknesses、Novelty Score。
- 支持 Accept/Reject/Borderline 决策输出。

### 5. 科学声明验证 (VerificationAgent)
- 基于 CIBER 方法，自动提取论文中的关键声明。
- 多视角证据检索（支撑/反驳），使用 Semantic Scholar API。
- 输出裁定：Strongly Supported / Refuted / Controversial / Unverified。

### 6. 文献背景分析 (Literature Grounding)
- **ResearchAgent 新能力**: 实时搜索相似已有工作验证"新颖性"声明。
- 自动生成 Prior Art 搜索查询。
- 输出 Literature Grounding Report（是否真正创新 vs 增量改进）。

### 7. HuggingFace 模型集成 (New)
- **HuggingFaceAgent**: 自动搜索论文关联的 HuggingFace 模型。
- 获取模型下载量、点赞数、Model Card 元数据。
- 自动检测是否存在官方实现 (meta-llama, google, openai 等)。

### 8. OpenReview 审稿意见(New)
- **OpenReviewAgent**: 从 OpenReview 获取论文审稿评分与意见。
- 支持 ICLR, NeurIPS, ICML, AAAI 等顶会。
- 自动计算平均评分，提取决策结果 (Accept/Reject)。
- 使用 LLM 总结审稿人指出的薄弱点和优点。

### 9. Paper2Code 代码生成 (ReproAgent)
- **多阶段流水线**: Planning → **Environment Inference** → Analysis → Generation → Verification。
- **环境自动推断 (New)**: 基于论文年份和代码特征，自动推断 PyTorch/TensorFlow 版本并生成 `Dockerfile` 或 `conda environment.yaml`。
- **超参智能提取 (New)**: 深度解析附录与实验章节，提取微调超参生成 structured `config.yaml`。
- **自愈调试 (New)**: `VerificationNode` 集成 Self-Healing Debugger，自动分类错误（语法/依赖/逻辑）并利用 LLM 尝试修复。
- 从论文方法章节自动生成代码骨架。
- Docker 隔离执行与细粒度验证。

#### DeepCode 架构增强 (v2.0)
基于 [DeepCode](https://arxiv.org/abs/2512.07921) 论文的设计理念，ReproAgent 新增以下核心能力：

| 模块 | 功能描述 |
|------|----------|
| **Blueprint Distillation** | 将论文压缩为结构化 Blueprint（~2000 tokens），包含架构类型、模块层次、核心算法等 |
| **Stateful Code Memory** | 跨文件上下文追踪，基于 AST 的符号索引，依赖感知的文件生成顺序 |
| **CodeRAG** | 关键词匹配的代码模式检索，内置 10+ PyTorch/Transformer 模板 |
| **Multi-Agent Orchestrator** | 4 个专用 Agent 协调：Planning、Coding、Debugging、Verification |
| **Self-Healing Debugger** | 错误分类（语法/依赖/逻辑）+ LLM 自动修复循环 |

**架构对比**:

```
Legacy Pipeline:
  PaperContext → PlanningNode → AnalysisNode → GenerationNode → VerificationNode

Orchestrator Pipeline (New):
  PaperContext → BlueprintDistillation → PlanningAgent → CodingAgent ⟷ DebuggingAgent
                                              ↓               ↓
                                        CodeMemory + RAG    VerificationAgent
```

## 🆚 与 AlphaXiv / DeepCode 的主要区别

### vs AlphaXiv
- **定位**：PaperBot 面向"论文+代码+复现"的多 Agent 深度分析与报告生成；AlphaXiv 更偏论文聚合/推荐。
- **代码与工程维度**：PaperBot 会自动发现/分析仓库，输出工程影响力（stars、last commit、可复现性）；AlphaXiv 主要提供论文元信息/摘要。
- **学者追踪与报告**：支持学者订阅、自动检测新论文、生成 Markdown/学术模板报告（含影响力评分、代码要点）；AlphaXiv 无学者追踪与工程报告链路。
- **可复现/实验**：内置 ExperimentManager，记录 git commit、依赖快照，支持学术模式/本地数据源、数据集校验脚本；AlphaXiv 不提供实验与复现闭环。
- **会议抓取与代码提取**：ConferenceResearchAgent 直接抓取顶会论文并尝试提取 GitHub 链接，带并发/重试/兜底；AlphaXiv 不聚焦抓取代码资源。
- **模板与模式**：学术/生产模式切换，paper/academic 模板可选，render-report 支持 meta 自动发现；AlphaXiv 模板化/报告定制能力有限。

### vs DeepCode (HKUDS)
[DeepCode](https://github.com/HKUDS/DeepCode) 是港大数据智能实验室发布的开源 Paper2Code 框架，在 PaperBench 基准上达到 73.5% 的 SOTA 性能。

| 特性 | PaperBot | DeepCode |
|------|----------|----------|
| **核心定位** | 学者追踪 + 论文分析 + 代码复现 | 纯 Paper2Code 生成 |
| **Agent 架构** | 多 Agent 协作（Research/Code/Quality） | 模块化 Agent 流水线 |
| **执行后端** | Docker + E2B 云沙箱 | 本地执行 |
| **影响力分析** | PIS 评分、引用速度、情感分析 | 无 |
| **学者追踪** | 支持 | 不支持 |
| **自愈调试** | VerificationNode 错误分类修复 | Error Correction 模块 |
| **Blueprint 蒸馏** | ✅ 已集成 | ✅ 原创设计 |
| **状态化代码记忆** | ✅ CodeMemory + AST 索引 | ✅ Stateful Memory |
| **知识注入 (RAG)** | ✅ 关键词匹配（无向量依赖） | ✅ 向量检索 |

**PaperBot v2.0 已完整借鉴并实现 DeepCode 的核心设计**，同时保持了自身的多 Agent 学者追踪能力。

## 🏗️ 系统架构

![System Architecture](public/asset/arcv2.png)

> **P3 架构升级 (Coordinator v2)**: 
> 引入了 `ScoreShareBus` (评分共享总线) 和 `FailFastEvaluator` (快速失败评估器)。
> - 支持阶段间评分共享与订阅。
> - 在流水线早期自动拦截低质量/无代码/空壳仓库论文，节省计算资源。

> **P4 多后端 LLM 架构 (New)**:
> 支持多种 LLM 后端与成本路由：
> - **OpenAI / DeepSeek**: GPT-4o, GPT-4o-mini, DeepSeek-V3
> - **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus (Native SDK)
> - **Ollama**: 本地 Llama3, DeepSeek-Coder (免费)
>
> 使用 `ModelRouter` 根据任务类型自动选择最优模型。

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 学者追踪 (Scholar Tracking)

**配置订阅**:
编辑 `config/scholar_subscriptions.yaml` 添加你想追踪的学者：
```yaml
subscriptions:
  scholars:
    - name: "Dawn Song"
      semantic_scholar_id: "1741101"
  settings:
    check_interval: "weekly"
    min_influence_score: 50
    reporting:
      template: "paper_report.md.j2"
      persist_history: true
```

**运行追踪**:
```bash
# 追踪所有订阅学者，生成报告
python main.py track

# 仅查看追踪状态摘要
python main.py track --summary

# 强制重新检测指定学者（忽略缓存）
python main.py track --scholar-id 1741101 --force

# Dry-run 模式（不生成文件，仅打印结果）
python main.py track --dry-run

# 指定配置文件
python main.py track --config my_subscriptions.yaml

# 学术模式 & 模板
python main.py track --mode academic --report-template academic_report.md.j2

# 使用本地数据集（覆盖 data_source）
python main.py track --mode academic --data-source local --dataset-path datasets/processed/sample_sentiment.csv
```

### 3. 会议论文下载

```bash
# 下载 CCS 2023 论文（使用 ConferenceResearchAgent）
python main.py --conference ccs --year 23

# 下载 NDSS 2023 论文
python main.py --conference ndss --year 23
```

### 4. 实验与报告渲染
```bash
# 运行实验
python main.py run-exp --config config/experiments/exp_sentiment.yaml

# 渲染最新实验报告（自动选取 output/experiments 最新 meta）
python main.py render-report --template academic_report.md.j2

# 或指定 meta
python main.py render-report --meta output/experiments/xxx_meta.json --template paper_report.md.j2
```

### 5. 数据集校验
```bash
python validate_datasets.py
# 检查 datasets/processed/*.csv 是否包含 text/label，metadata 是否含 license/source
```

### 6. 可复现性验证（Repro）
```bash
# 学者追踪时启用可复现性验证（需 Docker、本地镜像可配置）
python main.py track --mode academic --repro

# 自定义报告模板
python main.py track --mode academic --repro --report-template academic_report.md.j2
```
配置项（settings.yaml/env）：
- `repro.docker_image`: 基线镜像，默认 python:3.10-slim
- `repro.cpu_shares` / `repro.mem_limit`: 资源限制
- `repro.timeout_sec`: 超时（秒）
- `repro.network`: 是否允许容器出网（默认禁用）
- `repro.executor`: 执行器选择 `docker` / `e2b` / `auto`（默认 auto）

**E2B 云沙箱配置**（可选）：
- `E2B_API_KEY`: E2B API 密钥（从 [e2b.dev](https://e2b.dev) 获取）
- `PAPERBOT_EXECUTOR`: 强制指定执行器 (`docker`/`e2b`/`auto`)
- `PAPERBOT_E2B_TEMPLATE`: E2B 沙箱模板（默认 `Python3`）
- `PAPERBOT_E2B_TIMEOUT`: E2B 沙箱超时秒数（默认 300）

使用 E2B 的优势：
- 无需本地安装 Docker
- 基于 microVM 的更强隔离
- 云端弹性扩展
- 预装常用 ML 库

报告中会追加"可复现性验证"区块，展示状态、命令、耗时、日志摘要。

ReproAgent 可复现性验证整体流程如下图所示：

![ReproAgent Reproducibility Flow](public/asset/repoagent.png)

### 7. 论文深度评审 (ReviewerAgent)
```bash
# 对论文进行深度评审
python main.py review --title "Attention Is All You Need" --abstract "We propose a new architecture..."

# 输出到 JSON 文件
python main.py review --title "..." --abstract "..." --output review_result.json
```

### 8. 科学声明验证 (VerificationAgent)
```bash
# 验证论文中的科学声明
python main.py verify --title "Paper Title" --abstract "Paper abstract..."

# 指定提取声明数量
python main.py verify --title "..." --abstract "..." --num-claims 5 --output verify_result.json
```

### 9. Paper2Code 代码生成
```bash
# 从论文生成代码骨架（Legacy 模式）
python main.py gen-code --title "Paper Title" --abstract "We propose..." --output-dir ./my_code

# 使用 DeepCode 增强模式（Blueprint + CodeMemory + RAG）
python main.py gen-code --title "..." --abstract "..." --use-orchestrator --output-dir ./output

# 提供方法章节内容以获得更好的代码生成
python main.py gen-code --title "..." --abstract "..." --method "The model consists of..." --output-dir ./output

# 禁用 RAG 模式检索
python main.py gen-code --title "..." --abstract "..." --use-orchestrator --no-rag --output-dir ./output
```

**CLI 选项说明**:
| 选项 | 描述 |
|------|------|
| `--use-orchestrator` | 启用多 Agent 协调模式（Blueprint + CodeMemory + RAG） |
| `--no-rag` | 禁用 CodeRAG 模式检索 |
| `--method` | 提供论文方法章节内容 |
| `--output-dir` | 输出目录（默认 ./generated_code） |

## 📂 目录结构

```
PaperBot/
│
├── src/                               # 源代码目录
│   └── paperbot/                      # 主包
│       ├── __init__.py                # 统一导出
│       │
│       ├── core/                      # 核心抽象层
│       │   ├── abstractions/          # Executable, ExecutionResult
│       │   ├── pipeline/              # Pipeline 抽象
│       │   ├── di/                    # 依赖注入
│       │   ├── errors/                # 统一错误
│       │   ├── collaboration/         # 协作总线
│       │   ├── report_engine/         # 报告引擎
│       │   ├── llm_client.py          # LLM 客户端
│       │   ├── state.py               # 状态管理
│       │   └── workflow_coordinator.py # 工作流协调器
│       │
│       ├── agents/                    # Agent 层
│       │   ├── base.py                # BaseAgent
│       │   ├── mixins/                # 共享 Mixin
│       │   │   ├── semantic_scholar.py
│       │   │   ├── text_parsing.py
│       │   │   └── json_parser.py
│       │   ├── prompts/               # Prompt 模板
│       │   ├── state/                 # Agent 状态
│       │   ├── research/              # ResearchAgent
│       │   ├── code_analysis/         # CodeAnalysisAgent
│       │   ├── quality/               # QualityAgent
│       │   ├── review/                # ReviewerAgent
│       │   ├── verification/          # VerificationAgent
│       │   ├── documentation/         # DocumentationAgent
│       │   ├── conference/            # ConferenceResearchAgent
│       │   └── scholar_tracking/      # 学者追踪相关 Agents
│       │
│       ├── infrastructure/            # 基础设施层
│       │   ├── llm/                   # LLM 客户端
│       │   │   ├── base.py            # LLMClient 兼容层
│       │   │   ├── router.py          # P4: ModelRouter
│       │   │   └── providers/         # P4: 多后端提供商
│       │   │       ├── base.py        # LLMProvider ABC
│       │   │       ├── openai_provider.py
│       │   │       ├── anthropic_provider.py
│       │   │       └── ollama_provider.py
│       │   ├── api_clients/           # 外部 API
│       │   ├── storage/               # 存储
│       │   └── services/              # 业务服务
│       │
│       ├── domain/                    # 领域模型层
│       │   ├── paper.py               # PaperMeta, CodeMeta
│       │   ├── scholar.py             # Scholar
│       │   └── influence/             # 影响力计算
│       │       ├── result.py
│       │       ├── calculator.py
│       │       ├── metrics/
│       │       ├── analyzers/         # P2: 动态分析器
│       │       │   ├── citation_context.py
│       │       │   ├── dynamic_pis.py
│       │       │   └── code_health.py
│       │       └── weights.py
│       │
│       ├── workflows/                 # 工作流层
│       │   ├── feed.py                # 信息流
│       │   ├── filters.py             # 筛选器
│       │   ├── scheduler.py           # 调度器
│       │   ├── scholar_tracking.py    # 学者追踪工作流
│       │   └── nodes/                 # 工作流节点
│       │
│       ├── presentation/              # 展示层
│       │   ├── cards.py               # 信息卡片
│       │   ├── cli/                   # CLI
│       │   └── reports/               # 报告生成
│       │
│       ├── repro/                     # 复现模块 (DeepCode 增强)
│       │   ├── repro_agent.py         # 主 Agent（支持 Legacy/Orchestrator 模式）
│       │   ├── orchestrator.py        # 多 Agent 协调器
│       │   ├── docker_executor.py     # Docker 执行器
│       │   ├── e2b_executor.py        # E2B 云沙箱执行器
│       │   ├── models.py              # 数据模型（Blueprint, PaperContext 等）
│       │   ├── nodes/                 # 处理节点
│       │   │   ├── blueprint_node.py  # Blueprint 蒸馏节点
│       │   │   ├── planning_node.py   # 规划节点
│       │   │   ├── generation_node.py # 代码生成节点（集成 Memory+RAG）
│       │   │   └── verification_node.py # 验证节点
│       │   ├── agents/                # 专用 Agent
│       │   │   ├── planning_agent.py  # 规划 Agent
│       │   │   ├── coding_agent.py    # 编码 Agent
│       │   │   ├── debugging_agent.py # 调试 Agent
│       │   │   └── verification_agent.py # 验证 Agent
│       │   ├── memory/                # 状态化代码记忆
│       │   │   ├── code_memory.py     # 跨文件上下文
│       │   │   └── symbol_index.py    # AST 符号索引
│       │   └── rag/                   # 代码知识检索
│       │       ├── knowledge_base.py  # 关键词匹配知识库
│       │       └── patterns/          # 内置代码模式
│       │
│       └── utils/                     # 工具函数
│           ├── logger.py
│           ├── downloader.py
│           ├── retry_helper.py
│           ├── json_parser.py
│           └── text_processing.py
│
├── config/                            # 配置文件
│   ├── config.yaml
│   ├── settings.py
│   ├── models.py
│   ├── scholar_subscriptions.yaml
│   └── top_venues.yaml
│
├── tests/                             # 测试目录
│   ├── unit/                          # 单元测试
│   ├── integration/                   # 集成测试
│   └── e2e/                           # 端到端测试
│
├── output/                            # 输出目录
│   ├── reports/
│   └── experiments/
│
├── cache/                             # 缓存目录
│
├── datasets/                          # 数据集
│   ├── metadata/
│   └── processed/
│
├── public/                            # 静态资源
│   └── asset/
│
├── main.py                            # 入口点
├── requirements.txt                   # 依赖
└── README.md
```

## 🏛️ 架构设计原则

PaperBot 采用分层架构设计，遵循关注点分离原则：

| 层级 | 目录 | 职责 |
|------|------|------|
| **Core** | `core/` | 核心抽象（Executable、Pipeline、DI、Errors、Collaboration） |
| **Domain** | `domain/` | 业务实体（Paper、Scholar、Influence） |
| **Infrastructure** | `infrastructure/` | 外部依赖封装（LLM、API、Storage） |
| **Agents** | `agents/` | 智能代理实现 |
| **Workflows** | `workflows/` | 业务流程编排 |
| **Presentation** | `presentation/` | 用户接口（CLI、Reports） |

### 核心抽象

- **Executable**: 统一的执行单元接口，Agent 和 Node 都实现此接口
- **ExecutionResult**: 标准化的执行结果封装
- **Pipeline**: 声明式流水线，支持阶段编排
- **Container**: 轻量级依赖注入容器
- **Result**: 函数式错误处理（类似 Rust 的 Result 类型）
- **LLMProvider**: P4 新增，统一的 LLM 后端抽象接口
- **ModelRouter**: P4 新增，基于任务类型的成本路由器

### 使用示例

```python
# 导入组件
from paperbot.workflows import ScholarTrackingWorkflow
from paperbot.domain.paper import PaperMeta
from paperbot.domain.influence.result import InfluenceResult

# 使用工作流
workflow = ScholarTrackingWorkflow(config)
report_path, influence, data = await workflow.analyze_paper(paper)

# 使用影响力计算
from paperbot.domain.influence.calculator import InfluenceCalculator
calculator = InfluenceCalculator()
result = calculator.calculate(paper, code_meta)
```

### P4 多模型路由示例

```python
# 使用 ModelRouter 进行任务化成本优化
from paperbot.infrastructure.llm import ModelRouter, TaskType

# 从环境变量创建路由器
router = ModelRouter.from_env()

# 实体提取使用便宜模型 (gpt-4o-mini)
provider = router.get_provider(TaskType.EXTRACTION)
result = provider.invoke_simple("Extract entities", text)

# 复杂推理使用强模型 (Claude 3.5)
reasoning = router.get_provider(TaskType.REASONING)
analysis = reasoning.invoke_simple("Analyze this paper", abstract)
```

## 📊 论文分析流水线

![Paper Analysis Pipeline](public/asset/workflowv2.png)

## 🔄 学者追踪工作流

![Scholar Tracking Workflow](public/asset/scholar.png)

## 🎨 学者追踪 UI 设计预览

> 下图为 PaperBot 学者追踪系统的初版 UI 设计稿，用于展示 Dashboard、论文卡片、学者卡片、信息流事件和筛选面板等关键界面。

1. 主控制台 Dashboard 概览

  ![PaperBot UI 1](asset/ui/1.png)

2. 论文信息卡片与列表视图

  ![PaperBot UI 2](asset/ui/2.png)

3. 学者画像与统计指标视图

  ![PaperBot UI 3](asset/ui/3.png)

4. 学者动态信息流与事件 Feed

  ![PaperBot UI 4](asset/ui/4.png)

5. 高级筛选条件与研究领域面板

  ![PaperBot UI 5](asset/ui/5.png)

6. 综合样例界面 / 交互细节补充

  ![PaperBot UI 6](asset/ui/6.png)

## 🛠 配置说明

主要配置文件位于 `config/` 目录下：
- `scholar_subscriptions.yaml`: 学者订阅列表及追踪设置。
- `config.yaml`: 全局系统配置。
- `settings.py`: Python 配置模块。

### 环境变量

#### LLM 配置 (必选其一)
- `OPENAI_API_KEY`: OpenAI / DeepSeek API Key
- `ANTHROPIC_API_KEY`: Anthropic Claude API Key (可选)
- `LLM_DEFAULT_MODEL`: 默认模型 (default: gpt-4o-mini)
- `LLM_REASONING_MODEL`: 推理模型 (default: claude-3-5-sonnet-20241022)
- `LLM_REQUEST_TIMEOUT`: 请求超时秒数 (default: 1800)

#### GitHub 与外部服务
- `GITHUB_TOKEN`: 用于 GitHub API 调用（提高限流阈值）

#### E2B 云沙箱（可选）
- `E2B_API_KEY`: E2B API 密钥（[e2b.dev](https://e2b.dev) 注册获取）
- `PAPERBOT_EXECUTOR`: 执行器选择 (`docker`/`e2b`/`auto`，默认 auto)
- `PAPERBOT_E2B_TEMPLATE`: E2B 沙箱模板（默认 Python3）
- `PAPERBOT_E2B_TIMEOUT`: E2B 沙箱超时（秒）

#### 协作主持人（可选，默认关闭）
- `PAPERBOT_HOST_ENABLED`: 是否启用主持人协作（true/false）
- `PAPERBOT_HOST_API_KEY`: 主持人 LLM Key（未设置则回落到 OPENAI_API_KEY）
- `PAPERBOT_HOST_MODEL`: 主持人模型名称（默认 gpt-4o-mini）
- `PAPERBOT_HOST_BASE_URL`: 主持人 LLM Base URL（可为空）

#### 报告引擎（可选，生成 HTML/PDF）
- `PAPERBOT_RE_ENABLED`: 是否启用 ReportEngine
- `PAPERBOT_RE_API_KEY`: LLM Key（未设置则回落到 OPENAI_API_KEY）
- `PAPERBOT_RE_MODEL`: 模型名称（默认 gpt-4o-mini）
- `PAPERBOT_RE_BASE_URL`: 自定义 Base URL
- `PAPERBOT_RE_OUTPUT_DIR`: 输出目录（默认 output/reports）
- `PAPERBOT_RE_TEMPLATE_DIR`: 模板目录（默认 core/report_engine/templates）
- `PAPERBOT_RE_PDF_ENABLED`: PDF 导出开关（true/false）
- `PAPERBOT_RE_MAX_WORDS`: 总字数预算

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/

# 运行特定测试
pytest tests/unit/test_pipeline.py -v

# 运行 DeepCode 增强模块测试
pytest tests/unit/repro/ tests/integration/test_repro_deepcode.py -v
```

## 🗺️ Roadmap

### Phase 1: Core Enhancements ✅ (已完成)

| 模块 | 状态 | 描述 |
|------|------|------|
| **E2B 云沙箱** | ✅ | 支持 Docker/E2B 双执行器后端 |
| **Blueprint 蒸馏** | ✅ | 将论文压缩为 ~2000 tokens 结构化上下文 |
| **Stateful Code Memory** | ✅ | 跨文件上下文追踪 + AST 符号索引 |
| **CodeRAG** | ✅ | 关键词匹配模式检索（无 embedding 依赖） |
| **Multi-Agent Orchestrator** | ✅ | 4 专用 Agent 并行执行 |
| **Self-Healing Debugger** | ✅ | 错误分类 + LLM 自动修复循环 |

### Phase 2: UI/UX 现代化 (计划中)

基于 [Vercel AI SDK](https://ai-sdk.dev/) 和 [Gemini CLI](https://github.com/google-gemini/gemini-cli) 的设计理念，计划推出两套交互界面：

#### 2.1 终端 UI (Node.js CLI)

受 Gemini CLI 启发，构建类似 Claude Code 的终端交互体验：

| 特性 | 技术栈 | 描述 |
|------|--------|------|
| **富终端渲染** | [Ink](https://github.com/vadimdemedes/ink) | React for CLI，渐进式流输出 |
| **流式响应** | WebSocket / SSE | 实时显示分析进度 |
| **MCP 支持** | [Model Context Protocol](https://modelcontextprotocol.io/) | 可扩展工具集成 |
| **多模型路由** | LLM 后端 | 自动选择最优模型 |

**预期命令示例**：
```bash
# 交互式学者追踪
paperbot track --interactive

# 流式论文分析
paperbot analyze --paper "Attention Is All You Need" --stream

# Paper2Code 生成
paperbot gen-code --title "..." --abstract "..." --live
```

#### 2.2 Web UI (Next.js + Vercel AI SDK)

基于 [Vercel AI SDK](https://vercel.com/docs/ai-sdk) 构建现代 Web 界面：

| 特性 | 实现 | 描述 |
|------|------|------|
| **流式聊天** | `useChat` | 实时流式响应 |
| **多 Provider** | AI SDK Core | OpenAI/Anthropic/Gemini 统一接口 |
| **Server Components** | Next.js 15 | 服务端流式渲染 |
| **实时进度** | `useObject` | 结构化数据流 |

**技术选型**：
- **前端**: Next.js 15 + TypeScript + Tailwind CSS
- **AI 集成**: `@ai-sdk/openai`, `@ai-sdk/anthropic`
- **流式传输**: `streamText`, `streamObject`
- **后端通信**: REST API / tRPC 调用 Python 服务

#### 2.3 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     UI 层 (Node.js/Next.js)                  │
├────────────────────────┬────────────────────────────────────┤
│   Terminal UI (Ink)    │        Web UI (Next.js)            │
│   - 流式输出            │   - useChat streaming              │
│   - MCP Tools          │   - React Server Components        │
│   - 键盘导航            │   - Dashboard 可视化               │
└────────────────────────┴────────────────────────────────────┘
                              ↓ HTTP/WebSocket ↓
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway (Python/FastAPI)               │
├─────────────────────────────────────────────────────────────┤
│   /api/track       - 学者追踪                                │
│   /api/analyze     - 论文分析 (SSE 流)                       │
│   /api/gen-code    - Paper2Code                             │
│   /api/review      - 深度评审                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PaperBot Core (Python)                    │
│   Agents | Workflows | Orchestrator | ReproAgent            │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: 高级功能 (未来规划)

| 功能 | 优先级 | 描述 |
|------|--------|------|
| **论文 PDF 解析增强** | P1 | 图表/公式/表格精准提取 |
| **多轮对话 Agent** | P1 | 基于对话历史的持续分析 |
| **批量处理管道** | P2 | 会议论文批量分析与对比 |
| **知识图谱构建** | P2 | 学者/论文/主题关系可视化 |
| **自定义评审模板** | P3 | 用户可配置评审标准 |
| **插件系统** | P3 | 第三方扩展支持 |

### 参考资源

- [Vercel AI SDK Documentation](https://ai-sdk.dev/docs/introduction)
- [Gemini CLI GitHub](https://github.com/google-gemini/gemini-cli)
- [Ink - React for CLIs](https://github.com/vadimdemedes/ink)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 🙏 致谢

特别感谢 [Qc-TX](https://github.com/Qc-TX) 对爬虫脚本的完善与贡献！
多 Agent 协作与深度研究流程的部分实践参考了 [BettaFish](https://github.com/666ghj/BettaFish) InsightEngine 的公开实现。

## 📄 License

MIT License
