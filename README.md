# PaperBot: 顶会论文分析与学者追踪框架

## 📚 概述

PaperBot 是一个专为计算机领域设计的智能论文分析框架。它不仅支持从四大安全顶会（IEEE S&P、NDSS、ACM CCS、USENIX Security）自动获取论文，还新增了**学者追踪系统**，能够自动监测指定学者的最新发表，进行多 Agent 深度分析，并生成包含影响力评分（PIS）的详细报告。

此外，PaperBot 维护了一个 **AI for Science & LLM Papers Collection**，收录了相关的顶会论文与代码实现：[AI4S Repository](https://github.com/jerry609/AI4S)。

## ✨ 核心功能

### 1. 学者追踪与智能分析 (New!)
- **全自动追踪**: 定期监测指定学者的最新论文（基于 Semantic Scholar）。
- **Deep Research 模式**: 引入迭代式反思循环（Reflection Loop），对发现的论文和研究方向进行多轮检索与验证，构建更完整的学者画像。
- **多 Agent 协作**:
  - **Research Agent**: 提取论文核心贡献与摘要。
  - **Code Analysis Agent**: 自动发现并分析关联 GitHub 仓库，评估代码质量与可复现性。
  - **Quality Agent**: 综合评估论文质量。
- **影响力评分 (PIS)**: 基于学术指标（引用、顶会）与工程指标（代码、Stars）计算 PaperBot Impact Score。
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

### 4. 深度评审 (ReviewerAgent) 🆕
- **DeepReview 模式**: 模拟人工同行评审流程（初筛 → 深度批评 → 决策）。
- 输出结构化评审报告：Summary、Strengths、Weaknesses、Novelty Score。
- 支持 Accept/Reject/Borderline 决策输出。

### 5. 科学声明验证 (VerificationAgent) 🆕
- 基于 CIBER 方法，自动提取论文中的关键声明。
- 多视角证据检索（支撑/反驳），使用 Semantic Scholar API。
- 输出裁定：Strongly Supported / Refuted / Controversial / Unverified。

### 6. 文献背景分析 (Literature Grounding) 🆕
- **ResearchAgent 新能力**: 实时搜索相似已有工作验证"新颖性"声明。
- 自动生成 Prior Art 搜索查询。
- 输出 Literature Grounding Report（是否真正创新 vs 增量改进）。

### 7. Paper2Code 代码生成 (ReproAgent) 🆕
- **多阶段流水线**: Planning → Analysis → Generation → Verification。
- 从论文方法章节自动生成代码骨架。
- Docker 隔离执行与细粒度验证（语法检查、导入检查、单元测试、冒烟运行）。
- 迭代修复：基于错误反馈自动重试。

## 🆚 与 AlphaXiv 的主要区别

- **定位**：PaperBot 面向“论文+代码+复现”的多 Agent 深度分析与报告生成；AlphaXiv 更偏论文聚合/推荐。  
- **代码与工程维度**：PaperBot 会自动发现/分析仓库，输出工程影响力（stars、last commit、可复现性）；AlphaXiv 主要提供论文元信息/摘要。  
- **学者追踪与报告**：支持学者订阅、自动检测新论文、生成 Markdown/学术模板报告（含影响力评分、代码要点）；AlphaXiv 无学者追踪与工程报告链路。  
- **可复现/实验**：内置 ExperimentManager，记录 git commit、依赖快照，支持学术模式/本地数据源、数据集校验脚本；AlphaXiv 不提供实验与复现闭环。  
- **会议抓取与代码提取**：ConferenceResearchAgent 直接抓取顶会论文并尝试提取 GitHub 链接，带并发/重试/兜底；AlphaXiv 不聚焦抓取代码资源。  
- **模板与模式**：学术/生产模式切换，paper/academic 模板可选，render-report 支持 meta 自动发现；AlphaXiv 模板化/报告定制能力有限。
## 🏗️ 系统架构

![System Architecture](public/asset/arcv2.png)

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
python main.py run-exp --config ExperimentManager/configs/exp_sentiment.yaml

# 渲染最新实验报告（自动选取 output/experiments 最新 meta）
python main.py render-report --template academic_report.md.j2
# 或指定 meta
python main.py render-report --meta output/experiments/xxx_meta.json --template paper_report.md.j2
```

### 5. 数据集校验
```bash
python scripts/validate_datasets.py
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

报告中会追加“可复现性验证”区块，展示状态、命令、耗时、日志摘要。

ReproAgent 可复现性验证整体流程如下图所示：

![ReproAgent Reproducibility Flow](public/asset/repoagent.png)

### 7. 论文深度评审 (ReviewerAgent) 🆕
```bash
# 对论文进行深度评审
python main.py review --title "Attention Is All You Need" --abstract "We propose a new architecture..."

# 输出到 JSON 文件
python main.py review --title "..." --abstract "..." --output review_result.json
```

### 8. 科学声明验证 (VerificationAgent) 🆕
```bash
# 验证论文中的科学声明
python main.py verify --title "Paper Title" --abstract "Paper abstract..."

# 指定提取声明数量
python main.py verify --title "..." --abstract "..." --num-claims 5 --output verify_result.json
```

### 9. Paper2Code 代码生成 🆕
```bash
# 从论文生成代码骨架
python main.py gen-code --title "Paper Title" --abstract "We propose..." --output-dir ./my_code

# 提供方法章节内容以获得更好的代码生成
python main.py gen-code --title "..." --abstract "..." --method "The model consists of..." --output-dir ./output
```

## 📂 目录结构

> **注意**: 项目正在进行架构重构，新代码位于 `src/paperbot/`，旧代码保留在根目录以保持向后兼容。

### 新架构 (src/paperbot/) - 推荐

```
PaperBot/
│
├── src/
│   └── paperbot/                      # 主包
│       ├── __init__.py                # 统一导出
│       │
│       ├── core/                      # 核心抽象层
│       │   ├── abstractions/          # Executable, ExecutionResult
│       │   │   ├── __init__.py
│       │   │   └── executable.py
│       │   ├── pipeline/              # Pipeline 抽象
│       │   │   ├── __init__.py
│       │   │   ├── pipeline.py
│       │   │   └── context.py
│       │   ├── di/                    # 依赖注入
│       │   │   ├── __init__.py
│       │   │   ├── container.py
│       │   │   └── bootstrap.py
│       │   ├── errors/                # 统一错误
│       │   │   ├── __init__.py
│       │   │   └── errors.py
│       │   ├── collaboration/         # 协作总线
│       │   │   ├── __init__.py
│       │   │   ├── bus.py
│       │   │   ├── host.py
│       │   │   └── messages.py
│       │   ├── report_engine/         # 报告引擎
│       │   │   ├── __init__.py
│       │   │   ├── engine.py
│       │   │   ├── renderers/
│       │   │   └── templates/
│       │   └── state.py
│       │
│       ├── agents/                    # Agent 层
│       │   ├── __init__.py
│       │   ├── base.py                # BaseAgent
│       │   ├── mixins/                # 共享 Mixin
│       │   │   ├── __init__.py
│       │   │   ├── semantic_scholar.py
│       │   │   ├── text_parsing.py
│       │   │   └── json_parser.py
│       │   ├── prompts/               # Prompt 模板
│       │   │   ├── __init__.py
│       │   │   ├── research.py
│       │   │   ├── code_analysis.py
│       │   │   └── quality.py
│       │   ├── state/                 # Agent 状态
│       │   │   ├── __init__.py
│       │   │   └── base_state.py
│       │   ├── research/              # ResearchAgent
│       │   │   ├── __init__.py
│       │   │   └── agent.py
│       │   ├── code_analysis/         # CodeAnalysisAgent
│       │   │   ├── __init__.py
│       │   │   └── agent.py
│       │   ├── quality/               # QualityAgent
│       │   │   ├── __init__.py
│       │   │   └── agent.py
│       │   ├── review/                # ReviewerAgent
│       │   │   ├── __init__.py
│       │   │   └── agent.py
│       │   ├── verification/          # VerificationAgent
│       │   │   ├── __init__.py
│       │   │   └── agent.py
│       │   └── conference/            # ConferenceResearchAgent
│       │       ├── __init__.py
│       │       └── agent.py
│       │
│       ├── infrastructure/            # 基础设施层
│       │   ├── __init__.py
│       │   ├── llm/                   # LLM 客户端
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── anthropic.py
│       │   │   └── openai.py
│       │   ├── api_clients/           # 外部 API
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── semantic_scholar.py
│       │   │   └── github.py
│       │   ├── storage/               # 存储
│       │   │   ├── __init__.py
│       │   │   ├── cache.py
│       │   │   └── report_store.py
│       │   └── services/              # 业务服务
│       │       ├── __init__.py
│       │       ├── paper_fetcher.py
│       │       └── scholar_tracker.py
│       │
│       ├── domain/                    # 领域模型层
│       │   ├── __init__.py
│       │   ├── paper.py               # PaperMeta, CodeMeta
│       │   ├── scholar.py             # Scholar
│       │   ├── influence/             # 影响力计算
│       │   │   ├── __init__.py
│       │   │   ├── result.py
│       │   │   ├── calculator.py
│       │   │   ├── metrics/
│       │   │   └── weights.py
│       │   └── events.py              # 领域事件
│       │
│       ├── workflows/                 # 工作流层
│       │   ├── __init__.py
│       │   ├── coordinator.py         # ScholarWorkflowCoordinator
│       │   ├── scholar_tracking.py    # 学者追踪工作流
│       │   ├── paper_review.py        # 论文评审工作流
│       │   └── nodes/                 # 工作流节点
│       │       ├── __init__.py
│       │       ├── fetch_node.py
│       │       ├── analysis_node.py
│       │       └── report_node.py
│       │
│       ├── presentation/              # 展示层
│       │   ├── __init__.py
│       │   ├── cli/                   # CLI
│       │   │   ├── __init__.py
│       │   │   ├── main.py
│       │   │   └── commands/
│       │   ├── reports/               # 报告生成
│       │   │   ├── __init__.py
│       │   │   ├── generator.py
│       │   │   ├── writer.py
│       │   │   └── templates/
│       │   └── api/                   # REST API (可选)
│       │       ├── __init__.py
│       │       └── routes.py
│       │
│       ├── repro/                     # 复现模块
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── docker_executor.py
│       │   ├── models.py
│       │   └── nodes/
│       │
│       └── utils/                     # 工具函数
│           ├── __init__.py
│           ├── retry_helper.py
│           ├── text_processing.py
│           └── downloader.py
│
├── config/                            # 配置文件
│   ├── config.yaml
│   ├── scholar_subscriptions.yaml
│   └── top_venues.yaml
│
├── tests/                             # 测试目录
│   ├── __init__.py
│   ├── unit/                          # 单元测试
│   │   ├── test_abstractions.py
│   │   ├── test_pipeline.py
│   │   ├── test_di_container.py
│   │   ├── test_errors.py
│   │   ├── test_agents/
│   │   └── test_domain/
│   ├── integration/                   # 集成测试
│   │   ├── test_workflow.py
│   │   └── test_api_clients.py
│   └── e2e/                           # 端到端测试
│       └── test_full_pipeline.py
│
├── output/                            # 输出目录
│   ├── reports/
│   └── experiments/
│
├── cache/                             # 缓存目录
│
├── main.py                            # 入口点
├── requirements.txt
├── pyproject.toml                     # 包配置
└── README.md
```

### 旧架构 (根目录) - 兼容保留

```
PaperBot/
├── main.py                 # 统一入口脚本
├── config/                 # 配置文件
│   ├── scholar_subscriptions.yaml
│   └── settings.py
├── agents/                 # 🔧 智能 Agent 模块
│   ├── base_agent.py              # Agent 基类 (Template Method)
│   ├── mixins/                    # 共享 Mixin
│   │   ├── semantic_scholar.py    # S2 API 客户端
│   │   └── text_parsing.py        # 文本解析工具
│   ├── state/                     # 状态管理 (BettaFish 启发)
│   │   ├── base_state.py          # 状态基类
│   │   └── research_state.py      # 研究状态 (段落级进度)
│   ├── research_agent.py          # 论文分析 + State 集成
│   ├── reviewer_agent.py          # 深度评审 (DeepReview)
│   ├── verification_agent.py      # 声明验证 (CIBER)
│   └── ...
├── repro/                  # Paper2Code 代码复现 (Node 管线)
│   ├── repro_agent.py             # 管线协调器
│   ├── nodes/                     # 🆕 4阶段节点
│   │   ├── base_node.py           # 节点基类
│   │   ├── planning_node.py       # Phase 1: 规划
│   │   ├── analysis_node.py       # Phase 2: 分析
│   │   ├── generation_node.py     # Phase 3: 生成
│   │   └── verification_node.py   # Phase 4: 验证
│   ├── docker_executor.py         # Docker 沙箱
│   └── models.py                  # 数据模型
├── core/                   # 核心工作流
│   ├── workflow_coordinator.py
│   └── collaboration/             # Agent 协作
│       ├── coordinator.py         # 协调器
│       └── messages.py            # 消息模型
├── scholar_tracking/       # 学者追踪
├── influence/              # 影响力评分
├── reports/                # 报告生成
├── utils/                  # 通用工具
├── tests/                  # 测试
└── output/                 # 生成的报告
```

### 架构设计原则

新架构遵循分层设计：

| 层级 | 目录 | 职责 |
|------|------|------|
| **Core** | `core/` | 核心抽象（Executable、Pipeline、DI、Errors） |
| **Domain** | `domain/` | 业务实体（Paper、Scholar、Influence） |
| **Infrastructure** | `infrastructure/` | 外部依赖封装（LLM、API、Storage） |
| **Agents** | `agents/` | 智能代理实现 |
| **Workflows** | `workflows/` | 业务流程编排 |
| **Presentation** | `presentation/` | 用户接口（CLI、Reports、API） |

### 使用新架构

```python
# 导入新架构组件
from src.paperbot import (
    ScholarTrackingWorkflow,
    PaperMeta,
    InfluenceResult,
    Pipeline,
    Container,
)

# 使用工作流
workflow = ScholarTrackingWorkflow(config)
result = await workflow.analyze_paper(paper)
```

## � 论文分析流水线

![Paper Analysis Pipeline](public/asset/workflowv2.png)

## �🔄 学者追踪工作流

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

### 环境变量
- `OPENAI_API_KEY`: 用于 LLM 分析（可选）。
- `GITHUB_TOKEN`: 用于 GitHub API 调用（提高限流阈值）。
- 协作主持人（可选，默认关闭）：
  - `PAPERBOT_HOST_ENABLED`: 是否启用主持人协作（true/false）
  - `PAPERBOT_HOST_API_KEY`: 主持人 LLM Key（未设置则回落到 OPENAI_API_KEY）
  - `PAPERBOT_HOST_MODEL`: 主持人模型名称（默认 gpt-4o-mini）
  - `PAPERBOT_HOST_BASE_URL`: 主持人 LLM Base URL（可为空）
- 报告引擎（可选，生成 HTML/PDF）：
  - `PAPERBOT_RE_ENABLED`: 是否启用 ReportEngine
  - `PAPERBOT_RE_API_KEY`: LLM Key（未设置则回落到 OPENAI_API_KEY）
  - `PAPERBOT_RE_MODEL`: 模型名称（默认 gpt-4o-mini）
  - `PAPERBOT_RE_BASE_URL`: 自定义 Base URL
  - `PAPERBOT_RE_OUTPUT_DIR`: 输出目录（默认 output/reports）
  - `PAPERBOT_RE_TEMPLATE_DIR`: 模板目录（默认 core/report_engine/templates）
  - `PAPERBOT_RE_PDF_ENABLED`: PDF 导出开关（true/false）
  - `PAPERBOT_RE_MAX_WORDS`: 总字数预算

## 🙏 致谢

特别感谢 [Qc-TX](https://github.com/Qc-TX) 对爬虫脚本的完善与贡献！
多 Agent 协作与深度研究流程的部分实践参考了 [BettaFish](https://github.com/666ghj/BettaFish) InsightEngine 的公开实现。



