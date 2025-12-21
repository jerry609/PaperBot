# PaperBot：顶会论文分析与学者追踪框架

PaperBot 是一个面向计算机领域的研究工作流工具链：支持从顶会自动获取论文、持续追踪学者发表、通过多 Agent 完成论文/代码深度分析与评审，并生成包含影响力评分（PIS）的结构化报告；同时提供 Paper2Code（ReproAgent）能力，用于从论文生成代码骨架并进行验证。

此外，本仓库包含一个 **AI for Science & LLM Papers Collection**（`AI4S/`），收录相关顶会论文与代码实现（也可参考 [AI4S Repository](https://github.com/jerry609/AI4S)）。

## 概述

- **后端**：Python（Agents / Workflows）+ FastAPI（SSE 流式）
- **客户端**：Terminal UI（Ink/React）+ Web Dashboard（Next.js）
- **数据源**：Semantic Scholar（可选 API Key）、会议官网/出版方页面、GitHub、（可选）OpenReview、（可选）本地/混合数据源

> 改造的详细评估与可执行计划见：`docs/PLAN.md`

## 核心功能

### 1. 学者追踪与智能分析

- **全自动追踪**：定期监测指定学者的最新论文（基于 Semantic Scholar；支持离线/本地数据源回退）。
- **Deep Research 模式**：支持迭代式反思循环（Reflection Loop），对论文与方向进行多轮检索与总结，构建更完整的学者画像。
- **多 Agent 协作**：
  - **ResearchAgent**：提炼论文贡献点、摘要与方法概括，并支持 Literature Grounding（新颖性/先验工作验证）。
  - **CodeAnalysisAgent**：发现并分析关联 GitHub 仓库，评估工程质量与可复现性。
  - **Quality/Reviewer/Verification**：质量评估、深度评审、声明验证等。
- **影响力评分（PIS）**：
  - **静态指标**：引用数、venue 等级、GitHub Stars 等。
  - **动态指标**：Citation Velocity（引用增速）与 Momentum Score（趋势动量）。
  - **引用语境情感**：可选，基于 LLM 区分正面/负面/中性引用语境（未配置时自动降级）。
- **自动化报告**：生成 Markdown 报告（支持模板渲染与学术模板）。

### 2. 顶会论文获取

- 支持四大安全顶会论文下载与元数据提取：
  - IEEE Symposium on Security and Privacy (IEEE S&P)
  - Network and Distributed System Security Symposium (NDSS)
  - ACM Conference on Computer and Communications Security (ACM CCS)
  - USENIX Security Symposium
- 支持“智能下载”（批量下载 + 动态并发调整 + 缓存命中统计）。

### 3. 代码深度分析

- 自动提取论文/页面/PDF 中的代码仓库链接（优先 GitHub）。
- **深度健康检查**（Code Health）：
  - **空壳检测**：识别仅含 README/占位内容的仓库。
  - **文档覆盖率**：评估 README/API 文档/示例完整性。
  - **依赖风险扫描**：识别高风险/过时依赖（基于工具链与策略配置）。

### 4. 深度评审（ReviewerAgent）

- **DeepReview 模式**：模拟同行评审流程（初筛 → 深度批评 → 决策）。
- 输出结构化评审报告：Summary、Strengths、Weaknesses、Novelty Score 等。
- 支持 Accept/Reject/Borderline 等决策输出。

### 5. 科学声明验证（VerificationAgent）

- 基于“抽取关键声明 → 检索证据 → 支撑/反驳/争议判定”的流程。
- 多视角证据检索：支持 Semantic Scholar API。
- 输出裁定：Strongly Supported / Refuted / Controversial / Unverified。

### 6. 文献背景分析（Literature Grounding）

- 用于验证“新颖性/相关工作覆盖”：自动生成 prior-art 查询，在学术数据库检索并生成对比结论。

### 7. HuggingFace 模型集成（可选）

- **HuggingFaceAgent**：搜索论文关联的 HuggingFace 模型。
- 获取下载量、点赞数、Model Card 元数据等。
- 依赖：`huggingface_hub`（未安装时会提示并降级）。

### 8. OpenReview 审稿意见（可选）

- **OpenReviewAgent**：从 OpenReview 获取审稿评分与意见（适用于 ICLR/NeurIPS/ICML/AAAI 等）。
- 自动计算平均评分、提取决策结果，并用 LLM 总结优缺点。
- 依赖：`openreview-py`（未安装时会提示并降级）。

### 9. Paper2Code 代码生成（ReproAgent）

- **多阶段流水线**：Planning → Analysis → Generation → Verification。
- **自愈调试**：Verification/Debugging 结合错误分类与修复循环（语法/依赖/逻辑）。
- **执行后端**：支持 Docker 与 E2B 云沙箱（可选）。

#### DeepCode 架构增强（v2.0）

借鉴 DeepCode 的设计理念，ReproAgent 引入以下核心能力：

| 模块 | 功能描述 |
|------|----------|
| **Blueprint Distillation** | 将论文压缩为结构化 Blueprint（包含架构类型、模块层次、关键算法等） |
| **Stateful Code Memory** | 跨文件上下文追踪，基于 AST 的符号索引与依赖感知生成顺序 |
| **CodeRAG** | 代码模式检索（关键词匹配为主），内置常见 PyTorch/Transformer 模式 |
| **Multi-Agent Orchestrator** | 多专用 Agent 协同（Planning/Coding/Debugging/Verification） |
| **Self-Healing Debugger** | 错误分类（语法/依赖/逻辑）+ 自动修复循环 |

架构对比：

```text
Legacy Pipeline:
  PaperContext → PlanningNode → AnalysisNode → GenerationNode → VerificationNode

Orchestrator Pipeline:
  PaperContext → BlueprintDistillation → PlanningAgent → CodingAgent ⟷ DebuggingAgent
                                              ↓               ↓
                                        CodeMemory + RAG    VerificationAgent
```

## 界面预览

### Terminal UI（Ink）

![PaperBot CLI Demo](asset/ui/paperbot%20cli%20demo.jpg)

### Web Dashboard（Next.js）

![Dashboard](asset/ui/dashboard.jpg)

更多界面截图（论文分析/学者画像/Wiki/DeepCode Studio 等）：

1. 论文深度分析视图

![Paper Analysis](asset/ui/paper.jpg)

2. 学者画像与统计指标

![Scholar Profile](asset/ui/scholar2.jpg)

3. Wiki 知识库

![Wiki Knowledge Base](asset/ui/wiki.jpg)

4. DeepCode Studio（代码复现）

![DeepCode Studio](asset/ui/deepcode.jpg)

## 与 AlphaXiv / DeepCode 的主要区别（对标）

### vs AlphaXiv

- **定位**：PaperBot 面向“论文 + 代码 + 复现 + 报告”的多 Agent 工作流；AlphaXiv 更偏论文聚合/推荐。
- **工程维度**：PaperBot 会发现/分析代码仓库并输出工程指标与健康度；AlphaXiv 通常聚焦论文元信息/摘要。
- **学者追踪与报告**：PaperBot 支持订阅与持续追踪并生成报告；AlphaXiv 通常不提供完整闭环。

### vs DeepCode

| 特性 | PaperBot | DeepCode |
|------|----------|----------|
| **核心定位** | 学者追踪 + 论文分析 + 代码复现 | Paper2Code 生成 |
| **Agent 架构** | 多 Agent 协作（研究/代码/质量/评审/验证） | 模块化 Agent 流水线 |
| **执行后端** | Docker + E2B（可选） | 多为本地执行 |
| **影响力分析** | PIS、引用速度、（可选）引用情感 | 通常不包含 |
| **学者追踪** | 支持 | 通常不包含 |
| **自愈调试** | 支持 | 支持（实现细节不同） |
| **Blueprint/Memory/RAG** | 已集成 | 已集成 |

## 系统架构

![System Architecture](public/asset/arcv2.png)

### Coordinator v2（架构升级）

引入协作与快速失败机制，降低低质量任务的资源消耗：

- **ScoreShareBus**：阶段间评分共享与订阅
- **FailFastEvaluator**：早期拦截低质量/无代码/空壳仓库等情况

### Multi-LLM Backend（多后端架构）

支持多种 LLM 后端与成本路由（由 `ModelRouter` 进行任务类型路由）：

- **OpenAI / 兼容接口**：如 `gpt-4o`、`gpt-4o-mini` 等
- **Anthropic**：如 Claude 系列
- **Ollama**：本地模型（可选）

## 快速开始

### 1) 环境准备

- Python >= 3.8（推荐 3.10）
- Node.js >= 18（可选：Terminal UI / Web）
- Docker（可选：复现验证；或使用 E2B）

安装 Python 依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选依赖：

```bash
pip install jinja2
pip install openreview-py huggingface_hub
```

### 2) 配置环境变量

由于仓库忽略规则限制，示例文件使用 `env.example`：

```bash
cp env.example .env
```

至少配置一个 LLM Key（如 `OPENAI_API_KEY`），否则涉及 LLM 的能力将不可用/自动降级。

### 3) 启动 API 服务器（CLI/Web 都依赖它）

```bash
python -m uvicorn src.paperbot.api.main:app --reload --port 8000
```

已实现端点：

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/track` | GET | 学者追踪（SSE） |
| `/api/analyze` | POST | 论文分析（SSE） |
| `/api/gen-code` | POST | Paper2Code（SSE） |
| `/api/review` | POST | 深度评审（SSE） |
| `/api/chat` | POST | AI 对话（SSE） |

### 4) 运行 Terminal UI（Node CLI）

```bash
cd cli
npm install
npm run build
npm start
```

如后端不在默认地址，设置：

- `PAPERBOT_API_URL=http://<host>:8000`

### 5) 运行 Web Dashboard

```bash
cd web
npm install
npm run dev
```

## 常用命令（Python 入口：`main.py`）

### 学者追踪

```bash
python main.py track --summary
python main.py track
python main.py track --scholar-id 1741101
python main.py track --force
```

### 顶会论文下载

```bash
python main.py --conference ccs --year 23
python main.py --conference sp --year 23
python main.py --conference ndss --year 23
python main.py --conference usenix --year 23
```

### 深度评审 / 声明验证

```bash
python main.py review --title "..." --abstract "..."
python main.py verify --title "..." --abstract "..." --num-claims 5
```

### Paper2Code

```bash
python main.py gen-code --title "..." --abstract "..." --output-dir ./output
python main.py gen-code --title "..." --abstract "..." --use-orchestrator --output-dir ./output
```

### 实验与报告渲染（ExperimentRunner）

```bash
python main.py run-exp --config config/experiments/exp_sentiment.yaml
python main.py render-report --template academic_report.md.j2
```

## Roadmap（Plan 摘要）

- **Phase 1（P0）稳定性与一致性**：收敛重复实现、统一网络层与并发模型、补齐解析契约测试
- **Phase 2（P1）数据与运营能力**：DB 持久化、任务队列/调度、指标与告警、成本治理
- **Phase 3（P2/P3）平台化与企业级治理**：Source Registry、插件化、多租户/权限/审计/配额、可观测性与合规

> 详细评估与可执行计划见：`docs/PLAN.md`。

## 文档索引（Docs）

- **总体计划（Plan）**：`docs/PLAN.md`
- **数据集说明**：`datasets/README.md`
- **Web Dashboard**：`web/README.md`

## 目录结构（完整）

```text
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
│       │   │   ├── router.py          # ModelRouter (成本路由)
│       │   │   └── providers/         # 多后端适配器 (OpenAI, Anthropic, Ollama)
│       │   │       ├── base.py        # LLMProvider ABC
│       │   │       ├── openai_provider.py
│       │   │       ├── anthropic_provider.py
│       │   │       └── ollama_provider.py
│       │   ├── api_clients/           # 外部 API
│       │   ├── storage/               # 存储
│       │   └── services/              # 业务服务
│       │
│       ├── api/                       # FastAPI 后端
│       │   ├── main.py                # FastAPI 应用
│       │   ├── streaming.py           # SSE 流式工具
│       │   └── routes/                # API 路由
│       │       ├── track.py           # 学者追踪
│       │       ├── analyze.py         # 论文分析
│       │       ├── gen_code.py        # Paper2Code
│       │       ├── review.py          # 深度评审
│       │       └── chat.py            # AI 对话
│       │
│       ├── domain/                    # 领域模型层
│       │   ├── paper.py               # PaperMeta, CodeMeta
│       │   ├── scholar.py             # Scholar
│       │   └── influence/             # 影响力计算
│       │       ├── result.py
│       │       ├── calculator.py
│       │       ├── metrics/
│       │       ├── analyzers/         # 动态分析器 (Citation Context, PIS)
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
├── cli/                               # Node.js 终端 UI
│   ├── src/
│   │   ├── index.tsx                  # 入口点
│   │   ├── components/                # React/Ink 组件
│   │   │   ├── App.tsx
│   │   │   ├── ChatView.tsx
│   │   │   ├── TrackView.tsx
│   │   │   ├── AnalyzeView.tsx
│   │   │   └── GenCodeView.tsx
│   │   ├── hooks/                     # React Hooks
│   │   └── utils/                     # 工具函数
│   │       ├── api.ts                 # API 客户端
│   │       └── banner.ts              # oh-my-logo 横幅
│   ├── package.json
│   └── tsconfig.json
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
├── main.py                            # Python 入口点
├── requirements.txt                   # 依赖
└── README.md
```

## 测试

```bash
pytest -q
```

## 致谢

- 感谢 [Qc-TX](https://github.com/Qc-TX) 对爬虫脚本的贡献
- 多 Agent 协作与深度研究流程部分实践参考了 [BettaFish](https://github.com/666ghj/BettaFish) InsightEngine 的公开实现

## License

MIT，见 `LICENSE`。


