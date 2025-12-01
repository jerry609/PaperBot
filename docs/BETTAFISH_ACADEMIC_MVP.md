## BettaFish 学术版（“学术版微舆”）MVP 设计文档

### 1. 项目背景与目标

- **原项目定位（舆情产品版）**  
  BettaFish 是一个多 Agent 舆情分析系统，基于 `QueryEngine / MediaEngine / InsightEngine / ReportEngine / MindSpider` 以及 ForumEngine “论坛”协作机制，可面向企业/机构做实时舆情分析、品牌声誉监测和决策支持。

- **学术版定位（学术版微舆）**  
  面向课题组 / 实验室 / 教学场景的“学术研究版”，重点支持：
  - **可复现实验**：同一数据与配置下，多人多机可得到一致或高度一致结果；
  - **合规数据使用**：以公开数据集或内部审核数据为主，实时爬虫变为可选；
  - **论文/课程友好**：能直接产出论文风格的图表与报告结构；
  - **对比实验便利**：可方便切换模型、Prompt、多 Agent 策略并记录结果。

- **MVP 总体目标**  
  在不大改 BettaFish 现有架构的前提下，交付一条“**从数据集 → 实验配置 → 多模型评测 → 学术报告**”的最小闭环，并将 MindSpider 从“强依赖”降级为“可选数据生产工具”。

---

### 2. MVP 核心诉求与范围

#### 2.1 学术版核心诉求

1. **可复现**  
   - 同一数据集 + 同一 YAML/JSON 配置，在不同环境重跑，结果差异可控；  
   - 实验配置、数据版本、模型版本可被记录和溯源。

2. **合规数据源优先**  
   - 默认使用本地/数据库中的 **静态数据集**（`datasets/processed`）；  
   - MindSpider 作为可选爬虫模块，默认不自动运行，文档中强调合规与 ToS。

3. **论文导向的报告能力**  
   - 内置 IMRaD 风格的报告模板（摘要、引言、方法、结果、讨论、结论）；  
   - 可直接插入表格与图表（情感指标、时间序列、平台对比等）。

4. **对比实验能力**  
   - 通过配置切换：LLM、情感模型、Prompt 策略、多 Agent 协作模式；  
   - 实验结果统一记录，便于做 ablation / baseline 对比。

#### 2.2 MVP 必做范围

- 引入 **数据集优先** 的数据层（`datasets/` + DataSource 抽象）；  
- 将 MindSpider 调整为“数据生产工具”，支持导出数据集而非直接驱动系统；  
- 为 InsightEngine / QueryEngine / ReportEngine 增加 **学术模式** 与对应 Prompt / 模板；  
- 新增 `ExperimentManager` 子模块，用 YAML 配置驱动实验并记录指标；  
- 实现一个完整的 **情感模型对比实验** + **学术报告生成** Demo。

#### 2.3 非目标（MVP 暂不覆盖）

- 不全面重写 UI，仅增设简单的“舆情模式 / 学术模式”切换；  
- 不做大规模重构（如全面 nodes/tools/prompts 拆分）；  
- 不引入复杂的实验管理平台（如完整版 MLFlow），只做轻量实现；  
- 不新增复杂外部数据源（Twitter/X、Google Trends 等），仍以现有能力为主。

---

### 3. 系统改造概览

MVP 改造分为三大部分：

1. **数据层改造：从在线爬虫到可控数据集**  
   - 新增 `datasets/` 目录，用于存放原始/清洗后的数据集与元信息；  
   - 在 `InsightEngine/tools/search.py` 新增 **DataSource 抽象**：
     - `BaseDataSource` / `DBDataSource` / `LocalFileDataSource` / `HybridDataSource`；  
   - 在配置中引入 `DATA_SOURCE_TYPE` 和 `DATASET_NAME`，默认 `local`；
   - 为 MindSpider 增加导出脚本，将采集数据转为静态数据集。

2. **Agent & Prompt 改造：从舆情报告到学术报告**  
   - InsightEngine：增加 `mode="academic"`，在流程中显式生成 RQ、假设和方法；  
   - Sentiment 模型：通过配置在学术模式下支持多模型对比，输出统一指标表；  
   - QueryEngine / MediaEngine：增加可选的文献检索和小规模多模态案例分析；  
   - ReportEngine：新增学术报告模板，支持 IMRaD 结构与学术图表。

3. **实验管理 & 评测：ExperimentManager 最小闭环**  
   - `ExperimentManager/configs/*.yaml` 描述实验（数据集、模型、指标、种子等）；  
   - `runner.py` 统一读取配置，驱动数据加载、模型调用和指标计算；  
   - 输出 `results.csv` + `meta.json`，并与 ReportEngine 打通生成学术报告。

---

### 4. 数据层 MVP 设计

#### 4.1 目录与数据组织

```bash
datasets/
  ├── raw/           # 原始数据（MindSpider 导出或外部数据）
  ├── processed/     # 清洗后的数据（去敏、去重、打标签）
  └── metadata/      # 元信息（来源、采集时间、字段说明、许可）
```

- `datasets/metadata/{dataset_name}.yaml` 示例字段：
  - `name`、`description`、`source`、`collection_time`、`license`；  
  - 字段含义（列名 -> 说明）、标签体系说明等。

#### 4.2 DataSource 抽象

- 在 `InsightEngine/tools/search.py` 新增：
  - `BaseDataSource`：定义统一接口，如 `query(query_params)`、`get_comments(topic_id, limit)`；  
  - `DBDataSource`：封装现有 PostgreSQL/MySQL 查询逻辑；  
  - `LocalFileDataSource`：从 `datasets/processed/*.csv` / `*.parquet` 中加载数据；  
  - `HybridDataSource`：按配置决定部分数据来自 DB，部分来自本地文件。

- 在 `InsightEngine/utils/config.py` 中新增配置：

```python
DATA_SOURCE_TYPE = "local"           # local / db / hybrid
DATASET_NAME = "weibo_political_2020"
```

#### 4.3 MindSpider 角色调整

- 保留 `MindSpider/` 目录和功能，但对学术版做两点约定：
  - 默认不自动启动爬虫， README-学术版中必须有合规提示；  
  - 新增脚本：
    - `scripts/export_mindspider_to_datasets.py`：将 DB 数据导出到 `datasets/raw/*.jsonl`；  
    - `scripts/clean_and_label_dataset.py`：完成文本预处理、脱敏和标签生成，写入 `datasets/processed/`。

---

### 5. Agent 与 Prompt 学术化设计

#### 5.1 InsightEngine：数据分析管线

- 在 `InsightEngine/prompts/prompts.py` 中新增学术风格 Prompt 集合：
  - 用于生成：
    - 研究问题（RQ）；  
    - 假设（Hypothesis）；  
    - 方法说明（Methodology）；  
    - 限制与未来工作（Limitations & Future Work）。
  - 通过 `PROMPT_STYLES` 切换：

```python
PROMPT_STYLES = {
    "opinion_monitoring": "...",  # 原舆情模式
    "academic_analysis": "...",   # 学术模式
    "teaching_demo": "...",       # 教学示例
}
```

- 在 `InsightEngine/agent.py` 中：
  - 增加 `mode="academic"` 分支；  
  - 在主流程前插入“生成 RQ + 分析框架”的节点：
    - 将用户自然语言需求转为若干 RQ（如平台差异、时间演化等）；  
    - 后续情感分析、主题分析围绕这些 RQ 组织输出。

- 与 Sentiment 模型结合：
  - 在 `InsightEngine/tools/sentiment_analyzer.py` 中引入实验配置：

```python
SENTIMENT_CONFIG = {
    "experiments": [
        {"name": "bert_lora", "model_type": "bert", "checkpoint": "..."},
        {"name": "qwen_small", "model_type": "qwen", "checkpoint": "..."},
        {"name": "ml_baseline", "model_type": "ml", "checkpoint": "..."},
    ],
    "metrics": ["accuracy", "f1_macro", "f1_pos", "f1_neg"],
}
```

- 输出时，将各模型在同一数据集上的结果整理成 **实验结果表**，直接供论文使用。

#### 5.2 QueryEngine & MediaEngine：检索与多模态案例

- **QueryEngine（文献/案例检索扩展）**  
  - 在 `tools/` 中增加 `literature_search.py`（可选）：
    - 从本地 `papers/` 或导出的 BibTeX/向量库中检索文献；  
    - 返回包含 `title / authors / year / venue / abstract / url` 的列表；  
  - Prompt 要求生成规范引用格式（APA/MLA），便于论文撰写。

- **MediaEngine（小规模多模态分析）**  
  - 保留现有多模态能力，但在学术文档中强调：  
    - 针对少量图片/视频案例，而非大规模平台抓取；  
    - 适合作为教学示例或案例研究补充材料。

#### 5.3 ReportEngine：学术报告模板

- 在 `ReportEngine/report_template/` 新增：

```bash
report_template/
  ├── 学术研究报告_社会舆情.md
  ├── 学术研究报告_多模态分析.md
  └── 课程作业报告模板.md
```

- 模板结构遵循 IMRaD：
  - 摘要；  
  - 引言（研究背景、相关工作）；  
  - 方法（数据集、预处理、模型、实验设计）；  
  - 结果（表格 + 图表）；  
  - 讨论（发现、启示、局限）；  
  - 结论与未来工作。

- 在 `nodes/template_selection_node.py` 中：
  - 增加 `report_style` 字段（`"academic"` / `"business"`）；  
  - 当 `report_style="academic"` 时，仅在学术模板中进行选择。

- 图表输出：
  - 模板保留图表占位符，由 `renderers/chart_to_svg.py` 渲染：  
    - 情感比例条形图；  
    - 情感随时间变化的折线图；  
    - 平台/群体对比柱状图；  
  - MVP 阶段可先实现基础类型，词云等高阶可后续扩展。

---

### 6. 实验管理（ExperimentManager）MVP

#### 6.1 目录结构

```bash
ExperimentManager/
  ├── configs/
  │   ├── exp_sentiment_model_compare.yaml
  │   └── exp_topic_model_compare.yaml
  ├── runner.py       # 统一读取配置，调度 Engine/模型
  ├── metrics.py      # 统一计算/保存指标
  └── logger.py       # 记录日志、随机种子、版本信息
```

#### 6.2 配置示例（情感模型对比）

```yaml
experiment_name: "weibo_covid_sentiment_2020"
dataset: "weibo_covid_2020"
task: "sentiment_classification"
models:
  - name: "bert_lora"
    type: "bert"
    checkpoint: "SentimentAnalysisModel/WeiboSentiment_Finetuned/BertChinese-Lora"
  - name: "qwen_small"
    type: "qwen"
    checkpoint: "SentimentAnalysisModel/WeiboSentiment_SmallQwen"
  - name: "ml_baseline"
    type: "ml"
    checkpoint: "SentimentAnalysisModel/WeiboSentiment_ML"
metrics:
  - accuracy
  - f1_macro
  - f1_pos
  - f1_neg
seed: 42
output_dir: "experiments/weibo_covid_sentiment_2020/"
```

#### 6.3 Runner 责任

1. 读取 YAML 配置；  
2. 根据 `dataset` 调用 DataSource 加载 `datasets/processed/` 中的数据；  
3. 对每个模型：
   - 初始化对应情感模型；  
   - 运行预测并收集标签；  
4. 调用 `metrics.py` 计算所有指定指标；  
5. 将结果写入：
   - `results.csv`（每模型一行）；  
   - `meta.json`（时间、Git 提交、依赖版本、随机种子等）。

这些结果再交给 ReportEngine，生成包含“实验配置 + 指标表 + 图表 + 讨论”的学术报告。

---

### 7. 根目录与前端改造要点

- **配置与环境**
  - 在全局 `config.py` 中新增：

```python
PROJECT_MODE = "academic"          # academic / production
DEFAULT_REPORT_STYLE = "academic"
DATA_SOURCE_TYPE = "local"         # local / db / hybrid
```

  - 在 `.env.example` 中添加对应环境变量说明；
  - 在 README 中新增“Academic Edition / 学术版说明”章节，说明：
    - 数据集优先、本地分析、多模型对比等特点；  
    - 爬虫合规性提示；  
    - 如何运行学术 demo。

- **前端 / 接口（如 Flask + templates）**
  - 在 UI 层增加一个模式切换：
    - “舆情分析模式” vs “学术研究模式”；  
  - 学术模式下，额外显示：
    - 研究问题（可选文本）；  
    - 目标数据集选择器；  
    - 是否生成长篇学术报告的开关。

---

### 8. 实施阶段与验收标准

#### 8.1 实施阶段

- **阶段 1：数据 & 配置改造（约 1–2 周）**
  - 建立 `datasets/` 目录与 metadata 约定；  
  - 完成 DataSource 抽象和 `DATA_SOURCE_TYPE` 参数接入；  
  - 增加 MindSpider → datasets 导出与清洗脚本。

- **阶段 2：学术 Prompt 与报告模板（约 1–2 周）**
  - 为 InsightEngine/QueryEngine 增加学术 Prompt；  
  - 在 ReportEngine 中完成至少 1 套学术报告模板；  
  - 手动构造一批结果，验证报告生成质量。

- **阶段 3：ExperimentManager MVP（约 1–2 周）**
  - 搭建 ExperimentManager 目录与 `runner/metrics/logger`；  
  - 跑通一个完整的“情感模型对比”实验；  
  - 将结果对接学术报告模板。

- **阶段 4：前端 & 文档打磨（约 1–2 周）**
  - 前端增加学术模式入口；  
  - 在 README / 文档中写清楚学术版使用方法；  
  - 对 1–2 个真实数据集进行端到端试跑并记录案例。

#### 8.2 MVP 验收标准

- **功能性**
  - 至少 1 个数据集可通过 `DATASET_NAME` 加载；  
  - 至少 2–3 个情感模型可在该数据集上统一评测；  
  - 一条命令可完成：加载数据 → 跑实验 → 生成结果文件与学术报告。

- **可复现性**
  - 在至少两台机器上，使用同一配置重跑实验，结果在可接受误差范围内一致；  
  - `meta.json` 中记录了足够的信息（数据集版本、模型版本、随机种子、Git 提交）以复现实验。

- **论文友好度**
  - 报告中包含：数据集介绍、模型配置、量化指标表、至少 1 张图表和文字讨论；  
  - 报告结构满足常见学术论文/课程作业要求，可直接作为草稿基础。

- **扩展性准备**
  - 通过新增 YAML 配置即可扩展到新的数据集或模型，而无需大改代码；  
  - Agent / DataSource / Report 模块边界清晰，后续可逐步引入更多任务与中间件。


