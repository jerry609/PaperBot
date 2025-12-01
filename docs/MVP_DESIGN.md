## PaperBot 学者追踪 & 多 Agent 分析 MVP 设计文档

### 1. 项目背景与目标

- **背景**  
  - PaperBot 目前已具备：论文下载、ResearchAgent / CodeAnalysisAgent / QualityAgent / DocumentationAgent 等基础能力。  
  - BettaFish 项目提供了：多 Agent “论坛”协作机制、节点化架构、Report IR 中间表示等成熟实践。  
  - 《学者追踪与影响力分析系统设计方案》明确了：以学者为中心的长期追踪、影响力量化和智能推送需求。

- **MVP 总体目标**  
  在不大规模重构现有 PaperBot 的前提下，交付一个“**学者追踪 + 多 Agent 论文分析报告**”的最小可行版本，满足：
  - 支持少量学者订阅配置；
  - 定期自动发现这些学者的新论文；
  - 通过现有多 Agent 对新论文进行自动分析；
  - 计算一个简化版 PaperBot Impact Score（PIS）；
  - 生成结构化的 Markdown 报告文件并本地存档。

---

### 2. MVP 范围与非目标

#### 2.1 必做范围

- **学者订阅配置**
  - **配置文件**：新增 `config/scholar_subscriptions.yaml`（示例参考 SCHOLAR_TRACKING_DESIGN）：
    - `scholars`: 学者列表（name、semantic_scholar_id、可选 keywords）。
    - `settings`: 检查周期（`daily` / `weekly`）、`min_influence_score`、报告输出目录等。
  - **解析逻辑**：启动时读取 & 校验配置，缺失关键字段直接报错（早失败）。

- **学术元数据获取（优先 Semantic Scholar）**
  - 实现轻量版 `SemanticScholarAgent`：
    - 输入：`semantic_scholar_id`。
    - 输出：该学者最近 N 篇论文的元数据（论文 ID、标题、作者列表、年份、venue、引用数等）。
  - 本地维护论文缓存（JSON）：  
    - `cache/scholar_papers/{scholar_id}.json`  
    - 保存已处理过的论文 ID 列表，用于新论文检测。

- **新论文检测（PaperTrackerAgent）**
  - 对每个被订阅学者：
    - 拉取当前论文列表；
    - 与本地缓存对比，得到 `new_papers`；
    - 将 `new_papers` 传入后续分析流水线；
    - 更新缓存。

- **论文分析流水线（复用现有多 Agent）**
  - 统一由 `WorkflowCoordinator`（MVP 简化版）协调：
    1. **ResearchAgent**：  
       - 输入：论文元数据（标题、作者、venue、年份、URL、ID 等）；  
       - 输出：扩展摘要、初步代码仓库链接（从论文链接、GitHub、PapersWithCode 等获取，MVP 可采用简单启发式）。
    2. **CodeAnalysisAgent**：  
       - 输入：代码仓库链接（若存在）；  
       - 输出：轻量级代码质量与可复现性分析（语言、行数/文件数、star 数、最后更新日期等，必要时调用 LLM 提示进行主观评价）。
    3. **QualityAgent**：  
       - 综合论文信息 + 代码分析结果，生成面向用户的质量&价值评估文字。
    4. **InfluenceCalculator（新增）**：  
       - 基于论文元数据和代码元数据计算简化版 PIS；
       - 输出总分和维度拆解说明。
    5. **DocumentationAgent**：  
       - 根据预设报告模板，生成 Markdown 格式的完整报告文本。

- **影响力评分（简化版 PIS）**
  - **Academic Impact \(I_a\)**（论文本身）：  
    - `citations`: 论文引用数（可直接用 Semantic Scholar 提供值）；  
    - `is_top_conf`: 是否顶会/顶刊（根据 venue 名单简单匹配，如 CCS/S&P/NDSS/USENIX 等）。  
  - **Engineering Impact \(I_e\)**（代码 & 工程）：  
    - `has_code`: 是否存在公开代码仓库（0 / 1）；  
    - `repo_stars`: GitHub Star 数（可做区间映射，例如 0–10、10–100、100+）。  
  - **MVP 综合评分公式（不包含趋势项 \(I_t\)）**：
    \[
    Score = w_1 \cdot I_a + w_2 \cdot I_e
    \]
    - 初始建议：\(w_1 = 0.6, w_2 = 0.4\)，最后线性映射到 0–100。  
  - 输出字段示例：
    - `total_score`: 0–100；  
    - `academic_score`: 学术子分；  
    - `engineering_score`: 工程子分；  
    - `explanation`: 一段简短解释评分来源。

- **报告生成与输出**
  - 输出目录：默认 `reports/`，可通过配置覆盖。  
  - 文件命名：`reports/{scholar_name}/{YYYY-MM-DD}_{paper_id}.md`。  
  - 内容结构（固定模板，方便后续迁移到 IR）：  
    1. 标题：论文标题；  
    2. 元信息：作者、年份、venue、链接、Semantic Scholar ID、arXiv ID（如有）；  
    3. **执行摘要**：3–5 段，对论文核心贡献和适用场景的总结；  
    4. **代码与可复现性**：仓库链接 + 关键指标表 + LLM 评价；  
    5. **影响力评分（PIS）**：总分、维度拆解、简要解释；  
    6. **推荐级别**：例如“强烈推荐深入阅读 / 建议关注 / 可选阅读”等。

- **运行方式与调度**
  - 新增 CLI 入口（可在 `main.py` 中注册）：  
    - 示例：`python main.py track_scholars --config config/scholar_subscriptions.yaml`  
  - CLI 完成一次完整检测与报告生成；  
  - 定时调度通过系统层（如 crontab）实现，MVP 内部不实现复杂 Scheduler。

#### 2.2 非目标（MVP 暂不覆盖）

- 不实现 BettaFish 式多轮“论坛辩论”协作（ForumCoordinator），仅使用顺序流水线。  
- 不对现有 BaseAgent 做全面节点化重构（`nodes/`、`tools/`、`prompts/` 目录拆分仅作为后续规划）。  
- 不落地完整 Report IR 架构与 PDF/HTML 渲染，仅输出 Markdown。  
- 不接入高级中间件（CodeBERT、漏洞检测模型、复杂图分析等）。  
- 不提供 Web/Streamlit UI，仅通过 CLI + 本地文件交付结果。  

---

### 3. 系统架构（MVP）

#### 3.1 模块概览

- **配置管理层**
  - 解析现有 `config/config.yaml`；  
  - 新增 `config/scholar_subscriptions.yaml` 并提供访问接口（如 `get_scholar_subscriptions()`）。

- **学者追踪子系统**
  - **ScholarProfileAgent（轻量）**  
    - 负责：  
      - 读取订阅学者配置；  
      - 管理学者级缓存文件路径；  
    - 核心接口：  
      - `list_tracked_scholars() -> List[Scholar]`  
      - `load_scholar_cache(scholar_id) -> Set[paper_id]`  
      - `save_scholar_cache(scholar_id, paper_ids: Set[str])`
  - **SemanticScholarAgent**  
    - 封装 Semantic Scholar API 调用；  
    - 核心接口：`fetch_papers_by_author(author_id, limit_n) -> List[PaperMeta]`。
  - **PaperTrackerAgent**  
    - 核心职责：新论文检测与触发分析；  
    - 流程：  
      1. 遍历订阅学者；  
      2. 获取当前论文列表；  
      3. 与缓存做差集，得到 `new_papers`；  
      4. 调用分析流水线；  
      5. 合并新论文 ID 并更新缓存。

- **多 Agent 分析子系统（复用）**
  - **ResearchAgent**：补全摘要和潜在代码仓库；  
  - **CodeAnalysisAgent**：轻量代码与工程质量分析；  
  - **QualityAgent**：综合质量评价；  
  - **DocumentationAgent**：报告文案生成。

- **影响力计算子系统**
  - **InfluenceCalculator**（新增模块/类）：  
    - 输入：`paper_meta`、`code_meta`；  
    - 输出：`InfluenceResult`（总分 + 维度分 + 文字解释）。

- **工作流协调层**
  - **WorkflowCoordinator（MVP 版）**：  
    - 统一封装一个高层接口：  
      - `run_paper_pipeline(paper_meta) -> (report_markdown: str, influence_result: InfluenceResult)`  
    - 内部顺序调用 ResearchAgent → CodeAnalysisAgent → QualityAgent → InfluenceCalculator → DocumentationAgent；
    - 简单容错：个别步骤失败时写入日志，并尽量降级继续。

- **输出与通知层**
  - **ReportWriter**：  
    - 负责文件命名、目录创建和 Markdown 写入；  
  - **Notifier（占位）**：  
    - MVP 可仅记录“TODO: email / Slack 通知”，不实际发送。

---

### 4. 典型数据流程（一次执行）

1. **读取配置**
   - 从 `config/scholar_subscriptions.yaml` 中加载学者列表和全局参数；
   - 校验配置合法性（ID 缺失、重复等）。

2. **学者级循环**
   - 对每个学者：
     1. 从缓存加载该学者历史论文 ID 集合；  
     2. 调用 `SemanticScholarAgent` 获取最近 N 篇论文；  
     3. 计算差集，得到 `new_papers` 列表。

3. **论文级流水线**
   - 对每一篇 `new_paper`：
     1. 将 `PaperMeta` 交给 `WorkflowCoordinator.run_paper_pipeline()`；  
     2. 内部依次触发 ResearchAgent / CodeAnalysisAgent / QualityAgent / InfluenceCalculator / DocumentationAgent；  
     3. 获取 `report_markdown` 和 `InfluenceResult`。

4. **过滤与写入**
   - 根据 `InfluenceResult.total_score` 与 `min_influence_score`：  
     - 若分数低于阈值，可选择：  
       - a) 仍生成报告，但标记为“低优先级”；  
       - b) 仅在日志记录，不写报告文件（MVP 可采用 a 或 b 中的一个简单策略）。  
     - 若分数高于阈值：  
       - 通过 `ReportWriter` 写入 `reports/{scholar_name}/{date}_{paper_id}.md`。

5. **更新缓存**
   - 将本次检测出的新论文 ID 并入缓存；  
   - 将新的 ID 集合写回对应的 JSON 文件。

---

### 5. 技术选型与实现约束

- **编程语言与依赖**
  - Python（复用现有 PaperBot 项目环境）；
  - HTTP 客户端：`requests` 或项目内已存在的封装；
  - YAML 解析：`PyYAML`（若已在 `requirements.txt` 中则直接复用）；
  - 本地缓存：JSON 文件，无需额外数据库依赖。

- **外部 API**
  - **Semantic Scholar API**：  
    - 初期只使用公开 author / paper 查询接口；  
    - 简单处理限流和错误，出现异常时记录日志并跳过该次查询。
  - **GitHub API（可选）**：  
    - 用于获取仓库 star/fork 等指标；  
    - 若未配置 Token，则可退化为仅通过 `git ls-remote` 或页面抓取/忽略工程指标。

- **日志与容错**
  - 复用 `utils/logger.py`（如存在）；  
  - 关键事件均记录日志：  
    - 每次运行的学者数、新论文数；  
    - 每篇论文的最终 PIS 评分和报告路径；  
    - API 调用失败和降级信息。

---

### 6. 迭代计划与验收标准

#### 6.1 开发阶段划分

- **阶段 1：学者追踪骨架（约 1–2 天）**
  - 完成 `scholar_subscriptions.yaml` 解析与校验；
  - 实现 `ScholarProfileAgent` + `SemanticScholarAgent` + `PaperTrackerAgent`；
  - 能在日志中输出：每个学者的新论文列表。

- **阶段 2：多 Agent 分析对接（约 2–4 天）**
  - 打通 `new_papers` 与 Research/CodeAnalysis/Quality/Documentation Agents 的流水线；
  - 实现 `WorkflowCoordinator.run_paper_pipeline`；
  - 对单篇论文可完整生成 Markdown 报告文件。

- **阶段 3：影响力评分与过滤（约 1–2 天）**
  - 实现 `InfluenceCalculator` 简化公式；
  - 接入 `min_influence_score` 配置，并在流水线中应用。

- **阶段 4：文档与打磨（约 1–2 天）**
  - 在 `docs/` 或项目主 `README` 中添加“学者追踪使用指南”；
  - 对 2–3 位真实学者进行试运行，验证管线稳定性和报告可读性。

#### 6.2 MVP 验收标准

- **功能性**
  - 至少支持 1 名学者订阅；
  - 在学者产生新论文时，系统能够在一次运行中识别出新论文，并为每篇新论文生成一份 Markdown 报告；
  - 报告中包含：摘要、代码链接（如有）、基本代码指标、简化 PIS 评分与简短解释。

- **可用性**
  - 用户无需修改代码，仅通过编辑 `scholar_subscriptions.yaml` 即可增删学者；
  - 在 API 出错或单篇论文处理失败时，系统不会整体崩溃，而是略过问题项并继续处理其他项。

- **可扩展性准备**
  - 模块边界清晰，后续可以在不改动外部接口的前提下：  
    - 将评分模型替换为更复杂的多维指标；  
    - 引入 ForumCoordinator，实现多 Agent 论坛辩论；  
    - 引入 Report IR 架构和 PDF/HTML 渲染；  
    - 接入更多数据源（DBLP、PapersWithCode、Twitter/X 等）。


