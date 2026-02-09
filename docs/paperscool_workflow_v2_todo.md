# Papers.cool Workflow V2 TODO（UI + 可插拔源 + DailyPaper + Scheduler）

在已完成 MVP（单源检索、聚合、摘要、CLI/API）的基础上，V2 已全部落地（可插拔源、DailyPaper、Scheduler/Feed、Web UI）。

## 1) 可插拔源注入（Source Injection）

- [x] 抽象统一的 `TopicSearchSource` 接口
- [x] 增加 `TopicSearchSourceRegistry`（注册/创建/列出源）
- [x] 将现有 `papers.cool` 适配为默认 Source
- [x] workflow 支持 `sources=[...]` 选择源
- [x] CLI/API 新增 `sources` 参数

**验收**
- 不改 workflow 主逻辑即可注入新源
- 默认行为与当前 MVP 兼容

## 2) DailyPaper 风格日报输出

- [x] 定义日报结构（JSON）
- [x] 实现 Markdown 渲染器
- [x] 支持 `markdown/json/both` 输出格式
- [x] 支持写盘（含文件命名策略）
- [x] 提供 CLI/API 调用入口

**验收**
- 可一键生成当日摘要与 Top 论文
- JSON 可程序消费，Markdown 可直接阅读

## 3) Scheduler / Feed Pipeline 接入

- [x] ARQ 新增 `daily_papers_job`
- [x] ARQ 新增 `cron_daily_papers`（可配置开启、时间）
- [x] 将日报高亮项桥接为 feed 推荐事件
- [x] 记录 event log（start/result/error）

**验收**
- 定时执行后有可追踪事件
- feed 中可看到 DailyPaper 推荐条目

## 4) UI 集成（当前 Web）

- [x] 新增 Workflow 页面（参数配置 + 执行按钮 + 结果可视化）
- [x] 新增结果面板（查询高亮、Top 论文、日报预览）
- [x] 新增 API 代理路由（search / daily）
- [x] 侧栏新增入口

**UI 设计原则（MVP）**
- 先做“参数化工作流面板 + 执行结果面板”
- 暂不做 n8n/coze 式全自由拖拽编排（成本高，后续迭代）
- 预留节点化扩展点：Source -> Search -> Rank -> Digest -> Schedule

## 5) 交付提交节奏

- [x] Commit A: docs - V2 todo
- [x] Commit B: feat - source injection
- [x] Commit C: feat - dailypaper output + cli/api
- [x] Commit D: feat - scheduler/feed integration
- [x] Commit E: feat(web) - ui integration
