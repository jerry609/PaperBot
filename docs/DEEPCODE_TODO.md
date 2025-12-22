# DeepCode TODO (PaperBot)

本文件是 DeepCode（论文 → 可运行工程 → 可验证证据）的长期迭代清单。每完成一个功能，请把对应条目从 `[ ]` 更新为 `[x]`，并在 “Progress Log” 追加一行记录（日期 + 简述 + 关联文件/接口/commit）。

## North Star（不变的验收标准）

- **可复制命令**：任意“成功”都必须有可复制的命令与参数。
- **环境锁定**：能够复现相同依赖/镜像/版本（至少记录；后续支持 lockfile）。
- **证据齐全**：logs + metrics + artifacts + diff + step 状态均可追溯。
- **可回滚**：所有改动可回滚到某次 run/step 的快照。
- 指标：Repro Success Rate / Time to First Running / Evidence Coverage

## 当前已完成（已落地）

- [x] Studio 四分栏（Workspace / Runbook / Timeline / Blueprint）+ 小屏 Tab 化：`web/src/app/studio/page.tsx`
- [x] 四分栏可折叠/展开（按钮 + 仍支持拖拽）：`web/src/app/studio/page.tsx`
- [x] Runbook：Paper2Code（SSE 进度 → Timeline）：`web/src/components/studio/RunbookPanel.tsx`、`web/src/app/api/gen-code/route.ts`
- [x] Runbook：Smoke（docker/e2b 可选）+ 日志流入 Timeline：`web/src/components/studio/RunbookPanel.tsx`、`src/paperbot/api/routes/runbook.py`
- [x] Smoke 记录落库（Run + Step）：`src/paperbot/infrastructure/stores/models.py`、`src/paperbot/api/routes/runbook.py`

---

## v1（Studio 扎实化）— 目标：Evidence Coverage 基础闭环

### 1) 数据模型与存储（Run / Step / Artifact / Snapshot）

- [x] `RunbookStepModel`（runbook_steps）表：step 生命周期与配置
- [ ] 新增 `ArtifactModel` 表（产物索引）
  - [ ] 字段：`run_id`、`step_id`(可空)、`type`(log/metric/report/file/zip)、
    `path_or_uri`、`mime`、`size_bytes`、`sha256`、`metadata_json`、`created_at`
  - [ ] DoD：可记录 outputDir、报告、曲线图、导出的 evidence 包
- [ ] 新增 `SnapshotModel`（可选，或复用 Artifact）
  - [ ] 定义：对 workspace 文件的快照/patch（用于回滚）
  - [ ] DoD：至少能把 “本次 step 修改的 diff” 挂到 run/step

### 2) Runbook Step 模型（前端）

- [ ] 统一 Step 状态机（pending/running/success/failed/error/skipped）
  - [ ] 前端类型：StepDefinition、StepRun、StepResult、StepParams
  - [ ] DoD：Runbook 不是堆卡片，而是 “可配置步骤列表”
- [ ] 增加固定步骤卡（先跑通 smoke，逐步扩展）
  - [ ] [x] Smoke（已具备 docker/e2b + logs）
  - [ ] Install：识别 requirements/pyproject，选择网络策略（docker 默认禁网）
  - [ ] Data：数据下载/缓存/mini-sample（先 stub）
  - [ ] Train(Mini)：小批量、短 epoch（后续 GPU）
  - [ ] Eval：输出指标/表格（最少写入 artifacts）
  - [ ] Report：生成 evidence summary（markdown/json）
- [ ] Step 参数面板（seed/batch/device/fp16/timeout）
  - [ ] DoD：参数可保存到 run.metadata/step.metadata

### 3) Workspace：真实文件树 + Diff Staging（从“虚拟编辑器”升级）

- [ ] 后端：文件系统 API（严格限制在允许目录）
  - [ ] `GET /api/runbook/projects`（可选：列出最近项目/输出目录）
  - [ ] `GET /api/runbook/projects/{id}/files`（树）
  - [ ] `GET /api/runbook/projects/{id}/file?path=...`（读）
  - [ ] `POST /api/runbook/projects/{id}/file`（写：只允许 workspace 内）
  - [ ] 安全：路径规范化、防穿越、白名单根目录
- [ ] 前端：文件树组件（左侧或 Workspace 顶部）
  - [ ] 支持搜索/最近文件/仅显示改动
- [ ] Diff Staging（必须）
  - [ ] 生成 diff（基于“原始快照 vs 当前”）
  - [ ] Apply/Reject（按文件/按 hunk）
  - [ ] 一键回滚到某次 step 的快照

### 4) Evidence Timeline：按 Run/Step 组织 + 过滤

- [ ] Timeline 按 `run_id` 分组，并展示 step 节点（Smoke/Install/…）
- [ ] 过滤器：失败/警告/只看某 step/只看某 executor
- [ ] 支持点击 step → 跳转到 Runbook 对应卡片 + 展开 logs

### 5) EvidencePack（导出/共享）

- [ ] `POST /api/runbook/runs/{run_id}/export`（返回 zip 或生成 artifact）
  - [ ] 包含：run.json、steps.json、logs、metrics、patch/diff、关键产物索引
  - [ ] DoD：能把一个 run 的证据打包给导师/同学复现

---

## v2（空闲 GPU + 队列化）— 目标：夜间自动跑、白天看结果

### 1) Queue 化（ARQ）

- [ ] 将 step 执行改为队列任务（Smoke/Install/Train…）
- [ ] 任务状态与重试：pending/running/completed/failed + retry/backoff
- [ ] UI：Repro Queue 页面（队列、占用、失败聚类）

### 2) Executor 扩展（Local + SSH Fleet）

- [ ] 新增 executor：`ssh_docker`
  - [ ] 远端 runner：rsync 项目 → docker run → 拉回 artifacts/logs
  - [ ] 远端资源探测（可选：nvidia-smi / docker info）
- [ ] 统一 executor 能力矩阵（docker/e2b/ssh_docker）

### 3) GPU Scheduler（策略）

- [ ] 本机：读取 `nvidia-smi`，低占用阈值触发
- [ ] 远端：节点心跳 + 资源上报
- [ ] Admission control（min_vram/timeout/interruptible）
- [ ] Preemption（train 可暂停，优先跑 smoke）

---

## v3（Brief/Trends/Profile）— 目标：让系统懂你的方向并自动筛选

### 1) Research Profile（显式 + 隐式）

- [ ] Profile 编辑：主题树/偏好/禁忌/常用数据集&指标/目标 venue
- [ ] 隐式反馈：收藏/不相关/加入复现/阅读时长（形成权重）

### 2) Daily Brief

- [ ] 订阅源：关键词/作者/venue/类别/排除词
- [ ] 去重/合并（同论文多来源）
- [ ] 质量打分：相关性×新颖×影响×可复现性信号
- [ ] 一键加入 Repro Queue（夜间跑 smoke）

### 3) Trends

- [ ] 主题聚类（按周/月）
- [ ] 上升词/代表作/代表作者
- [ ] 可复现性地图（能跑通的比例、常见失败类型）

---

## Progress Log（每次完成请追加）

- YYYY-MM-DD: ...
