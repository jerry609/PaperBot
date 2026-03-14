# Codex Agent 循环迭代上限与自适应预算设计方案（中文版）

## 1. 背景与问题

当前系统在 Codex 工具循环中使用固定 `max_iterations` 作为安全上限。该机制必要，但在两类场景下体验不理想：

1. **假性未完成**：任务持续有进展，但在临近上限时被硬截断，出现  
   `Agent loop did not finish within N iterations.`
2. **无效消耗**：任务明显停滞（重复调用或无新增产出）仍继续迭代，浪费时间与成本。

目标是构建“**硬上限 + 自适应 + 可诊断**”的混合策略：  
既保留确定性安全边界，也让系统自动判断“该提前停”还是“可小幅延长”。

---

## 2. 设计目标

1. **保留硬保护**：任何情况下都必须受最大迭代上限约束。
2. **提高完成率**：在“持续进展”场景下，允许有界自动延长，减少误判失败。
3. **降低无效循环**：在“明显停滞”场景下，提前终止并给出明确原因。
4. **可观测可解释**：失败信息应包含 reason code、关键计数器、最后步骤摘要。
5. **最小改动复用**：优先复用现有 `CodexDispatcher`、`agent_board`、`on_step`、任务日志链路。

---

## 3. 现状与可复用资产

### 3.1 后端执行链路

1. `CodexDispatcher` 已实现工具循环主逻辑（普通/沙箱两条路径）。
2. `dispatch_with_tools` 与 `dispatch_with_sandbox_tools` 都支持 `max_iterations` 入参。
3. `agent_board` 通过 `_resolve_codex_max_iterations()` 读取环境变量并传给执行层。
4. `ExecutorAgent.execute(..., max_iterations=25)` 已向下透传。

### 3.2 现有防循环能力

1. `MAX_REPEAT_TOOL_CALLS`：重复相同工具调用阈值。  
2. `MAX_TOOL_ERRORS`：工具错误次数阈值。  
3. 达到迭代上限时统一报错：`Agent loop did not finish within N iterations.`

### 3.3 可复用事件与日志链路

1. `on_step`、`on_think` 已接入任务日志与 SSE。  
2. `TaskDetailPanel` 已展示 `task.lastError` 与执行日志，可直接承接更细粒度错误信息。  

### 3.4 现有测试基础

1. `tests/unit/test_codex_tool_loop.py` 已覆盖“触达迭代上限”场景。  
2. `test_executor_agent.py`、`test_verification.py`、`test_e2e_execution.py` 已覆盖 dispatcher 调用形态。

---

## 4. 方案总览（混合预算策略）

引入三层控制：

1. **硬上限（Hard Cap）**  
   继续使用 `CODEX_MAX_ITERATIONS`，永不取消。

2. **停滞提前终止（Stagnation Early Stop）**  
   连续 N 步“无有效进展”则提前失败，避免空转。

3. **有界自动扩展（Bounded Auto-Extend）**  
   临近上限但仍有持续进展时，按小步扩展迭代预算（受总扩展上限约束）。

---

## 5. 详细设计

### 5.1 新增策略对象（复用 Policy 模式）

在 `codex_dispatcher.py` 新增：

1. `ToolLoopPolicy`（建议 dataclass）
2. `ToolLoopPolicy.from_env(requested_max_iterations)`  

字段建议：

1. `hard_max_iterations: int`
2. `auto_budget_enabled: bool`（默认 `false`）
3. `stagnation_steps: int`（默认 6~8）
4. `auto_extend_steps: int`（默认 5）
5. `max_total_extension: int`（默认 20）
6. `near_limit_window: int`（默认 3）

环境变量建议：

1. `CODEX_MAX_ITERATIONS`（已有）
2. `CODEX_LOOP_AUTO_BUDGET`
3. `CODEX_LOOP_STAGNATION_STEPS`
4. `CODEX_LOOP_AUTO_EXTEND_STEPS`
5. `CODEX_LOOP_AUTO_EXTEND_MAX`
6. `CODEX_LOOP_NEAR_LIMIT_WINDOW`

> 说明：默认保持与现状一致（仅硬上限），自适应开关默认关闭，避免行为突变。

### 5.2 新增进展跟踪器（LoopProgressTracker）

在 dispatcher 内新增轻量跟踪器（无需新模块）：

1. `consecutive_no_progress_steps`
2. `last_progress_step`
3. `unique_tool_signatures_seen`
4. `files_written_count_last`
5. `total_extensions_used`

“有效进展”判定（只用现有信号）：

1. `write_file` 且写入成功
2. `update_subtask(done=true)`
3. `run_command` 且返回成功（`exit_code: 0`）
4. 新的工具签名（非重复）
5. `task_done`（直接完成）

### 5.3 循环控制逻辑

每个 step 后执行：

1. 更新进展状态
2. 若 `consecutive_no_progress_steps >= stagnation_steps`：提前失败  
   - reason: `stagnation_detected`
3. 若达到原始上限附近且存在持续进展，且 `auto_budget_enabled=true`：  
   - 增加可用步数（`auto_extend_steps`）  
   - 但 `total_extensions_used <= max_total_extension`
4. 最终仍未完成则保留原有硬上限失败  
   - reason: `max_iterations_exhausted`

### 5.4 统一两条循环路径（减少重复实现）

现有 `dispatch_with_tools` 与 `dispatch_with_sandbox_tools` 存在高度重复。  
建议抽取共享私有函数，例如：

1. `_run_tool_loop_core(...) -> CodexResult`

把差异通过参数注入：

1. tools 列表
2. tool_executor 对象
3. system prompt 构造函数
4. 是否记录 cache metrics
5. on_step/on_think 回调

这样可确保新策略在两条路径一致生效。

### 5.5 结构化失败信息（可诊断）

建议保留 `CodexResult.error` 文本兼容，同时增加结构化诊断字段（任选其一）：

1. 扩展 `CodexResult` 增加 `diagnostics: Dict[str, Any]`
2. 或在 `error` 里拼接标准化 JSON 片段（不推荐）

`diagnostics` 示例字段：

1. `reason_code`：`stagnation_detected` / `max_iterations_exhausted` / `repeat_tool_calls` / `too_many_tool_errors`
2. `hard_max_iterations`
3. `effective_iterations_budget`
4. `steps_executed`
5. `consecutive_no_progress_steps`
6. `tool_error_count`
7. `repeat_call_count`
8. `last_tool_name`

### 5.6 agent_board 与前端展示（复用现有链路）

后端：

1. 在失败分支将 `reason_code` 写入 `_append_task_log(..., details=...)`
2. SSE `progress` 事件附加 `failure_reason`、`diagnostics`

前端：

1. `TaskDetailPanel` 优先展示友好文案：
   - `stagnation_detected` -> “长时间无有效进展，已自动停止”
   - `max_iterations_exhausted` -> “达到迭代上限，任务未完成”
2. 保留原始 `lastError` 作为可展开详情

---

## 6. 兼容性与风险控制

1. 默认行为不变：自适应功能默认关闭。
2. `max_iterations` 参数签名保持不变，调用方无需改造。
3. 失败 reason 增强为增量信息，不破坏旧前端读取逻辑。
4. 若策略解析失败，回落到当前硬上限逻辑。

---

## 7. 测试计划

### 7.1 单元测试（dispatcher）

在 `test_codex_tool_loop.py` 新增：

1. `stagnation` 提前终止用例（多步无进展）
2. 近上限且有进展时自动扩展成功完成
3. 自动扩展达到上限后仍失败
4. reason_code 与 diagnostics 字段断言

### 7.2 路由/集成测试（agent_board）

新增/扩展：

1. 失败 diagnostics 可进入 task log details
2. SSE 事件包含 `failure_reason`
3. 前端映射字段不缺失（最小 contract 验证）

### 7.3 回归测试

确保现有测试继续通过：

1. max iterations 失败断言
2. timeout 行为
3. on_step 回调行为
4. executor agent 调用签名兼容性

---

## 8. 分阶段落地

### Phase 1（推荐先做）

1. 引入 `ToolLoopPolicy` + `LoopProgressTracker`
2. 增加 `stagnation_detected` 与 diagnostics
3. 自适应扩展逻辑先不启用（或仅在测试环境启用）

### Phase 2

1. 打开 `CODEX_LOOP_AUTO_BUDGET`（预发）
2. 观察指标：完成率、平均步数、失败原因分布、触顶率、超时率

### Phase 3

1. 根据数据调参后生产开启
2. 再评估是否调整默认 `CODEX_MAX_ITERATIONS`

---

## 9. 验收标准

1. 硬上限始终有效。
2. 停滞任务可提前结束，且原因明确。
3. 持续进展任务可在有界扩展下完成。
4. 旧调用方与旧前端不因接口变化中断。
5. 新增测试覆盖关键分支并通过 CI。

---

## 10. 关键结论

“手动硬上限”不是与“智能评估”冲突，而是其安全边界。  
最优实践是：**硬上限 + 自适应预算 + 停滞检测 + 结构化诊断**。  
本方案可在复用当前架构的前提下实现上述能力，风险低、落地快。

