# Phase 2B: 并行执行 + 依赖 DAG & Phase 2E: KV-Cache 优化 — 实施计划

> 日期：2026-03-11
> 基于：`docs/proposals/agent-board-vs-manus-analysis.md` v2

---

## Phase 2B：并行执行 + 依赖 DAG

### 现状分析

1. **依赖数据已存在但被丢弃**：`ClaudeCommander.decompose()` 明确要求 Claude 返回 `"dependencies": list of task titles`（`claude_commander.py:94`），但 `_plan_stream()` 在构建 `AgentTask` 时从未读取此字段（`agent_board.py:710-721`），Stage 1.5 fallback 中更是硬编码 `"dependencies": []`（`agent_board.py:860`）。

2. **执行循环完全顺序**：`_run_all_stream_sandbox` 使用 `for i, task in enumerate(planning_tasks)`（`agent_board.py:890`），每个任务 await 完成后才开始下一个。

3. **并行基础设施已存在但未使用**：`CodexDispatcher.dispatch_parallel()` 方法存在（`codex_dispatcher.py:384-389`），使用 `asyncio.gather`。但该方法调用旧版 `dispatch()` 而非 `dispatch_with_sandbox_tools()`，需适配。

4. **共享沙箱并发安全**：所有任务共享同一 VM（`SharedSandbox`）。并行 `write_file` / `run_command` 需要注意路径冲突。论文复现中，不同任务通常操作不同文件/目录，冲突风险低。

---

### 实施步骤

#### Step 1：AgentTask 模型添加 `dependencies` 字段

**文件**：`src/paperbot/api/routes/agent_board.py`

```python
class AgentTask(BaseModel):
    # ... 现有字段 ...
    dependencies: List[str] = Field(default_factory=list)   # 新增：依赖的任务 title 列表
```

#### Step 2：保留 decompose 返回的 dependencies

**文件**：`src/paperbot/api/routes/agent_board.py`，`_plan_stream()` 中创建 AgentTask 处（~line 710）

```python
# 当前：完全忽略 dependencies
task = AgentTask(
    id=f"task-{uuid.uuid4().hex[:8]}",
    title=task_data.get("title", "Untitled"),
    description=task_data.get("description", ""),
    # ...
    dependencies=task_data.get("dependencies", []),  # 新增
)
```

同时修改 Stage 1.5 fallback（~line 851），从 `planning_tasks` 读取而非硬编码空：

```python
tasks_data = [
    {
        "title": t.title,
        "description": t.description,
        "difficulty": ...,
        "acceptance_criteria": ...,
        "dependencies": t.dependencies,  # 从 AgentTask 读取
    }
    for t in planning_tasks
]
```

#### Step 3：实现 DAG 拓扑排序

**新增文件**：`src/paperbot/infrastructure/swarm/task_dag.py`

```python
"""DAG-based task scheduler for parallel execution."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from ...api.routes.agent_board import AgentTask


class TaskDAG:
    """Builds a dependency graph from AgentTask list and yields execution batches."""

    def __init__(self, tasks: List[AgentTask]):
        self._tasks = {t.title: t for t in tasks}
        self._task_by_id = {t.id: t for t in tasks}
        # Build adjacency: task_title -> set of titles that depend on it
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        # In-degree: how many unfinished dependencies each task has
        self._in_degree: Dict[str, int] = {}

        for t in tasks:
            # Resolve dependencies by title, skip unknown
            valid_deps = [d for d in t.dependencies if d in self._tasks]
            self._in_degree[t.title] = len(valid_deps)
            for dep in valid_deps:
                self._dependents[dep].add(t.title)

    def topological_batches(self) -> List[List[AgentTask]]:
        """Return tasks grouped into batches that can execute in parallel.

        Each batch contains tasks whose dependencies are all in earlier batches.
        Tasks within a batch have no mutual dependencies and can run concurrently.

        Returns empty list if there's a cycle (falls back to sequential).
        """
        in_degree = dict(self._in_degree)
        batches: List[List[AgentTask]] = []

        # Start with tasks that have no dependencies
        ready = deque(title for title, deg in in_degree.items() if deg == 0)

        while ready:
            batch_titles = list(ready)
            ready.clear()
            batch = [self._tasks[title] for title in batch_titles]
            batches.append(batch)

            for title in batch_titles:
                for dependent in self._dependents.get(title, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        ready.append(dependent)

        # Check for cycles: if any task has in_degree > 0, there's a cycle
        scheduled = sum(len(b) for b in batches)
        if scheduled < len(self._tasks):
            # Cycle detected — fall back to single batch (sequential-safe)
            remaining = [
                self._tasks[t]
                for t in in_degree
                if in_degree[t] > 0
            ]
            batches.append(remaining)

        return batches

    @property
    def is_trivial(self) -> bool:
        """True if all tasks are independent (no dependencies)."""
        return all(deg == 0 for deg in self._in_degree.values())
```

#### Step 4：重写 Stage 2 执行循环

**文件**：`src/paperbot/api/routes/agent_board.py`，`_run_all_stream_sandbox()` 的 Stage 2 部分（~line 877-1009）

**核心改动**：将 `for i, task in enumerate(planning_tasks)` 替换为 DAG 批次执行。

```python
from ..infrastructure.swarm.task_dag import TaskDAG

# ── Stage 2: Execute Tasks via ExecutorAgent (DAG-parallel) ──
dag = TaskDAG(planning_tasks)
batches = dag.topological_batches()

yield StreamEvent(
    type="progress",
    data={
        "phase": "executing",
        "message": f"Executing {total} tasks in {len(batches)} batch(es)...",
        "total": total,
        "batches": len(batches),
        "paper_slug": slug,
    },
)

executor_agent = ExecutorAgent(dispatcher)
completed_count = 0

for batch_idx, batch in enumerate(batches):

    # --- 单任务 batch → 走原来的串行路径（避免 gather 开销）---
    if len(batch) == 1:
        task = batch[0]
        # ... 现有单任务执行逻辑（不变）...
        completed_count += 1
        continue

    # --- 多任务 batch → asyncio.gather 并行 ---
    # 为每个任务创建 step-event queue 和 on_step 回调
    queues: Dict[str, asyncio.Queue[StreamEvent]] = {}
    futures: Dict[str, asyncio.Task] = {}

    for task in batch:
        task.status = "in_progress"
        task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
        task.progress = 10
        task.updated_at = _now_iso()
        _append_task_log(task, event="executor_started", phase="executing",
                         level="info", message=f"Executor started (batch {batch_idx + 1}).")
        yield StreamEvent(
            type="progress",
            data={"event": "executor_started", "task_id": task.id,
                  "task": task.model_dump(), "batch": batch_idx},
        )

        q: asyncio.Queue[StreamEvent] = asyncio.Queue()
        queues[task.id] = q

        # Capture task in closure
        def _make_on_step(_task, _q):
            async def _on_step(step, tool_name, args, observation):
                _task.progress = min(12 + (step * 2), 65)
                _task.updated_at = _now_iso()
                obs_preview = observation if len(observation) <= 200 else f"{observation[:200]}..."
                _append_task_log(_task, event="tool_call", phase="executing",
                                 level="info",
                                 message=f"[step {step}] {tool_name}({_summarize_args(args)})",
                                 details={"tool": tool_name, "args_keys": sorted(args.keys()),
                                          "observation_preview": obs_preview})
                await _q.put(StreamEvent(
                    type="progress",
                    data={"event": "tool_step", "task_id": _task.id,
                          "task": _task.model_dump(), "step": step,
                          "tool": tool_name, "observation_preview": obs_preview},
                ))
            return _on_step

        wisdom = list(commander.wisdom.learnings) if commander.wisdom.learnings else None
        futures[task.id] = asyncio.create_task(
            executor_agent.execute(
                task, shared, slug,
                on_step=_make_on_step(task, q),
                wisdom=wisdom,
                max_iterations=max_iterations,
            )
        )

    # Drain all queues until all futures done
    while any(not f.done() for f in futures.values()) or any(not q.empty() for q in queues.values()):
        for q in queues.values():
            while not q.empty():
                yield q.get_nowait()
        await asyncio.sleep(0.3)
    # Final drain
    for q in queues.values():
        while not q.empty():
            yield q.get_nowait()

    # Collect results
    for task in batch:
        try:
            result = futures[task.id].result()
        except Exception as exc:
            result = CodexResult(task_id=task.id, success=False, error=str(exc))
        task.generated_files = result.files_generated
        task.codex_output = result.output if result.success else result.error
        task.progress = 70
        completed_count += 1
        # ... 后续 verification/repair 逻辑不变 ...
```

#### Step 5：处理并行 VM 访问安全

**问题**：多个 Executor 同时对同一 VM 的 `run_command` 可能互相影响（如 `pip install` 并发、`cd` 冲突）。

**方案**：每个任务的工作目录隔离。

```python
# SandboxToolExecutor 中，为每个任务在 paper_slug 下创建子目录
# 例如 /home/user/{slug}/task-001/ vs /home/user/{slug}/task-002/

# run_command 的 cwd 默认设为任务子目录
# 共享文件（如 requirements.txt）放在 paper root
```

**实际风险评估**：
- `write_file` 路径不同 → 无冲突
- `pip install` → 全局操作，需串行化（可用 `asyncio.Lock`）
- `run_command` → 需指定 cwd，避免 cd 冲突

```python
# 在 SharedSandbox 或 SandboxToolExecutor 中添加安装锁
_install_lock = asyncio.Lock()

async def execute_run_command(self, command, cwd):
    if self._is_install_command(command):
        async with _install_lock:
            return self._sandbox.commands.run(command, cwd=cwd)
    return self._sandbox.commands.run(command, cwd=cwd)
```

#### Step 6：前端 AgentBoard 展示 batch 信息

**文件**：`web/src/components/studio/AgentBoard.tsx`

SSE 事件已包含 `batch` 字段。前端可在 DAG 可视化中用分组展示并行执行的任务：

```tsx
// 用 @xyflow/react 的 group node 展示 batch
// 同一 batch 内的任务节点水平排列
// 不同 batch 之间用 edge 连接
```

这是可选的 UI 增强，不影响后端功能。

---

### 测试计划

```python
# tests/unit/test_task_dag.py

def test_linear_dependencies():
    """A → B → C 应产生 3 个 batch，每个 1 个任务。"""

def test_diamond_dependencies():
    """A → B, A → C, B → D, C → D 应产生 3 个 batch: [A], [B,C], [D]。"""

def test_all_independent():
    """无依赖 → 1 个 batch 包含所有任务。"""

def test_cycle_detection():
    """A → B → A 应回退到安全模式（不崩溃）。"""

def test_unknown_dependency_ignored():
    """依赖不存在的任务名 → 视为无依赖。"""

def test_single_task():
    """单任务 → 1 个 batch。"""
```

### 风险与回退

| 风险 | 缓解 |
|------|------|
| VM 并发 `pip install` 冲突 | `asyncio.Lock` 串行化安装命令 |
| 依赖 title 拼写不一致（Claude 返回略有不同） | 模糊匹配（`difflib.get_close_matches`）或按索引回退 |
| 并行后 wisdom 注入不完整（同 batch 任务无法互相学习） | 设计如此：同 batch 任务独立，跨 batch 才需要 wisdom |
| 循环依赖导致死锁 | `topological_batches()` 检测循环并回退为顺序 |
| 沙箱资源竞争（CPU/内存） | 限制 batch 内最大并行数（默认 3） |

---
---

## Phase 2E：KV-Cache 优化

### 现状分析

1. **CodexDispatcher（OpenAI）**：`dispatch_with_sandbox_tools()` 每次循环迭代调用 `client.chat.completions.create()`，messages 列表持续追加（✅ 天然 append-only）。但 system prompt 每次都是 `_sandbox_workspace_system_prompt()` 的新字符串实例（无影响，OpenAI 按内容匹配缓存）。

2. **OpenAI Prompt Caching 机制**：OpenAI 自 GPT-4o 起自动启用 Prompt Caching（[文档](https://platform.openai.com/docs/guides/prompt-caching)）：
   - **自动生效**：不需要代码改动，只要请求前缀 ≥ 1024 tokens 且与前一次请求前缀相同
   - 缓存 token 价格为输入价格的 50%
   - 缓存有效期：5-10 分钟（同一 organization）
   - **当前代码已天然受益**：`dispatch_with_sandbox_tools()` 中 messages 列表是 append-only 的，system prompt + user prompt 前缀在同一任务的多次迭代中完全相同

3. **ClaudeCommander（Anthropic）**：`decompose()` 和 `review()` 使用 Anthropic API。Anthropic Prompt Caching 需要显式标记 `cache_control: {"type": "ephemeral"}`。当前未使用。

4. **已有 in-memory 缓存**：`LLMService` 有 SHA256 哈希缓存，但仅用于应用层（摘要、趋势分析），未用于 Agent Board。

5. **`_compress_messages()`**：当 messages 超过 `MAX_MESSAGES` 时，会压缩中间消息为摘要。这会破坏 KV-cache（前缀变了），但只在长会话中触发，是必要的上下文窗口管理。

---

### 实施步骤

#### Step 1：确认 OpenAI 自动缓存已生效（无需改动）

OpenAI 的 Prompt Caching 对 GPT-4o / GPT-4.1 / o-series 自动生效。验证方式：

```python
# 在 dispatch_with_sandbox_tools 的 response 中检查 usage
usage = response.usage
if hasattr(usage, "prompt_tokens_details"):
    cached = usage.prompt_tokens_details.cached_tokens
    log.debug("KV-cache hit: %d/%d tokens cached", cached, usage.prompt_tokens)
```

**文件**：`src/paperbot/infrastructure/swarm/codex_dispatcher.py`，在 `dispatch_with_sandbox_tools()` 的 response 处理后（~line 432）添加日志：

```python
response = await asyncio.wait_for(
    client.chat.completions.create(...),
    timeout=self.dispatch_timeout_seconds,
)

# 新增：KV-cache 命中率日志
if hasattr(response, "usage") and response.usage:
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    if details and hasattr(details, "cached_tokens"):
        log.debug(
            "Step %d: %d/%d prompt tokens cached (%.0f%%)",
            step, details.cached_tokens, usage.prompt_tokens,
            (details.cached_tokens / max(usage.prompt_tokens, 1)) * 100,
        )
```

#### Step 2：确保 _compress_messages 不过早破坏缓存前缀

**当前行为**：`_compress_messages()` 保留 `head = messages[:2]`（system + 首条 user），压缩中间部分。这保持了 system prompt 前缀稳定。

**优化**：增大 `MAX_MESSAGES` 阈值，延迟压缩触发点。

```python
# codex_dispatcher.py 顶部常量
MAX_MESSAGES = 60  # 当前值（确认）→ 可提升到 80
```

**权衡**：更大的 MAX_MESSAGES = 更高的缓存命中率 + 更多 token 消耗。建议保持当前值，仅在监控显示频繁压缩时调整。

#### Step 3：Anthropic Prompt Caching（ClaudeCommander）

**文件**：`src/paperbot/infrastructure/swarm/claude_commander.py`

Anthropic API 支持通过 `cache_control` 标记可缓存的内容块。对于 `decompose()` 和 `review()`，system prompt 是稳定的缓存目标。

**3a. decompose() 添加缓存（~line 98）**：

```python
# 当前
response = await asyncio.wait_for(
    client.messages.create(
        model=self.model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ),
    timeout=self.request_timeout_seconds,
)

# 优化后：用 cache_control 标记 system prompt
response = await asyncio.wait_for(
    client.messages.create(
        model=self.model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    ),
    timeout=self.request_timeout_seconds,
)
```

**3b. review() 添加 system prompt + 缓存**：

当前 `review()` 没有 system prompt（直接发 user message）。应添加：

```python
system = (
    "You are a senior code reviewer evaluating AI-generated code for a paper reproduction. "
    "You assess correctness, completeness relative to the task description, and code quality. "
    "Return JSON only."
)

response = await asyncio.wait_for(
    client.messages.create(
        model=self.model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    ),
    timeout=self.request_timeout_seconds,
)
```

**3c. 大型 user prompt 的缓存（高级）**：

如果同一 session 中多次 review（多个任务），每次 review 的 prompt 前半段（任务列表、项目上下文）可能重复。可将其标记为可缓存：

```python
messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": project_context,  # 跨 review 稳定的部分
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": task_specific_prompt,  # 每次不同
            },
        ],
    }
]
```

**注意**：Anthropic 缓存需要前缀 ≥ 1024 tokens（Claude 3.5 Sonnet）或 ≥ 2048 tokens（Claude 3 Opus）。system prompt 通常不到 1024 tokens，需要与 user message 前缀组合才能触发缓存。

#### Step 4：稳定 system prompt 格式

**原则**：system prompt 必须是**纯字面量**，不含时间戳、随机 ID 等动态内容。

**当前状态**：✅ 已满足。`_sandbox_workspace_system_prompt()` 返回纯静态字符串，无动态插值。

**需验证**：
- `_tool_system_prompt()` — ✅ 纯静态
- `decompose()` 的 system — ✅ 纯静态
- `review()` — 需新增，确保纯静态

#### Step 5：监控 + 成本追踪

添加一个轻量 metrics 收集器，追踪缓存命中率和 token 成本：

**文件**：`src/paperbot/infrastructure/swarm/codex_dispatcher.py`

```python
# 在类级别添加
class _CacheMetrics:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.cached_prompt_tokens = 0

    def record(self, usage):
        if not usage:
            return
        self.total_prompt_tokens += usage.prompt_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        if details and hasattr(details, "cached_tokens"):
            self.cached_prompt_tokens += details.cached_tokens

    @property
    def hit_rate(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.cached_prompt_tokens / self.total_prompt_tokens

    def report(self) -> str:
        return (
            f"KV-cache: {self.cached_prompt_tokens}/{self.total_prompt_tokens} "
            f"tokens cached ({self.hit_rate:.0%})"
        )
```

在 session 结束时通过 SSE 报告：

```python
yield StreamEvent(
    type="progress",
    data={
        "event": "cache_stats",
        "phase": "complete",
        "cache_hit_rate": metrics.hit_rate,
        "total_prompt_tokens": metrics.total_prompt_tokens,
        "cached_prompt_tokens": metrics.cached_prompt_tokens,
    },
)
```

---

### 成本节省估算

| 场景 | 每任务迭代数 | 无缓存 token 成本 | 有缓存后 | 节省 |
|------|------------|-----------------|---------|------|
| **CodexDispatcher（OpenAI GPT-4o）** | ~15 迭代/任务 | ~$0.08/任务 | ~$0.05/任务 | **~37%** |
| **ClaudeCommander.decompose()** | 1 次/session | ~$0.02 | ~$0.015 | **~25%** |
| **ClaudeCommander.review()** | N 次/session | ~$0.01×N | ~$0.007×N | **~30%** |
| **5 任务 session 总计** | — | ~$0.52 | ~$0.34 | **~35%** |

> 注：OpenAI 缓存自动生效（50% 折扣），Anthropic 缓存需显式标记（90% 折扣但需 cache write 成本）。

---

### 测试计划

```python
# tests/unit/test_kv_cache_metrics.py

def test_cache_metrics_recording():
    """验证 _CacheMetrics 正确记录和计算命中率。"""

def test_cache_metrics_zero_division():
    """无数据时 hit_rate 返回 0.0。"""

# tests/unit/test_decompose_cache_control.py

@pytest.mark.asyncio
async def test_decompose_sends_cache_control():
    """验证 decompose() 发送 cache_control 参数。"""
    # Mock anthropic client, 检查 system 参数格式

@pytest.mark.asyncio
async def test_review_has_system_prompt():
    """验证 review() 包含 system prompt。"""
```

---

### 实施优先级

| 步骤 | 工作量 | 影响 | 建议 |
|------|-------|------|------|
| Step 1: 确认 OpenAI 自动缓存 | 5 min | 高（验证已有收益） | **立即做** |
| Step 3a: decompose() cache_control | 10 min | 低（单次调用） | 可选 |
| Step 3b: review() 添加 system prompt | 15 min | 中（多次调用受益） | **推荐** |
| Step 5: 监控指标 | 20 min | 中（可观测性） | **推荐** |
| Step 2: MAX_MESSAGES 调优 | 5 min | 低（仅长会话受益） | 按需 |
| Step 3c: user prompt 前缀缓存 | 30 min | 低（需重构 prompt） | 后续 |

---

## 实施顺序建议

1. **先做 Phase 2E Step 1 + Step 5**（30 min）— 添加缓存监控，获取基线数据
2. **再做 Phase 2B Step 1-3**（2 hr）— DAG 模块 + 测试，不改执行逻辑
3. **Phase 2E Step 3b**（15 min）— review() 添加 system prompt
4. **Phase 2B Step 4**（3 hr）— 重写执行循环，最大改动，需充分测试
5. **Phase 2B Step 5**（1 hr）— 并发安全（安装锁）
6. **Phase 2B Step 6**（可选）— 前端 batch 可视化
