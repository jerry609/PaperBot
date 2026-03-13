# Phase 1A 实施计划：为 Codex Worker 添加 Tool-Use Agent Loop

## 概述

将 `CodexDispatcher.dispatch()` 从单次 LLM 调用改造为 **迭代式工具调用循环**（CodeAct 模式）。Worker 不再一次性生成所有代码文本，而是通过 OpenAI function calling 逐步执行：读取文件 → 写入代码 → 运行命令 → 观察结果 → 决定下一步。

## Phase 1A 范围收敛（安全优先）

为降低改造风险，Phase 1A 采用受控范围：

1. 第一批默认启用工具：`read_file`、`write_file`、`list_files`、`search_files`、`update_subtask`、`task_done`
2. `run_command` 仅在 **沙箱可用且显式开启** 时可用（`CODEX_ENABLE_RUN_COMMAND=true`）
3. 无沙箱时 `run_command` **Fail-Closed**（直接返回错误，不执行本地 subprocess）
4. 保持向后兼容：可通过 `CODEX_TOOL_USE=false` 一键回退到现有单次生成路径

## 当前状态

```
ClaudeCommander.build_codex_prompt()
        ↓
CodexDispatcher.dispatch()
        ↓
  单次 client.chat.completions.create()  ← 一次性文本生成
        ↓
  正则提取代码块 → 写入文件
        ↓
  CodexResult
```

**问题**：Worker 看不到工作区、不执行代码、不验证结果、无法自我修正。

## 目标状态

```
ClaudeCommander.build_codex_prompt()
        ↓
CodexDispatcher.dispatch()
        ↓
  ┌─→ client.chat.completions.create(tools=WORKER_TOOLS)
  │        ↓
  │   finish_reason == "tool_calls"?
  │     YES → ToolExecutor.execute(tool_name, args)
  │              ↓ 观察结果追加到 messages
  │              ↓ 通过 on_step 回调发出 SSE 心跳
  │              └─→ 回到顶部（下一次迭代）
  │     NO  → finish_reason == "stop" → 返回 CodexResult
  └──────────────────────────────────────┘
  安全上限：MAX_ITERATIONS = 25
```

---

## 文件清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `src/paperbot/infrastructure/swarm/worker_tools.py` | 工具定义 + ToolExecutor 类 |
| **修改** | `src/paperbot/infrastructure/swarm/codex_dispatcher.py` | dispatch() 改为 agent loop |
| **修改** | `src/paperbot/infrastructure/swarm/__init__.py` | 导出新类 |
| **修改** | `src/paperbot/api/routes/agent_board.py` | 接入沙箱，传递 on_step 回调 |
| **修改** | `src/paperbot/infrastructure/swarm/claude_commander.py` | 更新 system prompt 提示使用工具 |
| **新建** | `tests/unit/test_tool_executor.py` | ToolExecutor 单元测试 |
| **新建** | `tests/unit/test_codex_tool_loop.py` | Agent loop 集成测试 |
| **修改** | `tests/unit/test_swarm_timeouts.py` | 适配新接口签名 |

---

## 步骤分解

### Step 1：新建 `worker_tools.py` — 工具定义 + 执行器

**文件**：`src/paperbot/infrastructure/swarm/worker_tools.py`

#### 1.1 工具定义常量

定义 OpenAI function calling 格式的工具 schema：

```python
CODING_WORKER_TOOLS: List[dict]
```

共 7 个工具：

| 工具名 | 用途 | 参数 |
|--------|------|------|
| `read_file` | 读取工作区内文件内容 | `path: str` |
| `write_file` | 创建或覆盖文件 | `path: str, content: str` |
| `list_files` | 列出目录结构 | `path: str = "."` |
| `run_command` | 在沙箱中执行 shell 命令（Phase 1A 默认关闭） | `command: str` |
| `search_files` | 在文件中搜索文本/正则 | `pattern: str, glob: str = "*"` |
| `update_subtask` | 标记子任务完成状态 | `subtask_id: str, done: bool, notes?: str` |
| `task_done` | 宣告任务完成 | `summary: str, files_changed: list[str]` |

**设计决策**：

- `read_file` 截断 12000 字符，防止上下文爆炸
- `write_file` 通过路径安全检查，禁止 `..` 和绝对路径
- `run_command` 仅在沙箱可用且 `CODEX_ENABLE_RUN_COMMAND=true` 时启用
- 无沙箱时 `run_command` 返回错误（Fail-Closed），不做本地命令执行
- `search_files` 使用 `subprocess.run(["grep", ...])` 而非 Python 实现，输出限 6000 字符
- `task_done` 返回哨兵值 `"TASK_COMPLETE"` 终止循环

#### 1.2 ToolExecutor 类

```python
class ToolExecutor:
    """将 LLM 的工具调用路由到实际操作。"""

    def __init__(
        self,
        workspace: Path,
        sandbox: Optional[BaseExecutor],
        task: Optional[AgentTask] = None,
    ):
        self.workspace = workspace
        self.sandbox = sandbox
        self.task = task
        self.files_written: List[str] = []
        self.tool_log: List[Dict[str, Any]] = []  # 记录所有工具调用

    async def execute(self, tool_name: str, args: dict) -> str:
        """路由工具调用，返回观察结果字符串。"""
```

**方法清单**：

| 方法 | 操作 | 返回 |
|------|------|------|
| `_tool_read_file` | `workspace / path` → 读取 → 截断 | 文件内容或 "File not found" |
| `_tool_write_file` | 路径安全检查 → `mkdir -p` → 写入 | "Written N chars to path" |
| `_tool_list_files` | `iterdir()` → 排序 → 前 100 项 | "d src/\nf main.py\n..." |
| `_tool_run_command` | 委托 `sandbox.run()`（无沙箱则拒绝） | "exit_code: N\nstdout..." 或错误信息 |
| `_tool_search_files` | `grep -rn pattern workspace` | 匹配行或 "(no matches)" |
| `_tool_update_subtask` | 修改 `task.subtasks` 中的 `done` 字段 | "Subtask 'X' marked done" |
| `_tool_task_done` | 返回哨兵值 | `"TASK_COMPLETE"` |

**关键安全措施**：

```python
def _safe_path(self, rel: str) -> Optional[Path]:
    """确保路径不逃逸出 workspace。"""
    if not rel or ".." in Path(rel).parts:
        return None
    p = (self.workspace / rel).resolve()
    if not str(p).startswith(str(self.workspace.resolve())):
        return None
    return p
```

**run_command 执行策略（Fail-Closed）**：

```python
async def _tool_run_command(self, args: dict) -> str:
    enable_run = os.getenv("CODEX_ENABLE_RUN_COMMAND", "false").lower() == "true"
    if not enable_run:
        return "Error: run_command is disabled (set CODEX_ENABLE_RUN_COMMAND=true to enable)"

    if self.sandbox and self.sandbox.available():
        result = self.sandbox.run(
            workdir=self.workspace,
            commands=[args["command"]],
            timeout_sec=120,
        )
        return f"exit_code: {result.exit_code}\n{result.logs}"

    return "Error: run_command requires a sandbox executor (docker/e2b)"
```

---

### Step 2：修改 `codex_dispatcher.py` — Agent Loop

**文件**：`src/paperbot/infrastructure/swarm/codex_dispatcher.py`

#### 2.1 新增 `dispatch_with_tools()` 方法

保留原有 `dispatch()` 作为降级路径，新增 `dispatch_with_tools()`：

```python
async def dispatch_with_tools(
    self,
    task_id: str,
    prompt: str,
    workspace: Path,
    *,
    sandbox: Optional[BaseExecutor] = None,
    task: Optional["AgentTask"] = None,
    on_step: Optional[Callable[[int, str, dict, str], Awaitable[None]]] = None,
    max_iterations: int = 25,
) -> CodexResult:
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `sandbox` | `Optional[BaseExecutor]` | 复用 repro 的执行器，可为 None（降级） |
| `task` | `Optional[AgentTask]` | 传入任务对象以支持 `update_subtask` |
| `on_step` | `Optional[Callable]` | 每次工具执行后的回调，用于发 SSE 心跳 |
| `max_iterations` | `int` | 安全上限，防止无限循环 |

#### 2.2 Agent Loop 核心逻辑

```python
async def dispatch_with_tools(self, task_id, prompt, workspace, **kwargs):
    import openai, json
    from .worker_tools import CODING_WORKER_TOOLS, ToolExecutor

    client = openai.AsyncOpenAI(api_key=self.api_key)
    tool_exec = ToolExecutor(workspace, kwargs.get("sandbox"), kwargs.get("task"))
    on_step = kwargs.get("on_step")
    max_iter = kwargs.get("max_iterations", 25)

    messages = [
        {"role": "system", "content": self._tool_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    for step in range(max_iter):
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=CODING_WORKER_TOOLS,
                max_tokens=4096,
            ),
            timeout=self.dispatch_timeout_seconds,
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message.model_dump())
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                observation = await tool_exec.execute(name, args)

                if on_step:
                    await on_step(step, name, args, observation)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

                if observation == "TASK_COMPLETE":
                    return CodexResult(
                        task_id=task_id, success=True,
                        output=args.get("summary", ""),
                        files_generated=tool_exec.files_written,
                    )

        elif choice.finish_reason == "stop":
            text = choice.message.content or ""
            messages.append({"role": "assistant", "content": text})
            if tool_exec.files_written:
                return CodexResult(task_id=task_id, success=True,
                                   output=text, files_generated=tool_exec.files_written)
            # 降级到原有的正则提取
            files = self._persist_output(task_id, text, workspace)
            return CodexResult(task_id=task_id, success=True,
                               output=text, files_generated=files)

    return CodexResult(
        task_id=task_id,
        success=len(tool_exec.files_written) > 0,
        output=f"Reached iteration limit ({max_iter})",
        files_generated=tool_exec.files_written,
        error=f"Agent loop did not call task_done within {max_iter} iterations",
    )
```

#### 2.2.1 边界条件处理（必须实现）

除 `tool_calls` / `stop` 外，需明确处理以下分支，防止循环失控或静默失败：

1. `finish_reason in {"length", "content_filter"}`：立即返回失败，并记录可观测错误
2. 工具参数 JSON 解析失败：返回结构化错误给模型（而不是抛异常中断）
3. 未知工具名：返回 `"unknown tool"` 观察结果，并计入异常计数
4. 连续重复工具调用（相同 `tool_name + args`）超过阈值：提前中止并返回失败
5. 空响应/无 `choices`：返回失败，避免进入死循环

建议追加保护参数：

```python
MAX_REPEAT_TOOL_CALLS = 3
MAX_TOOL_ERRORS = 5
```

#### 2.3 新 System Prompt

```python
def _tool_system_prompt(self) -> str:
    return (
        "You are an expert coding agent working in a sandbox environment. "
        "You have tools to read/write files, run shell commands, and search code.\n\n"
        "## Workflow\n"
        "1. First use list_files and read_file to understand the existing project structure\n"
        "2. Write your implementation using write_file\n"
        "3. Use run_command to install dependencies and run tests\n"
        "4. If tests fail, read the error, fix the code, and retry\n"
        "5. Use update_subtask to mark each acceptance criterion as done\n"
        "6. When all criteria pass, call task_done with a summary\n\n"
        "## Rules\n"
        "- Always verify your code compiles/runs before calling task_done\n"
        "- One tool call per step, observe the result before deciding next action\n"
        "- Keep files focused and well-structured\n"
        "- If a command fails, diagnose and fix — do not retry blindly\n"
    )
```

#### 2.4 保持向后兼容

原有 `dispatch()` 方法保持不变。`agent_board.py` 切换到调用 `dispatch_with_tools()`。如果 `CODEX_TOOL_USE=false` 环境变量设置，降级回 `dispatch()`。

```python
async def dispatch_auto(self, task_id, prompt, workspace, **kwargs) -> CodexResult:
    """自动选择：有沙箱/工具则用 tool loop，否则降级。"""
    use_tools = os.getenv("CODEX_TOOL_USE", "true").lower() != "false"
    if use_tools:
        return await self.dispatch_with_tools(task_id, prompt, workspace, **kwargs)
    return await self.dispatch(task_id, prompt, workspace)
```

---

### Step 3：修改 `agent_board.py` — 接入沙箱 + SSE 回调

**文件**：`src/paperbot/api/routes/agent_board.py`

#### 3.1 添加沙箱工厂

在文件顶部 helper 区域新增：

```python
def _get_sandbox() -> Optional["BaseExecutor"]:
    """复用 repro 中已有的 Docker/E2B 执行器。"""
    executor_type = os.getenv("PAPERBOT_EXECUTOR", "auto")
    try:
        if executor_type == "e2b":
            from ...repro.e2b_executor import E2BExecutor
            return E2BExecutor()
        elif executor_type == "docker":
            from ...repro.docker_executor import DockerExecutor
            return DockerExecutor(image="python:3.11-slim")
        else:  # auto
            try:
                from ...repro.e2b_executor import E2BExecutor
                e2b = E2BExecutor()
                if e2b.available():
                    return e2b
            except Exception:
                pass
            try:
                from ...repro.docker_executor import DockerExecutor
                docker = DockerExecutor(image="python:3.11-slim")
                if docker.available():
                    return docker
            except Exception:
                pass
    except Exception:
        log.warning("No sandbox available; run_command will be fail-closed")
    return None
```

#### 3.2 修改 `_run_all_stream()` 中的 dispatch 调用

将现有的：

```python
dispatch_coro = dispatcher.dispatch(task.id, prompt, workspace)
```

改为：

```python
sandbox = _get_sandbox()

async def _on_step(step: int, tool_name: str, args: dict, observation: str):
    """每次工具调用后发出 SSE 事件。"""
    task.progress = min(10 + step * 3, 65)
    task.updated_at = _now_iso()
    obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
    _append_task_log(
        task,
        event="tool_call",
        phase="codex_running",
        level="info",
        message=f"[step {step}] {tool_name}({_summarize_args(args)})",
        details={"tool": tool_name, "args_keys": list(args.keys()),
                 "observation_preview": obs_preview},
    )

result = await dispatcher.dispatch_with_tools(
    task.id, prompt, workspace,
    sandbox=sandbox,
    task=task,
    on_step=_on_step,
)
```

#### 3.3 移除心跳 while 循环

工具调用循环内部已经自然地产生 SSE 事件（通过 `on_step` 回调），不再需要外部的 5 秒心跳轮询。删除现有的：

```python
dispatch_task = asyncio.ensure_future(dispatch_coro)
while not dispatch_task.done():
    await asyncio.sleep(5)
    ...
result = dispatch_task.result()
```

直接替换为：

```python
result = await dispatcher.dispatch_with_tools(...)
```

同样更新 `_execute_task_stream()` 中的单任务执行。

#### 3.4 与现有 Agent Board 强化项兼容（必须保持）

接入 tool loop 时，不得破坏近期已落地的两项能力：

1. **工作区路径安全**：继续复用 `agent_board.py` 中的 `workspace_dir` 校验与允许根目录策略（禁止绕过）
2. **会话持久化**：继续通过 `PipelineSessionStore` 持久化 session/task 状态，`on_step` 产生的关键状态变化也需要落盘

建议在 `_on_step` 中追加轻量持久化：

```python
_persist_session(session, checkpoint="tool_step", status="running")
```

---

### Step 4：修改 `claude_commander.py` — 更新 Prompt

**文件**：`src/paperbot/infrastructure/swarm/claude_commander.py`

#### 4.1 更新 `build_codex_prompt()`

在生成的 prompt 末尾添加子任务信息（之前 Worker 看不到）：

```python
# 注入子任务列表，让 Worker 知道需要完成什么
if task.get("subtasks"):
    parts.append("## Subtasks (Acceptance Criteria)")
    for sub in task["subtasks"]:
        status = "✓" if sub.get("done") else "○"
        parts.append(f"- [{status}] {sub['id']}: {sub['title']}")
    parts.append("")
    parts.append("Call update_subtask for each criterion as you complete it.")
```

移除末尾的 `"Output complete file contents with filenames."` —— 这是旧的一次性模式指令。替换为：

```python
parts.append(
    "Use the provided tools to explore the workspace, write code, "
    "run tests, and verify correctness. Call task_done when finished."
)
```

---

### Step 5：更新 `__init__.py` 导出

**文件**：`src/paperbot/infrastructure/swarm/__init__.py`

```python
from .worker_tools import CODING_WORKER_TOOLS, ToolExecutor

__all__ = [
    ...
    "CODING_WORKER_TOOLS",
    "ToolExecutor",
]
```

---

### Step 6：编写测试

#### 6.1 `tests/unit/test_tool_executor.py`

测试 `ToolExecutor` 的每个工具方法，无需真实 LLM 或沙箱。

| 测试用例 | 验证 |
|---------|------|
| `test_read_file_existing` | 读取已存在文件，返回内容 |
| `test_read_file_not_found` | 返回 "File not found" |
| `test_read_file_path_traversal_blocked` | `../../etc/passwd` → 返回 None |
| `test_write_file_creates_dirs` | 自动创建父目录 |
| `test_write_file_records_in_files_written` | 写入后追踪到列表 |
| `test_write_file_path_escape_blocked` | `../outside.py` → 拒绝 |
| `test_list_files_empty_dir` | 返回 "(empty directory)" |
| `test_list_files_with_entries` | 返回 "d" / "f" 前缀列表 |
| `test_run_command_with_sandbox` | 使用 FakeExecutor，返回 exit_code + logs |
| `test_run_command_no_sandbox_rejected` | 无沙箱时返回 fail-closed 错误 |
| `test_run_command_disabled_rejected` | 未开启 `CODEX_ENABLE_RUN_COMMAND` 时拒绝执行 |
| `test_search_files` | 搜索匹配和无匹配场景 |
| `test_update_subtask` | 修改 subtask.done 状态 |
| `test_task_done_returns_sentinel` | 返回 "TASK_COMPLETE" |

**测试模式**：使用 `tmp_path` fixture 作为 workspace，FakeExecutor 继承 `BaseExecutor`：

```python
class _FakeExecutor(BaseExecutor):
    def available(self) -> bool:
        return True

    def run(self, workdir, commands, timeout_sec=300, **kw):
        return ExecutionResult(
            status="success", exit_code=0,
            logs=f"ran: {commands[0]}"
        )
```

#### 6.2 `tests/unit/test_codex_tool_loop.py`

测试 `dispatch_with_tools()` 的 agent loop 逻辑。

| 测试用例 | 验证 |
|---------|------|
| `test_tool_loop_write_and_done` | LLM 调用 write_file → task_done → 成功 |
| `test_tool_loop_text_fallback` | LLM 直接返回文本（finish_reason=stop）→ 降级提取 |
| `test_tool_loop_max_iterations` | 达到迭代上限 → 返回部分结果 |
| `test_tool_loop_timeout` | API 超时 → CodexResult.success=False |
| `test_tool_loop_on_step_called` | 每次工具调用后 on_step 回调被触发 |

**Mock 方式**：用 `monkeypatch` 替换 `openai` 模块，模拟 tool_calls 响应：

```python
class _FakeChoice:
    def __init__(self, finish_reason, tool_calls=None, content=None):
        self.finish_reason = finish_reason
        self.message = types.SimpleNamespace(
            tool_calls=tool_calls,
            content=content,
            model_dump=lambda: {"role": "assistant", "content": content,
                                "tool_calls": [...]},
        )
```

#### 6.3 更新 `tests/unit/test_swarm_timeouts.py`

现有 4 个测试仅测试原有 `dispatch()`，无需修改（因为保留了向后兼容）。可选新增 `test_codex_tool_dispatch_timeout` 测试 `dispatch_with_tools()` 的超时行为。

---

## 上下文管理策略

### 消息历史增长控制

每次工具调用会追加 2 条消息（assistant + tool result），25 次迭代可能产生 50+ 条消息。需要控制：

```python
# 在 agent loop 中，当消息过多时压缩早期工具结果
MAX_MESSAGES = 60

if len(messages) > MAX_MESSAGES:
    # 保留 system + user + 最近 40 条，中间的压缩为摘要
    preserved_head = messages[:2]                    # system + user
    preserved_tail = messages[-(MAX_MESSAGES - 4):]  # 最近的消息
    summary = _summarize_middle(messages[2:-len(preserved_tail)])
    messages = preserved_head + [
        {"role": "user", "content": f"[Previous tool interactions summary]\n{summary}"}
    ] + preserved_tail
```

补充要求：

- `_summarize_middle()` 必须是**确定性摘要**（固定模板，不依赖额外 LLM 调用），避免成本和随机性
- 摘要中至少保留：工具名、核心参数、exit_code/错误、关键文件路径
- 单次摘要最大 1200 chars，超过则再截断并标记 `[truncated]`

### 工具输出截断

| 工具 | 最大输出长度 | 理由 |
|------|------------|------|
| `read_file` | 12,000 chars | 约 3k tokens，大文件截断 |
| `run_command` | 8,000 chars | 匹配 ExecutionResult.logs 上限 |
| `search_files` | 6,000 chars | grep 结果可能很长 |
| `list_files` | 100 entries | 防止巨型目录 |

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `CODEX_TOOL_USE` | `"true"` | 设为 `"false"` 降级到原有一次性模式 |
| `CODEX_ENABLE_RUN_COMMAND` | `"false"` | 是否开放 `run_command`（建议灰度开启） |
| `CODEX_MAX_ITERATIONS` | `25` | Agent loop 最大迭代次数 |
| `CODEX_TOOL_TIMEOUT_SECONDS` | `120` | 单次工具执行超时 |
| `PAPERBOT_EXECUTOR` | `"auto"` | 沙箱类型：`docker` / `e2b` / `auto` |
| `CODEX_DISPATCH_TIMEOUT_SECONDS` | `180` | 单次 LLM API 调用超时（已有） |

---

## 实施顺序和依赖

```
Step 1: worker_tools.py (新建)
  ├── 无依赖，可独立开发和测试
  └── 产出: ToolExecutor, CODING_WORKER_TOOLS

Step 2: codex_dispatcher.py (修改)
  ├── 依赖 Step 1 的 ToolExecutor
  └── 产出: dispatch_with_tools(), dispatch_auto()

Step 3: agent_board.py (修改)
  ├── 依赖 Step 2 的 dispatch_with_tools()
  └── 产出: 接入沙箱、on_step 回调、移除心跳轮询

Step 4: claude_commander.py (修改)
  ├── 无硬依赖，可与 Step 1-3 并行
  └── 产出: 更新 prompt 注入子任务

Step 5: __init__.py (修改)
  ├── 依赖 Step 1
  └── 产出: 导出新符号

Step 6: 测试 (新建)
  ├── test_tool_executor.py — 依赖 Step 1
  ├── test_codex_tool_loop.py — 依赖 Step 1 + 2
  └── test_swarm_timeouts.py — 适配检查
```

**建议开发顺序**：Step 1 → Step 6.1（测试先行）→ Step 2 → Step 6.2 → Step 4 → Step 5 → Step 3

---

## 风险和缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 不调用工具，直接输出文本 | 无文件生成 | `finish_reason=stop` 分支降级到正则提取 |
| LLM 进入死循环（反复调用同一工具） | API 费用浪费 | `MAX_ITERATIONS` 上限 + 连续重复检测 |
| 沙箱不可用（无 Docker、无 E2B key） | `run_command` 不可执行 | Fail-Closed + 明确错误提示 + 保留非命令工具继续运行 |
| 工具输出撑爆上下文 | API 报错 | 每个工具输出严格截断 |
| 路径穿越安全问题 | 读写工作区外文件 | `_safe_path()` 做 resolve + startswith 检查 |
| 原有 `dispatch()` 调用方受影响 | 回归 | 保留原方法不改，新增 `dispatch_with_tools()` |

---

## 灰度指标与退出条件（新增）

Phase 1A 不建议一次性全量开启，建议按 canary 灰度并持续观察：

1. `tool_loop_success_rate`（task_done 或有效文件产出成功率）>= 85%
2. `avg_tool_steps_per_task` <= 20（防止循环膨胀）
3. `tool_error_rate` <= 10%
4. `timeout_rate`（dispatch/tool timeout）<= 5%
5. `fallback_rate`（回退到旧 dispatch）可控且逐步下降
6. 单任务平均 token 成本不高于基线 +30%

若连续两个观察窗口不满足阈值，自动回退 `CODEX_TOOL_USE=false`。

---

## 验收标准

1. **工具调用可用**：`dispatch_with_tools()` 能通过工具读写文件、运行命令
2. **沙箱集成**：Docker 或 E2B 可用时，`run_command` 在沙箱内执行
3. **降级正常**：无沙箱或未开启 `CODEX_ENABLE_RUN_COMMAND` 时，`run_command` fail-closed 且有清晰错误
4. **向后兼容**：原有 `dispatch()` 不受影响，现有测试通过
5. **SSE 可见**：每次工具调用通过 `on_step` 产生 SSE 事件，前端可观察
6. **安全**：路径穿越、命令注入被阻止
7. **持久化兼容**：Tool loop 过程不破坏 `PipelineSessionStore` 的会话恢复能力
8. **测试覆盖**：`test_tool_executor.py`、`test_codex_tool_loop.py` 新增边界分支与 fail-closed 用例全部通过
