# Phase 1B 实施计划：Sandbox-as-Workspace 架构

> **核心理念**：VM 是唯一的工作区。所有 Agent 在 VM 内直接读写执行，文件系统是通信总线。本地仅在成功后下载产出。

## 1. 架构概述

### 1.1 与旧架构的根本区别

```
旧模式（当前）:
  Agent → 本地 workspace 写文件 → 上传到 sandbox → 验证 → 结果留在本地
  问题：两份文件源、上传同步延迟、desync

新模式（Manus-like Sandbox-as-Workspace）:
  Agent → sandbox VM 直接写文件 → 直接执行 → 观察输出 → 修复 → 成功后下载到本地
  优势：单一文件源、零 desync、Agent 所见即所得
```

### 1.2 整体拓扑

```
一个用户
│
└── 一个 Ubuntu VM Sandbox（E2B Firecracker microVM，长期存活）
    │  ← 唯一的文件系统，所有 Agent 直接操作
    │
    ├── /home/user/attn-is-all-you-need-a9b1/    ← 论文 A
    │   ├── .plan/                                ← Planner 写入
    │   ├── .status/                              ← Executor 写入（任务间通信）
    │   ├── .knowledge/                           ← Knowledge Manager 写入
    │   ├── src/                                  ← Executor 生成的代码
    │   ├── tests/
    │   ├── requirements.txt
    │   └── ...
    │
    ├── /home/user/resnet-deep-residual-b3c2/     ← 论文 B（完全隔离）
    │   └── ...
    │
    └── .sandbox-meta/                            ← 全局元数据（可选）
```

### 1.3 数据流

```
                    ┌───────────────────────────────────────────────┐
                    │           E2B Sandbox VM (长期存活)            │
                    │                                               │
  Planner ─────────►│  .plan/roadmap.md, tasks.json                │
  (Claude)         │                                               │
                    │         ↓ 读取计划                            │
  Executor 1 ──────►│  src/model.py, src/train.py                  │
  (Codex)          │  .status/task-001.json                        │
                    │         ↓ 读取前序状态                         │
  Executor 2 ──────►│  tests/test_model.py                         │
  (Codex)          │  .status/task-002.json                        │
                    │         ↓ 全部文件可见                         │
  Verification ────►│  pytest -q → 通过 ✓                          │
                    │         ↓                                     │
  Knowledge ───────►│  .knowledge/summary.md                       │
  Manager          │                                               │
                    └──────────────────┬────────────────────────────┘
                                       │ 成功后下载
                                       ▼
                              本地 workspace/
                              └── {paper_slug}/
                                  ├── src/
                                  ├── tests/
                                  └── ...
```

**关键约束**：
- **无本地写入**：Agent 的 `write_file` 直接写 VM，不经过本地 Path
- **无上传步骤**：没有 "先本地生成再上传" 的环节
- **Fail-closed**：无 sandbox 时，所有文件/命令工具返回错误，不降级到本地
- **下载是最后一步**：只在验证通过后才把 VM 中的文件拉回本地

---

## 2. 已就绪组件（Phase 1A 产出）

| 组件 | 状态 | 说明 |
|------|------|------|
| `paper_slug.py` | ✅ 已实现 | 论文目录名生成，8 个测试通过 |
| `PersistentSandboxManager` | ✅ 已实现 | 一个用户一个长期存活 VM |
| `E2BExecutor.sandbox_cwd` | ✅ 已实现 | 支持指定工作目录 |
| `BoardSession.sandbox_paper_cwd` | ✅ 已实现 | `/home/user/{paper_slug}` |
| `CodexDispatcher.dispatch_with_tools` | ✅ 已实现 | CodeAct tool loop（25 轮） |
| `ToolExecutor` (7 tools) | ✅ 已实现 | 但文件操作走本地 Path（需改） |
| `SandboxRuntime` | ✅ 已实现 | 依赖检测、验证策略 |
| `ClaudeCommander` | ✅ 已实现 | 任务分解、review、wisdom |
| `_upload_files` 过滤 | ✅ 已实现 | 跳过二进制/大文件 |
| 本地模块检测 | ✅ 已实现 | `_scan_workspace_local_modules` |

---

## 3. 差距分析

| # | 差距 | 当前行为 | 目标行为 |
|---|------|---------|---------|
| **G1** | **ToolExecutor 文件操作在本地** | `write_file` → `local_path.write_text()` | `write_file` → `sandbox.files.write()` |
| **G2** | **无共享 VM 抽象** | 每次操作自带 E2BExecutor | 统一 `SharedSandbox` 封装 |
| **G3** | **Planner 不写文件** | `decompose()` 纯文本输出 | Planner 写 `.plan/` 到 VM |
| **G4** | **无 Knowledge Manager** | 任务完成后无整理 | KM 整理 `.knowledge/` |
| **G5** | **无下载步骤** | 文件已在本地（因为本地写的） | 成功后从 VM 下载到本地 |
| **G6** | **验证在 agent_board 中 ad-hoc** | 手动拼 pip install + verify | 统一 `VerificationPolicy` |
| **G7** | **无修复循环** | 验证失败 → 记录日志结束 | 验证失败 → repair prompt → 重试 |
| **G8** | **前端无 paper scope 文件浏览** | 无 | 新增 sandbox 文件浏览 API |

---

## 4. 核心设计

### 4.1 SharedSandbox：用户级 VM 抽象

封装 `PersistentSandboxManager` / `E2BExecutor`，提供论文目录 scope 的文件操作。

```python
# src/paperbot/infrastructure/swarm/shared_sandbox.py

class SharedSandbox:
    """用户级共享 VM。所有 Agent 通过此类操作 VM 文件系统。

    设计原则：
    - VM 是唯一文件源（single source of truth）
    - 所有路径操作自动 scope 到 /home/user/{paper_slug}/
    - 无本地文件操作
    """

    BASE_DIR = "/home/user"

    def __init__(self, executor: E2BExecutor):
        self.executor = executor

    @property
    def alive(self) -> bool: ...

    def paper_root(self, slug: str) -> str:
        return f"{self.BASE_DIR}/{slug}"

    # --- 文件操作（全部在 VM 内） ---
    def read_file(self, slug: str, path: str) -> Optional[str]: ...
    def write_file(self, slug: str, path: str, content: str) -> bool: ...
    def list_files(self, slug: str, path: str = ".") -> List[str]: ...
    def search_files(self, slug: str, pattern: str, glob: str = "*") -> str: ...

    # --- 命令执行 ---
    def run_in_paper(self, slug: str, command: str, timeout_sec: int = 120) -> ExecutionResult: ...

    # --- 下载产出到本地 ---
    def download_paper(self, slug: str, local_dir: Path,
                       skip_dirs: Set[str] = {".plan", ".status", ".knowledge", "__pycache__"}
                       ) -> List[str]: ...

    # --- 生命周期 ---
    def ensure_paper_dir(self, slug: str) -> bool: ...
    def list_papers(self) -> List[str]: ...
    def teardown(self) -> None: ...
```

`download_paper()` 是**唯一的**文件从 VM → 本地的路径，仅在验证成功后调用。

### 4.2 SandboxToolExecutor：面向 Agent 的工具接口

替代当前 `ToolExecutor`（本地 Path）。所有文件操作通过 `SharedSandbox` 路由到 VM。

```python
# src/paperbot/infrastructure/swarm/sandbox_tool_executor.py

class SandboxToolExecutor:
    """Agent 的工具执行器。所有操作在 VM 内完成。

    工具列表：
    - read_file(path) → 从 VM 读取
    - write_file(path, content) → 写入 VM
    - list_files(path?) → VM 目录列表
    - search_files(pattern, glob?) → VM 内 grep
    - run_command(command) → VM 内执行
    - update_subtask(subtask_id, done) → 更新任务状态
    - task_done(summary) → 标记任务完成
    """

    def __init__(self, sandbox: SharedSandbox, paper_slug: str, task: AgentTask):
        self.sandbox = sandbox
        self.slug = paper_slug
        self.task = task
        self.files_written: List[str] = []  # 跟踪 Agent 写了哪些文件

    async def execute(self, tool_name: str, args: dict) -> str: ...

    # --- 关键区别：所有文件操作直接走 VM ---

    async def _tool_write_file(self, args: dict) -> str:
        path = self._sanitize_path(args["path"])
        ok = self.sandbox.write_file(self.slug, path, args["content"])
        if ok:
            self.files_written.append(path)
        return f"Written {len(args['content'])} chars to {path}" if ok else "Error: write failed"

    async def _tool_read_file(self, args: dict) -> str:
        path = self._sanitize_path(args["path"])
        content = self.sandbox.read_file(self.slug, path)
        return content if content else f"File not found: {args['path']}"

    async def _tool_run_command(self, args: dict) -> str:
        # run_command 始终可用（VM 就是工作区）
        # 不再需要 CODEX_ENABLE_RUN_COMMAND 环境变量
        if not self.sandbox.alive:
            return "Error: sandbox not available"
        result = self.sandbox.run_in_paper(self.slug, args["command"])
        return f"exit_code: {result.exit_code}\n{result.logs}"[:8000]

    def _sanitize_path(self, rel: str) -> Optional[str]:
        """阻止 .. 和绝对路径，防止跨论文访问。"""
        ...
```

**关键区别**：
- `run_command` **始终可用**（不需要 `CODEX_ENABLE_RUN_COMMAND=true`），因为 VM 就是工作区
- `write_file` 直接写 VM，Agent 写完可以立即 `run_command` 执行，零延迟
- `read_file` 直接从 VM 读，Agent 能看到其他 Agent 的产出

### 4.3 三个 Agent 角色

#### 4.3.1 Planner Agent

```python
# src/paperbot/infrastructure/swarm/agents/planner.py

class PlannerAgent:
    """在 VM 的 .plan/ 目录中写入结构化计划。

    输入：context_pack（论文的提取结果）
    输出：.plan/roadmap.md, .plan/tasks.json, .plan/context.md
    """

    async def plan(
        self,
        sandbox: SharedSandbox,
        paper_slug: str,
        context_pack: dict,
        on_step: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        # 1. 读取 VM 中已有的项目结构（可能有前序工作）
        existing = sandbox.list_files(paper_slug, ".")

        # 2. Claude 分解任务
        tasks = await self.commander.decompose(context_pack)

        # 3. 直接写入 VM 的 .plan/ 目录
        sandbox.run_in_paper(paper_slug, "mkdir -p .plan")
        sandbox.write_file(paper_slug, ".plan/roadmap.md", self._build_roadmap(...))
        sandbox.write_file(paper_slug, ".plan/tasks.json", json.dumps(tasks, ...))
        sandbox.write_file(paper_slug, ".plan/context.md", self._build_context_summary(...))

        return tasks
```

#### 4.3.2 Executor Sub-Agent

```python
# src/paperbot/infrastructure/swarm/agents/executor.py

class ExecutorAgent:
    """在 VM 中实现代码。

    读取 .plan/ → 实现代码 → 写入 .status/{task_id}.json

    所有操作通过 SandboxToolExecutor，直接在 VM 内完成。
    Agent 写的每一行代码都立刻在 VM 中可执行。
    """

    async def execute(
        self,
        task: AgentTask,
        sandbox: SharedSandbox,
        paper_slug: str,
        *,
        on_step: Optional[Callable] = None,
    ) -> CodexResult:
        # 1. 从 VM 读取计划和前序状态
        plan = sandbox.read_file(paper_slug, ".plan/roadmap.md") or ""
        context = sandbox.read_file(paper_slug, ".plan/context.md") or ""
        prior = self._read_prior_status(sandbox, paper_slug)

        # 2. 创建 VM-native 工具执行器
        tool_exec = SandboxToolExecutor(sandbox, paper_slug, task)

        # 3. CodeAct 工具循环
        #    Agent: write_file("src/model.py", "...") → 文件立刻在 VM 中
        #    Agent: run_command("python src/model.py") → 直接执行，零延迟
        #    Agent: 观察输出 → 修复 → 再次 run_command
        result = await self.dispatcher.dispatch_with_tools_executor(
            task_id=task.id,
            prompt=prompt,
            tool_executor=tool_exec,
            max_iterations=25,
        )

        # 4. 写入状态文件到 VM
        sandbox.run_in_paper(paper_slug, "mkdir -p .status")
        sandbox.write_file(paper_slug, f".status/{task.id}.json", json.dumps({
            "task_id": task.id,
            "success": result.success,
            "files_generated": tool_exec.files_written,
            "summary": (result.output or "")[:1000],
        }))

        return result
```

#### 4.3.3 Knowledge Manager

```python
# src/paperbot/infrastructure/swarm/agents/knowledge_manager.py

class KnowledgeManager:
    """整理 VM 中论文目录的产出。

    在所有 Executor 完成后运行：
    1. 扫描 VM 中的文件
    2. 生成 .knowledge/summary.md, conventions.md, learnings.md
    3. 清理临时文件（.status/）
    4. 更新 Commander wisdom
    """

    async def curate(
        self,
        sandbox: SharedSandbox,
        paper_slug: str,
        completed_tasks: List[AgentTask],
    ) -> Dict[str, str]:
        all_files = sandbox.list_files(paper_slug, ".")
        # ... 分析文件内容，生成知识文件 ...
        sandbox.run_in_paper(paper_slug, "mkdir -p .knowledge")
        sandbox.write_file(paper_slug, ".knowledge/summary.md", summary)
        # ... 清理 .status/ ...
        sandbox.run_in_paper(paper_slug, "rm -rf .status")
        return written
```

### 4.4 验证与修复循环

```python
# src/paperbot/infrastructure/swarm/verification.py

class VerificationPolicy:
    enabled: bool
    commands: List[str]       # e.g. ["pip install -r requirements.txt", "pytest -q"]
    timeout_seconds: int
    max_repair_attempts: int

    @classmethod
    def from_sandbox_env(cls, sandbox: SharedSandbox, paper_slug: str) -> "VerificationPolicy":
        """从 VM 中的项目文件自动推断验证命令。"""
        files = sandbox.list_files(paper_slug, ".")
        commands = []
        if "requirements.txt" in files:
            commands.append("pip install -r requirements.txt")
        if any(f in files for f in ("pyproject.toml", "setup.py", "requirements.txt")):
            commands.append("PYTHONPATH=. pytest -q")
        elif "package.json" in files:
            commands.append("npm install && npm test")
        return cls(enabled=bool(commands), commands=commands, ...)

def run_verification(
    sandbox: SharedSandbox,
    paper_slug: str,
    policy: VerificationPolicy,
    attempt: int = 0,
) -> VerificationResult:
    """在 VM 的论文目录中执行验证命令。"""
    combined = " && ".join(policy.commands)
    result = sandbox.run_in_paper(paper_slug, combined, timeout_sec=policy.timeout_seconds)
    return VerificationResult(
        passed=result.success,
        exit_code=result.exit_code,
        logs=result.logs[:4000],
        attempt=attempt,
    )
```

**修复循环**：

```python
async def verify_and_repair(
    sandbox: SharedSandbox,
    paper_slug: str,
    policy: VerificationPolicy,
    dispatcher: CodexDispatcher,
    tool_executor: SandboxToolExecutor,
    on_step: Optional[Callable] = None,
) -> VerificationResult:
    """验证 → 失败则修复 → 重试，最多 N 次。"""
    for attempt in range(policy.max_repair_attempts + 1):
        result = run_verification(sandbox, paper_slug, policy, attempt)
        if result.passed:
            return result
        if attempt < policy.max_repair_attempts:
            # 用失败日志构造修复 prompt
            repair_prompt = build_repair_prompt(result.logs, result.commands_run)
            await dispatcher.dispatch_with_tools_executor(
                task_id=f"repair-{attempt}",
                prompt=repair_prompt,
                tool_executor=tool_executor,
            )
    return result  # 最终结果（可能仍失败）
```

### 4.5 下载产出到本地

**这是 sandbox-as-workspace 的关键新步骤**。只在验证通过后执行。

```python
# SharedSandbox.download_paper()

def download_paper(
    self,
    slug: str,
    local_dir: Path,
    skip_dirs: Set[str] = {".plan", ".status", ".knowledge", "__pycache__", ".git"},
) -> List[str]:
    """将 VM 中论文的产出文件下载到本地。

    仅下载代码文件，跳过 Agent 通信目录（.plan, .status, .knowledge）。
    """
    downloaded = []
    files = self._recursive_list(slug)  # VM 中递归列出所有文件
    for remote_path in files:
        parts = Path(remote_path).parts
        if any(p in skip_dirs for p in parts):
            continue
        content = self.read_file(slug, remote_path)
        if content is not None:
            local_path = local_dir / remote_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content)
            downloaded.append(remote_path)
    return downloaded
```

### 4.6 agent_board.py 主流程重构

```python
async def _run_all_stream(session: BoardSession, ...):
    """Phase 1B 主流程（sandbox-as-workspace）。"""

    sandbox = _get_or_create_shared_sandbox(session)
    if not sandbox or not sandbox.alive:
        yield sse_event("error", {"message": "Sandbox unavailable. Cannot proceed."})
        return  # Fail-closed，不降级

    slug = session.paper_slug_name

    # ── Stage 1: Planner ──
    yield sse_event("stage", {"stage": "planning", "paper_slug": slug})
    sandbox.ensure_paper_dir(slug)
    planner = PlannerAgent(commander)
    tasks = await planner.plan(sandbox, slug, context_pack, on_step=...)
    yield sse_event("plan_written", {"tasks_count": len(tasks), "paper_slug": slug})

    # ── Stage 2: Execute Tasks ──
    for i, task in enumerate(tasks):
        yield sse_event("executor_started", {"task_id": task.id, "index": i, "total": len(tasks)})

        executor = ExecutorAgent(dispatcher)
        result = await executor.execute(task, sandbox, slug, on_step=...)

        yield sse_event("executor_finished", {"task_id": task.id, "success": result.success})

    # ── Stage 3: Verification + Repair ──
    policy = VerificationPolicy.from_sandbox_env(sandbox, slug)
    if policy.enabled:
        tool_exec = SandboxToolExecutor(sandbox, slug)
        vresult = await verify_and_repair(sandbox, slug, policy, dispatcher, tool_exec, ...)
        yield sse_event("verify_finished", {"passed": vresult.passed, ...})
    else:
        yield sse_event("verify_skipped", {"reason": "no_commands"})
        vresult = None

    # ── Stage 4: Knowledge Manager ──
    km = KnowledgeManager(commander)
    await km.curate(sandbox, slug, tasks)
    yield sse_event("knowledge_curated", {"paper_slug": slug})

    # ── Stage 5: Download to Local（仅在验证通过后） ──
    if vresult is None or vresult.passed:
        local_dir = session.workspace_dir / slug
        downloaded = sandbox.download_paper(slug, local_dir)
        yield sse_event("download_complete", {
            "paper_slug": slug,
            "files_count": len(downloaded),
            "local_dir": str(local_dir),
        })
    else:
        yield sse_event("download_skipped", {
            "reason": "verification_failed",
            "paper_slug": slug,
        })

    yield sse_event("session_complete", {"paper_slug": slug})
```

### 4.7 Fail-Closed 策略

**无 sandbox 时不降级到本地工作区**。这是与旧架构最根本的区别。

```python
# SandboxToolExecutor

async def _tool_write_file(self, args):
    if not self.sandbox.alive:
        return "Error: sandbox not available. Cannot write files."
    ...

async def _tool_run_command(self, args):
    if not self.sandbox.alive:
        return "Error: sandbox not available. Cannot execute commands."
    ...

# agent_board.py: _run_all_stream
if not sandbox or not sandbox.alive:
    yield sse_event("error", {"message": "Sandbox unavailable"})
    return  # 直接结束，不降级
```

**不再需要的环境变量**：
- `CODEX_ENABLE_RUN_COMMAND` — sandbox-as-workspace 中 run_command 始终可用
- 降级逻辑（"无 sandbox 时回退到本地 dispatch_with_tools"）不再存在

---

## 5. VM 文件系统约定

```
/home/user/
├── {paper_slug_A}/                     ← 论文 A
│   ├── .plan/                          ← Planner Agent 写入
│   │   ├── roadmap.md                  ← 结构化 roadmap
│   │   ├── tasks.json                  ← 任务分解（ID、依赖、描述）
│   │   └── context.md                  ← 论文上下文摘要
│   │
│   ├── .status/                        ← Executor 写入（任务间通信，完成后清理）
│   │   ├── task-abc123.json            ← 任务 A 的完成状态
│   │   └── task-def456.json
│   │
│   ├── .knowledge/                     ← Knowledge Manager 写入（持久）
│   │   ├── summary.md                  ← 项目总结
│   │   ├── conventions.md              ← 代码规范
│   │   └── learnings.md                ← 经验教训
│   │
│   ├── src/                            ← Executor 生成的代码
│   ├── tests/
│   ├── requirements.txt
│   └── ...
│
├── {paper_slug_B}/                     ← 论文 B（完全隔离）
│   └── ...
│
└── .sandbox-meta/                      ← 全局元数据（可选）
    └── active_papers.json
```

**隔离保证**：
- `SandboxToolExecutor._sanitize_path()` 阻止 `..` 和绝对路径
- `run_in_paper()` 的 `cwd` 固定为论文根目录
- 每个 Agent 绑定 `paper_slug`，无法跨论文操作

---

## 6. 前端集成

### 6.1 新增 API：sandbox 文件浏览

```python
@router.get("/sandbox/papers")
async def list_sandbox_papers():
    """列出 VM 中所有论文目录。"""

@router.get("/sandbox/papers/{paper_slug}/files")
async def list_paper_files(paper_slug: str, path: str = "."):
    """列出 VM 中指定论文的文件树。"""

@router.get("/sandbox/papers/{paper_slug}/file")
async def read_paper_file(paper_slug: str, path: str):
    """读取 VM 中指定文件内容（Monaco 编辑器预览）。"""
```

### 6.2 前端使用

```typescript
// AgentBoard.tsx
// session 创建时返回 paper_slug
const session = await createSession(...)
// session.paper_slug = "attention-is-all-you-need-a9b1"

// 文件浏览器直接读 VM 中的文件
const files = await fetch(`/api/sandbox/papers/${session.paper_slug}/files`)

// Monaco 编辑器预览 VM 中的文件
const content = await fetch(`/api/sandbox/papers/${session.paper_slug}/file?path=src/model.py`)
```

### 6.3 SSE 事件 Payload Schema

```typescript
// 沙箱生命周期
{ event: "sandbox_init", data: { status: "ready" | "unavailable" } }

// Planner
{ event: "plan_written", data: { paper_slug, tasks_count, files: string[] } }

// Executor
{ event: "executor_started", data: { task_id, paper_slug, index, total } }
{ event: "tool_call", data: { task_id, step, tool, args_summary, observation_preview } }
{ event: "executor_finished", data: { task_id, success, files_written: string[] } }

// 验证
{ event: "verify_started", data: { commands: string[], attempt } }
{ event: "verify_finished", data: { passed, exit_code, attempt, logs_preview } }
{ event: "verify_skipped", data: { reason: "disabled" | "no_commands" } }

// 修复
{ event: "repair_started", data: { attempt, reason } }
{ event: "repair_finished", data: { attempt, files_changed: string[] } }

// Knowledge Manager
{ event: "knowledge_curated", data: { paper_slug, files_written: string[] } }

// 下载
{ event: "download_complete", data: { paper_slug, files_count, local_dir } }
{ event: "download_skipped", data: { reason: "verification_failed" } }

// 完成
{ event: "session_complete", data: { paper_slug } }
```

---

## 7. 文件变更清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `src/paperbot/infrastructure/swarm/shared_sandbox.py` | 用户级 VM 抽象，文件操作 + 下载 |
| **新建** | `src/paperbot/infrastructure/swarm/sandbox_tool_executor.py` | VM-native 工具执行器 |
| **新建** | `src/paperbot/infrastructure/swarm/verification.py` | VerificationPolicy + verify_and_repair |
| **新建** | `src/paperbot/infrastructure/swarm/agents/__init__.py` | agents 子包 |
| **新建** | `src/paperbot/infrastructure/swarm/agents/planner.py` | Planner Agent |
| **新建** | `src/paperbot/infrastructure/swarm/agents/executor.py` | Executor Sub-Agent |
| **新建** | `src/paperbot/infrastructure/swarm/agents/knowledge_manager.py` | Knowledge Manager |
| **修改** | `src/paperbot/infrastructure/swarm/worker_tools.py` | 标记 `ToolExecutor` 为 legacy，不再用于 1B 流程 |
| **修改** | `src/paperbot/infrastructure/swarm/codex_dispatcher.py` | 新增 `dispatch_with_tools_executor()` |
| **修改** | `src/paperbot/infrastructure/swarm/claude_commander.py` | 新增 `build_repair_prompt()` |
| **修改** | `src/paperbot/infrastructure/swarm/__init__.py` | 导出新符号 |
| **修改** | `src/paperbot/repro/e2b_executor.py` | 新增 `upload_single_file()`、`download_file()`、`recursive_list()` |
| **修改** | `src/paperbot/api/routes/agent_board.py` | 5 阶段主流程 + sandbox 文件浏览 API + fail-closed |
| **新建** | `tests/unit/test_shared_sandbox.py` | SharedSandbox 测试 |
| **新建** | `tests/unit/test_sandbox_tool_executor.py` | SandboxToolExecutor 测试 |
| **新建** | `tests/unit/test_verification.py` | 验证 + 修复循环测试 |
| **新建** | `tests/unit/test_planner_agent.py` | Planner 测试 |
| **新建** | `tests/unit/test_executor_agent.py` | Executor 测试 |
| **新建** | `tests/unit/test_knowledge_manager.py` | Knowledge Manager 测试 |

---

## 8. 实施顺序

```
Step 1: shared_sandbox.py（新建）
  └── SharedSandbox 类：文件读写 + 命令执行 + download_paper

Step 2: sandbox_tool_executor.py（新建）
  └── SandboxToolExecutor：替代 ToolExecutor，所有操作走 VM

Step 3: verification.py（新建）
  └── VerificationPolicy.from_sandbox_env() + run_verification + verify_and_repair

Step 4: agents/planner.py（新建）
  └── 在 VM 中写 .plan/ 目录

Step 5: agents/executor.py（新建）
  └── 用 SandboxToolExecutor 在 VM 中实现代码

Step 6: agents/knowledge_manager.py（新建）
  └── 整理 VM 中的产出到 .knowledge/

Step 7: codex_dispatcher.py 修改
  └── dispatch_with_tools_executor() 接受 SandboxToolExecutor

Step 8: claude_commander.py 修改
  └── build_repair_prompt()

Step 9: e2b_executor.py 扩展
  └── download_file(), recursive_list()

Step 10: agent_board.py 重构
  └── 5 阶段主流程 + fail-closed + sandbox 文件浏览 API

Step 11: 测试（6 个新文件）

Step 12: worker_tools.py
  └── 标记 ToolExecutor 为 legacy
```

---

## 9. 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `E2B_API_KEY` | (无) | E2B API 密钥（**必须**，sandbox-as-workspace 无降级） |
| `PAPERBOT_SANDBOX_MODE` | `"persistent"` | VM 生命周期模式 |
| `CODEX_TOOL_USE` | `"true"` | 工具循环开关 |
| `CODEX_ENABLE_VERIFICATION` | `"true"` | 验证 gate |
| `CODEX_VERIFY_COMMANDS` | (自动检测) | 验证命令 |
| `CODEX_VERIFY_TIMEOUT_SECONDS` | `180` | 验证超时 |
| `CODEX_MAX_REPAIR_ATTEMPTS` | `2` | 最大修复次数 |

**已移除**：
- ~~`CODEX_ENABLE_RUN_COMMAND`~~：sandbox-as-workspace 中 run_command 始终可用

---

## 10. 回滚策略

| 级别 | 操作 | 效果 |
|------|------|------|
| L1 | `CODEX_ENABLE_VERIFICATION=false` | 关闭验证，保留 VM 执行 |
| L2 | 设 `PAPERBOT_SANDBOX_MODE=disabled` | agent_board 返回 503，提示需要 sandbox |
| L3 | `CODEX_TOOL_USE=false` | 回退到 1A 路径（本地单次生成，无 tool loop） |

**注意**：L3 是唯一会回退到本地文件操作的方式，仅作为紧急回滚。正常运行**必须**有 sandbox。

---

## 11. 验收标准

1. **VM 是唯一工作区**：Agent 的 `write_file` 直接写 VM，不经本地
2. **零 desync**：Agent 写完文件后立刻能 `run_command` 执行，无需上传
3. **跨 Agent 可见**：Executor 2 能读到 Executor 1 在 VM 中写的文件
4. **Planner 写 .plan/**：计划文件在 VM 中，Executor 从 VM 读取
5. **验证在 VM 中**：`pytest` 等命令在论文目录下运行
6. **修复循环**：验证失败 → Agent 在 VM 中修复 → 重新验证 → 最多 N 次
7. **成功后下载**：验证通过后 `download_paper()` 拉文件到本地
8. **失败不下载**：验证失败时不下载，文件留在 VM 中供调试
9. **Fail-closed**：无 sandbox 时不降级，返回错误
10. **论文隔离**：不同论文的文件在 `/home/user/{paper_slug}/` 下互不干扰
11. **前端 scope**：文件浏览 API 返回 VM 中指定论文的文件树
