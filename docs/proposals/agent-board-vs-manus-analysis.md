# Agent Board 架构分析：与 Manus 对比及优化方案

## 一、当前功能拆解

系统采用**三层架构**：

| 层级 | 组件 | 功能 |
|------|------|------|
| **规划层** | `ClaudeCommander.decompose()` | 将 Context Pack 发送给 Claude API → 获取 JSON 任务数组 |
| **执行层** | `CodexDispatcher.dispatch()` | 将 Prompt 发送给 OpenAI → 提取代码块 → 写入文件 |
| **审查层** | `ClaudeCommander.review()` | 将输出发回 Claude → 获取 approve/reject 的 JSON |

### 已实现的功能

- 基于 Context Pack 的任务分解（带 Roadmap 降级回退）
- 带心跳 SSE 的顺序任务执行
- 从 LLM 输出中提取文件（基于正则的代码块解析）
- 基础 AI 审查（通过/拒绝）
- 工作区安全（危险路径屏蔽、允许根目录）
- 通过 `PipelineSessionStore` 的会话持久化
- 用户审查文档生成（`_build_user_review_doc`）
- 人工审查端点（approve/request_changes）

### 占位/缺失的功能

- `swarm/aci/` — 空目录（MCP/工具集成）
- `swarm/adapters/` — 空目录（Agent 适配器）
- `swarm/analytics/` — 空目录（分析模块）
- 已有的 `core/collaboration/` 框架（`AgentCoordinator`、`ScoreShareBus`、`CollaborationBus`、`HostOrchestrator`）**完全未接入 Agent Board**

---

## 二、当前核心问题

### P1：没有真正的代码执行 —— "盲写代码"

Codex 通过 `chat.completions.create` 生成代码，但从不执行。Prompt 中写了"运行测试验证正确性"，但实际上没有沙箱、没有 REPL、没有反馈循环。纯文本输入输出，生成的代码未经验证就直接写入磁盘。

### P2：没有上下文感知 —— 任务互相隔离

每次 Codex 调度只拿到自己的任务描述和累积的"智慧"（仅仅是类似 `"Completed: X -- output length: Y chars"` 的完成记录）。Worker 无法：
- 读取现有项目文件
- 查看之前任务的输出
- 理解跨文件依赖
- 访问实际代码库结构

### P3：仅支持顺序执行

`_run_all_stream` 在 `for` 循环中逐个处理任务。尽管 `CodexDispatcher` 中存在 `dispatch_parallel`，但从未被调用。独立任务无法并行运行。

### P4：没有重试或迭代循环

如果 Codex 生成了低质量代码，Claude 审查 → 拒绝 → 转入 "human_review"。没有自动重试并携带反馈。系统对每个任务是一次性的。

### P5：审查过于浅层

`commander.review()` 将输出截断为 8000 字符，问一个简单的 yes/no 问题。没有静态分析、没有测试执行、没有 diff 审查、没有结构化评分标准。

### P6：没有文件依赖图

任务从分解中获得 `dependencies` 字段，但 `_run_all_stream` 完全忽略它 —— 任务按数组顺序运行，不考虑依赖关系。

### P7：智慧积累过于简单

`accumulate_wisdom()` 仅存储 `"Completed: {title} -- output length: N chars"`。没有提取实际的学习经验、模式或约定。

---

## 三、与 Manus 逐项对比

### 3.1 沙箱执行环境

**Manus**：每个 Agent 运行在隔离的云端 VM（E2B Firecracker 微虚拟机）中，拥有完整的 Linux 环境 —— 带 sudo 权限的 shell、Python/Node 解释器、文件系统、包管理器。Agent 可以 `pip install`、运行测试、启动服务器并实时观察结果。用户关闭浏览器后任务仍在后台继续运行。

**当前系统**：`CodexDispatcher` 调用 `chat.completions.create` → 解析文本输出 → 写入文件。没有执行，没有验证。讽刺的是，代码库中已经有 `repro/docker_executor.py` 和 `repro/e2b_executor.py` —— 只是没有接入 Agent Board。

**差距**：需要为每个 Codex Worker 包装一个沙箱，使其能实际运行和测试生成的代码。

### 3.2 工具调用的 Agent 循环（CodeAct 模式）

**Manus**：通过函数调用使用 29 个工具，在迭代循环中运行：
- **Shell 工具**：`execute_command`、`view_shell`、`wait_for_process`、`write_to_process`、`kill_process`
- **浏览器工具**：`browser_navigate`、`browser_click`、`browser_type`、`browser_read`
- **文件工具**：`file_read`、`file_write`、`file_list`
- **规划工具**：`todo_create`、`todo_update`、`todo_complete`
- **通信工具**：`message_user`、`attach_file`
- **部署工具**：`deploy_expose_port`

每次迭代：**选择工具 → 在沙箱中执行 → 观察结果 → 决定下一步动作**。每次迭代一个工具调用，循环直到完成。

**当前系统**：一次性文本生成。没有工具调用，没有观察-行动循环。Worker 在单次 LLM 调用中生成所有内容。

**差距**：这是最根本的区别。需要将 `CodexDispatcher.dispatch()` 从单次 API 调用转变为工具调用循环：

```python
# Manus 的做法（伪代码）
while not task_complete:
    action = llm.decide_next_action(context, available_tools)
    result = sandbox.execute(action)
    context.append(observation=result)
```

### 3.3 上下文感知的工具状态机

**Manus**：不会一次暴露所有 29 个工具。使用**状态机**根据当前阶段动态显示/隐藏工具。规划阶段 → 仅显示规划工具。编码阶段 → 文件 + shell 工具。浏览器研究阶段 → 浏览器工具。这防止 LLM 被不相关选项淹没，提高动作选择准确率。

同时在解码时 mask token logits，根据状态强制/阻止特定工具的选择。

**当前系统**：完全没有工具管理（因为根本没有工具）。`core/collaboration/` 中有带 Agent 注册的 `AgentCoordinator`，但没有工具级别的状态管理。

**差距**：添加工具时，不要把所有工具都塞进每个 Prompt。根据阶段进行门控。

### 3.4 KV-Cache 优化

**Manus**：将 KV-cache 命中率视为**最重要的生产指标**。关键技术：
- 保持 System Prompt 前缀稳定（不在开头放时间戳）
- 工具定义稳定并置于前部
- 仅追加上下文（从不重写历史）
- 缓存 token 成本 0.30 USD/MTok vs 未缓存 3 USD/MTok — **10 倍差异**

**当前系统**：每次 `ClaudeCommander` 调用都从头构建全新 Prompt。没有缓存意识。每次 `decompose()`、`review()` 和 `build_codex_prompt()` 都按全价未缓存定价付费。

**差距**：重构 Prompt 使 System Prompt + 工具定义成为稳定前缀。使用 Anthropic 的 Prompt Caching 或 OpenAI 的缓存补全。

### 3.5 文件系统作为扩展记忆

**Manus**：将沙箱文件系统作为**无限持久上下文**。Agent 将中间结果、计划、笔记和数据写入文件，需要时再读回。这完全绕过了上下文窗口限制。

核心模式：不是把所有内容塞进 Prompt，而是写入 `todo.md`、`notes.md`、`progress.json`，然后按需选择性读回。

**当前系统**：`WisdomEntry` 将学习经验以 Python 列表形式存储在内存中。进程重启后所有智慧丢失。`repro/memory/` 中的 `CodeMemory` 和 `SymbolIndex` 正确实现了此功能，但未接入 Agent Board。

**差距**：给 Worker 一个暂存目录。将计划、中间结果和经验教训作为文件持久化到工作区。

### 3.6 Wide Research —— 并行通用 Agent

**Manus**：其 "Wide Research" 功能启动 **100+ 个并行 Agent**，每个在自己的 VM 中，每个都是完整的通用 Manus 实例。与你的系统中 Commander 分配固定角色不同，每个 Manus 子 Agent 可以动态适应。协作协议处理：
- 跨 Agent 的任务分发
- 结果聚合（转为电子表格、网页）
- 去重和冲突解决
- 异步完成（Agent 在不同时间完成）

**当前系统**：`_run_all_stream` 中的顺序 `for` 循环。`dispatch_parallel` 存在但从未被调用。没有结果聚合或冲突解决。

**差距**：你的 `AgentCoordinator` 和 `CollaborationBus` 正是为此设计的，但仍未接入。

### 3.7 实时可观察性和人工接管

**Manus**：用户可以实时观看 Agent 的屏幕（浏览器、终端、文件变化）。用户可以在**任意时刻接管** —— 在终端中输入、在浏览器中点击、编辑文件 —— 然后将控制权交还 Agent。

**当前系统**：SSE 事件仅包含状态文字（`"Codex still running (45%)"`、`"Claude started review"`）。无法看到 Worker 实际在做什么。人工审查端点存在，但是事后批准，不是实时介入。

**差距**：添加结构化执行日志，显示实际运行的命令、写入的文件和遇到的错误 —— 而不仅仅是阶段标签。

### 3.8 TodoList 驱动的规划

**Manus**：有显式的 `todo_create`、`todo_update`、`todo_complete` 工具。Agent 在工作过程中维护实时任务列表，勾选已完成项目并添加执行过程中发现的新项目。这给 Agent 提供了**进度自感知**。

**当前系统**：任务和子任务在 `decompose()` 期间预先创建，Worker 从不更新。Worker 甚至不知道存在什么子任务 —— 它只是拿到一个 Prompt。

**差距**：让 Worker 在工作过程中更新自己的子任务状态。将 Todo 状态反馈到 Agent 循环中。

---

## 四、差距优先级总结

| 优先级 | Manus 功能 | 当前状态 | 工作量 | 影响 |
|--------|-----------|---------|--------|------|
| 1 | 工具调用 Agent 循环（CodeAct） | 完全缺失 | 高 | 变革性 |
| 2 | 沙箱执行（VM/Docker） | 代码已有，未接入 | 中 | 关键 |
| 3 | 文件系统作为记忆 | `CodeMemory` 已有，未接入 | 低 | 高 |
| 4 | 并行 Agent 执行 | `dispatch_parallel` 已有，未调用 | 低 | 高 |
| 5 | 上下文感知的工具状态机 | 尚无工具 | 中 | 高 |
| 6 | KV-Cache 优化 | 无缓存意识 | 低 | 成本节省 |
| 7 | 实时可观察性 + 接管 | 仅 SSE 标签 | 中 | 用户体验 |
| 8 | 自更新 Todo/进度 | 静态子任务 | 低 | 中 |

---

## 五、优化方案

### 第一阶段：赋予 Worker 工具和环境（关键）

**A. 为 Codex Worker 添加工具调用**

将单次 `chat.completions.create` 调用替换为工具调用循环。Worker 应能够：

```python
# 给每个 Codex Worker 的工具
WORKER_TOOLS = [
    "read_file",        # 读取现有项目文件
    "write_file",       # 写入/修改文件
    "list_directory",   # 浏览工作区结构
    "run_command",      # 在沙箱中执行（测试、linter）
    "search_codebase",  # 在项目中 grep/find
]
```

这将 Worker 从"生成文本"转变为"在环境中行动"。OpenAI 的 API 支持工具调用 —— 你已经在用它了，但只用于文本生成。

**B. 添加执行沙箱**

代码库中已有 `repro/docker_executor.py` 和 `repro/e2b_executor.py`。将其中一个接入 Agent Board，使 Worker 能实际运行生成的代码：

```python
# 在 codex_dispatcher.py 中
async def dispatch(self, task_id, prompt, workspace):
    # 1. 生成代码（现有）
    # 2. 新增：在沙箱中运行代码
    # 3. 新增：如果测试失败，将错误反馈给 LLM
    # 4. 新增：迭代最多 N 次
```

### 第二阶段：添加反馈循环（高影响）

**A. 带审查反馈的重试**

当 Claude 拒绝时，不是直接丢给 human_review，而是将反馈回传给 Codex：

```python
MAX_RETRIES = 3
for attempt in range(MAX_RETRIES):
    result = await dispatcher.dispatch(task_id, prompt, workspace)
    review = await commander.review(task_dict, result.output)
    if review.approved:
        break
    # 将反馈注入下一次尝试
    prompt += f"\n\n## 审查反馈（第 {attempt+1} 次尝试）\n{review.feedback}"
```

**B. 测试驱动验证**

文件生成后，在 AI 审查之前运行自动化检查：
- Python：`pytest`、`pyright`、`ruff`
- TypeScript：`tsc --noEmit`、`eslint`、`vitest`
- 通用：文件存在、import 可解析、无语法错误

### 第三阶段：接入协作框架

你已经构建了 `AgentCoordinator`、`ScoreShareBus`、`CollaborationBus`、`HostOrchestrator` —— 它们闲置未用。接入它们：

```python
# 在 _run_all_stream 中，用以下替换顺序循环：
coordinator = AgentCoordinator()
coordinator.register("commander", commander_agent)
for task in planning_tasks:
    coordinator.register(f"worker-{task.id}", create_worker(task))

# 使用 ScoreShareBus 实现快速失败：如果某个 Worker 质量下降，
# 停止并重新规划，而不是继续浪费 API 调用
```

### 第四阶段：依赖感知调度

尊重分解中的 `dependencies` 字段：

```python
# 从任务依赖构建 DAG
dag = build_dependency_graph(planning_tasks)
# 按拓扑顺序执行，并行化独立任务
for batch in dag.topological_batches():
    results = await dispatcher.dispatch_parallel(batch, workspace)
```

### 第五阶段：丰富上下文注入

**A. 项目感知的 Prompt**

分派前扫描工作区并注入上下文：

```python
async def build_codex_prompt(self, task, workspace):
    # 现有 Prompt 构建...

    # 新增：注入相关的现有文件
    relevant_files = await self._find_relevant_files(task, workspace)
    for path, content in relevant_files:
        parts.append(f"## 现有文件: {path}\n```\n{content}\n```")

    # 新增：注入依赖任务的输出
    for dep_output in task.get("dependency_outputs", []):
        parts.append(f"## 来自 {dep_output['title']} 的输出\n{dep_output['summary']}")
```

**B. 真正的智慧积累**

```python
def accumulate_wisdom(self, task, output, review):
    # 让 Claude 提取实际学习经验
    learnings = await self._extract_learnings(task, output, review)
    self.wisdom.learnings.extend(learnings.patterns)
    self.wisdom.conventions.extend(learnings.conventions)
    self.wisdom.gotchas.extend(learnings.gotchas)
```

---

## 六、关键结论

最大的差距是 **工具调用 Agent 循环 + 沙箱执行**的组合：你的 Worker 是文本生成器，Manus 的 Worker 是在沙箱中拥有工具的自主 Agent。其他所有功能都建立在这个基础之上。

好消息是，你的代码库中已经有很多构建模块（`docker_executor`、`e2b_executor`、`CodeMemory`、`AgentCoordinator`、`dispatch_parallel`），只是没有接入 Agent Board。优化的核心工作是**连接现有组件**，而非从零开始。

---

## 参考来源

- [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [Introducing Wide Research — Manus Blog](https://manus.im/blog/introducing-wide-research)
- [How Manus Uses E2B for Virtual Computers](https://e2b.dev/blog/how-manus-uses-e2b-to-provide-agents-with-virtual-computers)
- [Manus AI: Features, Architecture & More — DataCamp](https://www.datacamp.com/blog/manus-ai)
- [From Mind to Machine: The Rise of Manus AI (arXiv)](https://arxiv.org/html/2505.02024v1)
- [Manus Unveiled: Internal Prompts, Workflows, and Tool Configurations](https://medium.com/@joycebirkins/manus-unveiled-dive-into-internal-prompts-workflows-and-tool-configurations-6ee9a7e0e708)

---
---

# v2 更新：Phase 1B 实施后的重新评估（2026-03-11）

> Phase 1B（执行沙箱计划）已基本实现（~90%）。以下内容反映当前代码真实状态，对 v1 中每项差距重新评估完成度。

---

## v2-一、架构升级概览

系统已从 v1 的"三层文本生成"升级为 **Sandbox-as-Workspace** 架构：

| 层级 | 组件 | 功能 | v1→v2 变化 |
|------|------|------|-----------|
| **规划层** | `PlannerAgent` + `ClaudeCommander.decompose()` | Claude 分解任务 → 写入 `.plan/` 到 VM | **新增** PlannerAgent 类 |
| **执行层** | `ExecutorAgent` + `CodexDispatcher.dispatch_with_sandbox_tools()` | CodeAct 工具循环，在 VM 内执行代码 | **重写**：单次生成 → 工具调用循环 |
| **验证层** | `VerificationPolicy` + `RepairLoop` | 在 VM 内运行测试 → 失败则自动修复 | **新增** |
| **E2E 层** | `E2EExecution` + E2E RepairLoop | 端到端运行 → 失败则用 Codex 修复 | **新增** |
| **知识层** | `KnowledgeManager` | 提取经验写入 `.knowledge/` | **新增**（stub） |
| **审查层** | `ClaudeCommander.review()` + Human Review | AI 审查 + 人工审查端点 | 不变 |

### Agent 间通信协议（VM 文件系统）

```
/home/user/{paper_slug}/
├── .plan/
│   ├── roadmap.md        # PlannerAgent 写入：项目路线图
│   ├── context.md        # PlannerAgent 写入：论文上下文摘要
│   └── tasks.json        # PlannerAgent 写入：完整任务列表
├── .status/
│   └── {task_id}.json    # ExecutorAgent 写入：任务完成状态
├── .knowledge/
│   └── learnings.json    # KnowledgeManager 写入（目前 stub）
└── src/                  # 生成的代码
```

### Phase 1B 后新增的功能

- ✅ **Sandbox-as-Workspace**：E2B/Docker VM 作为唯一真实来源，fail-closed 设计
- ✅ **CodeAct 工具调用循环**：`dispatch_with_sandbox_tools()` 最多 50 次迭代
- ✅ **7 个 VM 原生工具**：`read_file`、`write_file`、`list_files`、`run_command`、`search_files`、`update_subtask`、`task_done`
- ✅ **Agent 类架构**：PlannerAgent 写 `.plan/`、ExecutorAgent 读 `.plan/` 写 `.status/`
- ✅ **Stage 1.5 回退守卫**：沙箱超时时从 session state 重建 `.plan/`
- ✅ **验证 + 修复循环**：`VerificationPolicy` 运行测试，失败 → Codex 修复（最多 2 轮）
- ✅ **E2E 执行 + 修复**：`E2EExecution` 端到端运行 → 失败 → Codex 修复（最多 2 轮）
- ✅ **文件浏览 API**：`/api/agent-board/sessions/{id}/files` 可浏览 VM 文件
- ✅ **下载端点**：`/api/agent-board/sessions/{id}/download` 打包 VM 文件
- ✅ **结构化执行日志**：SSE 包含 `tool_step` 事件（工具名、参数、结果）
- ✅ **Commander wisdom 注入**：已完成任务的学习传递给后续 Executor

### 仍缺失/占位的功能

- ❌ `swarm/aci/` — 空目录（MCP/工具集成）
- ❌ `swarm/adapters/` — 空目录（Agent 适配器）
- ❌ `swarm/analytics/` — 空目录（分析模块）
- ❌ `core/collaboration/` 框架（`AgentCoordinator`、`ScoreShareBus`）**仍未接入 Agent Board**
- ❌ `worker_tools.py` 旧版 ToolExecutor **未标记为 legacy**

---

## v2-二、v1 核心问题状态更新

### ~~P1：没有真正的代码执行~~ → ✅ 已解决

Codex Worker 现在通过 `SandboxToolExecutor` 在 VM 内直接执行代码。`run_command` 工具可运行任意命令，`write_file` 直接写入 VM 文件系统。每个工具调用的结果作为 observation 反馈给 LLM。

### ~~P2：没有上下文感知~~ → ✅ 大幅改善

ExecutorAgent 从 `.plan/` 读取 roadmap 和 context，从 `.status/` 读取已完成任务状态，通过 `list_files`/`read_file` 工具访问项目文件。`wisdom` 参数传递跨任务学习经验。

**残余问题**：wisdom 积累仍是 stub（仅存储 title + output length），未提取真正的模式和经验。

### P3：仅支持顺序执行 → ⚠️ 未解决

`_run_all_stream_sandbox` 仍在 `for` 循环中逐个处理任务。`dispatch_parallel` 存在但未被调用。独立任务无法并行。

### ~~P4：没有重试或迭代循环~~ → ✅ 已解决

- **工具循环内**：CodeAct 循环最多 50 次迭代，LLM 可自主诊断错误并重试
- **验证修复**：`VerificationPolicy` 测试失败 → Codex 修复 → 重新验证（最多 2 轮）
- **E2E 修复**：端到端失败 → Codex 修复 → 重跑（最多 2 轮）

### ~~P5：审查过于浅层~~ → ✅ 大幅改善

审查不再仅靠 Claude 的文本判断。`VerificationPolicy` 实际运行测试：
- `pytest -q`（Python）
- `node main.js`（Node.js）
- 自定义验证命令

E2E 验证提供真实的执行输出作为审查依据。

**残余问题**：没有静态分析（pyright/ruff/eslint）集成。

### P6：没有文件依赖图 → ⚠️ 未解决

任务的 `dependencies` 字段仍被忽略。顺序执行掩盖了此问题，但无法支持并行化。

### P7：智慧积累过于简单 → ⚠️ 部分改善

Commander 现在通过 `wisdom` 参数将学习传递给后续 Executor prompt。但 `accumulate_wisdom()` 仍仅存储 `"Completed: {title} -- output length: N chars"`，未提取真正的技术经验。

---

## v2-三、与 Manus 逐项对比（重新评估）

### 3.1 沙箱执行环境 → ✅ 已实现

**当前系统（v2）**：通过 `SharedSandbox` 封装 E2B 或 Docker executor，每个 session 共享同一 VM。`PersistentSandboxManager` 管理 VM 生命周期（自动创建、空闲回收）。Agent 可在 VM 内 `pip install`、运行代码、读写文件。

**剩余差距**：
- Manus 每个 Agent 有独立 VM；当前所有 Agent 共享同一 VM（适合论文复现场景）
- Manus 支持后台持续运行（关闭浏览器后继续）；当前 VM 空闲后回收

### 3.2 工具调用的 Agent 循环（CodeAct 模式） → ✅ 已实现

**当前系统（v2）**：`CodexDispatcher.dispatch_with_sandbox_tools()` 实现了完整的 CodeAct 循环：

```python
# 实际代码 (codex_dispatcher.py:~391)
while iteration < max_iterations:
    response = await client.chat.completions.create(
        model=self.model, messages=messages,
        tools=tool_executor.tool_definitions(), tool_choice="auto",
    )
    if tool_calls:
        for tc in tool_calls:
            result = tool_executor.execute(tc.function.name, args)
            messages.append({"role": "tool", "content": result})
    else:
        break  # LLM chose to stop
```

7 个 VM 原生工具（vs Manus 的 29 个）：

| 工具 | 功能 | 对应 Manus 工具 |
|------|------|---------------|
| `read_file` | 读取 VM 文件 | `file_read` |
| `write_file` | 写入 VM 文件 | `file_write` |
| `list_files` | 列出目录 | `file_list` |
| `run_command` | 执行 shell 命令 | `execute_command` |
| `search_files` | 搜索文件内容 | （内建于 shell） |
| `update_subtask` | 更新子任务状态 | `todo_update` |
| `task_done` | 标记任务完成 | `todo_complete` |

**剩余差距**：
- 缺少浏览器工具（`browser_navigate`、`browser_click` 等）
- 缺少进程管理工具（`wait_for_process`、`kill_process`、`write_to_process`）
- 缺少部署工具（`deploy_expose_port`）
- 缺少用户通信工具（`message_user`、`attach_file`）

### 3.3 上下文感知的工具状态机 → ❌ 未实现

**当前系统（v2）**：所有 7 个工具始终全部暴露，不区分阶段。`SandboxToolExecutor.tool_definitions()` 返回固定列表。

**差距**：当工具数量增加时需要状态机门控。当前 7 个工具尚可管理，但扩展到浏览器/部署工具后将成为问题。

### 3.4 KV-Cache 优化 → ❌ 未实现

**当前系统（v2）**：每次 `dispatch_with_sandbox_tools` 调用从头构建 messages。每次 `decompose()`、`review()` 按全价付费。

**差距**：纯成本优化。功能正确但价格高。

### 3.5 文件系统作为扩展记忆 → ✅ 已实现

**当前系统（v2）**：这是 Sandbox-as-Workspace 架构的核心：
- `.plan/roadmap.md` — 项目路线图（持久化在 VM）
- `.plan/context.md` — 论文上下文摘要
- `.plan/tasks.json` — 完整任务列表
- `.status/{task_id}.json` — 每个任务的完成状态
- `.knowledge/learnings.json` — 经验教训（KnowledgeManager 写入）
- Executor 可通过 `read_file`/`list_files` 访问所有之前的代码和输出

**剩余差距**：
- KnowledgeManager 仍是 stub，`learnings.json` 内容有限
- 没有 `notes.md` 等 Agent 自由笔记机制

### 3.6 Wide Research —— 并行通用 Agent → ❌ 未实现

**当前系统（v2）**：仍是顺序 `for` 循环。`dispatch_parallel` 存在但未被调用。`AgentCoordinator` 和 `CollaborationBus` 仍未接入。

**差距**：这是最大的架构差距之一。论文复现场景中任务通常有依赖关系，但环境安装、测试编写等可并行。

### 3.7 实时可观察性和人工接管 → ⚠️ 部分实现

**当前系统（v2）**：
- ✅ SSE 包含结构化 `tool_step` 事件（工具名、参数、结果摘要）
- ✅ 文件浏览 API 可查看 VM 内文件
- ✅ 前端 `AgentBoard` 展示任务状态、执行日志、生成文件
- ❌ 没有实时终端/浏览器共享
- ❌ 没有用户接管（无法在执行中介入）

**差距**：可观察性已大幅提升（从纯文本标签到结构化工具日志），但实时交互仍缺失。

### 3.8 TodoList 驱动的规划 → ✅ 已实现

**当前系统（v2）**：
- `update_subtask` 工具让 Executor 在工作中更新子任务状态
- `task_done` 工具标记任务完成并写入摘要
- `.status/{task_id}.json` 持久化完成状态
- 后续 Executor 从 `.status/` 读取前序任务完成情况

**剩余差距**：Executor 无法动态创建新子任务（只能更新预定义的）。

---

## v2-四、差距优先级总结（重新评估）

| 优先级 | Manus 功能 | v1 状态 | v2 状态 | 剩余工作 |
|--------|-----------|---------|---------|---------|
| 1 | 工具调用 Agent 循环（CodeAct） | ❌ 完全缺失 | ✅ **已实现** | 增加更多工具类型 |
| 2 | 沙箱执行（VM/Docker） | ❌ 未接入 | ✅ **已实现** | 稳定性优化 |
| 3 | 文件系统作为记忆 | ❌ 未接入 | ✅ **已实现** | KnowledgeManager 从 stub 升级 |
| 4 | TodoList 驱动规划 | ❌ 静态子任务 | ✅ **已实现** | 支持动态创建子任务 |
| 5 | 实时可观察性 | ❌ 仅 SSE 标签 | ⚠️ **部分实现** | 实时终端共享、用户接管 |
| 6 | 验证+修复循环 | ❌ 无重试 | ✅ **已实现** | 添加静态分析 |
| 7 | 并行 Agent 执行 | ❌ 未调用 | ❌ **未实现** | 接入 dispatch_parallel + DAG |
| 8 | 上下文感知工具状态机 | ❌ 无工具 | ❌ **未实现** | 工具数量增加后需要 |
| 9 | KV-Cache 优化 | ❌ 无缓存意识 | ❌ **未实现** | 成本优化 |
| 10 | 真正的智慧提取 | ❌ 简单存储 | ⚠️ **部分实现** | LLM 提取模式/经验 |
| 11 | 协作框架接入 | ❌ 未接入 | ❌ **未实现** | 接入 AgentCoordinator |

### 完成度评估

- **v1 总完成度**：约 20%（仅有规划分解和基础 SSE）
- **v2 总完成度**：约 65%（核心执行循环、沙箱、验证修复均已就位）

---

## v2-五、下一阶段优化方案

### Phase 2A：智慧提取升级（低工作量，高影响）

当前 `accumulate_wisdom()` 仅存储 `"Completed: {title} -- output length: N chars"`。应让 Claude 提取真正的技术经验：

```python
# claude_commander.py
async def accumulate_wisdom(self, task, codex_result, sandbox):
    # 让 Claude 分析执行结果，提取：
    # - 安装了哪些依赖、版本
    # - 遇到的错误及解决方案
    # - 代码架构约定（命名、目录结构）
    # - 环境特殊要求
    learnings = await self._extract_learnings(task, codex_result)
    self.wisdom.learnings.extend(learnings)
    # 写入 .knowledge/learnings.json
    knowledge_mgr.persist(sandbox, slug, learnings)
```

### Phase 2B：并行执行 + 依赖 DAG（中工作量，高影响）

```python
# 构建依赖 DAG
dag = build_dependency_graph(planning_tasks)
for batch in dag.topological_batches():
    # 同一批次内的任务并行执行
    results = await asyncio.gather(*[
        executor_agent.execute(task, shared, slug)
        for task in batch
    ])
```

需要：
1. 解析 `dependencies` 字段构建 DAG
2. 拓扑排序产生执行批次
3. 同批次内 `asyncio.gather` 并行（共享同一 VM）
4. 处理并行写入冲突

### Phase 2C：协作框架接入（中工作量，中影响）

将 `core/collaboration/` 中的 `AgentCoordinator`、`ScoreShareBus`、`FailFastEvaluator` 接入：

```python
coordinator = AgentCoordinator()
score_bus = ScoreShareBus()

# 注册 Agent
coordinator.register("planner", planner_agent)
for task in tasks:
    coordinator.register(f"executor-{task.id}", executor_agent)

# ScoreShareBus 实现快速失败：
# 如果连续任务质量下降 → 停止并请求人工审查
score_bus.on_score_below(threshold=0.3, callback=request_human_review)
```

### Phase 2D：扩展工具集（高工作量，中影响）

按需添加工具类型：

| 工具类别 | 具体工具 | 优先级 |
|---------|---------|--------|
| 进程管理 | `wait_for_process`、`kill_process` | 高（长时间训练） |
| 环境管理 | `install_package`、`set_env_var` | 高（依赖安装） |
| 浏览器 | `browser_navigate`、`browser_screenshot` | 低（论文复现不需要） |
| 部署 | `expose_port`、`create_url` | 低（可后续添加） |

### Phase 2E：KV-Cache 优化（低工作量，成本节省）

```python
# 稳定 System Prompt 前缀
SYSTEM_PREFIX = """You are an AI coding agent working inside a VM sandbox..."""
TOOL_DEFINITIONS = [...]  # 工具定义放在前部，保持稳定

# 仅追加上下文，不重写历史
messages = [
    {"role": "system", "content": SYSTEM_PREFIX},
    # 工具定义作为 system message 的一部分
    *conversation_history,  # 仅追加
]
```

---

## v2-六、关键结论

Phase 1B 实施后，**最根本的差距已被弥合**：Worker 从"文本生成器"转变为"在 VM 中拥有工具的自主 Agent"。CodeAct 循环 + Sandbox-as-Workspace + 验证修复循环构成了与 Manus 同质的核心执行引擎。

**当前最大差距**按影响排序：
1. **并行执行**（顺序处理浪费时间，独立任务应并行）
2. **智慧提取**（当前仅 stub，跨任务学习近乎为零）
3. **协作框架**（已有 AgentCoordinator 等组件，仍闲置）
4. **工具状态机**（当前 7 个工具尚可，扩展后需要）

与 Manus 相比，论文复现这一垂直场景不需要浏览器工具和部署工具，因此实际功能覆盖度比通用对比显示的更高。核心的"规划 → 执行 → 验证 → 修复"循环已经与 Manus 的架构模式一致。
