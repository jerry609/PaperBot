# AgentSwarm Design Proposal

> **Date**: 2026-03-03
> **Status**: Draft → Under Review (#214)
> **Branch**: `feat/daily-push-epic-179`
> **Related**: `docs/AGENTSWARM_TODO.md`, README AgentSwarm 模块

---

## 1. Executive Summary

AgentSwarm 是 PaperBot 的多 Agent 协作平台，目标是将「论文 → 可运行、可验证的复现工程」流程从当前的单体 Orchestrator 模式升级为多 Agent 协作模式，并统一接入不同 Code Agent（Claude Code / Codex / OpenCode / OpenHands / Cursor / Devin）。

**核心定位**：所有竞品（Devin/OpenHands/SWE-Agent/Cursor/Codex/Claude Code/OpenCode）都是通用软件工程 Agent，**没有任何平台聚焦 paper→reproducible-code 场景**。AgentSwarm 填补这个空白。

```
                Generic SWE ←─────────────→ Domain-Specific
                     |                           |
  Autonomous    Devin, Codex                     │  ★ AgentSwarm
       ↑        OpenHands                        │  (Paper→Code)
       |        SWE-Agent, OpenCode              │
       |        Cursor, Claude Code              │
  Assisted      ↓                                │
```

---

## 2. Current State Assessment

### 2.1 What's Done (可复用)

| Component | Status | Location |
|-----------|--------|----------|
| Studio 3-panel UI (Papers / ReproLog / Files) | ✅ | `web/src/app/studio/page.tsx` |
| Runbook file management (CRUD + snapshot + diff + hunk revert) | ✅ | `api/routes/runbook.py` (800+ lines) |
| Studio Chat (Claude CLI subprocess, 3 modes) | ✅ | `api/routes/studio_chat.py` |
| Sandbox job queue + log/metrics streaming | ✅ | `api/routes/sandbox.py` |
| Paper2Code pipeline (Planning→Blueprint→Env→Gen→Verify) | ✅ | `repro/orchestrator.py` + nodes/ |
| CodeMemory (cross-file context + SymbolIndex) | ✅ | `repro/memory/` |
| CodeRAG (pattern retrieval, 10 built-in patterns) | ✅ | `repro/rag/` |
| Docker + E2B executors | ✅ | `repro/docker_executor.py`, `e2b_executor.py` |
| Context Pack generation (SSE) | ✅ | `api/routes/gen_code.py` |
| MCP client (frontend) | ✅ | `web/src/lib/mcp/` |
| 15+ Studio components (Monaco editor, diff viewer, etc.) | ✅ | `web/src/components/studio/` |
| Zustand store with paper/task/agent state | ✅ | `web/src/lib/store/studio-store.ts` |

### 2.2 Key Gaps (需要补齐)

| Gap | Impact | Priority |
|-----|--------|----------|
| 无多 Agent 运行时 — Orchestrator 驱动顺序执行 | 不能并行、不能自主迭代 | P0 |
| 无 Agent 间通信 — 共享 `Dict[str,Any]` 无类型 | 脆弱、不可观测 | P0 |
| 无持久化 Agent Session — 每次 request 新进程 | 无法断点续跑 | P1 |
| `ParallelOrchestrator` 是空壳 — 直接 fallback 顺序 | 浪费 GPU/token 时间 | P1 |
| Studio Chat 用 `--print` 单次调用 — 非多轮 session | 无法迭代修复 | P1 |
| 无 tool approval workflow — `--dangerously-skip-permissions` | 安全风险 | P2 |
| MCP 前端存在但未接入 chat/backend | 工具扩展被阻断 | P2 |
| CodeRAG 仅 10 个 hardcoded pattern + keyword 匹配 | 召回率低 | P2 |
| 无 Agent 活动可视化（terminal/DAG/timeline） | 用户无法观察/干预 | P2 |
| repro 模块不走 PaperBot 自身的 `LLMService` | 模型路由无法统一 | P1 |

---

## 3. Architecture Design

### 3.1 Overall Architecture

```
┌─────────────────────── Studio UI (Next.js) ─────────────────────┐
│  Papers │ AgentSwarm Dashboard │ Workspace │ Terminal │ Timeline  │
│         │  ┌─ Agent Cards ───┐ │           │ (xterm) │           │
│         │  │ 🧠 Planner      │ │  Monaco   │         │  Events   │
│         │  │ 🔧 Coder        │ │  Editor   │  Live   │  by Run/  │
│         │  │ ✅ Verifier     │ │  + Diff   │  Output │  Step     │
│         │  │ 🐛 Debugger     │ │           │         │           │
│         │  │ 🔍 Reviewer     │ │           │         │           │
│         │  └─────────────────┘ │           │         │           │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  AgentSwarm Gateway │  (FastAPI)
                    │  /api/swarm/*       │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌─── Agent Router ───────────────────────────┐
     │                                             │
     │  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
     │  │ Session    │  │ Message  │  │ Event    │ │
     │  │ Manager    │  │ Bus      │  │ Store    │ │
     │  └───────────┘  └──────────┘  └──────────┘ │
     │                                             │
     │  ┌──────────────────────────────────────┐   │
     │  │          Agent Adapters              │   │
     │  │  ┌────────┐ ┌──────┐ ┌───────────┐  │   │
     │  │  │Claude  │ │Codex │ │ OpenHands │  │   │
     │  │  │Code    │ │      │ │           │  │   │
     │  │  ├────────┤ ├──────┤ ├───────────┤  │   │
     │  │  │Cursor  │ │Devin │ │ Built-in  │  │   │
     │  │  │        │ │      │ │ (P2C)     │  │   │
     │  │  └────────┘ └──────┘ └───────────┘  │   │
     │  └──────────────────────────────────────┘   │
     └─────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Docker   │   │ E2B      │   │ SSH      │
        │ Sandbox  │   │ Cloud    │   │ Fleet    │
        └──────────┘   └──────────┘   └──────────┘
```

### 3.2 Core Components

#### 3.2.1 Agent Router — 任务分发器

```python
# src/paperbot/infrastructure/swarm/agent_router.py

class AgentRouter:
    """Route sub-tasks to the best available agent based on task type,
    model capability, and cost/latency requirements."""

    def __init__(self, adapters: Dict[str, AgentAdapter], config: RouterConfig):
        self.adapters = adapters
        self.config = config

    async def dispatch(self, task: SwarmTask) -> AgentResult:
        """Dispatch a task to the best agent."""
        adapter = self._select_adapter(task)
        session = await self.session_mgr.get_or_create(adapter, task.workspace)
        return await adapter.execute(session, task)

    def _select_adapter(self, task: SwarmTask) -> AgentAdapter:
        """Selection strategy:
        - blueprint/planning → Claude (high reasoning)
        - bulk code generation → Codex (fast, parallel)
        - config/boilerplate → Haiku (cheap, fast)
        - debugging with tool use → Claude Code (CLI tools)
        - verification → Built-in (no LLM needed for syntax/import)
        """
        ...
```

**路由策略参考竞品最佳实践**：

| Task Type | Default Agent | Rationale |
|-----------|--------------|-----------|
| Blueprint distillation | Claude Opus/Sonnet | 需要高推理能力理解论文 |
| File structure planning | Claude Code | 可用 tool use 创建目录/文件 |
| Config/boilerplate gen | Haiku / Codex | 快速、低成本 |
| Model architecture code | Claude Opus | 需要深度理解 + CodeRAG |
| Training loop code | Codex | 模式化、适合批量 |
| Debugging/repair | Claude Code | 需要运行命令、读错误日志 |
| Verification | Built-in (no LLM) | 语法检查/import 检查不需要 LLM |

#### 3.2.2 Agent Adapter — 统一接口

```python
# src/paperbot/infrastructure/swarm/adapters/base.py

class AgentAdapter(ABC):
    """Unified interface for different code agents."""

    @property
    @abstractmethod
    def agent_type(self) -> str: ...

    @abstractmethod
    async def execute(self, session: AgentSession, task: SwarmTask) -> AgentResult: ...

    @abstractmethod
    async def create_session(self, workspace: Path, config: dict) -> AgentSession: ...

    @abstractmethod
    async def destroy_session(self, session: AgentSession) -> None: ...
```

**已规划的 Adapter 实现**：

| Adapter | 接入方式 | 状态 |
|---------|---------|------|
| `BuiltInAdapter` | 直接调用现有 P2C nodes | v1 实现 |
| `ClaudeCodeAdapter` | Claude Agent SDK / CLI subprocess | v1 实现 |
| `CodexAdapter` | Codex App Server / MCP / CLI | v2 实现 |
| `OpenHandsAdapter` | OpenHands SDK / Docker | v2 实现 |
| `OpenCodeAdapter` | OpenCode CLI (Build/Plan agents) | v2 实现 |
| `CursorAdapter` | Cursor CLI (if available) | v3 评估 |
| `DevinAdapter` | Devin API (enterprise) | v3 评估 |

#### 3.2.3 Message Bus — Agent 间通信

取代当前的 `Dict[str, Any]` 共享上下文，采用 typed event-stream 模式（借鉴 OpenHands V1）：

```python
# src/paperbot/infrastructure/swarm/message_bus.py

@dataclass
class SwarmEvent:
    event_type: str          # "blueprint_ready", "file_generated", "error_found"
    source: str              # agent name
    timestamp: datetime
    payload: Dict[str, Any]  # typed per event_type

class MessageBus:
    """Event-driven communication between agents.
    Replaces the mutable Dict[str, Any] shared context."""

    async def publish(self, event: SwarmEvent) -> None: ...
    async def subscribe(self, event_type: str, handler: Callable) -> None: ...
    async def get_history(self, event_type: str = None) -> List[SwarmEvent]: ...
```

**关键 Event 类型**：

| Event | Producer | Consumer |
|-------|----------|----------|
| `blueprint_ready` | PlannerAgent | CoderAgent |
| `plan_ready` | PlannerAgent | CoderAgent, UI |
| `file_generated` | CoderAgent | VerifierAgent, CodeMemory |
| `verification_failed` | VerifierAgent | DebuggerAgent |
| `repair_applied` | DebuggerAgent | VerifierAgent (re-run) |
| `all_verified` | VerifierAgent | ReviewerAgent, UI |
| `review_complete` | ReviewerAgent | UI, EvidencePack |

#### 3.2.4 Session Manager — 持久化 Agent 会话

```python
# src/paperbot/infrastructure/swarm/session_manager.py

class AgentSession:
    session_id: str
    agent_type: str
    workspace: Path
    status: Literal["idle", "running", "paused", "terminated"]
    created_at: datetime
    events: List[SwarmEvent]   # deterministic replay (借鉴 OpenHands)

class SessionManager:
    """Manage persistent agent sessions with replay capability."""

    async def create(self, agent_type: str, workspace: Path) -> AgentSession: ...
    async def resume(self, session_id: str) -> AgentSession: ...
    async def pause(self, session_id: str) -> None: ...
    async def replay(self, session_id: str) -> List[SwarmEvent]: ...
```

---

## 4. UI/UX Design

### 4.1 Studio 升级 — 从 3-panel 到 5-zone

借鉴 Devin（cloud IDE）和 Cursor（Mission Control）的最佳实践：

```
┌────────────────────────────────────────────────────────────┐
│ Papers │     Agent Dashboard      │ Workspace  │ Terminal  │
│ (left) │                          │ (editor)   │ (xterm)   │
│        │  ┌──────────────────┐    │            │           │
│ paper  │  │ SwarmOrchestrator│    │  Monaco    │  Agent    │
│ list   │  │ ┌──┐ ┌──┐ ┌──┐  │    │  Editor    │  stdout   │
│        │  │ │P │→│C │→│V │  │    │            │           │
│ + add  │  │ │  │ │  │ │  │  │    │  + Diff    │  + stdin  │
│        │  │ └──┘ └──┘ └──┘  │    │  Viewer    │  (user    │
│        │  │   ↗       ↘     │    │            │  approval)│
│        │  │ ┌──┐     ┌──┐   │    │            │           │
│        │  │ │D │     │R │   │    │            │           │
│        │  │ └──┘     └──┘   │    │            │           │
│        │  └──────────────────┘    │            │           │
│        │                          │            │           │
│        │  ── Evidence Timeline ── │            │           │
│        │  [Run #3] Blueprint ✅   │            │           │
│        │  [Run #3] Gen model.py ✅│            │           │
│        │  [Run #3] Gen train.py 🔄│            │           │
│        │  [Run #3] Verify...  ⏳  │            │           │
└────────────────────────────────────────────────────────────┘

P=Planner  C=Coder  V=Verifier  D=Debugger  R=Reviewer
```

### 4.2 Agent Card UI

每个 Agent 显示为可交互卡片：

```
┌─────────────────────────────┐
│ 🧠 Planner (Claude Opus)    │
│ Status: ✅ Complete          │
│ Tokens: 12.4k in / 3.2k out│
│ Duration: 8.3s              │
│ Output: blueprint.json      │
│ [View Log] [View Output]    │
└─────────────────────────────┘
```

### 4.3 Interactive Planning（借鉴 Devin 2.0）

在代码生成前，展示可编辑的执行计划：

```
┌── Execution Plan (editable) ──────────────────┐
│                                                │
│ 📄 Paper: "Attention Is All You Need"          │
│                                                │
│ Step 1: Extract architecture from Section 3    │
│   → Multi-head attention + FFN + LayerNorm     │
│   → Model: Claude Opus (high reasoning)        │
│                                                │
│ Step 2: Generate file structure                │
│   ├── config.py (hyperparams from Table 3)     │
│   ├── model.py (Transformer architecture)      │
│   ├── data.py (WMT14 En-De loader)             │
│   ├── train.py (Adam, lr warmup from Sec 5.3)  │
│   └── evaluate.py (BLEU score)                 │
│   → Model: Claude Code (tool use)              │
│                                                │
│ Step 3: Verify against Table 2 results         │
│   → Expected: BLEU 28.4 on WMT14 En-De        │
│   → Sandbox: Docker (GPU optional)             │
│                                                │
│ [✏️ Edit Plan]  [▶️ Execute]  [💾 Save Template] │
└────────────────────────────────────────────────┘
```

---

## 5. API Design

### 5.1 New Endpoints

```
# Swarm orchestration
POST /api/swarm/sessions              # Create new swarm session
GET  /api/swarm/sessions/{id}         # Get session status + agent states
POST /api/swarm/sessions/{id}/start   # Start execution
POST /api/swarm/sessions/{id}/pause   # Pause (agents checkpoint)
POST /api/swarm/sessions/{id}/resume  # Resume from checkpoint
DELETE /api/swarm/sessions/{id}       # Terminate + cleanup

# Agent management
GET  /api/swarm/sessions/{id}/agents       # List agents in session
GET  /api/swarm/sessions/{id}/agents/{aid} # Agent detail + logs
POST /api/swarm/sessions/{id}/agents/{aid}/message  # Send message to agent

# Event stream
GET  /api/swarm/sessions/{id}/events  # SSE stream of all agent events

# Plan editing
GET  /api/swarm/sessions/{id}/plan           # Get current plan
PATCH /api/swarm/sessions/{id}/plan          # Edit plan
POST /api/swarm/sessions/{id}/plan/approve   # Approve plan → execute

# Evidence
GET  /api/swarm/sessions/{id}/evidence              # List artifacts
POST /api/swarm/sessions/{id}/evidence/export        # Export evidence pack
```

### 5.2 Backward Compatibility

现有 `/api/gen-code` 保持不变，内部路由到 `BuiltInAdapter`（包裹现有 P2C pipeline）。新 UI 使用 `/api/swarm/*`。

---

## 6. Agent-Computer Interface (ACI)

借鉴 SWE-Agent 的核心洞察：**Agent 需要为 LLM 优化的接口，而非人类接口**。

### 6.1 Paper-Specific ACI Commands

为论文理解设计专用工具（这是 AgentSwarm 区别于通用 coding agent 的核心）：

```python
# 论文理解工具
extract_equations(section: str) -> List[Equation]
get_hyperparameters() -> Dict[str, Any]
get_dataset_requirements() -> DatasetSpec
get_architecture_description(section: str) -> str
compare_results(table_number: int) -> ComparisonTable

# 代码生成工具
query_coderag(pattern_type: str) -> List[CodePattern]
get_file_context(filename: str) -> FileContext   # from CodeMemory
check_dependency_graph() -> DependencyTree
validate_imports(filename: str) -> ImportReport

# 验证工具
run_syntax_check(filename: str) -> CheckResult
run_smoke_test(command: str, timeout: int) -> TestResult
compare_output_vs_paper(metric: str, expected: float) -> VerificationResult
```

### 6.2 MCP Integration

将 ACI 工具注册为 MCP tools，使任何接入的 Agent 都能调用：

```python
# Paper-specific MCP tools
mcp_server = MCPServer("paperbot-paper-tools")

@mcp_server.tool("extract_equations")
async def extract_equations(section: str) -> List[dict]: ...

@mcp_server.tool("query_coderag")
async def query_coderag(pattern_type: str) -> List[dict]: ...

@mcp_server.tool("run_smoke_test")
async def run_smoke_test(command: str, timeout: int) -> dict: ...
```

---

## 7. Phased Roadmap

### Phase 1 (v1) — Foundation: Agent Runtime + Router

**目标**：替换 `Dict[str, Any]` 共享上下文，实现真正的多 Agent 并行执行。

| Task | Description | Est. |
|------|-------------|------|
| `swarm/message_bus.py` | Typed event-stream message bus | 3h |
| `swarm/agent_router.py` | Task → Agent 路由器 | 3h |
| `swarm/session_manager.py` | 持久化 Agent session + replay | 4h |
| `swarm/adapters/builtin.py` | 包裹现有 P2C nodes 为 adapter | 2h |
| `swarm/adapters/claude_code.py` | Claude Code SDK/CLI adapter | 4h |
| `swarm/orchestrator.py` | 新 Orchestrator（替代旧的，event-driven） | 4h |
| `api/routes/swarm.py` | Swarm API endpoints | 3h |
| repro/ LLM 统一 | 让 repro 模块走 `LLMService` | 2h |
| Tests | 单元测试 + 集成测试 | 4h |

**Deliverables**:
- `POST /api/swarm/sessions` 可创建 session
- PlannerAgent + CoderAgent 可并行子任务（独立文件生成）
- Event stream SSE 可观测 Agent 活动
- 现有 `/api/gen-code` 不受影响

### Phase 2 (v2) — Multi-Agent + External Agents

**目标**：接入 Codex/OpenHands，实现跨 Agent 协作。

| Task | Description | Est. |
|------|-------------|------|
| `swarm/adapters/codex.py` | Codex CLI/API adapter | 4h |
| `swarm/adapters/openhands.py` | OpenHands SDK adapter | 4h |
| Interactive planning UI | 可编辑执行计划 + approve/reject | 6h |
| Agent Dashboard UI | Agent 卡片 + DAG 可视化 | 6h |
| Terminal component | xterm.js agent stdout 实时流 | 4h |
| Evidence Timeline UI | Run/Step 分组 + 过滤 | 4h |
| Paper ACI tools | 论文专用 MCP tools (5-8 个) | 6h |
| SSH fleet executor | 远端 GPU 执行 | 4h |
| CodeRAG 升级 | Embedding-based retrieval | 4h |

**Deliverables**:
- 用户可选择用 Claude Code / Codex / OpenHands 执行子任务
- 可编辑执行计划
- Agent 活动实时可视化

### Phase 3 (v3) — Intelligence Layer

**目标**：领域知识积累，自动优化路由。

| Task | Description |
|------|-------------|
| Execution history analytics | 成功/失败模式分析，优化 Agent 路由 |
| Learned patterns in CodeRAG | 从成功生成中提取新 pattern |
| Paper-to-Repro success predictor | 根据论文特征预测复现难度 |
| GPU scheduler | nvidia-smi 空闲检测 + 自动调度 |
| Reproducibility leaderboard | 按领域/venue 展示可复现率 |
| Cursor/Devin adapter | 评估是否值得接入 |

---

## 8. Key Design Decisions

### Decision 1: Event-Sourced State vs. Shared Mutable State

**选择**：Event-Sourced（借鉴 OpenHands V1）

理由：
- 可确定性 replay（对学术复现至关重要）
- 可观测性（每个 Agent action 都有记录）
- 解耦（Agent 不直接修改共享状态）
- 已有 `event_log` 基础设施可复用

### Decision 2: Agent SDK vs. CLI Subprocess

**选择**：优先 Agent SDK，CLI 作为 fallback

理由：
- Claude Agent SDK 提供 programmatic tool use + streaming
- Codex 有 App Server protocol（比 CLI 更丰富）
- CLI subprocess 仅用于无 SDK 的 Agent（Cursor）

### Decision 3: Sandbox Per Agent vs. Shared Workspace

**选择**：Shared Workspace + Isolated Verification Sandbox

理由：
- Agent 间需要看到彼此生成的文件（CodeMemory 依赖此）
- 仅 Verification 阶段需要隔离 sandbox（防止副作用）
- 借鉴 Codex 的 sandbox inheritance 模式

### Decision 4: MCP vs. Custom Tool Protocol

**选择**：MCP（前端已有基础设施）

理由：
- 前端 `MCPClientManager` 已实现
- Claude Code / Codex / OpenHands 均支持 MCP
- Paper-specific tools 注册为 MCP server，所有 Agent 可调用
- 避免自造协议（Codex 选择自建 App Server protocol，但他们有资源维护）

---

## 9. File Structure

```
src/paperbot/infrastructure/swarm/
├── __init__.py
├── agent_router.py          # Task → Agent routing
├── message_bus.py           # Event-driven agent communication
├── session_manager.py       # Persistent agent sessions
├── orchestrator.py          # New event-driven orchestrator
├── models.py                # SwarmTask, SwarmEvent, AgentResult, etc.
├── adapters/
│   ├── __init__.py
│   ├── base.py              # AgentAdapter ABC
│   ├── builtin.py           # Wraps existing P2C pipeline
│   ├── claude_code.py       # Claude Code SDK/CLI
│   ├── codex.py             # Codex CLI/API (v2)
│   └── openhands.py         # OpenHands SDK (v2)
├── aci/                     # Agent-Computer Interface
│   ├── __init__.py
│   ├── paper_tools.py       # Paper-specific ACI tools
│   ├── code_tools.py        # Code generation tools
│   └── mcp_server.py        # MCP server exposing ACI tools
└── evidence/
    ├── __init__.py
    └── pack.py              # Evidence pack export

src/paperbot/api/routes/swarm.py  # Swarm API endpoints

web/src/components/studio/
├── AgentDashboard.tsx       # Agent cards + DAG visualization
├── AgentCard.tsx            # Single agent status card
├── ExecutionPlan.tsx        # Interactive plan editor
├── EvidenceTimeline.tsx     # Run/Step grouped timeline
└── TerminalPanel.tsx        # xterm.js agent output
```

---

## 10. Competitive Advantages

| Advantage | Details |
|-----------|---------|
| **Domain-specific ACI** | Paper-understanding tools (equation extraction, hyperparameter identification, result comparison) that no generic coding agent has |
| **Verification-centric** | Dedicated VerificationAgent + ReviewerAgent 对比论文报告结果 vs 代码输出 — 目前没有平台做这件事 |
| **Academic knowledge graph** | 可利用 PaperBot 已有的 Scholar tracking + citation network + venue 数据 |
| **Multi-agent router** | 根据任务特征自动选择最佳 Agent（Claude for reasoning, Codex for bulk, Haiku for boilerplate） |
| **Evidence traceability** | Event-sourced state + Evidence Pack 导出，满足学术复现的可追溯要求 |
| **Existing infrastructure** | CodeMemory, CodeRAG, Docker/E2B executors, Runbook file management 已完成 |

---

## Appendix: Competitive Landscape Summary

| Platform | UX | Agent Loop | Sandbox | Multi-Agent | Key Lesson |
|----------|-----|-----------|---------|-------------|------------|
| Devin | Cloud IDE | Planner→Coder→Critic | Cloud VM | MultiDevin parallel | Interactive planning, compound AI |
| OpenHands | Web IDE + CLI | Event-stream loop | Docker | Hierarchical delegation | Event-sourced state, benchmark rigor |
| SWE-Agent | CLI | ACI custom commands | Docker/SWE-ReX | Single (external multi) | LM-optimized interfaces > human interfaces |
| Cursor | IDE | MoE + subagents | Local | Background agents | Codebase indexing, tight feedback loop |
| Codex | Cloud + CLI | Responses API loop | Container/Seatbelt | Experimental spawn | Dual cloud/local, App Server protocol |
| Claude Code | CLI | Tool-use loop | Permission-based | Teams (experimental) | Thin wrapper, prefix caching, Unix composability |
| **OpenCode** | **CLI (Go+Bun)** | **Build/Plan agents** | **Local** | **Subagents** | **112K stars, 75+ providers, scans .claude/skills/** |

---

## Appendix B: Open-Source Orchestration Frameworks

| Framework | Key Pattern | PaperBot Relevance |
|-----------|-----------|-------------------|
| [VibeKanban](https://github.com/BloopAI/vibe-kanban) | Git worktree isolation per agent | Parallel Paper2Code jobs 互不冲突 |
| [MetaSwarm](https://github.com/dsifry/metaswarm) | Cross-model adversarial review + budget enforcement | Writer/Reviewer 用不同 LLM，USD circuit breaker |
| [Ruflo](https://github.com/ruvnet/ruflo) | Dual-mode Claude+Codex, WASM policy engine | 双 Agent 协作 pattern |
| [Oh-My-OpenCode](https://github.com/code-yeongyu/oh-my-opencode) | Intent Gate + Category Routing + Boulder Continuation | 编排模式直接借鉴（见 §11.2） |

---

## 11. OpenCode / OpenClaw / oMo 生态集成

### 11.1 OpenCode (opencode.ai)

**概述**：OpenCode 是增长最快的开源 coding agent（112K GitHub stars，250万月活），Go + Bun 架构，支持 75+ LLM provider。

**关键特性**：
- 原生 SKILL.md 支持（v1.0.190+）
- **主动扫描 `.claude/skills/` 目录** — 意味着放在这里的 skill 同时被 Claude Code 和 OpenCode 发现
- 两种 Agent: Build (full access) + Plan (read-only)
- LSP 深度集成（Global Event Bus）
- `AGENTS.md` 支持（vs Claude Code 的 `CLAUDE.md`）

**Adapter 规划** (#215)：
```python
class OpenCodeAdapter(AgentAdapter):
    """OpenCode CLI as execution backend.
    Useful for: local model routing (Ollama), non-Anthropic models."""
    agent_type = "opencode"
```

### 11.2 Oh-My-OpenCode (oMo)

**概述**：OpenCode 的多 Agent 编排插件层（1,208 TS 文件，143K 行），实现分层 Agent 系统。

**三层 Agent 架构**：
| Tier | Agents | Role |
|------|--------|------|
| Orchestrator | Sisyphus, Prometheus | 规划、委派、驱动 |
| Specialist | Oracle, Hephaestus, Momus, Metis | 架构、执行、审查、补漏 |
| Utility | Explore, Librarian | 搜索、文档 |

**PaperBot 应借鉴的三个模式** (#216)：

#### Pattern 1: Intent Gate Classification
在 dispatch 到 Agent 前先分类意图（research / implementation / investigation / review / tracking），避免误路由。
```python
class IntentGate:
    INTENTS = ["research", "implementation", "investigation", "review", "tracking"]
    async def classify(self, user_input: str) -> str: ...
```

#### Pattern 2: Category-based Model Routing
替代硬编码 model mapping，定义 capability categories:
```python
MODEL_CATEGORIES = {
    "orchestration": ["claude-opus-4-6", "kimi-k2.5"],
    "reasoning":     ["claude-opus-4-6", "gpt-5.3"],
    "speed":         ["claude-haiku-4-5", "minimax-text"],
    "implementation": ["codex", "claude-sonnet-4-6"],
}
```
Agent 声明需要哪个 category，Router 自动选择可用的最佳 model + fallback chain。

#### Pattern 3: Boulder Continuation Enforcer
长流水线中断时自动注入 continuation prompt（指数退避），配合 DebuggerAgent 形成自愈闭环。PaperBot 的 Paper2Code pipeline 经常在 verification 阶段失败，Boulder 模式可自动重试。

### 11.3 OpenClaw + ClawHub Skills 生态

**概述**：OpenClaw 是自治 AI Agent（100K stars），ClawHub 是其 skills marketplace（13,729 社区 skills）。

**可复用的学术 Skills**：

| ClawHub Skill | PaperBot 对应模块 | 复用策略 |
|---------------|-------------------|---------|
| `academic-deep-research` | `agents/research/` | Fork + adapt 评分体系 |
| `arxiv-reader` | `connectors/arxiv.py` | Fork + 接入现有 connector |
| `academic-research` (OpenAlex) | `adapters/openalex.py` | Fork + 接入现有 adapter |
| `agent-brain` (SQLite memory) | `services/memory/` | 参考模式 |
| `academic-writing-refiner` | 无 | 评估是否新增 |

**PaperBot 独有 Skills（需 build from scratch）**：

| Skill | 说明 | 现有代码 |
|-------|------|----------|
| `/reproduce` | 论文→代码完整 pipeline | `repro/orchestrator.py` |
| `/analyze-paper` | 5维评分 + 推荐分级 | `workflows/analysis/judge.py` |
| `/scholar-track` | 学者追踪 + PIS 评分 | `agents/scholar_tracking/` |
| `/daily-digest` | 每日推送生成 | `workflows/dailypaper.py` |
| `/extract-figures` | MinerU 图表提取 | `extractors/mineru_client.py` |
| `/verify-results` | 结果验证 | `repro/nodes/verification.py` |

### 11.4 统一 Skills 策略 (#217)

**核心原则**：一套 SKILL.md，多平台兼容。

```
.claude/skills/              ← Claude Code 扫描
                             ← OpenCode 也扫描（自动互通）
├── analyze-paper/SKILL.md
├── reproduce/SKILL.md
├── extract-figures/SKILL.md
├── verify-results/SKILL.md
├── scholar-track/SKILL.md
├── daily-digest/SKILL.md
├── arxiv-search/SKILL.md    ← adopt from ClawHub arxiv-reader
└── paperbot-conventions/SKILL.md  ← auto-invoked background knowledge

.claude/agents/              ← Claude Code 独有
├── paper-analyst.md
├── code-reproducer.md
└── research-explorer.md

AGENTS.md                    ← Codex / Cursor / Copilot / OpenCode 读取
```

**SKILL.md 兼容性保证**：frontmatter 保持最小公约数（`name` + `description` + `license` + `metadata`），不使用任何平台独有字段。平台特定逻辑放在 markdown body 中作为 conditional instructions。

**发布策略**：将 PaperBot 独有 skills 提交到 ClawHub（Fork → PR），扩大 PaperBot 在 Agent 生态中的影响力。

---

## 12. Updated File Structure

```
src/paperbot/infrastructure/swarm/
├── __init__.py
├── agent_router.py          # Task → Agent routing
├── message_bus.py           # Event-driven agent communication
├── session_manager.py       # Persistent agent sessions
├── orchestrator.py          # New event-driven orchestrator
├── models.py                # SwarmTask, SwarmEvent, AgentResult, etc.
├── intent_gate.py           # Intent classification (from oMo)
├── category_router.py       # Category-based model routing (from oMo)
├── boulder_continuation.py  # Auto-retry for failed pipeline steps (from oMo)
├── adversarial_review.py    # Cross-model writer/reviewer (from MetaSwarm)
├── adapters/
│   ├── __init__.py
│   ├── base.py              # AgentAdapter ABC
│   ├── builtin.py           # Wraps existing P2C pipeline
│   ├── claude_code.py       # Claude Code SDK/CLI
│   ├── codex.py             # Codex App Server / MCP / CLI (v2)
│   ├── opencode.py          # OpenCode Build/Plan agents (v2)
│   ├── openhands.py         # OpenHands SDK (v2)
│   └── ...
├── aci/                     # Agent-Computer Interface
│   ├── __init__.py
│   ├── paper_tools.py       # Paper-specific ACI tools
│   ├── code_tools.py        # Code generation tools
│   └── mcp_server.py        # MCP server exposing ACI tools
└── evidence/
    ├── __init__.py
    └── pack.py              # Evidence pack export

.claude/skills/              # Unified Skills (Claude Code + OpenCode)
├── analyze-paper/SKILL.md
├── reproduce/SKILL.md
├── extract-figures/SKILL.md
├── verify-results/SKILL.md
├── scholar-track/SKILL.md
├── daily-digest/SKILL.md
├── arxiv-search/SKILL.md
└── paperbot-conventions/SKILL.md

.claude/agents/              # Custom subagents (Claude Code)
├── paper-analyst.md
├── code-reproducer.md
└── research-explorer.md

AGENTS.md                    # Cross-platform agent instructions
```

---

## Appendix C: Skills Ecosystem Landscape

| Ecosystem | Skills Count | Format | Discovery Path | PaperBot 策略 |
|-----------|-------------|--------|---------------|---------------|
| Claude Code | Built-in + custom | SKILL.md | `.claude/skills/` | 主要 target |
| OpenCode | Native (v1.0.190) | SKILL.md | `.claude/skills/` + `.opencode/skills/` | 自动兼容 |
| OpenClaw ClawHub | 13,729 | SKILL.md | `~/.openclaw/skills/` | 复用学术 skills + 发布独有 skills |
| Codex | Automations | AGENTS.md | `AGENTS.md` walk | 通过 AGENTS.md 兼容 |
| Cursor | Rules | `.cursor/rules/` | `.cursor/rules/*.mdc` | 不主动支持 |
| VS Code Copilot | Agent Skills | SKILL.md | `.github/skills/` | 自动兼容 |

### Key References

**Agent Platforms:**
- [OpenCode](https://github.com/opencode-ai/opencode) — 112K stars, 75+ providers
- [Oh-My-OpenCode](https://github.com/code-yeongyu/oh-my-opencode) — Multi-agent orchestration plugin
- [OpenClaw](https://github.com/openclaw/openclaw) — 100K stars autonomous agent

**Skills Ecosystem:**
- [Agent Skills spec](https://agentskills.io/specification) — SKILL.md open standard
- [ClawHub](https://github.com/openclaw/clawhub) — 13K+ community skills
- [Anthropic Skills](https://github.com/anthropics/skills) — Official skill examples
- [AGENTS.md](https://agents.md/) — Linux Foundation open standard
- [Awesome OpenClaw Skills](https://github.com/VoltAgent/awesome-openclaw-skills) — 5,494 curated skills
