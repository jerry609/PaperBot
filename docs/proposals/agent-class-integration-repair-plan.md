# Agent Class Integration Repair Plan (Option A)

> Generated: 2026-03-11
> Branch: `feature/AgentSwarm`
> Prerequisite: `e2e-bugfix-repair-plan.md` (apply first)

---

## Problem Statement

`_run_all_stream_sandbox()` in `agent_board.py` bypasses the agent classes (`PlannerAgent`, `ExecutorAgent`) and inlines all logic directly. This causes:

1. **`.plan/` is never written** — `PlannerAgent` is imported but never called; `roadmap.md`, `tasks.json`, `context.md` are never created in the VM
2. **`.plan/` reads return empty strings** — lines 853-854 read `roadmap.md` and `context.md` which don't exist, so executor prompts lack project-level context
3. **`tasks.json` is dead data** — written by `PlannerAgent` but never consumed anywhere
4. **Code duplication** — prompt construction in `agent_board.py:856-899` duplicates `ExecutorAgent._build_prompt()` (lines 109-167)
5. **`ExecutorAgent` is never instantiated** — all execution goes through inline `dispatch_with_sandbox_tools` calls
6. **Cross-task context is broken** — later tasks can't see the roadmap or what earlier tasks accomplished, because `.plan/` and `.status/` are never populated

### Current vs Intended Data Flow

```
CURRENT (broken):
  _plan_stream() → commander.decompose() → session.tasks (server memory only)
  _run_all_stream_sandbox() → inline prompt build → dispatch (no .plan/, no .status/)

INTENDED (Option A):
  _plan_stream() → commander.decompose() → session.tasks + PlannerAgent writes .plan/
  _run_all_stream_sandbox() → ExecutorAgent reads .plan/ → dispatch → writes .status/
```

---

## Repair Steps

### Step 1: Integrate PlannerAgent into `_plan_stream()`

**File:** `agent_board.py`, function `_plan_stream()` (line 670)

**Current:** `_plan_stream()` calls `commander.decompose()` and only stores tasks in `session.tasks`. No VM writes.

**Change:** After task creation, call `PlannerAgent.plan()` to write `.plan/` files when a sandbox is available.

```python
# --- AFTER existing task-creation loop (around line 724) ---

# Write plan to sandbox if available
shared = _get_shared_sandbox(session)
if shared and shared.alive:
    slug = session.paper_slug_name
    shared.ensure_paper_dir(slug)
    planner = PlannerAgent(_get_commander())
    await planner.plan(shared, slug, pack)
```

**Why here and not in `_run_all_stream_sandbox`:** Planning and execution are separate API calls (`/plan` then `/run`). The plan must be in the VM *before* `/run` is called so that `ExecutorAgent` can read it.

**Edge case:** If no sandbox is available at planning time, `_run_all_stream_sandbox()` must write `.plan/` as a fallback before iterating tasks (see Step 3).

### Step 2: Replace inline execution with ExecutorAgent

**File:** `agent_board.py`, function `_run_all_stream_sandbox()`, Stage 2 (lines 816-965)

**Current:** ~150 lines of inline prompt building, tool executor creation, dispatcher calls, and result handling.

**Change:** Replace with `ExecutorAgent.execute()`.

Before (lines 842-957, simplified):
```python
# Current inline code
tool_exec = SandboxToolExecutor(shared, slug, task)
prompt_parts = [...]  # 40+ lines of prompt building
prompt = "\n".join(prompt_parts)
result = await dispatcher.dispatch_with_sandbox_tools(
    task_id=task.id, prompt=prompt, tool_executor=tool_exec, ...
)
```

After:
```python
executor_agent = ExecutorAgent(dispatcher)
result = await executor_agent.execute(
    task, shared, slug,
    on_step=_on_step,
    max_iterations=max_iterations,
)
```

**What this preserves:**
- SSE event emission (via `on_step` callback)
- Session persistence checkpoints
- Task status transitions (in_progress, done, human_review)
- File merge from `tool_exec.files_written`

**What must change in `ExecutorAgent`:** The `on_step` callback signature must be adapted (see Step 4).

### Step 3: Add `.plan/` fallback write in `_run_all_stream_sandbox()`

If `_plan_stream()` ran without a sandbox (e.g., sandbox was provisioned later), `.plan/` won't exist. Add a guard at the start of Stage 2:

```python
# At the top of Stage 2, before the task loop
plan_exists = shared.read_file(slug, ".plan/roadmap.md")
if not plan_exists:
    log.info("No .plan/ found in VM, writing plan files...")
    pack = _load_context_pack(session.context_pack_id)
    if pack:
        planner = PlannerAgent(commander)
        # Build task data from session.tasks for the roadmap
        tasks_data = [
            {
                "title": t.title,
                "description": t.description,
                "difficulty": next((tag for tag in t.tags if tag in ("easy", "medium", "hard")), "medium"),
                "acceptance_criteria": [s.get("title", "") for s in t.subtasks],
                "dependencies": [],
            }
            for t in planning_tasks
        ]
        # Write roadmap and context directly (skip decompose, tasks already exist)
        shared.run_in_paper(slug, "mkdir -p .plan")
        shared.write_file(slug, ".plan/roadmap.md", planner._build_roadmap(pack, tasks_data, []))
        shared.write_file(slug, ".plan/context.md", planner._build_context_summary(pack))
        shared.write_file(slug, ".plan/tasks.json", json.dumps(tasks_data, indent=2, ensure_ascii=False))
```

### Step 4: Adapt ExecutorAgent for SSE streaming

**File:** `agents/executor.py`

**Current:** `ExecutorAgent.execute()` accepts `on_step` as a simple `Callable` and passes it to `dispatch_with_sandbox_tools`. The route handler needs richer events (task status, session persistence).

**Change:** Add a `on_progress` callback parameter for higher-level events, keeping `on_step` for tool-level events.

```python
async def execute(
    self,
    task: "AgentTask",
    sandbox: SharedSandbox,
    paper_slug: str,
    *,
    on_step: Optional[Callable[..., Awaitable[None]]] = None,
    on_progress: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    max_iterations: int = 25,
) -> CodexResult:
```

The route handler passes in a `on_progress` that emits SSE events and persists session state:

```python
async def _on_progress(event_name: str, data: dict) -> None:
    _persist_session(session, checkpoint=event_name, status="running")
    await step_events_queue.put(StreamEvent(type="progress", data=data))
```

### Step 5: Inject Commander wisdom into ExecutorAgent prompt

**File:** `agents/executor.py`, method `_build_prompt()`

**Current:** `_build_prompt()` includes plan, context, and prior status, but NOT commander wisdom (learnings/conventions). The inline code in `agent_board.py:886-890` adds wisdom but `ExecutorAgent` does not.

**Change:** Add a `wisdom` parameter to `_build_prompt()`:

```python
def _build_prompt(
    self,
    task: "AgentTask",
    plan: str,
    context: str,
    prior_status: str,
    wisdom: Optional[List[str]] = None,
) -> str:
    parts = [...]  # existing code

    # NEW: inject commander wisdom
    if wisdom:
        parts.append("## Context from Previous Tasks")
        for learning in wisdom[-5:]:
            parts.append(f"- {learning}")
        parts.append("")

    parts.append("## Instructions\n...")
    return "\n".join(parts)
```

In `execute()`, pass wisdom from the commander:

```python
prompt = self._build_prompt(
    task, plan, context, prior,
    wisdom=getattr(self.dispatcher, '_commander_wisdom', None),
)
```

**Alternative:** Pass wisdom explicitly from the route handler via a new parameter to `execute()`, keeping ExecutorAgent independent of Commander:

```python
result = await executor_agent.execute(
    task, shared, slug,
    on_step=_on_step,
    wisdom=commander.wisdom.learnings,
)
```

This is cleaner — use this approach.

### Step 6: Make `tasks.json` consumable

**File:** `agents/executor.py`

**Current:** `tasks.json` is written by PlannerAgent but never read.

**Option A (minimal):** Read `tasks.json` in `_read_prior_status()` to show all planned tasks (not just completed ones) so the executor knows the full scope.

```python
def _read_prior_status(self, sandbox: SharedSandbox, slug: str) -> str:
    # Read full task list from .plan/tasks.json for context
    tasks_json = sandbox.read_file(slug, ".plan/tasks.json")
    all_tasks = []
    if tasks_json:
        try:
            all_tasks = json.loads(tasks_json)
        except Exception:
            pass

    # Read completed status from .status/
    entries = sandbox.list_files(slug, STATUS_DIR)
    completed = {}
    for f in entries:
        if not f.endswith(".json"):
            continue
        content = sandbox.read_file(slug, f"{STATUS_DIR}/{f}")
        if not content:
            continue
        try:
            d = json.loads(content)
            completed[d.get("title", "")] = d
        except Exception:
            pass

    # Build status string
    lines: List[str] = []
    for t in all_tasks:
        title = t.get("title", "")
        if title in completed:
            d = completed[title]
            marker = "done" if d.get("success") else "failed"
            lines.append(f"- [{marker}] {title}: {d.get('summary', '')[:100]}")
        else:
            lines.append(f"- [pending] {title}")

    return "\n".join(lines) if lines else "(no prior tasks)"
```

**Option B (remove):** Delete `tasks.json` writes from PlannerAgent entirely. The roadmap already contains all task information in human-readable form.

**Recommendation:** Option A — it gives executors structured awareness of all tasks, not just the text roadmap.

### Step 7: Delete inline code from agent_board.py

After Steps 1-6, the following inline code blocks in `_run_all_stream_sandbox()` become dead code and should be removed:

| Lines | Description | Replaced by |
|-------|-------------|-------------|
| 842-899 | Inline prompt building (SandboxToolExecutor, prompt_parts, wisdom injection) | `ExecutorAgent.execute()` |
| 935-943 | `asyncio.create_task(dispatcher.dispatch_with_sandbox_tools(...))` | `ExecutorAgent.execute()` (internally calls dispatch) |
| 945-956 | Event queue drain loop + result extraction | Simplified via `on_progress` callback |
| 958-961 | File merge from `tool_exec.files_written` | `ExecutorAgent.execute()` already merges |

The Stage 2 task loop shrinks from ~150 lines to ~50 lines:

```python
for i, task in enumerate(planning_tasks):
    task.status = "in_progress"
    task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
    task.progress = 10
    task.updated_at = _now_iso()
    # ... SSE event + persist (keep as-is) ...

    executor_agent = ExecutorAgent(dispatcher)
    result = await executor_agent.execute(
        task, shared, slug,
        on_step=_on_step,
        wisdom=commander.wisdom.learnings,
        max_iterations=max_iterations,
    )

    merged_files = result.files_generated
    task.generated_files = merged_files
    task.codex_output = result.output if result.success else result.error
    task.progress = 70
    task.updated_at = _now_iso()
    # ... rest of status handling (keep as-is) ...
```

### Step 8: Update tests

| Test file | Changes |
|-----------|---------|
| `test_planner_agent.py` | Already passes — no changes needed |
| `test_executor_agent.py` | Add test for `wisdom` parameter in prompt; verify `.plan/` reads feed into prompt |
| `test_agent_board_route.py` | Update to verify `PlannerAgent.plan()` is called during planning; verify `ExecutorAgent.execute()` is called during execution |

**New test: `test_plan_written_before_execution`**
```python
@pytest.mark.asyncio
async def test_plan_files_exist_when_executor_runs():
    """Verify that .plan/roadmap.md exists in VM before ExecutorAgent runs."""
    # Setup: PlannerAgent writes .plan/
    # Then: ExecutorAgent reads .plan/ and gets non-empty roadmap in prompt
    ...
```

**New test: `test_executor_receives_wisdom`**
```python
@pytest.mark.asyncio
async def test_executor_prompt_includes_wisdom():
    """Verify commander wisdom appears in ExecutorAgent's prompt."""
    ...
    result = await agent.execute(task, shared, "slug", wisdom=["Use PyTorch not TF"])
    assert "Use PyTorch not TF" in dispatcher.last_prompt
```

---

## Implementation Order

| Step | Scope | Risk | Effort |
|------|-------|------|--------|
| 1. PlannerAgent in `_plan_stream` | `agent_board.py` | Low — additive, no existing behavior changes | Small |
| 3. `.plan/` fallback guard | `agent_board.py` | Low — defensive, handles edge case | Small |
| 5. Wisdom param in ExecutorAgent | `executor.py` | Low — additive parameter | Small |
| 6. `tasks.json` consumption | `executor.py` | Low — improves context, no breaking change | Medium |
| 4. `on_progress` callback in ExecutorAgent | `executor.py` | Medium — interface change | Medium |
| 2. Replace inline execution | `agent_board.py` | Medium — structural refactor, must preserve SSE streaming | Large |
| 7. Delete inline dead code | `agent_board.py` | Low — cleanup after Step 2 verified | Small |
| 8. Update tests | `tests/unit/` | Low | Medium |

**Total:** ~100 lines added to agent classes, ~100 lines removed from route handler. Net reduction in codebase complexity.

---

## Validation

```bash
# Unit tests
PYTHONPATH=src pytest -q \
  tests/unit/test_planner_agent.py \
  tests/unit/test_executor_agent.py \
  tests/unit/test_agent_board_route.py \
  tests/unit/test_e2e_execution.py \
  tests/unit/test_knowledge_manager.py

# Integration: verify SSE events are unchanged
# (Manual test with a running sandbox)
```

### SSE Contract Check

The refactor must NOT change the SSE event types or data shapes emitted to the frontend. The frontend (`AgentBoard.tsx:567-707`) parses these events by `event` field name. Verify:

- `executor_started` — still emitted with `task_id`, `task`, `index`, `total`
- `tool_step` — still emitted with `task_id`, `step`, `tool`, `observation_preview`
- `executor_finished` / `executor_failed` — still emitted with `task_id`, `task`, `files_written`
- `task_reviewed` — still emitted with `task_id`, `approved`, `feedback`

No frontend changes required.

---

## Architecture After Repair

```
_plan_stream()
  └─ commander.decompose() → session.tasks
  └─ PlannerAgent.plan() → .plan/roadmap.md, .plan/tasks.json, .plan/context.md

_run_all_stream_sandbox()
  ├─ Stage 1: ensure_paper_dir()
  ├─ Stage 1.5: fallback .plan/ write (if not already present)
  ├─ Stage 2: ExecutorAgent.execute() × N
  │   ├─ reads .plan/roadmap.md, .plan/context.md
  │   ├─ reads .status/*.json (prior task results)
  │   ├─ builds prompt with wisdom
  │   ├─ dispatch_with_sandbox_tools (CodeAct loop)
  │   └─ writes .status/{task_id}.json
  ├─ Stage 3: Verification + repair (unchanged)
  ├─ Stage 4: E2E execution + repair (unchanged)
  ├─ Stage 5: KnowledgeManager.curate() (already using agent class)
  └─ Stage 6: download_paper() (unchanged)
```

All three agent classes (`PlannerAgent`, `ExecutorAgent`, `KnowledgeManager`) are now active participants. The VM file system (`.plan/`, `.status/`, `.knowledge/`) serves as the inter-agent communication protocol, as originally designed.
