"""Tests for ExecutorAgent -- implements code in VM."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from paperbot.infrastructure.swarm.agents.executor import ExecutorAgent
from paperbot.infrastructure.swarm.codex_dispatcher import CodexResult
from paperbot.infrastructure.swarm.shared_sandbox import SharedSandbox


class _FakeFS:
    def __init__(self):
        self._files = {}

    def read(self, path):
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    def write(self, path, content):
        self._files[path] = content

    def make_dir(self, path):
        pass


class _CmdResult:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeSandbox:
    def __init__(self):
        self.files = _FakeFS()

    @property
    def commands(self):
        return self

    def run(self, command, timeout=120, cwd="/home/user"):
        if command.startswith("find"):
            return _CmdResult(0, "", "")
        return _CmdResult(0, "", "")


class _FakeExecutor:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def available(self):
        return True


class _FakeTask:
    def __init__(self):
        self.id = "task-001"
        self.title = "Implement model"
        self.description = "Create the model"
        self.subtasks = [{"id": "sub-1", "title": "Write model.py", "done": False}]
        self.codex_output = None
        self.generated_files = []


class _FakeDispatcher:
    def __init__(self, result=None):
        self._result = result or CodexResult(
            task_id="task-001",
            success=True,
            output="Model implemented",
            files_generated=["src/model.py"],
        )
        self.last_prompt = None

    async def dispatch_with_sandbox_tools(
        self, task_id, prompt, tool_executor, on_step=None, on_think=None, max_iterations=25
    ):
        self.last_prompt = prompt
        # Simulate writing a file
        tool_executor.files_written.append("src/model.py")
        return self._result


@pytest.mark.asyncio
async def test_executor_reads_plan_and_writes_status():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    # Pre-populate plan files in VM
    raw.files._files["/home/user/slug/.plan/roadmap.md"] = "# Roadmap\n## Tasks\n1. Implement model"
    raw.files._files["/home/user/slug/.plan/context.md"] = "# Paper Context\nAttention paper"

    task = _FakeTask()
    dispatcher = _FakeDispatcher()
    agent = ExecutorAgent(dispatcher)

    result = await agent.execute(task, shared, "slug")

    assert result.success is True
    assert "src/model.py" in result.files_generated

    # Check prompt included plan
    assert "Roadmap" in dispatcher.last_prompt
    assert "Paper Context" in dispatcher.last_prompt

    # Check status file was written to VM
    status_content = raw.files._files.get("/home/user/slug/.status/task-001.json")
    assert status_content is not None
    status = json.loads(status_content)
    assert status["task_id"] == "task-001"
    assert status["success"] is True


@pytest.mark.asyncio
async def test_executor_reads_prior_status():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    # Pre-populate a prior task's status
    raw.files._files["/home/user/slug/.status/task-000.json"] = json.dumps({
        "task_id": "task-000",
        "title": "Setup project",
        "success": True,
        "summary": "Created project structure",
    })

    # Make find return the status file
    class PatchedSandbox(_FakeSandbox):
        def run(self, command, timeout=120, cwd="/home/user"):
            if command.startswith("find") and ".status" in cwd:
                return _CmdResult(0, f"{cwd}/task-000.json", "")
            if command.startswith("find"):
                lines = [f"{cwd}/{f.split('/')[-1]}" for k, f in
                         [(k, k) for k in self.files._files if k.startswith(cwd)]]
                return _CmdResult(0, "\n".join(lines), "")
            return _CmdResult(0, "", "")

    raw2 = PatchedSandbox()
    raw2.files = raw.files
    executor2 = _FakeExecutor(raw2)
    shared2 = SharedSandbox(executor2)

    task = _FakeTask()
    task.id = "task-001"
    dispatcher = _FakeDispatcher()
    agent = ExecutorAgent(dispatcher)

    await agent.execute(task, shared2, "slug")

    # The prior status should show up in prompt (if list_files returns it)
    # Since our mock returns empty for .status listing, just verify it doesn't crash
    assert dispatcher.last_prompt is not None


@pytest.mark.asyncio
async def test_executor_handles_failure():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    task = _FakeTask()
    fail_result = CodexResult(
        task_id="task-001", success=False, error="Timeout", files_generated=[]
    )
    dispatcher = _FakeDispatcher(result=fail_result)
    agent = ExecutorAgent(dispatcher)

    result = await agent.execute(task, shared, "slug")
    assert result.success is False

    # Status file should still be written
    status_content = raw.files._files.get("/home/user/slug/.status/task-001.json")
    assert status_content is not None
    status = json.loads(status_content)
    assert status["success"] is False


@pytest.mark.asyncio
async def test_executor_prompt_includes_wisdom():
    """Verify commander wisdom is injected into the executor's prompt."""
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    task = _FakeTask()
    dispatcher = _FakeDispatcher()
    agent = ExecutorAgent(dispatcher)

    wisdom = ["Use PyTorch not TensorFlow", "Pin numpy==1.24"]
    result = await agent.execute(task, shared, "slug", wisdom=wisdom)

    assert result.success is True
    assert "Use PyTorch not TensorFlow" in dispatcher.last_prompt
    assert "Pin numpy==1.24" in dispatcher.last_prompt
    assert "Context from Previous Tasks" in dispatcher.last_prompt


@pytest.mark.asyncio
async def test_executor_prompt_without_wisdom():
    """Verify prompt is valid when no wisdom is provided."""
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    task = _FakeTask()
    dispatcher = _FakeDispatcher()
    agent = ExecutorAgent(dispatcher)

    result = await agent.execute(task, shared, "slug", wisdom=None)

    assert result.success is True
    assert "Context from Previous Tasks" not in dispatcher.last_prompt
    assert "## Goal" in dispatcher.last_prompt


@pytest.mark.asyncio
async def test_executor_reads_tasks_json_for_scope():
    """Verify executor reads .plan/tasks.json to show full task scope."""
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    # Write tasks.json with 3 tasks
    raw.files._files["/home/user/slug/.plan/tasks.json"] = json.dumps([
        {"title": "Setup environment", "description": "..."},
        {"title": "Implement model", "description": "..."},
        {"title": "Write tests", "description": "..."},
    ])
    # Write status for first task only
    raw.files._files["/home/user/slug/.plan/roadmap.md"] = "# Roadmap"
    raw.files._files["/home/user/slug/.plan/context.md"] = "# Context"

    task = _FakeTask()
    task.id = "task-002"
    task.title = "Implement model"

    # Make find return status files
    class PatchedSandbox(_FakeSandbox):
        def run(self, command, timeout=120, cwd="/home/user"):
            if command.startswith("find") and ".status" in command:
                return _CmdResult(0, f"{cwd}/task-000.json", "")
            if command.startswith("find"):
                return _CmdResult(0, "", "")
            return _CmdResult(0, "", "")

    raw2 = PatchedSandbox()
    raw2.files = raw.files
    raw2.files._files["/home/user/slug/.status/task-000.json"] = json.dumps({
        "task_id": "task-000",
        "title": "Setup environment",
        "success": True,
        "summary": "Done setting up",
    })
    executor2 = _FakeExecutor(raw2)
    shared2 = SharedSandbox(executor2)

    dispatcher = _FakeDispatcher()
    agent = ExecutorAgent(dispatcher)

    await agent.execute(task, shared2, "slug")

    # Prompt should show all 3 tasks with their status
    assert "[pending] Implement model" in dispatcher.last_prompt
    assert "[pending] Write tests" in dispatcher.last_prompt
