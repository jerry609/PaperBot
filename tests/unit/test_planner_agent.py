"""Tests for PlannerAgent -- writes plans to VM."""

import json
from unittest.mock import AsyncMock

import pytest

from paperbot.infrastructure.swarm.agents.planner import PlannerAgent
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


class _FakeCommander:
    async def decompose(self, context_pack):
        return [
            {
                "title": "Implement model",
                "description": "Create the neural network model",
                "difficulty": "medium",
                "acceptance_criteria": ["Model compiles", "Forward pass works"],
                "dependencies": [],
            },
            {
                "title": "Write training loop",
                "description": "Training and evaluation",
                "difficulty": "hard",
                "acceptance_criteria": ["Training runs"],
                "dependencies": ["Implement model"],
            },
        ]


@pytest.mark.asyncio
async def test_planner_writes_plan_files():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)
    commander = _FakeCommander()
    planner = PlannerAgent(commander)

    context_pack = {
        "objective": "Reproduce attention mechanism",
        "paper": {"title": "Attention Is All You Need", "year": 2017, "authors": ["Vaswani"]},
        "observations": [{"title": "Multi-head attention", "narrative": "Key mechanism"}],
        "warnings": [],
        "task_roadmap": [],
    }

    tasks = await planner.plan(shared, "attn-a9b1", context_pack)

    assert len(tasks) == 2
    assert tasks[0]["title"] == "Implement model"

    # Verify files were written to VM
    roadmap = raw.files._files.get("/home/user/attn-a9b1/.plan/roadmap.md")
    assert roadmap is not None
    assert "Implement model" in roadmap

    tasks_json = raw.files._files.get("/home/user/attn-a9b1/.plan/tasks.json")
    assert tasks_json is not None
    parsed = json.loads(tasks_json)
    assert len(parsed) == 2

    context_md = raw.files._files.get("/home/user/attn-a9b1/.plan/context.md")
    assert context_md is not None
    assert "Attention Is All You Need" in context_md


@pytest.mark.asyncio
async def test_planner_on_step_callback():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)
    commander = _FakeCommander()
    planner = PlannerAgent(commander)

    callback_data = []

    async def on_step(*args):
        callback_data.append(args)

    await planner.plan(
        shared,
        "slug",
        {"objective": "test", "paper": {}, "observations": [], "warnings": [], "task_roadmap": []},
        on_step=on_step,
    )

    assert len(callback_data) == 1
    assert callback_data[0][0] == "planner"
    assert callback_data[0][1] == "write_plan"
