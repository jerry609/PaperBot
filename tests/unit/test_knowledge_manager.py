"""Tests for KnowledgeManager -- curates outputs in VM."""

import pytest

from paperbot.infrastructure.swarm.agents.knowledge_manager import KnowledgeManager
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
        self._commands = []

    @property
    def commands(self):
        return self

    def run(self, command, timeout=120, cwd="/home/user"):
        self._commands.append((command, cwd))
        if command.startswith("find") and "-type f" in command:
            prefix = cwd.rstrip("/")
            lines = [k for k in sorted(self.files._files) if k.startswith(prefix)]
            return _CmdResult(0, "\n".join(lines), "")
        if command.startswith("find") and "-maxdepth" in command:
            prefix = cwd.rstrip("/")
            lines = [k for k in sorted(self.files._files) if k.startswith(prefix + "/")]
            return _CmdResult(0, "\n".join(lines), "")
        return _CmdResult(0, "", "")


class _FakeExecutor:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def available(self):
        return True


class _FakeCommander:
    class wisdom:
        learnings = ["Learned to use attention"]
        conventions = ["Use snake_case"]
        gotchas = ["Watch for OOM"]

    def accumulate_wisdom(self, task, output):
        pass


class _FakeTask:
    def __init__(self, title, status="done", codex_output="code", generated_files=None):
        self.id = f"task-{title.lower().replace(' ', '-')}"
        self.title = title
        self.description = f"Implement {title}"
        self.status = status
        self.codex_output = codex_output
        self.generated_files = generated_files or []


@pytest.mark.asyncio
async def test_curate_writes_knowledge_files():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    # Pre-populate some project files in VM
    raw.files._files["/home/user/slug/src/model.py"] = "class Model: pass"
    raw.files._files["/home/user/slug/requirements.txt"] = "torch"

    commander = _FakeCommander()
    km = KnowledgeManager(commander)

    tasks = [
        _FakeTask("Implement model", generated_files=["src/model.py"]),
        _FakeTask("Write tests", status="human_review", generated_files=["tests/test_model.py"]),
    ]

    written = await km.curate(shared, "slug", tasks)

    assert "summary.md" in written
    assert "conventions.md" in written
    assert "learnings.md" in written

    # Verify files in VM
    summary = raw.files._files.get("/home/user/slug/.knowledge/summary.md")
    assert summary is not None
    assert "Implement model" in summary

    conventions = raw.files._files.get("/home/user/slug/.knowledge/conventions.md")
    assert conventions is not None
    assert "snake_case" in conventions

    learnings = raw.files._files.get("/home/user/slug/.knowledge/learnings.md")
    assert learnings is not None
    assert "attention" in learnings


@pytest.mark.asyncio
async def test_curate_keeps_status_for_review_traceability():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    commander = _FakeCommander()
    km = KnowledgeManager(commander)

    await km.curate(shared, "slug", [])

    # .status should be preserved for review traceability.
    assert not any("rm -rf .status" in cmd for cmd, _ in raw._commands)


@pytest.mark.asyncio
async def test_curate_on_step_callback():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)

    commander = _FakeCommander()
    km = KnowledgeManager(commander)

    callbacks = []

    async def on_step(*args):
        callbacks.append(args)

    await km.curate(shared, "slug", [], on_step=on_step)

    assert len(callbacks) == 1
    assert callbacks[0][0] == "knowledge_manager"
