"""Tests for SandboxToolExecutor -- VM-native tool execution."""

import pytest

from paperbot.infrastructure.swarm.sandbox_tool_executor import SandboxToolExecutor
from paperbot.infrastructure.swarm.shared_sandbox import SharedSandbox
from paperbot.infrastructure.swarm.worker_tools import TASK_COMPLETE_SENTINEL


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
            lines = []
            prefix = cwd.rstrip("/")
            for k in sorted(self.files._files):
                if k.startswith(prefix + "/"):
                    lines.append(k)
            return _CmdResult(0, "\n".join(lines), "")
        if command.startswith("grep"):
            return _CmdResult(0, "(no matches)", "")
        if command.startswith("mkdir"):
            return _CmdResult(0, "", "")
        return _CmdResult(0, f"ran: {command}", "")


class _FakeExecutor:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def available(self):
        return True


def _make():
    raw = _FakeSandbox()
    executor = _FakeExecutor(raw)
    shared = SharedSandbox(executor)
    tool_exec = SandboxToolExecutor(shared, "test-slug")
    return tool_exec, raw


class TestSandboxToolExecutor:
    @pytest.mark.asyncio
    async def test_write_then_read(self):
        te, _ = _make()
        result = await te.execute("write_file", {"path": "hello.py", "content": "print(1)"})
        assert "Written" in result
        assert "hello.py" in te.files_written

        content = await te.execute("read_file", {"path": "hello.py"})
        assert content == "print(1)"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        te, _ = _make()
        result = await te.execute("read_file", {"path": "nope.py"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_list_files(self):
        te, raw = _make()
        raw.files._files["/home/user/test-slug/a.py"] = "a"
        result = await te.execute("list_files", {"path": "."})
        assert "a.py" in result

    @pytest.mark.asyncio
    async def test_list_empty(self):
        te, _ = _make()
        result = await te.execute("list_files", {})
        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_run_command(self):
        te, _ = _make()
        result = await te.execute("run_command", {"command": "echo hello"})
        assert "exit_code: 0" in result

    @pytest.mark.asyncio
    async def test_run_command_empty(self):
        te, _ = _make()
        result = await te.execute("run_command", {"command": ""})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_search_files(self):
        te, _ = _make()
        result = await te.execute("search_files", {"pattern": "test"})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_search_missing_pattern(self):
        te, _ = _make()
        result = await te.execute("search_files", {"pattern": ""})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        te, _ = _make()
        result = await te.execute("write_file", {"path": "../../../etc/passwd", "content": "x"})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_absolute_path_blocked(self):
        te, _ = _make()
        result = await te.execute("read_file", {"path": "/etc/passwd"})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_windows_absolute_path_blocked(self):
        te, _ = _make()
        result = await te.execute("read_file", {"path": "C:/Windows/System32/drivers/etc/hosts"})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_task_done(self):
        te, _ = _make()
        result = await te.execute("task_done", {"summary": "all good"})
        assert result == TASK_COMPLETE_SENTINEL

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        te, _ = _make()
        result = await te.execute("nonexistent_tool", {})
        assert "unknown tool" in result.lower()

    @pytest.mark.asyncio
    async def test_update_subtask(self):
        class FakeTask:
            subtasks = [{"id": "sub-1", "title": "do thing", "done": False}]

        te, _ = _make()
        te.task = FakeTask()
        result = await te.execute("update_subtask", {"subtask_id": "sub-1", "done": True})
        assert "done" in result.lower()
        assert te.task.subtasks[0]["done"] is True

    @pytest.mark.asyncio
    async def test_update_subtask_not_found(self):
        class FakeTask:
            subtasks = [{"id": "sub-1", "title": "do thing", "done": False}]

        te, _ = _make()
        te.task = FakeTask()
        result = await te.execute("update_subtask", {"subtask_id": "sub-99", "done": True})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_log_recorded(self):
        te, _ = _make()
        await te.execute("write_file", {"path": "x.py", "content": "x"})
        assert len(te.tool_log) == 1
        assert te.tool_log[0]["tool"] == "write_file"
