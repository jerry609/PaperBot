from __future__ import annotations

import types
from pathlib import Path

import pytest

from paperbot.infrastructure.swarm.worker_tools import TASK_COMPLETE_SENTINEL, LocalToolExecutor
from paperbot.repro.base_executor import BaseExecutor
from paperbot.repro.execution_result import ExecutionResult


class _FakeExecutor(BaseExecutor):
    def available(self) -> bool:
        return True

    def run(
        self,
        workdir: Path,
        commands: list[str],
        timeout_sec: int = 300,
        cache_dir: Path | None = None,
        record_meta: bool = True,
    ) -> ExecutionResult:
        return ExecutionResult(status="success", exit_code=0, logs=f"ran: {commands[0]}")


class _RecordingExecutor(BaseExecutor):
    def __init__(self, *, logs: str = "ok"):
        self.last_timeout_sec: int | None = None
        self.logs = logs

    def available(self) -> bool:
        return True

    def run(
        self,
        workdir: Path,
        commands: list[str],
        timeout_sec: int = 300,
        cache_dir: Path | None = None,
        record_meta: bool = True,
    ) -> ExecutionResult:
        self.last_timeout_sec = timeout_sec
        return ExecutionResult(status="success", exit_code=0, logs=self.logs)


@pytest.mark.asyncio
async def test_read_file_existing(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("read_file", {"path": "a.txt"})

    assert result == "hello"


@pytest.mark.asyncio
async def test_read_file_not_found(tmp_path):
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("read_file", {"path": "missing.txt"})

    assert "File not found" in result


@pytest.mark.asyncio
async def test_read_file_path_traversal_blocked(tmp_path):
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("read_file", {"path": "../etc/passwd"})

    assert "invalid path" in result.lower()


@pytest.mark.asyncio
async def test_write_file_creates_dirs_and_tracks_output(tmp_path):
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute(
        "write_file",
        {"path": "src/train.py", "content": "print('ok')\n"},
    )

    assert "Written" in result
    assert (tmp_path / "src/train.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert "src/train.py" in executor.files_written


@pytest.mark.asyncio
async def test_write_file_path_escape_blocked(tmp_path):
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("write_file", {"path": "../outside.py", "content": "x = 1"})

    assert "invalid path" in result.lower()
    assert not (tmp_path.parent / "outside.py").exists()


@pytest.mark.asyncio
async def test_list_files_empty_dir(tmp_path):
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("list_files", {"path": "."})

    assert "(empty directory)" in result


@pytest.mark.asyncio
async def test_list_files_with_entries(tmp_path):
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src/main.py").write_text("print('x')\n", encoding="utf-8")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("list_files", {"path": "."})

    assert "d src/" in result
    assert "f src/main.py" in result


@pytest.mark.asyncio
async def test_run_command_with_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_RUN_COMMAND", "true")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=_FakeExecutor())

    result = await executor.execute("run_command", {"command": "pytest -q"})

    assert "exit_code: 0" in result
    assert "ran: pytest -q" in result


@pytest.mark.asyncio
async def test_run_command_no_sandbox_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_RUN_COMMAND", "true")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("run_command", {"command": "pytest -q"})

    assert "requires an available sandbox" in result


@pytest.mark.asyncio
async def test_run_command_disabled_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_RUN_COMMAND", "false")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=_FakeExecutor())

    result = await executor.execute("run_command", {"command": "pytest -q"})

    assert "run_command is disabled" in result


@pytest.mark.asyncio
async def test_run_command_install_uses_extended_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_RUN_COMMAND", "true")
    sandbox = _RecordingExecutor(logs="Successfully installed numpy-1.0")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=sandbox)

    result = await executor.execute("run_command", {"command": "pip install numpy -q"})

    assert "exit_code: 0" in result
    assert sandbox.last_timeout_sec == 300


@pytest.mark.asyncio
async def test_run_command_non_install_uses_default_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_RUN_COMMAND", "true")
    sandbox = _RecordingExecutor()
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=sandbox)

    await executor.execute("run_command", {"command": "pytest -q"})

    assert sandbox.last_timeout_sec == 120


@pytest.mark.asyncio
async def test_run_command_pip_allowlist_blocks_unapproved_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_RUN_COMMAND", "true")
    monkeypatch.setenv("CODEX_PIP_ALLOWLIST", "numpy,pytest")
    sandbox = _RecordingExecutor()
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=sandbox)

    result = await executor.execute("run_command", {"command": "pip install numpy scipy"})

    assert "not in allowlist" in result
    assert "scipy" in result


@pytest.mark.asyncio
async def test_search_files_matches_and_no_matches(tmp_path):
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src/main.py").write_text("def train_model():\n    return 1\n", encoding="utf-8")
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result_match = await executor.execute(
        "search_files",
        {"pattern": "train_model", "glob": "*.py"},
    )
    result_none = await executor.execute(
        "search_files",
        {"pattern": "does_not_exist", "glob": "*.py"},
    )

    assert "src/main.py:1:def train_model()" in result_match
    assert result_none == "(no matches)"


@pytest.mark.asyncio
async def test_update_subtask(tmp_path):
    task = types.SimpleNamespace(subtasks=[{"id": "sub-1", "title": "Add API", "done": False}])
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None, task=task)

    result = await executor.execute("update_subtask", {"subtask_id": "sub-1", "done": True})

    assert "marked done" in result
    assert task.subtasks[0]["done"] is True


@pytest.mark.asyncio
async def test_task_done_returns_sentinel(tmp_path):
    executor = LocalToolExecutor(workspace=tmp_path, sandbox=None)

    result = await executor.execute("task_done", {"summary": "done"})

    assert result == TASK_COMPLETE_SENTINEL
