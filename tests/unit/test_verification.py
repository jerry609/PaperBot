"""Tests for verification module -- VerificationPolicy + run_verification."""

from unittest.mock import MagicMock, patch

import pytest

from paperbot.infrastructure.swarm.shared_sandbox import SharedSandbox
from paperbot.infrastructure.swarm.verification import (
    VerificationPolicy,
    VerificationResult,
    _detect_commands,
    run_verification,
    verify_and_repair,
)
from paperbot.repro.execution_result import ExecutionResult


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
    def __init__(self, file_list=None):
        self.files = _FakeFS()
        self._file_list = file_list or []
        self._last_cmd = None

    @property
    def commands(self):
        return self

    def run(self, command, timeout=120, cwd="/home/user"):
        self._last_cmd = (command, cwd)
        if command.startswith("find"):
            prefix = cwd.rstrip("/")
            lines = [f"{prefix}/{f}" for f in self._file_list]
            return _CmdResult(0, "\n".join(lines), "")
        if command.startswith("mkdir"):
            return _CmdResult(0, "", "")
        # Simulate verification commands
        if "pytest" in command:
            return _CmdResult(0, "1 passed", "")
        return _CmdResult(0, "", "")


class _FakeExecutor:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def available(self):
        return True


def _make_shared(file_list=None):
    raw = _FakeSandbox(file_list=file_list)
    executor = _FakeExecutor(raw)
    return SharedSandbox(executor), raw


class TestDetectCommands:
    def test_python_project(self):
        shared, _ = _make_shared(["pyproject.toml", "src", "tests"])
        cmds = _detect_commands(shared, "slug")
        assert any("pytest" in c for c in cmds)

    def test_requirements_txt(self):
        shared, _ = _make_shared(["requirements.txt"])
        cmds = _detect_commands(shared, "slug")
        assert any("pip install" in c for c in cmds)
        assert any("pytest" in c for c in cmds)

    def test_node_project(self):
        shared, _ = _make_shared(["package.json", "index.js"])
        cmds = _detect_commands(shared, "slug")
        assert any("npm" in c for c in cmds)

    def test_empty_project(self):
        shared, _ = _make_shared([])
        cmds = _detect_commands(shared, "slug")
        assert cmds == []


class TestVerificationPolicy:
    def test_from_sandbox_env_defaults(self):
        shared, _ = _make_shared(["requirements.txt", "pyproject.toml"])
        policy = VerificationPolicy.from_sandbox_env(shared, "slug")
        assert policy.enabled is True
        assert len(policy.commands) > 0

    @patch.dict("os.environ", {"CODEX_ENABLE_VERIFICATION": "false"})
    def test_disabled(self):
        shared, _ = _make_shared(["requirements.txt"])
        policy = VerificationPolicy.from_sandbox_env(shared, "slug")
        assert policy.enabled is False

    @patch.dict("os.environ", {"CODEX_VERIFY_COMMANDS": "make test && make lint"})
    def test_custom_commands(self):
        shared, _ = _make_shared([])
        policy = VerificationPolicy.from_sandbox_env(shared, "slug")
        assert policy.commands == ["make test", "make lint"]
        assert policy.enabled is True


class TestRunVerification:
    def test_passes(self):
        shared, _ = _make_shared([])
        policy = VerificationPolicy(
            enabled=True,
            commands=["pytest -q"],
            timeout_seconds=60,
        )
        result = run_verification(shared, "slug", policy, attempt=0)
        assert result.passed is True
        assert result.attempt == 0

    def test_no_commands(self):
        shared, _ = _make_shared([])
        policy = VerificationPolicy(enabled=True, commands=[])
        result = run_verification(shared, "slug", policy)
        assert result.passed is True

    def test_failed(self):
        # Create sandbox that fails pytest
        class FailSandbox:
            files = _FakeFS()

            @property
            def commands(self):
                return self

            def run(self, command, timeout=120, cwd="/home/user"):
                return _CmdResult(1, "FAILED", "error output")

        executor = _FakeExecutor(FailSandbox())
        shared = SharedSandbox(executor)
        policy = VerificationPolicy(
            enabled=True,
            commands=["pytest -q"],
            timeout_seconds=60,
        )
        result = run_verification(shared, "slug", policy)
        assert result.passed is False
        assert result.exit_code == 1


@pytest.mark.asyncio
async def test_verify_and_repair_dispatch_exception_returns_last_result():
    class FailSandbox:
        files = _FakeFS()

        @property
        def commands(self):
            return self

        def run(self, command, timeout=120, cwd="/home/user"):
            if command.startswith("find"):
                return _CmdResult(0, "", "")
            return _CmdResult(1, "FAILED", "error output")

    class _ExplodingDispatcher:
        async def dispatch_with_sandbox_tools(
            self, task_id, prompt, tool_executor, on_step=None, max_iterations=25
        ):
            raise RuntimeError("dispatcher unavailable")

    executor = _FakeExecutor(FailSandbox())
    shared = SharedSandbox(executor)
    policy = VerificationPolicy(
        enabled=True,
        commands=["pytest -q"],
        timeout_seconds=60,
        max_repair_attempts=2,
    )

    result = await verify_and_repair(
        sandbox=shared,
        paper_slug="slug",
        policy=policy,
        dispatcher=_ExplodingDispatcher(),
        tool_executor=MagicMock(),
    )

    assert result is not None
    assert result.passed is False
    assert result.exit_code == 1
