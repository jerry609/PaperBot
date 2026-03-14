from __future__ import annotations

from pathlib import Path

import pytest

from paperbot.infrastructure.swarm.sandbox_runtime import (
    SandboxRuntime,
    SandboxVerificationPolicy,
    SandboxRunResult,
    detect_missing_python_packages,
)
from paperbot.repro.base_executor import BaseExecutor
from paperbot.repro.execution_result import ExecutionResult


class _FakeExecutor(BaseExecutor):
    def __init__(self, outcomes: list[ExecutionResult], available: bool = True):
        self._outcomes = outcomes
        self._available = available

    def available(self) -> bool:
        return self._available

    def run(
        self,
        workdir: Path,
        commands: list[str],
        timeout_sec: int = 300,
        cache_dir: Path | None = None,
        record_meta: bool = True,
    ) -> ExecutionResult:
        if not self._outcomes:
            return ExecutionResult(status="error", exit_code=1, error="no more outcomes")
        return self._outcomes.pop(0)


def test_policy_disabled_when_sandbox_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    policy = SandboxVerificationPolicy.from_env(tmp_path, sandbox_available=False)

    assert policy.enabled is False


def test_policy_parses_env_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_VERIFY_COMMANDS", "pytest -q && pyright")
    policy = SandboxVerificationPolicy.from_env(tmp_path, sandbox_available=True)

    assert policy.enabled is True
    assert policy.commands == ["pytest -q && pyright"]


def test_policy_parses_newline_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_VERIFY_COMMANDS", "pytest -q\npyright")
    policy = SandboxVerificationPolicy.from_env(tmp_path, sandbox_available=True)

    assert policy.enabled is True
    assert policy.commands == ["pytest -q", "pyright"]


def test_policy_resolves_default_commands(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_VERIFY_COMMANDS", raising=False)
    monkeypatch.delenv("CODEX_SANDBOX_BOOTSTRAP_COMMANDS", raising=False)
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (tmp_path / "test_example.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (tmp_path / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "web/package.json").write_text('{"name":"web"}\n', encoding="utf-8")

    policy = SandboxVerificationPolicy.from_env(tmp_path, sandbox_available=True)

    assert "PYTHONPATH=. pytest -q" in policy.commands
    assert "cd web && npm run lint && npm run build" in policy.commands
    assert "pip install -q -r requirements.txt" in policy.bootstrap_commands


def test_policy_parses_bootstrap_env_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_BOOTSTRAP_COMMANDS", "pip install -q -r requirements.txt\npip install -q statsmodels")
    policy = SandboxVerificationPolicy.from_env(tmp_path, sandbox_available=True)

    assert policy.bootstrap_commands == [
        "pip install -q -r requirements.txt",
        "pip install -q statsmodels",
    ]


def test_policy_prefers_tests_directory_for_pytest_command(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_VERIFY_COMMANDS", raising=False)
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    policy = SandboxVerificationPolicy.from_env(tmp_path, sandbox_available=True)

    assert policy.commands[0] == "PYTHONPATH=. pytest -q tests"


@pytest.mark.asyncio
async def test_runtime_returns_unavailable_error(tmp_path):
    runtime = SandboxRuntime(executor=None, workspace=tmp_path)
    result = await runtime.run_command("pytest -q", timeout_seconds=10)

    assert result.success is False
    assert "unavailable" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_runtime_stops_after_first_failure(tmp_path):
    runtime = SandboxRuntime(
        executor=_FakeExecutor(
            outcomes=[
                ExecutionResult(status="failed", exit_code=2, logs="first failed"),
                ExecutionResult(status="success", exit_code=0, logs="second passed"),
            ]
        ),
        workspace=tmp_path,
    )

    results = await runtime.run_commands(
        ["pytest -q", "pyright"],
        timeout_seconds=20,
    )

    assert len(results) == 1
    assert results[0].exit_code == 2


def test_detect_missing_python_packages_from_verification_logs():
    results = [
        SandboxRunResult(
            command="pytest -q",
            status="failed",
            exit_code=2,
            logs=(
                "ModuleNotFoundError: No module named 'statsmodels'\n"
                "ModuleNotFoundError: No module named 'sqlalchemy'"
            ),
        ),
        SandboxRunResult(
            command="pytest -q",
            status="failed",
            exit_code=2,
            logs="ModuleNotFoundError: No module named 'sklearn'",
        ),
    ]

    packages = detect_missing_python_packages(results)

    assert packages == ["statsmodels", "sqlalchemy", "scikit-learn"]


def test_detect_missing_python_packages_ignores_local_modules(tmp_path):
    # Directories containing .py files are recognised as local packages.
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pipeline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pipeline" / "run.py").write_text("", encoding="utf-8")
    results = [
        SandboxRunResult(
            command="pytest -q tests",
            status="failed",
            exit_code=2,
            logs=(
                "ModuleNotFoundError: No module named 'src'\n"
                "ModuleNotFoundError: No module named 'pipeline'\n"
                "ModuleNotFoundError: No module named 'sqlalchemy'"
            ),
        )
    ]

    packages = detect_missing_python_packages(results, workspace=tmp_path)

    assert packages == ["sqlalchemy"]


def test_detect_missing_packages_known_local_modules_override():
    """known_local_modules parameter filters even without a workspace."""
    results = [
        SandboxRunResult(
            command="pytest -q tests",
            status="failed",
            exit_code=2,
            logs=(
                "ModuleNotFoundError: No module named 'src'\n"
                "ModuleNotFoundError: No module named 'pipeline'\n"
                "ModuleNotFoundError: No module named 'sqlalchemy'"
            ),
        )
    ]

    packages = detect_missing_python_packages(
        results, known_local_modules={"src", "pipeline"}
    )

    assert packages == ["sqlalchemy"]


def test_detect_missing_packages_scans_workspace_py_files(tmp_path):
    """Top-level .py files are treated as local modules."""
    (tmp_path / "models.py").write_text("class Foo: ...", encoding="utf-8")
    results = [
        SandboxRunResult(
            command="pytest -q",
            status="failed",
            exit_code=2,
            logs="ModuleNotFoundError: No module named 'models'",
        )
    ]

    packages = detect_missing_python_packages(results, workspace=tmp_path)

    assert packages == []


def test_detect_missing_packages_ignores_empty_dirs(tmp_path):
    """Empty directories (no .py files) are NOT treated as local packages."""
    (tmp_path / "vendor").mkdir()
    results = [
        SandboxRunResult(
            command="pytest -q",
            status="failed",
            exit_code=2,
            logs="ModuleNotFoundError: No module named 'vendor'",
        )
    ]

    packages = detect_missing_python_packages(results, workspace=tmp_path)

    assert packages == ["vendor"]
