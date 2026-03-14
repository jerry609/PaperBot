from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from ...repro.base_executor import BaseExecutor

MAX_SANDBOX_LOG_CHARS = 8000
DEFAULT_VERIFY_TIMEOUT_SECONDS = 180
DEFAULT_VERIFY_MAX_RETRIES = 1
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 300

_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_\.]+)['\"]")
_MODULE_PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "pil": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "sqlalchemy": "sqlalchemy",
    "statsmodels": "statsmodels",
}
_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", set()))


@dataclass
class SandboxRunResult:
    command: str
    status: str
    exit_code: int
    logs: str = ""
    error: Optional[str] = None
    executor_type: str = "none"

    @property
    def success(self) -> bool:
        return self.status == "success" and self.exit_code == 0


@dataclass
class SandboxVerificationPolicy:
    enabled: bool
    commands: List[str] = field(default_factory=list)
    bootstrap_commands: List[str] = field(default_factory=list)
    bootstrap_timeout_seconds: int = DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
    timeout_seconds: int = DEFAULT_VERIFY_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_VERIFY_MAX_RETRIES

    @classmethod
    def from_env(cls, workspace: Path, *, sandbox_available: bool) -> "SandboxVerificationPolicy":
        enabled_raw = os.getenv("CODEX_ENABLE_SANDBOX_VERIFY", "true").strip().lower()
        enabled = enabled_raw not in {"0", "false", "no", "off"}
        if not sandbox_available:
            enabled = False

        timeout_seconds = _parse_int_env(
            "CODEX_SANDBOX_VERIFY_TIMEOUT_SECONDS",
            DEFAULT_VERIFY_TIMEOUT_SECONDS,
            min_value=1,
            max_value=3600,
        )
        bootstrap_timeout_seconds = _parse_int_env(
            "CODEX_SANDBOX_BOOTSTRAP_TIMEOUT_SECONDS",
            DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS,
            min_value=1,
            max_value=3600,
        )
        max_retries = _parse_int_env(
            "CODEX_SANDBOX_MAX_RETRIES",
            DEFAULT_VERIFY_MAX_RETRIES,
            min_value=0,
            max_value=5,
        )

        bootstrap_env = (os.getenv("CODEX_SANDBOX_BOOTSTRAP_COMMANDS", "") or "").strip()
        if bootstrap_env:
            bootstrap_commands = _split_commands(bootstrap_env)
        else:
            bootstrap_commands = _resolve_default_bootstrap_commands(workspace)

        commands_env = (os.getenv("CODEX_SANDBOX_VERIFY_COMMANDS", "") or "").strip()
        if commands_env:
            commands = _split_commands(commands_env)
        else:
            commands = _resolve_default_verify_commands(workspace)

        return cls(
            enabled=enabled,
            commands=commands,
            bootstrap_commands=bootstrap_commands,
            bootstrap_timeout_seconds=bootstrap_timeout_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


class SandboxRuntime:
    """Runs commands through a configured sandbox executor with normalized results."""

    def __init__(self, executor: Optional[BaseExecutor], workspace: Path):
        self.executor = executor
        self.workspace = workspace

    def available(self) -> bool:
        return self.executor is not None and self.executor.available()

    async def run_command(self, command: str, *, timeout_seconds: int) -> SandboxRunResult:
        command = (command or "").strip()
        if not command:
            return SandboxRunResult(
                command="",
                status="error",
                exit_code=1,
                error="Command is required",
                executor_type=self._executor_type(),
            )

        if not self.available():
            return SandboxRunResult(
                command=command,
                status="error",
                exit_code=1,
                error="Sandbox executor is unavailable",
                executor_type=self._executor_type(),
            )

        try:
            result = await asyncio.to_thread(
                self.executor.run,
                workdir=self.workspace,
                commands=[command],
                timeout_sec=timeout_seconds,
            )
        except Exception as exc:
            return SandboxRunResult(
                command=command,
                status="error",
                exit_code=1,
                error=str(exc),
                executor_type=self._executor_type(),
            )

        logs = (result.logs or "").strip()
        if len(logs) > MAX_SANDBOX_LOG_CHARS:
            logs = logs[:MAX_SANDBOX_LOG_CHARS].rstrip() + "\n...[truncated]"
        return SandboxRunResult(
            command=command,
            status=result.status,
            exit_code=result.exit_code,
            logs=logs,
            error=result.error,
            executor_type=self._executor_type(),
        )

    async def run_commands(
        self, commands: List[str], *, timeout_seconds: int
    ) -> List[SandboxRunResult]:
        results: List[SandboxRunResult] = []
        for command in commands:
            result = await self.run_command(command, timeout_seconds=timeout_seconds)
            results.append(result)
            if not result.success:
                break
        return results

    def _executor_type(self) -> str:
        if self.executor is None:
            return "none"
        return self.executor.executor_type


def summarize_verification_results(
    results: List[SandboxRunResult], *, max_chars: int = 1200
) -> str:
    if not results:
        return "No verification commands were executed."

    lines: List[str] = []
    for item in results:
        status = "ok" if item.success else "failed"
        lines.append(f"- `{item.command}` -> {status} (exit_code={item.exit_code})")
        if item.error:
            lines.append(f"  error: {item.error}")
        elif item.logs:
            preview = item.logs.replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180].rstrip() + "..."
            lines.append(f"  logs: {preview}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def _split_commands(raw: str) -> List[str]:
    # Keep shell chaining (&&) in the same command so setup and verify
    # happen within one sandbox run context.
    if "\n" in raw:
        items = [line.strip() for line in raw.splitlines()]
        return [item for item in items if item]
    return [raw.strip()] if raw.strip() else []


def _resolve_default_verify_commands(workspace: Path) -> List[str]:
    commands: List[str] = []
    has_python_project = (
        (workspace / "pyproject.toml").exists()
        or (workspace / "requirements.txt").exists()
        or (workspace / "setup.py").exists()
    )
    if has_python_project:
        if (workspace / "tests").is_dir():
            commands.append("PYTHONPATH=. pytest -q tests")
        elif list(workspace.glob("test_*.py")) or list(workspace.glob("*_test.py")):
            commands.append("PYTHONPATH=. pytest -q")
    if (workspace / "web" / "package.json").exists():
        commands.append("cd web && npm run lint && npm run build")
    return commands


def _resolve_default_bootstrap_commands(workspace: Path) -> List[str]:
    commands: List[str] = []
    requirements = workspace / "requirements.txt"
    if requirements.exists():
        commands.append("pip install -q -r requirements.txt")
    elif (workspace / "pyproject.toml").exists():
        commands.append("pip install -q .")
    return commands


def detect_missing_python_packages(
    results: Sequence[SandboxRunResult],
    *,
    workspace: Optional[Path] = None,
    known_local_modules: Optional[set[str]] = None,
) -> List[str]:
    local_modules = _scan_workspace_local_modules(workspace)
    if known_local_modules:
        local_modules |= known_local_modules

    packages: List[str] = []
    seen: set[str] = set()
    for item in results:
        text = "\n".join(part for part in (item.logs, item.error or "") if part)
        for match in _MISSING_MODULE_RE.findall(text):
            module = match.split(".", 1)[0].strip().lower()
            if not module or module in _STDLIB_MODULES:
                continue
            if module in local_modules:
                continue
            package = _MODULE_PACKAGE_ALIASES.get(module, module)
            if package and package not in seen:
                seen.add(package)
                packages.append(package)
    return packages


def _scan_workspace_local_modules(workspace: Optional[Path]) -> set[str]:
    """Scan workspace top-level entries to identify local Python modules/packages.

    Any top-level directory that contains ``__init__.py`` or any ``.py`` file,
    as well as any top-level ``.py`` file, is considered a local module that
    should never be installed from PyPI.
    """
    local: set[str] = set()
    if workspace is None or not workspace.is_dir():
        return local

    try:
        entries = list(workspace.iterdir())
    except OSError:
        return local

    for entry in entries:
        name = entry.name.lower()
        if entry.is_dir():
            # A directory is treated as a local package if it looks like
            # Python source: has __init__.py or contains any .py file at
            # the top level.
            if (entry / "__init__.py").exists():
                local.add(name)
            elif any(child.suffix == ".py" for child in entry.iterdir() if child.is_file()):
                local.add(name)
        elif entry.is_file() and entry.suffix == ".py":
            local.add(entry.stem.lower())

    return local


def _parse_int_env(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = (os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))
