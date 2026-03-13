"""Verification policy and repair loop for sandbox-as-workspace.

Verification commands run inside the VM's paper directory.  If
verification fails, a repair loop dispatches the error back to the
agent for fixing, up to ``max_repair_attempts`` times.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ...repro.execution_result import ExecutionResult
from .shared_sandbox import SharedSandbox

if TYPE_CHECKING:
    from .codex_dispatcher import CodexDispatcher, CodexResult
    from .sandbox_tool_executor import SandboxToolExecutor

log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    passed: bool
    commands_run: List[str]
    exit_code: int = 0
    logs: str = ""
    duration_sec: float = 0.0
    attempt: int = 0


@dataclass
class VerificationPolicy:
    enabled: bool = True
    commands: List[str] = field(default_factory=list)
    timeout_seconds: int = 180
    max_repair_attempts: int = 2

    @classmethod
    def from_sandbox_env(
        cls,
        sandbox: SharedSandbox,
        paper_slug: str,
    ) -> "VerificationPolicy":
        """Auto-detect verification commands from project files in the VM."""
        enabled_raw = os.getenv("CODEX_ENABLE_VERIFICATION", "true").strip().lower()
        enabled = enabled_raw not in {"0", "false", "no", "off"}
        timeout = int(os.getenv("CODEX_VERIFY_TIMEOUT_SECONDS", "180"))
        max_repair = int(os.getenv("CODEX_MAX_REPAIR_ATTEMPTS", "2"))

        raw_commands = os.getenv("CODEX_VERIFY_COMMANDS", "").strip()
        if raw_commands:
            commands = [c.strip() for c in raw_commands.split("&&") if c.strip()]
        else:
            commands = _detect_commands(sandbox, paper_slug)

        return cls(
            enabled=enabled and bool(commands),
            commands=commands,
            timeout_seconds=timeout,
            max_repair_attempts=max_repair,
        )


def _detect_commands(sandbox: SharedSandbox, slug: str) -> List[str]:
    """Heuristically detect verification commands from the VM file listing."""
    files = sandbox.list_files(slug, ".")
    commands: List[str] = []

    has_requirements = "requirements.txt" in files
    has_python_project = any(f in files for f in ("pyproject.toml", "setup.py", "requirements.txt"))
    has_package_json = "package.json" in files

    if has_requirements:
        commands.append("pip install -q -r requirements.txt")

    if has_python_project:
        commands.append("PYTHONPATH=. pytest -q")
    elif has_package_json:
        commands.append("npm install && npm test")

    return commands


def run_verification(
    sandbox: SharedSandbox,
    paper_slug: str,
    policy: VerificationPolicy,
    attempt: int = 0,
) -> VerificationResult:
    """Run verification commands in the paper directory."""
    import time

    if not policy.commands:
        return VerificationResult(
            passed=True,
            commands_run=[],
            attempt=attempt,
        )

    start = time.time()
    combined = " && ".join(policy.commands)
    result = sandbox.run_in_paper(paper_slug, combined, timeout_sec=policy.timeout_seconds)
    duration = time.time() - start

    return VerificationResult(
        passed=result.success,
        commands_run=policy.commands,
        exit_code=result.exit_code,
        logs=result.logs[:4000] if result.logs else "",
        duration_sec=duration,
        attempt=attempt,
    )


async def verify_and_repair(
    sandbox: SharedSandbox,
    paper_slug: str,
    policy: VerificationPolicy,
    dispatcher: "CodexDispatcher",
    tool_executor: "SandboxToolExecutor",
    *,
    on_step: Optional[Callable[..., Awaitable[None]]] = None,
) -> VerificationResult:
    """Verify, and if it fails, dispatch repair prompts up to N times."""
    last_result: Optional[VerificationResult] = None

    for attempt in range(policy.max_repair_attempts + 1):
        vresult = run_verification(sandbox, paper_slug, policy, attempt=attempt)
        last_result = vresult

        if vresult.passed:
            return vresult

        if attempt >= policy.max_repair_attempts:
            break

        # Build repair prompt and re-run agent loop
        repair_prompt = _build_repair_prompt(
            commands=vresult.commands_run,
            logs=vresult.logs,
            attempt=attempt + 1,
        )
        from .sandbox_tool_executor import SANDBOX_WORKER_TOOLS

        try:
            await dispatcher.dispatch_with_sandbox_tools(
                task_id=f"repair-{attempt + 1}",
                prompt=repair_prompt,
                tool_executor=tool_executor,
                on_step=on_step,
            )
        except Exception:
            log.exception("Repair dispatch failed at attempt %d", attempt + 1)
            break

    return last_result  # type: ignore[return-value]


def _build_repair_prompt(commands: List[str], logs: str, attempt: int) -> str:
    return (
        f"## Repair Attempt {attempt}\n\n"
        "The previous implementation failed sandbox verification.\n\n"
        "## Verification Commands\n"
        f"```\n{' && '.join(commands)}\n```\n\n"
        "## Failure Output\n"
        f"```\n{logs[:6000]}\n```\n\n"
        "Diagnose the failure, fix the code using the available tools "
        "(read_file, write_file, run_command), and call task_done when fixed."
    )
