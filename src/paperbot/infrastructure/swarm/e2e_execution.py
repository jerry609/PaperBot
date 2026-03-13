"""End-to-end execution -- runs the paper's main entry point in the sandbox.

After all sub-tasks complete, this module detects the project's main script
(e.g., main.py, train.py, run.py), executes it inside the VM, and if it
fails, builds a diagnosis prompt so the Commander can direct Codex to fix
the code.  The fix-and-rerun loop repeats up to ``max_attempts`` times.

This mirrors how Manus works: write → run entire program → observe output →
fix → re-run until the full pipeline succeeds.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ...repro.execution_result import ExecutionResult
from .shared_sandbox import SharedSandbox

if TYPE_CHECKING:
    from .codex_dispatcher import CodexDispatcher
    from .sandbox_tool_executor import SandboxToolExecutor

log = logging.getLogger(__name__)

# Candidate entry-point filenames in priority order
_ENTRY_POINT_CANDIDATES = [
    "main.py",
    "train.py",
    "run.py",
    "run_experiment.py",
    "experiment.py",
    "reproduce.py",
    "demo.py",
    "evaluate.py",
    "eval.py",
    "test_main.py",
    "app.py",
    "index.py",
]

# Default timeout for running the full program (5 minutes)
_DEFAULT_RUN_TIMEOUT = 300

# Max output to capture for diagnosis
_MAX_OUTPUT_CHARS = 8000
_MAIN_GUARD_RE = re.compile(
    r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:',
    re.MULTILINE,
)


@dataclass
class E2EExecutionPolicy:
    """Controls end-to-end execution behavior."""

    enabled: bool = True
    entry_point: Optional[str] = None  # explicit entry point, or auto-detect
    entry_command: Optional[str] = None  # explicit command override (e.g., "bash run.sh")
    timeout_seconds: int = _DEFAULT_RUN_TIMEOUT
    max_repair_attempts: int = 3
    install_deps: bool = True  # pip install -r requirements.txt before running

    @classmethod
    def from_context(
        cls,
        sandbox: SharedSandbox,
        paper_slug: str,
        context_pack: Optional[dict] = None,
    ) -> "E2EExecutionPolicy":
        """Build policy from VM project state and optional context pack."""
        import os

        enabled = os.getenv("PAPERBOT_E2E_EXECUTION", "true").strip().lower()
        if enabled in ("0", "false", "no", "off"):
            return cls(enabled=False)

        timeout = int(os.getenv("PAPERBOT_E2E_TIMEOUT", str(_DEFAULT_RUN_TIMEOUT)))
        max_repair = int(os.getenv("PAPERBOT_E2E_MAX_REPAIR", "3"))

        # Check for explicit entry command in context pack
        entry_command = None
        entry_point = None
        if context_pack:
            entry_command = context_pack.get("entry_command")
            entry_point = context_pack.get("entry_point")

        # Auto-detect if not explicitly set
        if not entry_command and not entry_point:
            entry_point = detect_entry_point(sandbox, paper_slug)

        if not entry_command and not entry_point:
            # Do NOT disable E2E — fall through with entry_point=None.
            # The orchestrator will ask Commander to generate one.
            pass

        return cls(
            enabled=True,
            entry_point=entry_point,
            entry_command=entry_command,
            timeout_seconds=timeout,
            max_repair_attempts=max_repair,
        )


@dataclass
class E2EResult:
    """Result of an end-to-end execution attempt."""

    success: bool
    entry_point: str
    command: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
    attempt: int = 0
    repair_history: List[Dict[str, Any]] = field(default_factory=list)


def detect_entry_point(sandbox: SharedSandbox, paper_slug: str) -> Optional[str]:
    """Heuristically detect the project's main entry point from VM files.

    Strategy:
    1. Check for well-known filenames (main.py, train.py, etc.)
    2. Check for __main__.py in any package directory
    3. Check for shell scripts (run.sh, train.sh)
    4. Check Makefile for a 'run' or 'train' target
    """
    files = sandbox.list_files(paper_slug, ".")
    file_set = set(files)

    # 1. Well-known filenames in root
    for candidate in _ENTRY_POINT_CANDIDATES:
        if candidate in file_set:
            return candidate

    # 2. Check for __main__.py in subdirectories
    all_files = sandbox.list_files_recursive(paper_slug)
    for f in all_files:
        if f.endswith("__main__.py"):
            # Return the package directory for `python -m pkg`
            parts = f.split("/")
            if len(parts) >= 2:
                return f

    # 3. Shell scripts
    for script in ("run.sh", "train.sh", "experiment.sh", "start.sh"):
        if script in file_set:
            return script

    # 4. Check Makefile for run/train target
    if "Makefile" in file_set:
        makefile_content = sandbox.read_file(paper_slug, "Makefile")
        if makefile_content:
            for target in ("run:", "train:", "experiment:", "all:"):
                if target in makefile_content:
                    return f"Makefile:{target.rstrip(':')}"

    # 5. Look for any .py file with `if __name__` guard
    for f in all_files:
        if f.endswith(".py") and not f.startswith("test_") and not f.startswith("tests/"):
            content = sandbox.read_file(paper_slug, f)
            if content and _MAIN_GUARD_RE.search(content):
                return f

    return None


def build_run_command(policy: E2EExecutionPolicy) -> str:
    """Build the shell command to run the entry point."""
    if policy.entry_command:
        return policy.entry_command

    entry = policy.entry_point or "main.py"

    # Shell scripts
    if entry.endswith(".sh"):
        return f"bash {entry}"

    # Makefile targets
    if entry.startswith("Makefile:"):
        target = entry.split(":", 1)[1]
        return f"make {target}"

    # __main__.py → python -m package
    if entry.endswith("__main__.py"):
        parts = entry.replace("__main__.py", "").rstrip("/").split("/")
        pkg = ".".join(p for p in parts if p)
        if pkg:
            return f"python -m {pkg}"
        return f"python {entry}"

    # Files inside a package directory (e.g. src/evaluate.py using
    # "from src.module import ..." style imports) — run as module so that
    # Python resolves the package correctly.
    parts = entry.replace(".py", "").split("/")
    if len(parts) > 1 and all(p.isidentifier() for p in parts):
        return f"python -m {'.'.join(parts)}"

    # Regular Python file at project root
    return f"python {entry}"


def _build_deps_command(sandbox: SharedSandbox, paper_slug: str) -> Optional[str]:
    """Build dependency installation command if requirements exist."""
    files = sandbox.list_files(paper_slug, ".")
    if "requirements.txt" in files:
        return "pip install -q -r requirements.txt 2>&1"
    if "setup.py" in files:
        return "pip install -q -e . 2>&1"
    if "pyproject.toml" in files:
        return "pip install -q -e . 2>&1"
    if "package.json" in files:
        return "npm install 2>&1"
    return None


def _tail_truncate(text: str, max_chars: int) -> str:
    """Keep the tail of command output, where failure traces usually appear."""
    if len(text) <= max_chars:
        return text
    return "...[truncated]\n" + text[-max_chars:]


def run_e2e(
    sandbox: SharedSandbox,
    paper_slug: str,
    policy: E2EExecutionPolicy,
    attempt: int = 0,
) -> E2EResult:
    """Run the paper's entry point in the sandbox and capture output."""
    command = build_run_command(policy)
    entry = policy.entry_point or policy.entry_command or "unknown"

    # Install dependencies first if needed
    if policy.install_deps:
        deps_cmd = _build_deps_command(sandbox, paper_slug)
        if deps_cmd:
            log.info("Installing dependencies: %s", deps_cmd)
            sandbox.run_in_paper(paper_slug, deps_cmd, timeout_sec=120)

    log.info("E2E execution [attempt %d]: %s", attempt, command)
    start = time.time()
    result = sandbox.run_in_paper(paper_slug, command, timeout_sec=policy.timeout_seconds)
    duration = time.time() - start

    stdout = _tail_truncate(result.logs or "", _MAX_OUTPUT_CHARS)
    stderr = _tail_truncate(result.error or "", _MAX_OUTPUT_CHARS)

    return E2EResult(
        success=result.success,
        entry_point=entry,
        command=command,
        exit_code=result.exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_sec=duration,
        attempt=attempt,
    )


def _build_diagnosis_prompt(
    e2e_result: E2EResult,
    project_files: List[str],
) -> str:
    """Build a prompt for Commander to diagnose and direct a fix."""
    output_section = e2e_result.stdout
    if e2e_result.stderr:
        output_section += f"\n\n--- STDERR ---\n{e2e_result.stderr}"

    files_listing = "\n".join(f"- {f}" for f in project_files[:50])

    return (
        f"## End-to-End Execution Failed (Attempt {e2e_result.attempt + 1})\n\n"
        f"**Entry point**: `{e2e_result.entry_point}`\n"
        f"**Command**: `{e2e_result.command}`\n"
        f"**Exit code**: {e2e_result.exit_code}\n"
        f"**Duration**: {e2e_result.duration_sec:.1f}s\n\n"
        "## Output\n"
        f"```\n{_tail_truncate(output_section, 6000)}\n```\n\n"
        "## Project Files\n"
        f"{files_listing}\n\n"
        "## Instructions\n"
        "Analyze the error output above. Identify the root cause of the failure.\n"
        "Use the available tools to:\n"
        "1. Read the relevant source files to understand the issue\n"
        "2. Fix the code by writing corrected files\n"
        "3. If needed, run intermediate commands to verify your fix\n"
        "4. Call task_done when the fix is ready\n\n"
        "Focus on making the full program run successfully. "
        "Do NOT just add error suppression — fix the actual issue."
    )


async def run_e2e_with_repair(
    sandbox: SharedSandbox,
    paper_slug: str,
    policy: E2EExecutionPolicy,
    dispatcher: "CodexDispatcher",
    tool_executor_factory: Callable[[], "SandboxToolExecutor"],
    *,
    on_step: Optional[Callable[..., Awaitable[None]]] = None,
    on_attempt: Optional[Callable[[E2EResult], Awaitable[None]]] = None,
) -> E2EResult:
    """Run the entry point, and if it fails, use Commander-directed repair loop.

    Parameters
    ----------
    sandbox : SharedSandbox
        The VM sandbox wrapper.
    paper_slug : str
        Paper namespace in the VM.
    policy : E2EExecutionPolicy
        Execution and repair configuration.
    dispatcher : CodexDispatcher
        Dispatches repair tasks to Codex.
    tool_executor_factory : callable
        Factory to create a fresh SandboxToolExecutor per repair attempt.
    on_step : callback, optional
        Step callback for tool-use events during repair.
    on_attempt : callback, optional
        Called after each execution attempt (success or failure).
    """
    repair_history: List[Dict[str, Any]] = []
    result: Optional[E2EResult] = None

    for attempt in range(policy.max_repair_attempts + 1):
        result = run_e2e(sandbox, paper_slug, policy, attempt=attempt)

        repair_history.append({
            "attempt": attempt,
            "success": result.success,
            "exit_code": result.exit_code,
            "duration_sec": result.duration_sec,
            "stdout_preview": result.stdout[:500],
            "stderr_preview": result.stderr[:500],
        })
        result.repair_history = repair_history

        if on_attempt:
            await on_attempt(result)

        if result.success:
            log.info("E2E execution succeeded on attempt %d", attempt)
            return result

        if attempt >= policy.max_repair_attempts:
            log.warning(
                "E2E execution failed after %d attempts", attempt + 1
            )
            break

        # Dispatch repair to Codex
        log.info("E2E failed (attempt %d), dispatching repair...", attempt)
        project_files = sandbox.list_files_recursive(paper_slug)
        diagnosis_prompt = _build_diagnosis_prompt(result, project_files)

        tool_exec = tool_executor_factory()
        try:
            await dispatcher.dispatch_with_sandbox_tools(
                task_id=f"e2e-repair-{attempt + 1}",
                prompt=diagnosis_prompt,
                tool_executor=tool_exec,
                on_step=on_step,
            )
        except Exception:
            log.exception("E2E repair dispatch failed at attempt %d", attempt + 1)
            break

    if result is None:
        return E2EResult(
            success=False,
            entry_point=policy.entry_point or policy.entry_command or "unknown",
            command=build_run_command(policy),
            exit_code=1,
            stderr="No execution attempts were made (max_repair_attempts < 0?).",
        )
    return result
