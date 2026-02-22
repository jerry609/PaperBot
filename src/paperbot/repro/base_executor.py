# repro/base_executor.py
"""
Base class for code executors.

Note: Docker and E2B executors have been removed.
Code execution now uses Claude CLI integration.
"""

from pathlib import Path
from typing import List, Optional

from .execution_result import ExecutionResult


class BaseExecutor:
    """
    Stub executor class.

    Note: Actual code execution is now handled by Claude CLI.
    This class is kept for interface compatibility.
    """

    def available(self) -> bool:
        """Check if the executor is available and ready to use."""
        return False

    def run(
        self,
        workdir: Path,
        commands: List[str],
        timeout_sec: int = 300,
        cache_dir: Optional[Path] = None,
        record_meta: bool = True,
    ) -> ExecutionResult:
        """
        Execute commands in the sandbox environment.

        Note: This is a stub. Actual execution uses Claude CLI.
        """
        return ExecutionResult(
            success=False,
            logs="Executor not available. Use Claude CLI for code execution.",
            exit_code=1,
        )

    @property
    def executor_type(self) -> str:
        """Return the executor type identifier."""
        return "stub"
