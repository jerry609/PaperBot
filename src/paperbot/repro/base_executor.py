# repro/base_executor.py
"""
Abstract base class for code executors.

Provides a unified interface for different execution backends:
- DockerExecutor: Local Docker-based execution
- E2BExecutor: Cloud-based E2B sandbox execution
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .execution_result import ExecutionResult


class BaseExecutor(ABC):
    """
    Abstract base class for code execution backends.

    All executors must implement:
    - available(): Check if the executor is ready
    - run(): Execute commands in the sandbox
    """

    @abstractmethod
    def available(self) -> bool:
        """Check if the executor is available and ready to use."""
        pass

    @abstractmethod
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

        Args:
            workdir: Directory containing code to execute
            commands: List of shell commands to run
            timeout_sec: Maximum execution time in seconds
            cache_dir: Optional cache directory for dependencies
            record_meta: Whether to record runtime metadata

        Returns:
            ExecutionResult with status, logs, and metadata
        """
        pass

    @property
    def executor_type(self) -> str:
        """Return the executor type identifier."""
        return self.__class__.__name__
