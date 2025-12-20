# repro/e2b_executor.py
"""
E2B (e2b.dev) cloud sandbox executor for code verification.

E2B provides secure cloud-based code execution environments,
ideal for running untrusted code generated from papers.

Features:
- No local Docker required
- Secure microVM isolation
- Multi-language support
- Automatic dependency management
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base_executor import BaseExecutor
from .execution_result import ExecutionResult

logger = logging.getLogger(__name__)

# Lazy import E2B SDK
try:
    from e2b_code_interpreter import Sandbox
    HAS_E2B = True
except ImportError:
    Sandbox = None
    HAS_E2B = False


class E2BExecutor(BaseExecutor):
    """
    E2B cloud sandbox executor.

    Uses E2B's Code Interpreter sandbox for secure code execution.
    Requires E2B_API_KEY environment variable or explicit api_key.

    Advantages over Docker:
    - No local Docker installation needed
    - Better isolation (microVM-based)
    - Scalable cloud infrastructure
    - Pre-built environments with common ML libraries

    Usage:
        executor = E2BExecutor(api_key="your-api-key")
        result = executor.run(workdir=Path("./code"), commands=["python main.py"])
    """

    # E2B sandbox templates
    TEMPLATE_PYTHON = "Python3"
    TEMPLATE_NODEJS = "Node.js"

    def __init__(
        self,
        api_key: Optional[str] = None,
        template: str = TEMPLATE_PYTHON,
        timeout_sandbox: int = 300,
        keep_alive: bool = False,
    ):
        """
        Initialize E2B executor.

        Args:
            api_key: E2B API key (defaults to E2B_API_KEY env var)
            template: Sandbox template to use (default: Python3)
            timeout_sandbox: Sandbox lifetime in seconds
            keep_alive: Whether to keep sandbox alive between runs
        """
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.template = template
        self.timeout_sandbox = timeout_sandbox
        self.keep_alive = keep_alive
        self._sandbox: Optional[Any] = None

        if not HAS_E2B:
            logger.warning(
                "E2B SDK not installed. Install with: pip install e2b-code-interpreter"
            )

    def available(self) -> bool:
        """Check if E2B is available (SDK installed and API key set)."""
        return HAS_E2B and bool(self.api_key)

    def _get_sandbox(self) -> Optional[Any]:
        """Get or create a sandbox instance."""
        if not self.available():
            return None

        if self._sandbox is not None and self.keep_alive:
            return self._sandbox

        try:
            self._sandbox = Sandbox(
                api_key=self.api_key,
                timeout=self.timeout_sandbox,
            )
            return self._sandbox
        except Exception as e:
            logger.error(f"Failed to create E2B sandbox: {e}")
            return None

    def _upload_files(self, sandbox: Any, workdir: Path) -> Dict[str, str]:
        """
        Upload files from workdir to sandbox.

        Returns:
            Dict mapping local paths to sandbox paths
        """
        uploaded = {}

        for file_path in workdir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(workdir)
                sandbox_path = f"/home/user/{relative_path}"

                try:
                    # Ensure parent directory exists
                    parent_dir = str(relative_path.parent)
                    if parent_dir != ".":
                        sandbox.filesystem.make_dir(f"/home/user/{parent_dir}")

                    # Upload file content
                    content = file_path.read_text(errors="ignore")
                    sandbox.filesystem.write(sandbox_path, content)
                    uploaded[str(relative_path)] = sandbox_path

                except Exception as e:
                    logger.warning(f"Failed to upload {file_path}: {e}")

        return uploaded

    def run(
        self,
        workdir: Path,
        commands: List[str],
        timeout_sec: int = 300,
        cache_dir: Optional[Path] = None,
        record_meta: bool = True,
    ) -> ExecutionResult:
        """
        Execute commands in E2B sandbox.

        Args:
            workdir: Directory containing code to execute
            commands: List of shell commands to run
            timeout_sec: Maximum execution time per command
            cache_dir: Ignored for E2B (caching handled by E2B)
            record_meta: Whether to record runtime metadata

        Returns:
            ExecutionResult with status, logs, and metadata
        """
        if not self.available():
            return ExecutionResult(
                status="error",
                exit_code=1,
                error="E2B not available. Check API key and SDK installation.",
            )

        start_time = time.time()
        all_logs = []
        final_exit_code = 0

        try:
            # Create sandbox context
            with Sandbox(api_key=self.api_key, timeout=self.timeout_sandbox) as sandbox:
                # Upload files
                logger.info(f"Uploading files from {workdir} to E2B sandbox...")
                uploaded = self._upload_files(sandbox, workdir)
                logger.info(f"Uploaded {len(uploaded)} files")

                # Change to work directory
                sandbox.process.start("cd /home/user")

                # Execute commands
                for cmd in commands:
                    logger.debug(f"Executing: {cmd}")

                    try:
                        # Use process API for shell commands
                        result = sandbox.process.start(
                            cmd,
                            timeout=timeout_sec,
                            cwd="/home/user",
                        )

                        # Wait for completion
                        output = result.wait()

                        stdout = output.stdout or ""
                        stderr = output.stderr or ""
                        exit_code = output.exit_code

                        all_logs.append(f"$ {cmd}")
                        if stdout:
                            all_logs.append(stdout)
                        if stderr:
                            all_logs.append(f"[stderr] {stderr}")
                        all_logs.append(f"[exit_code: {exit_code}]")

                        if exit_code != 0:
                            final_exit_code = exit_code
                            # Continue execution for remaining commands

                    except Exception as cmd_error:
                        all_logs.append(f"$ {cmd}")
                        all_logs.append(f"[error] {str(cmd_error)}")
                        final_exit_code = 1

                duration = time.time() - start_time
                status = "success" if final_exit_code == 0 else "failed"

                result = ExecutionResult(
                    status=status,
                    exit_code=final_exit_code,
                    logs="\n".join(all_logs)[-8000:],  # Limit log size
                    duration_sec=duration,
                )

                if record_meta:
                    result.runtime_meta = {
                        "executor": "e2b",
                        "template": self.template,
                        "files_uploaded": len(uploaded),
                        "timeout_sec": timeout_sec,
                        "sandbox_timeout": self.timeout_sandbox,
                    }

                return result

        except Exception as e:
            logger.error(f"E2B execution error: {e}")
            return ExecutionResult(
                status="error",
                exit_code=1,
                error=str(e),
                duration_sec=time.time() - start_time,
            )

    def run_code(
        self,
        code: str,
        language: str = "python",
        timeout_sec: int = 60,
    ) -> ExecutionResult:
        """
        Execute code directly without file upload.

        Useful for quick code verification or REPL-style execution.

        Args:
            code: Source code to execute
            language: Programming language (default: python)
            timeout_sec: Execution timeout

        Returns:
            ExecutionResult with output
        """
        if not self.available():
            return ExecutionResult(
                status="error",
                exit_code=1,
                error="E2B not available",
            )

        start_time = time.time()

        try:
            with Sandbox(api_key=self.api_key, timeout=self.timeout_sandbox) as sandbox:
                # Use run_code for direct code execution
                execution = sandbox.run_code(code)

                # Collect output
                logs = []
                if execution.logs:
                    for log in execution.logs:
                        logs.append(str(log))

                if execution.error:
                    logs.append(f"[error] {execution.error}")
                    status = "failed"
                    exit_code = 1
                else:
                    status = "success"
                    exit_code = 0

                # Handle results (for expressions)
                if execution.results:
                    for result in execution.results:
                        if hasattr(result, 'text'):
                            logs.append(f"[result] {result.text}")

                return ExecutionResult(
                    status=status,
                    exit_code=exit_code,
                    logs="\n".join(logs),
                    duration_sec=time.time() - start_time,
                    runtime_meta={
                        "executor": "e2b",
                        "language": language,
                        "mode": "direct_code",
                    },
                )

        except Exception as e:
            return ExecutionResult(
                status="error",
                exit_code=1,
                error=str(e),
                duration_sec=time.time() - start_time,
            )

    def cleanup(self):
        """Clean up sandbox resources."""
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            self._sandbox = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
