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
from typing import Any, Dict, List, Optional

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
    TEMPLATE_PYTHON = os.getenv("E2B_TEMPLATE", "paperbot-repro")
    TEMPLATE_NODEJS = "Node.js"
    DEFAULT_SANDBOX_TIMEOUT_SECONDS = 3600
    MIN_SANDBOX_TIMEOUT_SECONDS = 60

    # Default root inside the sandbox where files are uploaded and commands run.
    DEFAULT_SANDBOX_HOME = "/home/user"

    def __init__(
        self,
        api_key: Optional[str] = None,
        template: Optional[str] = None,
        timeout_sandbox: Optional[int] = None,
        keep_alive: bool = False,
        sandbox_cwd: Optional[str] = None,
    ):
        """
        Initialize E2B executor.

        Args:
            api_key: E2B API key (defaults to E2B_API_KEY env var)
            template: Sandbox template to use (default: Python3)
            timeout_sandbox: Sandbox lifetime in seconds
            keep_alive: Whether to keep sandbox alive between runs
            sandbox_cwd: Working directory inside the sandbox.  Defaults to
                ``/home/user``.  When a *paper_slug* is provided by the caller
                this is typically ``/home/user/<slug>`` so that different paper
                reproductions are isolated inside one persistent VM.
        """
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.template = template or os.getenv("E2B_TEMPLATE", self.TEMPLATE_PYTHON)
        self.timeout_sandbox = self._resolve_sandbox_timeout(timeout_sandbox)
        self.keep_alive = keep_alive
        self.sandbox_cwd = sandbox_cwd or self.DEFAULT_SANDBOX_HOME
        self._sandbox: Optional[Any] = None
        self._sandbox_id: Optional[str] = None

        if not HAS_E2B:
            logger.warning("E2B SDK not installed. Install with: pip install e2b-code-interpreter")

    def available(self) -> bool:
        """Check if E2B is available (SDK installed and API key set)."""
        return HAS_E2B and bool(self.api_key)

    def _get_sandbox(self) -> Optional[Any]:
        """Get or create a sandbox instance."""
        if not self.available():
            return None

        if self._sandbox is not None and self.keep_alive:
            self._refresh_sandbox_timeout(self._sandbox)
            return self._sandbox

        if self.keep_alive and self._sandbox_id:
            connected = self._connect_sandbox_instance(self._sandbox_id)
            if connected is not None:
                self._refresh_sandbox_timeout(connected)
                self._sandbox = connected
                return connected

        try:
            # Set API key in environment (required for newer SDK versions)
            if self.api_key:
                os.environ["E2B_API_KEY"] = self.api_key

            sandbox = self._create_sandbox_instance()
            self._refresh_sandbox_timeout(sandbox)
            if self.keep_alive:
                self._sandbox = sandbox
                self._sandbox_id = self._extract_sandbox_id(sandbox) or self._sandbox_id
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create E2B sandbox: {e}")
            return None

    def attach_sandbox(self, sandbox_id: str) -> bool:
        """Attach to an existing sandbox by id for persistent-session reuse."""
        target = (sandbox_id or "").strip()
        if not target or not self.available():
            return False

        connected = self._connect_sandbox_instance(target)
        if connected is None:
            return False
        self._sandbox = connected
        self._sandbox_id = self._extract_sandbox_id(connected) or target
        self.keep_alive = True
        return True

    def ensure_sandbox(self) -> bool:
        """Ensure an active sandbox exists and capture its id."""
        sandbox = self._get_sandbox()
        if sandbox is None:
            return False
        self._sandbox_id = self._extract_sandbox_id(sandbox) or self._sandbox_id
        return True

    @property
    def sandbox_id(self) -> Optional[str]:
        return self._sandbox_id

    def _resolve_sandbox_timeout(self, timeout_sandbox: Optional[int]) -> int:
        if timeout_sandbox is not None:
            try:
                value = int(timeout_sandbox)
            except (TypeError, ValueError):
                value = self.DEFAULT_SANDBOX_TIMEOUT_SECONDS
            return max(self.MIN_SANDBOX_TIMEOUT_SECONDS, value)

        raw = os.getenv("E2B_SANDBOX_TIMEOUT_SECONDS", str(self.DEFAULT_SANDBOX_TIMEOUT_SECONDS))
        try:
            value = int((raw or "").strip())
        except (TypeError, ValueError):
            value = self.DEFAULT_SANDBOX_TIMEOUT_SECONDS
        return max(self.MIN_SANDBOX_TIMEOUT_SECONDS, value)

    def _refresh_sandbox_timeout(self, sandbox: Any) -> None:
        """Best-effort timeout refresh for long-running persistent sessions."""
        for candidate in (
            lambda: sandbox.set_timeout(self.timeout_sandbox),
            lambda: sandbox.set_timeout(timeout=self.timeout_sandbox),
            lambda: sandbox.setTimeout(self.timeout_sandbox),
            lambda: sandbox.setTimeout(timeout=self.timeout_sandbox),
        ):
            try:
                candidate()
                return
            except Exception:
                continue

    @staticmethod
    def _is_sandbox_not_found_error(error: Exception) -> bool:
        text = str(error or "").strip().lower()
        return (
            "sandbox was not found" in text
            or "sandbox not found" in text
            or "due to sandbox timeout" in text
        )

    def _recover_expired_sandbox(self) -> Optional[Any]:
        """Re-create sandbox after provider-side timeout/expiry."""
        self._sandbox = None
        self._sandbox_id = None
        try:
            if self.api_key:
                os.environ["E2B_API_KEY"] = self.api_key
            sandbox = self._create_sandbox_instance()
            self._refresh_sandbox_timeout(sandbox)
            if self.keep_alive:
                self._sandbox = sandbox
                self._sandbox_id = self._extract_sandbox_id(sandbox) or self._sandbox_id
            return sandbox
        except Exception as exc:
            logger.error("Failed to recover expired E2B sandbox: %s", exc)
            return None

    def _create_sandbox_instance(self) -> Any:
        # Prefer explicit template, but gracefully fall back if the SDK
        # version does not support the template argument.
        try:
            return Sandbox.create(template=self.template, timeout=self.timeout_sandbox)
        except (TypeError, AttributeError):
            pass

        try:
            return Sandbox.create(timeout=self.timeout_sandbox)
        except (TypeError, AttributeError):
            pass

        try:
            return Sandbox(
                api_key=self.api_key,
                template=self.template,
                timeout=self.timeout_sandbox,
            )
        except TypeError:
            return Sandbox(
                api_key=self.api_key,
                timeout=self.timeout_sandbox,
            )

    def _connect_sandbox_instance(self, sandbox_id: str) -> Optional[Any]:
        # Best-effort compatibility across E2B SDK versions.
        for candidate in (
            lambda: Sandbox.connect(sandbox_id),  # type: ignore[attr-defined]
            lambda: Sandbox.connect(id=sandbox_id),  # type: ignore[attr-defined]
            lambda: Sandbox.from_id(sandbox_id),  # type: ignore[attr-defined]
            lambda: Sandbox(sandbox_id=sandbox_id, api_key=self.api_key),
            lambda: Sandbox(id=sandbox_id, api_key=self.api_key),
        ):
            try:
                return candidate()
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_sandbox_id(sandbox: Any) -> Optional[str]:
        for attr in ("sandbox_id", "id"):
            value = getattr(sandbox, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        getter = getattr(sandbox, "get_id", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, str) and value.strip():
                    return value.strip()
            except Exception:
                pass
        return None

    # File patterns to skip during upload — binary assets, OS metadata,
    # build artefacts, and review/output docs that are not needed for
    # code execution inside the sandbox.
    _SKIP_DIRS = frozenset({
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", ".tox", ".eggs",
        "reviews", ".DS_Store",
    })
    _SKIP_SUFFIXES = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
        ".mp4", ".mp3", ".wav", ".avi", ".mov",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".whl", ".egg", ".pyc", ".pyo", ".so", ".dylib", ".dll",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    })
    _MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB per file

    def _upload_files(self, sandbox: Any, workdir: Path) -> Dict[str, str]:
        """
        Upload source files from workdir to sandbox.

        Files are placed under ``self.sandbox_cwd`` (e.g.
        ``/home/user/<paper_slug>/``).  Skips binary assets, OS metadata,
        build artefacts, and large files that are not needed for execution.

        Returns:
            Dict mapping local paths to sandbox paths
        """
        uploaded = {}
        cwd = self.sandbox_cwd.rstrip("/")

        # Get the filesystem API (new API uses 'files', old uses 'filesystem')
        fs = getattr(sandbox, "files", None) or getattr(sandbox, "filesystem", None)
        if fs is None:
            logger.warning("No filesystem API available in sandbox")
            return uploaded

        # Ensure the paper directory itself exists.
        if cwd != self.DEFAULT_SANDBOX_HOME:
            try:
                fs.make_dir(cwd)
            except Exception:
                pass

        for file_path in workdir.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip entire directory trees that are never useful in-sandbox.
            if any(part in self._SKIP_DIRS for part in file_path.relative_to(workdir).parts):
                continue

            # Skip binary / media / archive files.
            if file_path.suffix.lower() in self._SKIP_SUFFIXES:
                continue

            # Skip oversized files that would slow down upload.
            try:
                if file_path.stat().st_size > self._MAX_FILE_SIZE_BYTES:
                    logger.debug("Skipping oversized file: %s", file_path)
                    continue
            except OSError:
                continue

            relative_path = file_path.relative_to(workdir)
            sandbox_path = f"{cwd}/{relative_path}"

            try:
                self._ensure_remote_parent_dirs(fs, relative_path, base=cwd)
                content = file_path.read_text(errors="ignore")
                fs.write(sandbox_path, content)
                uploaded[str(relative_path)] = sandbox_path
            except Exception as e:
                logger.warning(f"Failed to upload {file_path}: {e}")

        return uploaded

    @staticmethod
    def _ensure_remote_parent_dirs(
        fs: Any, relative_path: Path, base: str = "/home/user"
    ) -> None:
        parent = relative_path.parent
        if str(parent) in {"", "."}:
            return

        current = Path(base)
        for segment in parent.parts:
            current = current / segment
            try:
                fs.make_dir(str(current))
            except Exception:
                # Directory may already exist.
                continue

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
        recovered_during_run = False

        try:
            # Set API key in environment (required for newer SDK versions)
            if self.api_key:
                os.environ["E2B_API_KEY"] = self.api_key

            sandbox = self._get_sandbox() if self.keep_alive else self._create_sandbox_instance()
            if sandbox is None:
                return ExecutionResult(
                    status="error",
                    exit_code=1,
                    error="Failed to obtain E2B sandbox instance.",
                )

            try:
                # Upload files
                logger.info(f"Uploading files from {workdir} to E2B sandbox...")
                uploaded = self._upload_files(sandbox, workdir)
                logger.info(f"Uploaded {len(uploaded)} files")

                # Execute commands
                for cmd in commands:
                    logger.debug(f"Executing: {cmd}")
                    retried_after_recovery = False

                    while True:
                        try:
                            # Use commands.run for shell commands (new API)
                            if hasattr(sandbox, "commands"):
                                result = sandbox.commands.run(
                                    cmd, timeout=timeout_sec, cwd=self.sandbox_cwd
                                )
                                stdout = result.stdout or ""
                                stderr = result.stderr or ""
                                exit_code = result.exit_code
                            elif hasattr(sandbox, "process"):
                                # Old API fallback
                                proc = sandbox.process.start(
                                    cmd, timeout=timeout_sec, cwd=self.sandbox_cwd
                                )
                                output = proc.wait()
                                stdout = output.stdout or ""
                                stderr = output.stderr or ""
                                exit_code = output.exit_code
                            else:
                                raise RuntimeError("Unknown E2B API version")

                            all_logs.append(f"$ {cmd}")
                            if stdout:
                                all_logs.append(stdout)
                            if stderr:
                                all_logs.append(f"[stderr] {stderr}")
                            all_logs.append(f"[exit_code: {exit_code}]")

                            if exit_code != 0:
                                final_exit_code = exit_code
                            break

                        except Exception as cmd_error:
                            # Handle provider-side timeout by rebuilding sandbox and retrying once.
                            if (
                                self.keep_alive
                                and not retried_after_recovery
                                and self._is_sandbox_not_found_error(cmd_error)
                            ):
                                recovered = self._recover_expired_sandbox()
                                if recovered is not None:
                                    recovered_during_run = True
                                    sandbox = recovered
                                    uploaded.update(self._upload_files(sandbox, workdir))
                                    retried_after_recovery = True
                                    all_logs.append(
                                        "[warning] Sandbox expired; recreated sandbox and retried command."
                                    )
                                    continue

                            # Some E2B SDK versions raise on non-zero exits while still
                            # attaching useful exit_code/stdout/stderr on the exception.
                            err_exit_code = int(getattr(cmd_error, "exit_code", 1) or 1)
                            err_stdout = str(getattr(cmd_error, "stdout", "") or "")
                            err_stderr = str(getattr(cmd_error, "stderr", "") or "")
                            all_logs.append(f"$ {cmd}")
                            if err_stdout:
                                all_logs.append(f"[stdout] {err_stdout}")
                            if err_stderr:
                                all_logs.append(f"[stderr] {err_stderr}")
                            all_logs.append(f"[error] {str(cmd_error)}")
                            final_exit_code = err_exit_code
                            break

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
                        "sandbox_id": self._extract_sandbox_id(sandbox),
                        "files_uploaded": len(uploaded),
                        "timeout_sec": timeout_sec,
                        "sandbox_timeout": self.timeout_sandbox,
                        "sandbox_recovered": recovered_during_run,
                    }

                return result
            finally:
                # Clean up sandbox
                if not self.keep_alive:
                    try:
                        if hasattr(sandbox, "kill"):
                            sandbox.kill()
                        elif hasattr(sandbox, "close"):
                            sandbox.close()
                    except Exception:
                        pass

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
            # Set API key in environment (required for newer SDK versions)
            if self.api_key:
                os.environ["E2B_API_KEY"] = self.api_key

            # Create sandbox using new API
            sandbox = self._get_sandbox() if self.keep_alive else self._create_sandbox_instance()
            if sandbox is None:
                return ExecutionResult(
                    status="error",
                    exit_code=1,
                    error="Failed to obtain E2B sandbox instance.",
                )

            try:
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
                        if hasattr(result, "text"):
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
                        "sandbox_id": self._extract_sandbox_id(sandbox),
                    },
                )
            finally:
                # Clean up sandbox
                if not self.keep_alive:
                    try:
                        if hasattr(sandbox, "kill"):
                            sandbox.kill()
                        elif hasattr(sandbox, "close"):
                            sandbox.close()
                    except Exception:
                        pass

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
            self._sandbox_id = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
