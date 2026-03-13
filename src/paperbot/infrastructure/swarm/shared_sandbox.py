"""Shared Sandbox -- user-level VM abstraction for sandbox-as-workspace.

All agents operate directly inside the VM. The file system is the single
source of truth. Local workspace is only used for downloading results
after successful verification.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...repro.base_executor import BaseExecutor
from ...repro.execution_result import ExecutionResult

log = logging.getLogger(__name__)

BASE_DIR = "/home/user"

# Directories created by agents for inter-agent communication.
_AGENT_DIRS = frozenset({".plan", ".status", ".knowledge"})

# Directories/files to skip when downloading results to local.
_DEFAULT_SKIP_DIRS = frozenset({
    "__pycache__", ".git",
    ".mypy_cache", ".pytest_cache", ".tox", ".eggs",
    "node_modules", ".venv", "venv",
})


class SharedSandbox:
    """User-level shared VM. All agent file operations go through this class.

    Design principles:
    - VM is the single source of truth (no local file writes by agents)
    - All path operations are scoped to /home/user/{paper_slug}/
    - download_paper() is the *only* path from VM → local
    """

    def __init__(self, executor: BaseExecutor):
        self.executor = executor

    @property
    def alive(self) -> bool:
        return self.executor is not None and self.executor.available()

    def paper_root(self, slug: str) -> str:
        return f"{BASE_DIR}/{slug}"

    # ----- File operations (all inside VM) -----

    def read_file(self, slug: str, path: str) -> Optional[str]:
        """Read a file from the paper directory in the VM."""
        full = f"{self.paper_root(slug)}/{path}"
        for attempt in range(2):
            fs = self._get_fs()
            if fs is None:
                if attempt == 0 and self.recover_if_expired():
                    continue
                return None
            try:
                content = fs.read(full)
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                return content
            except Exception as exc:
                if attempt == 0 and self._is_sandbox_expired(exc):
                    log.warning("SharedSandbox.read_file: sandbox expired, recovering...")
                    if self.recover_if_expired():
                        continue
                return None
        return None

    def write_file(self, slug: str, path: str, content: str) -> bool:
        """Write a file into the paper directory in the VM."""
        root = self.paper_root(slug)
        full = f"{root}/{path}"
        for attempt in range(2):
            fs = self._get_fs()
            if fs is None:
                if attempt == 0 and self.recover_if_expired():
                    continue
                return False
            try:
                parent = str(Path(path).parent)
                if parent not in ("", "."):
                    self._ensure_dir(fs, f"{root}/{parent}")
                fs.write(full, content)
                return True
            except Exception as exc:
                if attempt == 0 and self._is_sandbox_expired(exc):
                    log.warning("SharedSandbox.write_file: sandbox expired, recovering...")
                    if self.recover_if_expired():
                        continue
                log.warning("SharedSandbox.write_file failed for %s: %s", full, exc)
                return False
        return False

    def list_files(self, slug: str, path: str = ".") -> List[str]:
        """List entries under a path in the paper directory."""
        root = self.paper_root(slug)
        target = f"{root}/{path}" if path not in (".", "") else root
        result = self.run_in_paper(
            slug,
            f"find {shlex.quote(target)} -maxdepth 1 -not -name '.*' 2>/dev/null | head -100",
        )
        if not result.success or not result.logs:
            return []
        prefix = f"{root}/"
        entries = []
        for line in result.logs.strip().splitlines():
            line = line.strip()
            if not line or line == target:
                continue
            entries.append(line.replace(prefix, ""))
        return entries

    def list_files_recursive(self, slug: str) -> List[str]:
        """Recursively list all files in the paper directory."""
        root = self.paper_root(slug)
        result = self.run_in_paper(
            slug,
            f"find {shlex.quote(root)} -type f 2>/dev/null",
        )
        if not result.success or not result.logs:
            return []
        prefix = f"{root}/"
        return [
            line.strip().replace(prefix, "")
            for line in result.logs.strip().splitlines()
            if line.strip() and line.strip() != root
        ]

    def search_files(self, slug: str, pattern: str, glob: str = "*") -> str:
        """Search file contents by pattern in the paper directory."""
        root = self.paper_root(slug)
        result = self.run_command(
            f"grep -rn --include={shlex.quote(glob)} {shlex.quote(pattern)} "
            f"{shlex.quote(root)} 2>/dev/null | head -50",
            cwd=root,
        )
        output = result.logs.replace(f"{root}/", "") if result.logs else "(no matches)"
        return output[:6000]

    # ----- Command execution -----

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout_sec: int = 120,
    ) -> ExecutionResult:
        """Run a command inside the VM."""
        for attempt in range(2):
            if not self.alive:
                if attempt == 0 and self.recover_if_expired():
                    continue
                return ExecutionResult(
                    status="error",
                    exit_code=1,
                    error="Sandbox not available",
                )
            sandbox = self._get_raw_sandbox()
            if sandbox is None:
                return ExecutionResult(
                    status="error",
                    exit_code=1,
                    error="No sandbox instance",
                )

            try:
                if hasattr(sandbox, "commands"):
                    result = sandbox.commands.run(command, timeout=timeout_sec, cwd=cwd or BASE_DIR)
                    stdout = result.stdout or ""
                    stderr = result.stderr or ""
                    exit_code = result.exit_code
                elif hasattr(sandbox, "process"):
                    proc = sandbox.process.start(command, timeout=timeout_sec, cwd=cwd or BASE_DIR)
                    output = proc.wait()
                    stdout = output.stdout or ""
                    stderr = output.stderr or ""
                    exit_code = output.exit_code
                else:
                    return ExecutionResult(status="error", exit_code=1, error="Unknown sandbox API")

                logs = stdout
                if stderr:
                    logs = f"{logs}\n[stderr] {stderr}" if logs else f"[stderr] {stderr}"

                return ExecutionResult(
                    status="success" if exit_code == 0 else "failed",
                    exit_code=exit_code,
                    logs=logs.strip(),
                )
            except Exception as exc:
                if attempt == 0 and self._is_sandbox_expired(exc):
                    log.warning("SharedSandbox.run_command: sandbox expired, recovering...")
                    if self.recover_if_expired():
                        continue
                return ExecutionResult(
                    status="error",
                    exit_code=1,
                    error=str(exc),
                    logs=str(exc),
                )
        return ExecutionResult(status="error", exit_code=1, error="Sandbox recovery exhausted")

    def run_in_paper(
        self, slug: str, command: str, timeout_sec: int = 120
    ) -> ExecutionResult:
        """Run a command with cwd set to the paper root directory."""
        return self.run_command(command, cwd=self.paper_root(slug), timeout_sec=timeout_sec)

    # ----- Timeout / keepalive -----

    def refresh_timeout(self) -> bool:
        """Extend the sandbox TTL. Call periodically during long sessions."""
        executor = self.executor
        if executor is None:
            return False
        refresh = getattr(executor, "_refresh_sandbox_timeout", None)
        if not callable(refresh):
            return False
        sandbox = self._get_raw_sandbox()
        if sandbox is None:
            return False
        try:
            refresh(sandbox)
            log.debug("SharedSandbox: timeout refreshed")
            return True
        except Exception:
            log.debug("SharedSandbox: timeout refresh failed", exc_info=True)
            return False

    def recover_if_expired(self) -> bool:
        """Attempt to recover if the sandbox has expired.

        Returns True if the sandbox is alive (either was already alive or
        recovery succeeded).
        """
        if self.alive:
            return True
        recover = getattr(self.executor, "_recover_expired_sandbox", None)
        if callable(recover):
            try:
                new_sandbox = recover()
                if new_sandbox is not None:
                    log.info("SharedSandbox: recovered from expired sandbox")
                    return True
            except Exception:
                log.warning("SharedSandbox: recovery failed", exc_info=True)
        return False

    # ----- Lifecycle -----

    def ensure_paper_dir(self, slug: str) -> bool:
        """Ensure the paper directory exists in the VM."""
        result = self.run_command(f"mkdir -p {shlex.quote(self.paper_root(slug))}")
        return result.exit_code == 0

    def list_papers(self) -> List[str]:
        """List all paper directories in the VM."""
        result = self.run_command(
            f"ls -1d {BASE_DIR}/*/ 2>/dev/null | xargs -I{{}} basename {{}}"
        )
        if not result.success or not result.logs:
            return []
        return [d.strip() for d in result.logs.strip().splitlines() if d.strip()]

    # ----- Download results to local -----

    def download_paper(
        self,
        slug: str,
        local_dir: Path,
        skip_dirs: Optional[Set[str]] = None,
    ) -> List[str]:
        """Download paper outputs from VM to local directory.

        Only called after verification passes. Includes agent communication
        directories (.plan, .status, .knowledge) by default.
        """
        skip = skip_dirs if skip_dirs is not None else _DEFAULT_SKIP_DIRS
        files = self.list_files_recursive(slug)
        downloaded: List[str] = []

        for remote_path in files:
            parts = Path(remote_path).parts
            if any(p in skip for p in parts):
                continue
            content = self.read_file(slug, remote_path)
            if content is None:
                continue
            local_path = local_dir / remote_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding="utf-8")
            downloaded.append(remote_path)

        return downloaded

    # ----- Internal helpers -----

    def _get_fs(self) -> Optional[Any]:
        sandbox = self._get_raw_sandbox()
        if sandbox is None:
            return None
        return getattr(sandbox, "files", None) or getattr(sandbox, "filesystem", None)

    def _get_raw_sandbox(self) -> Optional[Any]:
        """Get the underlying E2B sandbox object from the executor."""
        # PersistentSandboxManager stores executor; executor stores _sandbox
        sandbox = getattr(self.executor, "_sandbox", None)
        if sandbox is not None:
            return sandbox
        # Fallback: the executor IS the sandbox (duck typing)
        if hasattr(self.executor, "commands") or hasattr(self.executor, "files"):
            return self.executor
        return None

    @staticmethod
    def _is_sandbox_expired(exc: Exception) -> bool:
        """Check if an exception indicates the sandbox has timed out."""
        text = str(exc).lower()
        return (
            "sandbox was not found" in text
            or "sandbox not found" in text
            or "due to sandbox timeout" in text
        )

    def _ensure_dir(self, fs: Any, path: str) -> None:
        """Create a directory in the VM, ignoring 'already exists' errors."""
        try:
            fs.make_dir(path)
        except Exception:
            pass

    def teardown(self) -> None:
        """Destroy the VM."""
        cleanup = getattr(self.executor, "cleanup", None)
        if callable(cleanup):
            cleanup()
