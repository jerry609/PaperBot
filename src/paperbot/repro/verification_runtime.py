from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


_MAX_LOG_CHARS = 4000


def _default_cache_dir() -> Path:
    return Path(os.getenv("PAPERBOT_VERIFICATION_RUNTIME_CACHE_DIR") or "output/runtime_envs")


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _requirements_hash(requirements_text: str) -> str:
    normalized = "\n".join(line.strip() for line in requirements_text.splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _trim_log(text: Optional[str], *, limit: int = _MAX_LOG_CHARS) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[-limit:]


def _requires_torch_cpu_index(requirements_text: str) -> bool:
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        package_name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
        if package_name in {"torch", "torchvision", "torchaudio"}:
            return True
    return False


def _pip_env(*, use_torch_cpu_index: bool = False) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PIP_NO_CACHE_DIR", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    if use_torch_cpu_index:
        env["PIP_INDEX_URL"] = "https://download.pytorch.org/whl/cpu"
        env["PIP_EXTRA_INDEX_URL"] = "https://pypi.org/simple"
    return env


@dataclass(frozen=True)
class PreparedVerificationRuntime:
    python_executable: str
    prepared: bool = False
    reused_cache: bool = False
    requirements_hash: str = ""
    requirements_path: Optional[str] = None
    environment_dir: Optional[str] = None
    failure_log_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "python_executable": self.python_executable,
            "prepared": self.prepared,
            "reused_cache": self.reused_cache,
            "requirements_hash": self.requirements_hash,
            "requirements_path": self.requirements_path,
            "environment_dir": self.environment_dir,
            "failure_log_path": self.failure_log_path,
            "error": self.error,
        }


class VerificationRuntimePreparationError(RuntimeError):
    def __init__(
        self,
        *,
        step: str,
        command: Sequence[str],
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        environment_dir: Optional[str],
        requirements_path: Optional[str],
        requirements_hash: str,
        failure_log_path: Optional[str],
    ) -> None:
        self.step = step
        self.command = [str(part) for part in command]
        self.returncode = returncode
        self.stdout = _trim_log(stdout)
        self.stderr = _trim_log(stderr)
        self.environment_dir = environment_dir
        self.requirements_path = requirements_path
        self.requirements_hash = requirements_hash
        self.failure_log_path = failure_log_path
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        detail = (self.stderr or self.stdout or "verification runtime preparation failed").strip()
        detail_line = detail.splitlines()[-1][:240]
        code = self.returncode if self.returncode is not None else "timeout"
        return f"{self.step} failed (exit {code}): {detail_line}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "environment_dir": self.environment_dir,
            "requirements_path": self.requirements_path,
            "requirements_hash": self.requirements_hash,
            "failure_log_path": self.failure_log_path,
            "error": str(self),
        }


def _write_failure_log(
    env_dir: Path,
    *,
    step: str,
    command: Sequence[str],
    returncode: Optional[int],
    stdout: str,
    stderr: str,
    requirements_path: Optional[Path],
    requirements_hash: str,
) -> str:
    env_dir.mkdir(parents=True, exist_ok=True)
    log_path = env_dir / ".paperbot_runtime_failure.json"
    payload = {
        "step": step,
        "command": [str(part) for part in command],
        "returncode": returncode,
        "stdout": _trim_log(stdout),
        "stderr": _trim_log(stderr),
        "environment_dir": str(env_dir),
        "requirements_path": str(requirements_path) if requirements_path else None,
        "requirements_hash": requirements_hash,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(log_path)


def _run_runtime_command(
    command: Sequence[str],
    *,
    step: str,
    env_dir: Path,
    requirements_path: Optional[Path],
    requirements_hash: str,
    timeout: int,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        failure_log_path = _write_failure_log(
            env_dir,
            step=step,
            command=command,
            returncode=None,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            requirements_path=requirements_path,
            requirements_hash=requirements_hash,
        )
        raise VerificationRuntimePreparationError(
            step=step,
            command=command,
            returncode=None,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            environment_dir=str(env_dir),
            requirements_path=str(requirements_path) if requirements_path else None,
            requirements_hash=requirements_hash,
            failure_log_path=failure_log_path,
        ) from exc
    except subprocess.CalledProcessError as exc:
        failure_log_path = _write_failure_log(
            env_dir,
            step=step,
            command=command,
            returncode=exc.returncode,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            requirements_path=requirements_path,
            requirements_hash=requirements_hash,
        )
        raise VerificationRuntimePreparationError(
            step=step,
            command=command,
            returncode=exc.returncode,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            environment_dir=str(env_dir),
            requirements_path=str(requirements_path) if requirements_path else None,
            requirements_hash=requirements_hash,
            failure_log_path=failure_log_path,
        ) from exc


def prepare_verification_runtime(
    output_dir: Path,
    *,
    base_python: str = "python3",
    prepare_requirements: bool = False,
    cache_dir: Optional[Path] = None,
    install_timeout: int = 600,
    extra_packages: Optional[Sequence[str]] = None,
    prefer_cpu_torch: bool = False,
) -> PreparedVerificationRuntime:
    requirements_path = output_dir / "requirements.txt"
    if not prepare_requirements or not requirements_path.exists():
        return PreparedVerificationRuntime(
            python_executable=base_python,
            prepared=False,
            requirements_path=str(requirements_path) if requirements_path.exists() else None,
        )

    requirements_text = requirements_path.read_text(encoding="utf-8").strip()
    if not requirements_text:
        return PreparedVerificationRuntime(
            python_executable=base_python,
            prepared=False,
            requirements_path=str(requirements_path),
        )

    resolved_cache_dir = (cache_dir or _default_cache_dir()).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    req_hash = _requirements_hash(requirements_text)
    env_dir = resolved_cache_dir / f"req_{req_hash}"
    env_python = _venv_python_path(env_dir)
    stamp_file = env_dir / ".paperbot_runtime_ready"
    failure_log_path = env_dir / ".paperbot_runtime_failure.json"
    if env_python.exists() and stamp_file.exists():
        return PreparedVerificationRuntime(
            python_executable=str(env_python),
            prepared=True,
            reused_cache=True,
            requirements_hash=req_hash,
            requirements_path=str(requirements_path),
            environment_dir=str(env_dir),
            failure_log_path=str(failure_log_path) if failure_log_path.exists() else None,
        )

    if env_dir.exists() and (not env_python.exists() or not stamp_file.exists()):
        import shutil

        shutil.rmtree(env_dir, ignore_errors=True)

    _run_runtime_command(
        [base_python, "-m", "venv", str(env_dir)],
        step="create_venv",
        env_dir=env_dir,
        requirements_path=requirements_path,
        requirements_hash=req_hash,
        timeout=install_timeout,
    )
    _run_runtime_command(
        [str(env_python), "-m", "pip", "install", "-U", "pip"],
        step="upgrade_pip",
        env_dir=env_dir,
        requirements_path=requirements_path,
        requirements_hash=req_hash,
        timeout=install_timeout,
        env=_pip_env(),
    )
    _run_runtime_command(
        [str(env_python), "-m", "pip", "install", "-r", str(requirements_path)],
        step="install_requirements",
        env_dir=env_dir,
        requirements_path=requirements_path,
        requirements_hash=req_hash,
        timeout=install_timeout,
        cwd=output_dir,
        env=_pip_env(
            use_torch_cpu_index=(prefer_cpu_torch and _requires_torch_cpu_index(requirements_text))
        ),
    )

    for package in list(extra_packages or []):
        _run_runtime_command(
            [str(env_python), "-m", "pip", "install", package],
            step=f"install_extra:{package}",
            env_dir=env_dir,
            requirements_path=requirements_path,
            requirements_hash=req_hash,
            timeout=install_timeout,
            env=_pip_env(),
        )

    if failure_log_path.exists():
        failure_log_path.unlink()
    stamp_file.write_text(req_hash + "\n", encoding="utf-8")
    return PreparedVerificationRuntime(
        python_executable=str(env_python),
        prepared=True,
        reused_cache=False,
        requirements_hash=req_hash,
        requirements_path=str(requirements_path),
        environment_dir=str(env_dir),
    )
