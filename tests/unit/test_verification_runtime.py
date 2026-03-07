from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
from paperbot.repro.verification_runtime import (
    VerificationRuntimePreparationError,
    _normalize_requirements_text,
    prepare_verification_runtime,
)


def _hash_requirements(text: str) -> str:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def test_prepare_verification_runtime_returns_base_python_without_requirements(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    runtime = prepare_verification_runtime(
        output_dir,
        base_python="python3",
        prepare_requirements=True,
        cache_dir=tmp_path / "cache",
    )
    assert runtime.python_executable == "python3"
    assert runtime.prepared is False


def test_prepare_verification_runtime_reuses_cached_env(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    requirements = "pytest\n"
    (output_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
    env_dir = tmp_path / "cache" / f"req_{_hash_requirements(requirements)}"
    python_path = env_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")
    (env_dir / ".paperbot_runtime_ready").write_text("ok\n", encoding="utf-8")
    runtime = prepare_verification_runtime(
        output_dir,
        base_python="python3",
        prepare_requirements=True,
        cache_dir=tmp_path / "cache",
    )
    assert runtime.prepared is True
    assert runtime.reused_cache is True
    assert runtime.environment_dir == str(env_dir)


def test_prepare_verification_runtime_surfaces_install_failure(monkeypatch, tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "requirements.txt").write_text("broken-package\n", encoding="utf-8")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["python3", "-m", "venv"]:
            env_dir = Path(cmd[-1])
            python_path = env_dir / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "install" in cmd and "-r" in cmd:
            install_path = Path(cmd[-1])
            assert install_path.exists()
            assert install_path.read_text(encoding="utf-8") == "broken-package\n"
            raise subprocess.CalledProcessError(
                1,
                cmd,
                output="",
                stderr="ERROR: Could not find a version that satisfies the requirement broken-package",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("paperbot.repro.verification_runtime.subprocess.run", _fake_run)
    try:
        prepare_verification_runtime(
            output_dir,
            base_python="python3",
            prepare_requirements=True,
            cache_dir=tmp_path / "cache",
        )
    except VerificationRuntimePreparationError as exc:
        assert exc.step == "install_requirements"
        assert "broken-package" in exc.stderr
        assert exc.failure_log_path is not None
        failure_log = Path(exc.failure_log_path)
        assert failure_log.exists()
        assert "broken-package" in failure_log.read_text(encoding="utf-8")
    else:
        raise AssertionError("Expected VerificationRuntimePreparationError")


def test_normalize_requirements_text_maps_known_packages_and_deduplicates():
    normalized = _normalize_requirements_text(
        "PIL\nnumpy\nunknown module\npillow\n# keep me\n"
    )

    assert normalized.splitlines() == ["Pillow", "numpy", "# keep me"]


def test_prepare_verification_runtime_installs_normalized_requirements(monkeypatch, tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "requirements.txt").write_text("PIL\nnumpy\n", encoding="utf-8")
    observed_install_path = None

    def _fake_run(cmd, **kwargs):
        nonlocal observed_install_path
        if cmd[:3] == ["python3", "-m", "venv"]:
            env_dir = Path(cmd[-1])
            python_path = env_dir / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "install" in cmd and "-r" in cmd:
            observed_install_path = Path(cmd[-1])
            assert observed_install_path.exists()
            assert observed_install_path.read_text(encoding="utf-8") == "Pillow\nnumpy\n"
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("paperbot.repro.verification_runtime.subprocess.run", _fake_run)
    runtime = prepare_verification_runtime(
        output_dir,
        base_python="python3",
        prepare_requirements=True,
        cache_dir=tmp_path / "cache",
    )

    assert runtime.prepared is True
    assert observed_install_path is not None
    assert observed_install_path.name == ".paperbot_requirements.txt"
