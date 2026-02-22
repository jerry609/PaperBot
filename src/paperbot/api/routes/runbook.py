"""
Runbook API Routes - File Management

Provides file management endpoints for DeepCode Studio:
- List/read/write files in project directories
- Snapshot creation and comparison
- File diff and revert functionality
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from paperbot.application.collaboration.message_schema import new_run_id
from paperbot.infrastructure.stores.models import AgentRunModel, ArtifactModel, Base
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider
from paperbot.repro.e2b_executor import E2BExecutor
from paperbot.repro.execution_result import ExecutionResult
from paperbot.repro.openai_ci_executor import OpenAICodeInterpreterExecutor

router = APIRouter()

ExecutorType = Literal["docker", "e2b", "openai_ci"]

_provider = SessionProvider()
Base.metadata.create_all(_provider.engine)


def _allowed_workdir(workdir: Path) -> bool:
    allowed_prefixes = [Path(tempfile.gettempdir()).resolve()]
    # macOS: /tmp -> /private/tmp, but tempfile.gettempdir() returns /var/folders/...
    # Add common temp paths for cross-platform compatibility
    for tmp_path in ["/tmp", "/private/tmp"]:
        try:
            resolved_tmp = Path(tmp_path).resolve()
            if resolved_tmp not in allowed_prefixes:
                allowed_prefixes.append(resolved_tmp)
        except Exception:
            pass
    extra = os.getenv("PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES", "").strip()
    if extra:
        for p in extra.split(","):
            p = p.strip()
            if p:
                allowed_prefixes.append(Path(p).expanduser().resolve())

    try:
        resolved = workdir.resolve()
    except Exception:
        return False

    for prefix in allowed_prefixes:
        try:
            if resolved == prefix or str(resolved).startswith(str(prefix) + os.sep):
                return True
        except Exception:
            continue

    return False


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    """
    Resolve a relative path within a root directory.
    Rejects path traversal.
    """
    target = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if not (target == root_resolved or str(target).startswith(str(root_resolved) + os.sep)):
        raise HTTPException(status_code=400, detail="invalid path")
    return target


class SmokeRequest(BaseModel):
    project_dir: str = Field(..., description="Project directory on the API host (typically /tmp/... from gen-code)")
    executor: ExecutorType = Field("openai_ci", description="Execution backend: docker, e2b, or openai_ci")
    allow_network: bool = Field(False, description="Allow network (pip install, downloads). Docker disables network by default.")
    timeout_sec: int = Field(300, ge=10, le=3600)
    docker_image: str = Field("python:3.10-slim", description="Docker image to use when executor=docker")
    model: Optional[str] = Field(None, description="OpenAI model for Code Interpreter (executor=openai_ci)")


class StepRequest(BaseModel):
    """Base request for all runbook steps."""
    project_dir: str = Field(..., description="Project directory on the API host")
    executor: ExecutorType = Field("openai_ci", description="Execution backend: docker, e2b, or openai_ci")
    allow_network: bool = Field(True, description="Allow network access")
    timeout_sec: int = Field(600, ge=10, le=7200)
    docker_image: str = Field("python:3.10-slim", description="Docker image for executor=docker")
    command_override: Optional[str] = Field(None, description="Override auto-detected command")
    model: Optional[str] = Field(None, description="OpenAI model for Code Interpreter (executor=openai_ci)")


class InstallRequest(StepRequest):
    """Request for install step."""
    pip_cache: bool = Field(True, description="Use pip cache directory")


class DataRequest(StepRequest):
    """Request for data preparation step."""
    data_cmd: Optional[str] = Field(None, description="Custom data command (alternative to command_override)")


class TrainRequest(StepRequest):
    """Request for training step."""
    mini_mode: bool = Field(True, description="Run in mini mode with limited epochs/samples")
    max_epochs: int = Field(2, ge=1, le=100, description="Max epochs for mini mode")
    max_samples: int = Field(100, ge=1, le=100000, description="Max samples for mini mode")


class EvalRequest(StepRequest):
    """Request for evaluation step."""
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")


class ReportRequest(BaseModel):
    """Request for report generation (no sandbox needed)."""
    project_dir: str = Field(..., description="Project directory")
    run_id: Optional[str] = Field(None, description="Run ID to include in report")
    output_format: str = Field("html", description="Output format: html or json")


class SmokeStartResponse(BaseModel):
    run_id: str
    status: str = "running"


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    exit_code: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None


@router.post("/runbook/smoke", response_model=SmokeStartResponse)
async def start_smoke(body: SmokeRequest) -> SmokeStartResponse:
    workdir = Path(body.project_dir)
    if not workdir.exists() or not workdir.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory on the API host")
    if not _allowed_workdir(workdir):
        raise HTTPException(
            status_code=403,
            detail="project_dir is not allowed; set PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES to allow additional roots",
        )

    run_id = new_run_id()
    started_at = datetime.now(timezone.utc).isoformat()
    commands = _build_smoke_commands(allow_network=(body.allow_network if body.executor == "docker" else body.allow_network))

    # Persist run + step records (long-term source of truth).
    with _provider.session() as session:
        run = AgentRunModel(
            run_id=run_id,
            workflow="runbook",
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            status="running",
            executor_type=body.executor,
            timeout_seconds=body.timeout_sec,
            paper_url=None,
            paper_id=None,
            metadata_json=json.dumps(
                {
                    "kind": "smoke",
                    "project_dir": str(workdir),
                    "allow_network": body.allow_network,
                    "docker_image": body.docker_image,
                    "model": body.model,
                },
                ensure_ascii=False,
            ),
        )
        session.merge(run)

        step = RunbookStepModel(
            run_id=run_id,
            step_name="smoke",
            status="running",
            executor_type=body.executor,
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            command="\n".join(commands),
            exit_code=None,
            error=None,
            metadata_json=json.dumps({"allow_network": body.allow_network}, ensure_ascii=False),
        )
        session.add(step)
        session.commit()

    logger = get_execution_logger()
    logger.start_run(run_id)
    logger.log(run_id, "info", f"Smoke started (executor={body.executor}, allow_network={body.allow_network})", source="system")
    logger.log(run_id, "info", f"Project dir: {str(workdir)}", source="system")

    monitor = get_resource_monitor()
    monitor.start_run(run_id, timeout_seconds=float(body.timeout_sec))

    asyncio.create_task(_run_smoke_async(run_id, workdir, body))
    return SmokeStartResponse(run_id=run_id, status="running")


@router.get("/runbook/detect-commands")
async def detect_commands(project_dir: str = Query(..., description="Project directory to analyze")) -> Dict[str, Any]:
    """Auto-detect commands for each step by analyzing project structure."""
    workdir = Path(project_dir)
    if not workdir.exists() or not workdir.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(workdir):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    return _detect_project_commands(workdir)


@router.get("/runbook/executor-status")
async def get_executor_status() -> Dict[str, Any]:
    """Check availability of each executor backend."""
    result: Dict[str, Any] = {
        "docker": {"available": False, "error": None},
        "e2b": {"available": False, "error": None},
        "openai_ci": {"available": False, "error": None},
    }

    # Check Docker
    try:
        import docker  # type: ignore
        client = docker.from_env()
        client.ping()
        result["docker"]["available"] = True
        client.close()
    except ImportError:
        result["docker"]["error"] = "Docker SDK not installed (pip install docker)"
    except Exception as e:
        result["docker"]["error"] = f"Docker not running: {str(e)[:100]}"

    # Check E2B
    api_key = _load_e2b_api_key()
    if api_key:
        try:
            from e2b_code_interpreter import Sandbox
            result["e2b"]["available"] = True
        except ImportError:
            result["e2b"]["error"] = "E2B SDK not installed (pip install e2b-code-interpreter)"
    else:
        result["e2b"]["error"] = "E2B_API_KEY not set"

    # Check OpenAI Code Interpreter
    openai_key = os.getenv("OPENAI_API_KEY")

    # Also check model endpoints for OpenAI key
    if not openai_key or openai_key.startswith("sk-or-"):
        try:
            from paperbot.infrastructure.stores.model_endpoint_store import ModelEndpointStore
            store = ModelEndpointStore(auto_create_schema=False)
            endpoints = store.list_endpoints(enabled_only=True, include_secrets=True)
            for ep in endpoints:
                if ep.get("vendor") == "openai" and ep.get("api_key"):
                    key = ep["api_key"]
                    if key and not key.startswith("***") and not key.startswith("sk-or-"):
                        openai_key = key
                        break
            store.close()
        except Exception:
            pass

    if openai_key:
        if openai_key.startswith("sk-or-"):
            result["openai_ci"]["error"] = "OpenRouter key detected - need direct OpenAI key for Code Interpreter"
        else:
            try:
                from openai import OpenAI
                result["openai_ci"]["available"] = True
            except ImportError:
                result["openai_ci"]["error"] = "OpenAI SDK not installed (pip install openai)"
    else:
        result["openai_ci"]["error"] = "OPENAI_API_KEY not set (configure in Settings > Model Endpoints)"

    return result


async def _start_step(
    step_name: str,
    body: StepRequest,
    commands: List[str],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> SmokeStartResponse:
    """Generic step starter - creates run/step records and launches async execution."""
    workdir = Path(body.project_dir)
    if not workdir.exists() or not workdir.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory on the API host")
    if not _allowed_workdir(workdir):
        raise HTTPException(
            status_code=403,
            detail="project_dir is not allowed; set PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES to allow additional roots",
        )

    run_id = new_run_id()
    metadata = {
        "kind": step_name,
        "project_dir": str(workdir),
        "allow_network": body.allow_network,
        "docker_image": body.docker_image,
        "model": body.model,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with _provider.session() as session:
        run = AgentRunModel(
            run_id=run_id,
            workflow="runbook",
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            status="running",
            executor_type=body.executor,
            timeout_seconds=body.timeout_sec,
            paper_url=None,
            paper_id=None,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        session.merge(run)

        step = RunbookStepModel(
            run_id=run_id,
            step_name=step_name,
            status="running",
            executor_type=body.executor,
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            command="\n".join(commands),
            exit_code=None,
            error=None,
            metadata_json=json.dumps({"allow_network": body.allow_network}, ensure_ascii=False),
        )
        session.add(step)
        session.commit()

    logger = get_execution_logger()
    logger.start_run(run_id)
    logger.log(run_id, "info", f"{step_name.title()} started (executor={body.executor}, allow_network={body.allow_network})", source="system")
    logger.log(run_id, "info", f"Project dir: {str(workdir)}", source="system")

    monitor = get_resource_monitor()
    monitor.start_run(run_id, timeout_seconds=float(body.timeout_sec))

    asyncio.create_task(_run_step_async(run_id, step_name, workdir, commands, body))
    return SmokeStartResponse(run_id=run_id, status="running")


@router.post("/runbook/install", response_model=SmokeStartResponse)
async def start_install(body: InstallRequest) -> SmokeStartResponse:
    """Start install step - installs dependencies."""
    workdir = Path(body.project_dir)
    detected = _detect_project_commands(workdir).get("install", {})
    commands = _build_install_commands(body, detected)
    return await _start_step("install", body, commands, {"pip_cache": body.pip_cache})


@router.post("/runbook/data", response_model=SmokeStartResponse)
async def start_data(body: DataRequest) -> SmokeStartResponse:
    """Start data step - prepares dataset."""
    workdir = Path(body.project_dir)
    detected = _detect_project_commands(workdir).get("data", {})
    commands = _build_data_commands(body, detected)
    return await _start_step("data", body, commands, {"data_cmd": body.data_cmd})


@router.post("/runbook/train", response_model=SmokeStartResponse)
async def start_train(body: TrainRequest) -> SmokeStartResponse:
    """Start train step - runs training (optionally in mini mode)."""
    workdir = Path(body.project_dir)
    detected = _detect_project_commands(workdir).get("train", {})
    commands = _build_train_commands(body, detected)
    return await _start_step(
        "train",
        body,
        commands,
        {"mini_mode": body.mini_mode, "max_epochs": body.max_epochs, "max_samples": body.max_samples},
    )


@router.post("/runbook/eval", response_model=SmokeStartResponse)
async def start_eval(body: EvalRequest) -> SmokeStartResponse:
    """Start eval step - runs evaluation."""
    workdir = Path(body.project_dir)
    detected = _detect_project_commands(workdir).get("eval", {})
    commands = _build_eval_commands(body, detected)
    return await _start_step("eval", body, commands, {"checkpoint_path": body.checkpoint_path})


@router.post("/runbook/report")
async def generate_report(body: ReportRequest) -> Dict[str, Any]:
    """Generate execution report - runs locally (no sandbox)."""
    workdir = Path(body.project_dir)
    if not workdir.exists() or not workdir.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(workdir):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    run_id = body.run_id or new_run_id()
    report_dir = workdir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Collect execution info from the project
    report_data: Dict[str, Any] = {
        "run_id": run_id,
        "project_dir": str(workdir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "logs": [],
    }

    # List Python files
    for py_file in workdir.rglob("*.py"):
        try:
            rel_path = py_file.relative_to(workdir)
            report_data["files"].append(str(rel_path))
        except ValueError:
            pass

    # Check for any log files
    for log_file in workdir.rglob("*.log"):
        try:
            rel_path = log_file.relative_to(workdir)
            report_data["logs"].append(str(rel_path))
        except ValueError:
            pass

    if body.output_format == "html":
        # Generate simple HTML report
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Execution Report - {run_id}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }}
        .file-list {{ font-family: monospace; font-size: 14px; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Execution Report</h1>
    <p class="timestamp">Generated: {report_data['generated_at']}</p>
    <p>Run ID: <code>{run_id}</code></p>
    <p>Project: <code>{workdir}</code></p>

    <div class="section">
        <h2>Files ({len(report_data['files'])})</h2>
        <div class="file-list">
            {'<br>'.join(report_data['files'][:50])}
            {'<br>...' + str(len(report_data['files']) - 50) + ' more' if len(report_data['files']) > 50 else ''}
        </div>
    </div>

    <div class="section">
        <h2>Log Files ({len(report_data['logs'])})</h2>
        <div class="file-list">
            {'<br>'.join(report_data['logs']) if report_data['logs'] else 'No log files found'}
        </div>
    </div>
</body>
</html>"""
        report_path = report_dir / f"report_{run_id}.html"
        report_path.write_text(html_content, encoding="utf-8")
        return {
            "ok": True,
            "run_id": run_id,
            "format": "html",
            "path": str(report_path),
            "file_count": len(report_data["files"]),
        }
    else:
        # JSON format
        report_path = report_dir / f"report_{run_id}.json"
        report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "ok": True,
            "run_id": run_id,
            "format": "json",
            "path": str(report_path),
            "file_count": len(report_data["files"]),
        }


def _snapshot_root() -> Path:
    root = Path(os.getenv("PAPERBOT_RUNBOOK_SNAPSHOT_DIR", "data/runbook_snapshots"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


class CreateSnapshotRequest(BaseModel):
    project_dir: str
    label: str = ""
    max_total_bytes: int = Field(5_000_000, ge=100_000, le=50_000_000)
    max_file_bytes: int = Field(200_000, ge=1_000, le=10_000_000)


@router.post("/runbook/snapshots")
async def create_snapshot(body: CreateSnapshotRequest):
    """
    Create a text snapshot of a project directory for diff/revert.

    Snapshot is stored on disk (data/runbook_snapshots) and indexed as an ArtifactModel (type=snapshot).
    """
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    allowed_ext = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg"}

    files: Dict[str, Dict[str, Any]] = {}
    skipped_large: List[str] = []
    skipped_binary: List[str] = []
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            p = Path(dirpath) / name
            rel = os.path.relpath(str(p), str(root))
            ext = p.suffix.lower()
            if ext and ext not in allowed_ext:
                continue
            try:
                size = p.stat().st_size
            except Exception:
                continue
            if size > body.max_file_bytes:
                skipped_large.append(rel)
                continue
            if total_bytes + size > body.max_total_bytes:
                break
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                try:
                    content = p.read_text(errors="ignore")
                except Exception:
                    skipped_binary.append(rel)
                    continue
            total_bytes += len(content.encode("utf-8", errors="ignore"))
            files[rel] = {"sha256": _sha256_text(content), "size_bytes": size, "content": content}
        if total_bytes >= body.max_total_bytes:
            break

    created_at = datetime.now(timezone.utc)
    snapshot_payload = {
        "version": 1,
        "project_dir": str(root.resolve()),
        "label": body.label,
        "created_at": created_at.isoformat(),
        "files": files,
        "skipped": {
            "too_large": sorted(skipped_large),
            "binary_or_unreadable": sorted(skipped_binary),
        },
        "limits": {
            "max_total_bytes": body.max_total_bytes,
            "max_file_bytes": body.max_file_bytes,
        },
    }

    run_id = new_run_id()
    with _provider.session() as session:
        run = AgentRunModel(
            run_id=run_id,
            workflow="runbook",
            started_at=created_at,
            ended_at=created_at,
            status="completed",
            executor_type=None,
            timeout_seconds=None,
            paper_url=None,
            paper_id=None,
            metadata_json=json.dumps({"kind": "snapshot", "project_dir": str(root), "label": body.label}, ensure_ascii=False),
        )
        session.merge(run)
        session.commit()

        snapshot_path = _snapshot_root() / f"{run_id}.json"
        snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False), encoding="utf-8")

        artifact = ArtifactModel(
            run_id=run_id,
            step_id=None,
            type="snapshot",
            path_or_uri=str(snapshot_path),
            mime="application/json",
            size_bytes=snapshot_path.stat().st_size if snapshot_path.exists() else None,
            sha256=None,
            created_at=created_at,
            metadata_json=json.dumps(
                {"project_dir": str(root.resolve()), "label": body.label, "file_count": len(files), "total_bytes": total_bytes},
                ensure_ascii=False,
            ),
        )
        session.add(artifact)
        session.commit()

        return {
            "snapshot_id": artifact.id,
            "run_id": run_id,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "skipped": snapshot_payload["skipped"],
        }


def _load_snapshot(snapshot_id: int) -> Dict[str, Any]:
    with _provider.session() as session:
        artifact = session.get(ArtifactModel, snapshot_id)
        if artifact is None or artifact.type != "snapshot":
            raise HTTPException(status_code=404, detail="snapshot not found")
        path = Path(artifact.path_or_uri)
        snap_root = _snapshot_root()
        try:
            resolved = path.resolve()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid snapshot path")
        if not (resolved == snap_root or str(resolved).startswith(str(snap_root) + os.sep)):
            raise HTTPException(status_code=400, detail="snapshot path not allowed")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="snapshot file missing")
        try:
            return json.loads(resolved.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(status_code=500, detail="failed to read snapshot")


@router.get("/runbook/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: int):
    payload = _load_snapshot(snapshot_id)
    return {
        "snapshot_id": snapshot_id,
        "project_dir": payload.get("project_dir"),
        "label": payload.get("label"),
        "created_at": payload.get("created_at"),
        "file_count": len((payload.get("files") or {}).keys()),
        "files": sorted((payload.get("files") or {}).keys()),
        "skipped": payload.get("skipped") or {},
    }


@router.get("/runbook/diff")
async def diff_file(
    snapshot_id: int = Query(...),
    project_dir: str = Query(...),
    path: str = Query(..., description="Relative file path within project_dir"),
):
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(snapshot_id)
    files = snapshot.get("files") or {}
    if path not in files:
        raise HTTPException(status_code=404, detail="file not found in snapshot")
    old_content = files[path].get("content", "")

    target = _resolve_under_root(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found on disk")

    try:
        new_content = target.read_text(encoding="utf-8")
    except Exception:
        new_content = target.read_text(errors="ignore")

    return {
        "snapshot_id": snapshot_id,
        "path": path,
        "old": old_content,
        "new": new_content,
    }


class RevertFileRequest(BaseModel):
    snapshot_id: int
    project_dir: str
    path: str


@router.post("/runbook/revert")
async def revert_file(body: RevertFileRequest):
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(body.snapshot_id)
    files = snapshot.get("files") or {}
    if body.path not in files:
        raise HTTPException(status_code=404, detail="file not found in snapshot")
    old_content = files[body.path].get("content", "")

    target = _resolve_under_root(root, body.path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(old_content, encoding="utf-8")
    return {"ok": True, "path": body.path}


# ──────────────────────────────────────────────────────────────────────────────
# File Operations
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/runbook/files")
async def list_project_files(
    project_dir: str = Query(..., description="Project directory on the API host"),
    recursive: bool = Query(True, description="List files recursively"),
    max_files: int = Query(2000, ge=1, le=20000),
):
    """
    List files under a project directory (best-effort).

    Notes:
    - This endpoint is intentionally restrictive and will only serve allowed roots.
    - Large directories (e.g. node_modules) are skipped by default.
    """
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}

    files: List[str] = []
    directories: List[str] = []

    if not recursive:
        for p in root.iterdir():
            if p.is_dir():
                directories.append(p.name)
            elif p.is_file():
                files.append(p.name)
        return {"project_dir": str(root), "files": sorted(files), "directories": sorted(directories)}

    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir != ".":
            directories.append(rel_dir)

        for name in filenames:
            if len(files) >= max_files:
                break
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            files.append(rel)
        if len(files) >= max_files:
            break

    return {
        "project_dir": str(root),
        "files": sorted(files),
        "directories": sorted(set(directories)),
        "truncated": len(files) >= max_files,
        "max_files": max_files,
    }


class ReadFileResponse(BaseModel):
    path: str
    content: str


@router.get("/runbook/file", response_model=ReadFileResponse)
async def read_project_file(
    project_dir: str = Query(..., description="Project directory on the API host"),
    path: str = Query(..., description="Relative file path within project_dir"),
    max_bytes: int = Query(2_000_000, ge=1, le=20_000_000),
):
    """Read a single file under project_dir (UTF-8 best effort)."""
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    target = _resolve_under_root(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    size = target.stat().st_size
    if size > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large ({size} bytes)")

    try:
        content = target.read_text(encoding="utf-8")
    except Exception:
        content = target.read_text(errors="ignore")

    return ReadFileResponse(path=path, content=content)


class WriteFileRequest(BaseModel):
    project_dir: str
    path: str
    content: str


@router.post("/runbook/file")
async def write_project_file(body: WriteFileRequest):
    """Write a file under project_dir (creates parents)."""
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    target = _resolve_under_root(root, body.path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")

    return {"ok": True, "path": body.path}


@router.get("/runbook/changes")
async def list_changes(
    snapshot_id: int = Query(...),
    project_dir: str = Query(...),
    max_files: int = Query(5000, ge=1, le=20000),
):
    """
    Compute file-level changes between a snapshot and the current project directory.

    Returns lists of changed/unchanged/added/removed files.
    """
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(snapshot_id)
    snap_files = snapshot.get("files") or {}
    snap_paths = set(snap_files.keys())

    ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    allowed_ext = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg"}

    changed: List[str] = []
    unchanged: List[str] = []
    removed: List[str] = []

    for rel in sorted(snap_paths):
        target = _resolve_under_root(root, rel)
        if not target.exists() or not target.is_file():
            removed.append(rel)
            continue
        try:
            content = target.read_text(encoding="utf-8")
        except Exception:
            content = target.read_text(errors="ignore")
        current_sha = _sha256_text(content)
        if current_sha != snap_files.get(rel, {}).get("sha256"):
            changed.append(rel)
        else:
            unchanged.append(rel)

    current_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            if len(current_files) >= max_files:
                break
            p = Path(dirpath) / name
            ext = p.suffix.lower()
            if ext and ext not in allowed_ext:
                continue
            rel = os.path.relpath(str(p), str(root))
            current_files.append(rel)
        if len(current_files) >= max_files:
            break

    added = sorted([p for p in current_files if p not in snap_paths])

    return {
        "snapshot_id": snapshot_id,
        "project_dir": str(root.resolve()),
        "changed": changed,
        "unchanged": unchanged,
        "added": added,
        "removed": removed,
    }


class DeleteFileRequest(BaseModel):
    project_dir: str
    path: str


@router.post("/runbook/delete")
async def delete_file(body: DeleteFileRequest):
    """Delete a single file under project_dir."""
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    target = _resolve_under_root(root, body.path)
    if not target.exists():
        return {"ok": True, "path": body.path, "deleted": False}
    if target.is_dir():
        raise HTTPException(status_code=400, detail="cannot delete a directory")

    target.unlink()
    return {"ok": True, "path": body.path, "deleted": True}


class RevertProjectRequest(BaseModel):
    snapshot_id: int
    project_dir: str
    delete_added: bool = True


@router.post("/runbook/revert-project")
async def revert_project(body: RevertProjectRequest):
    """Revert project files back to a snapshot (file-level)."""
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(body.snapshot_id)
    snap_files = snapshot.get("files") or {}
    snap_paths = set(snap_files.keys())

    restored = 0
    for rel, info in snap_files.items():
        target = _resolve_under_root(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(info.get("content", ""), encoding="utf-8")
        restored += 1

    deleted = 0
    if body.delete_added:
        ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
        allowed_ext = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg"}
        current_files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            for name in filenames:
                p = Path(dirpath) / name
                ext = p.suffix.lower()
                if ext and ext not in allowed_ext:
                    continue
                rel = os.path.relpath(str(p), str(root))
                current_files.append(rel)
        for rel in current_files:
            if rel in snap_paths:
                continue
            try:
                target = _resolve_under_root(root, rel)
                if target.exists() and target.is_file():
                    target.unlink()
                    deleted += 1
            except Exception:
                continue

    return {"ok": True, "restored": restored, "deleted": deleted}


class HunkPayload(BaseModel):
    before: str = ""
    after: str = ""
    old: str
    new: str


class RevertHunksRequest(BaseModel):
    snapshot_id: int
    project_dir: str
    path: str
    hunks: List[HunkPayload]


@router.post("/runbook/revert-hunks")
async def revert_hunks(body: RevertHunksRequest):
    """
    Revert selected hunks of a file back to the snapshot content.

    This operates purely on text lines. Each hunk includes:
    - before/after context (unchanged lines)
    - old/new core (changed region)
    """
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(body.snapshot_id)
    snap_files = snapshot.get("files") or {}
    if body.path not in snap_files:
        raise HTTPException(status_code=404, detail="file not found in snapshot")

    target = _resolve_under_root(root, body.path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found on disk")

    try:
        current_text = target.read_text(encoding="utf-8")
    except Exception:
        current_text = target.read_text(errors="ignore")

    current_lines = current_text.split("\n")

    def split_lines(s: str) -> List[str]:
        return [] if s == "" else s.split("\n")

    applied = 0
    failed: List[Dict[str, Any]] = []

    for i, hunk in enumerate(body.hunks):
        before = split_lines(hunk.before)
        after = split_lines(hunk.after)
        old = split_lines(hunk.old)
        new = split_lines(hunk.new)

        pattern = before + new + after
        replacement = before + old + after

        def find_once(pat: List[str]) -> Optional[int]:
            if not pat:
                return None
            hits: List[int] = []
            for start in range(0, len(current_lines) - len(pat) + 1):
                if current_lines[start : start + len(pat)] == pat:
                    hits.append(start)
                    if len(hits) > 1:
                        break
            if len(hits) == 1:
                return hits[0]
            return None

        start_idx = find_once(pattern)
        used_pattern = "context"
        if start_idx is None:
            # Fall back to matching core only if unique.
            start_idx = find_once(new)
            used_pattern = "core"

        if start_idx is None:
            failed.append({"index": i, "reason": "pattern_not_found_or_not_unique", "used": used_pattern})
            continue

        if used_pattern == "core":
            current_lines[start_idx : start_idx + len(new)] = old
        else:
            current_lines[start_idx : start_idx + len(pattern)] = replacement
        applied += 1

    new_text = "\n".join(current_lines)
    target.write_text(new_text, encoding="utf-8")

    return {"ok": True, "applied": applied, "failed": failed}


@router.get("/runbook/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    with _provider.session() as session:
        run = session.get(AgentRunModel, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        steps = (
            session.query(RunbookStepModel)
            .filter(RunbookStepModel.run_id == run_id)
            .order_by(RunbookStepModel.id.asc())
            .all()
        )
        return RunStatusResponse(
            run_id=run.run_id,
            status=run.status,
            exit_code=run.get_metadata().get("exit_code"),
            error=run.get_metadata().get("error"),
            started_at=run.started_at.isoformat() if run.started_at else None,
            finished_at=run.ended_at.isoformat() if run.ended_at else None,
            steps=[
                {
                    "id": s.id,
                    "name": s.step_name,
                    "status": s.status,
                    "executor": s.executor_type,
                    "exit_code": s.exit_code,
                    "error": s.error,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                }
                for s in steps
            ],
        )


async def _run_smoke_async(run_id: str, workdir: Path, body: SmokeRequest) -> None:
    logger = get_execution_logger()
    monitor = get_resource_monitor()

    try:
        loop = asyncio.get_running_loop()
        if body.executor == "docker":
            result = await loop.run_in_executor(None, _run_docker_smoke_blocking, run_id, workdir, body)
        elif body.executor == "e2b":
            result = await loop.run_in_executor(None, _run_e2b_smoke_blocking, run_id, workdir, body)
        else:
            result = await loop.run_in_executor(None, _run_openai_ci_smoke_blocking, run_id, workdir, body)

        status = "success" if result.success else ("failed" if result.status != "error" else "error")

        with _provider.session() as session:
            run = session.get(AgentRunModel, run_id)
            if run is not None:
                run.status = status
                run.ended_at = datetime.now(timezone.utc)
                meta = run.get_metadata()
                meta["exit_code"] = result.exit_code
                if result.error:
                    meta["error"] = result.error
                run.set_metadata(meta)

            step = (
                session.query(RunbookStepModel)
                .filter(RunbookStepModel.run_id == run_id, RunbookStepModel.step_name == "smoke")
                .order_by(RunbookStepModel.id.desc())
                .first()
            )
            if step is not None:
                step.status = status
                step.ended_at = datetime.now(timezone.utc)
                step.exit_code = result.exit_code
                step.error = result.error
            session.commit()

        if result.success:
            logger.log(run_id, "info", f"Smoke completed successfully (exit_code={result.exit_code})", source="system")
        else:
            if result.error:
                logger.log(run_id, "error", f"Smoke failed: {result.error}", source="system")
            logger.log(run_id, "error", f"Smoke failed (exit_code={result.exit_code})", source="system")
    except Exception as e:
        with _provider.session() as session:
            run = session.get(AgentRunModel, run_id)
            if run is not None:
                run.status = "error"
                run.ended_at = datetime.now(timezone.utc)
                meta = run.get_metadata()
                meta["error"] = str(e)
                run.set_metadata(meta)
            step = (
                session.query(RunbookStepModel)
                .filter(RunbookStepModel.run_id == run_id, RunbookStepModel.step_name == "smoke")
                .order_by(RunbookStepModel.id.desc())
                .first()
            )
            if step is not None:
                step.status = "error"
                step.ended_at = datetime.now(timezone.utc)
                step.exit_code = 1
                step.error = str(e)
            session.commit()
        logger.log(run_id, "error", f"Smoke runner crashed: {e}", source="system")
    finally:
        try:
            with _provider.session() as session:
                run = session.get(AgentRunModel, run_id)
                status = run.status if run is not None else "completed"
            monitor.stop_run(run_id, status=status)
        except Exception:
            pass
        try:
            logger.stop_run(run_id)
        except Exception:
            pass


async def _run_step_async(
    run_id: str, step_name: str, workdir: Path, commands: List[str], body: StepRequest
) -> None:
    """Generic async step runner - follows _run_smoke_async pattern but parameterized by step_name."""
    logger = get_execution_logger()
    monitor = get_resource_monitor()

    try:
        loop = asyncio.get_running_loop()
        if body.executor == "docker":
            result = await loop.run_in_executor(
                None, _run_docker_step_blocking, run_id, step_name, workdir, commands, body
            )
        elif body.executor == "e2b":
            result = await loop.run_in_executor(
                None, _run_e2b_step_blocking, run_id, step_name, workdir, commands, body
            )
        else:
            result = await loop.run_in_executor(
                None, _run_openai_ci_step_blocking, run_id, step_name, workdir, commands, body
            )

        status = "success" if result.success else ("failed" if result.status != "error" else "error")

        with _provider.session() as session:
            run = session.get(AgentRunModel, run_id)
            if run is not None:
                run.status = status
                run.ended_at = datetime.now(timezone.utc)
                meta = run.get_metadata()
                meta["exit_code"] = result.exit_code
                if result.error:
                    meta["error"] = result.error
                run.set_metadata(meta)

            step = (
                session.query(RunbookStepModel)
                .filter(RunbookStepModel.run_id == run_id, RunbookStepModel.step_name == step_name)
                .order_by(RunbookStepModel.id.desc())
                .first()
            )
            if step is not None:
                step.status = status
                step.ended_at = datetime.now(timezone.utc)
                step.exit_code = result.exit_code
                step.error = result.error
            session.commit()

        if result.success:
            logger.log(run_id, "info", f"{step_name.title()} completed successfully (exit_code={result.exit_code})", source="system")
        else:
            if result.error:
                logger.log(run_id, "error", f"{step_name.title()} failed: {result.error}", source="system")
            logger.log(run_id, "error", f"{step_name.title()} failed (exit_code={result.exit_code})", source="system")
    except Exception as e:
        with _provider.session() as session:
            run = session.get(AgentRunModel, run_id)
            if run is not None:
                run.status = "error"
                run.ended_at = datetime.now(timezone.utc)
                meta = run.get_metadata()
                meta["error"] = str(e)
                run.set_metadata(meta)
            step = (
                session.query(RunbookStepModel)
                .filter(RunbookStepModel.run_id == run_id, RunbookStepModel.step_name == step_name)
                .order_by(RunbookStepModel.id.desc())
                .first()
            )
            if step is not None:
                step.status = "error"
                step.ended_at = datetime.now(timezone.utc)
                step.exit_code = 1
                step.error = str(e)
            session.commit()
        logger.log(run_id, "error", f"{step_name.title()} runner crashed: {e}", source="system")
    finally:
        try:
            with _provider.session() as session:
                run = session.get(AgentRunModel, run_id)
                status = run.status if run is not None else "completed"
            monitor.stop_run(run_id, status=status)
        except Exception:
            pass
        try:
            logger.stop_run(run_id)
        except Exception:
            pass


def _build_smoke_commands(*, allow_network: bool) -> List[str]:
    commands: List[str] = []
    commands.append("python -V")
    commands.append("python -m pip --version || true")
    if allow_network:
        commands.append("test -f requirements.txt && python -m pip install -r requirements.txt || true")
    commands.append("python -m compileall -q .")
    return commands


def _detect_project_commands(workdir: Path) -> Dict[str, Dict[str, Any]]:
    """Analyze project structure to suggest commands for each step."""
    results: Dict[str, Dict[str, Any]] = {}

    # Install: requirements.txt, setup.py, pyproject.toml
    if (workdir / "requirements.txt").exists():
        results["install"] = {"detected": True, "command": "pip install -r requirements.txt"}
    elif (workdir / "setup.py").exists():
        results["install"] = {"detected": True, "command": "pip install -e ."}
    elif (workdir / "pyproject.toml").exists():
        results["install"] = {"detected": True, "command": "pip install -e ."}
    else:
        results["install"] = {"detected": False, "command": None}

    # Data: download_data.py, prepare_data.py, get_data.py
    data_patterns = ["download_data.py", "prepare_data.py", "get_data.py", "data.py"]
    for pattern in data_patterns:
        matches = list(workdir.rglob(pattern))
        if matches:
            rel_path = matches[0].relative_to(workdir)
            results["data"] = {"detected": True, "command": f"python {rel_path}"}
            break
    else:
        results["data"] = {"detected": False, "command": None}

    # Train: train.py, main.py
    if (workdir / "train.py").exists():
        results["train"] = {"detected": True, "command": "python train.py"}
    elif (workdir / "main.py").exists():
        results["train"] = {"detected": True, "command": "python main.py train"}
    elif (workdir / "run.py").exists():
        results["train"] = {"detected": True, "command": "python run.py"}
    else:
        results["train"] = {"detected": False, "command": None}

    # Eval: eval.py, evaluate.py, test.py
    eval_names = ["eval.py", "evaluate.py", "test.py"]
    for name in eval_names:
        if (workdir / name).exists():
            results["eval"] = {"detected": True, "command": f"python {name}"}
            break
    else:
        results["eval"] = {"detected": False, "command": None}

    return results


def _build_install_commands(body: InstallRequest, detected: Dict[str, Any]) -> List[str]:
    """Build commands for install step."""
    commands = ["python -V", "pip -V"]
    if body.command_override:
        commands.append(body.command_override)
    elif detected.get("command"):
        commands.append(detected["command"])
    else:
        commands.append("test -f requirements.txt && pip install -r requirements.txt || true")
    return commands


def _build_data_commands(body: DataRequest, detected: Dict[str, Any]) -> List[str]:
    """Build commands for data preparation step."""
    # Install dependencies first (each step runs in fresh container)
    commands = ["test -f requirements.txt && pip install -q -r requirements.txt || true"]
    if body.command_override:
        commands.append(body.command_override)
    elif body.data_cmd:
        commands.append(body.data_cmd)
    elif detected.get("command"):
        commands.append(detected["command"])
    else:
        commands.append("echo 'No data command detected'")
    return commands


def _build_train_commands(body: TrainRequest, detected: Dict[str, Any]) -> List[str]:
    """Build commands for training step."""
    # Install dependencies first (each step runs in fresh container)
    commands = ["test -f requirements.txt && pip install -q -r requirements.txt || true"]
    base = body.command_override or detected.get("command") or "python train.py"
    if body.mini_mode:
        # Try with mini mode flags, fall back to base command if flags not supported
        commands.append(f"{base} --max_epochs {body.max_epochs} --max_samples {body.max_samples} || {base}")
    else:
        commands.append(base)
    return commands


def _build_eval_commands(body: EvalRequest, detected: Dict[str, Any]) -> List[str]:
    """Build commands for evaluation step."""
    # Install dependencies first (each step runs in fresh container)
    commands = ["test -f requirements.txt && pip install -q -r requirements.txt || true"]
    base = body.command_override or detected.get("command") or "python eval.py"
    if body.checkpoint_path:
        commands.append(f"{base} --checkpoint {body.checkpoint_path}")
    else:
        commands.append(base)
    return commands


def _load_e2b_api_key() -> Optional[str]:
    """Load E2B API key from environment or saved settings."""
    # Check environment first
    key = os.getenv("E2B_API_KEY")
    if key:
        return key

    # Check saved settings file
    settings_file = Path.home() / ".paperbot" / "sandbox_settings.json"
    if settings_file.exists():
        try:
            import json
            settings = json.loads(settings_file.read_text())
            key = settings.get("e2b_api_key")
            if key:
                # Also set in environment for this session
                os.environ["E2B_API_KEY"] = key
                return key
        except Exception:
            pass
    return None


def _resolve_ci_model(requested: Optional[str]) -> str:
    if requested:
        return requested
    return os.getenv("PAPERBOT_CI_MODEL") or "gpt-4.1"


def _run_e2b_smoke_blocking(run_id: str, workdir: Path, body: SmokeRequest) -> ExecutionResult:
    logger = get_execution_logger()

    # Load API key from settings if not in environment
    api_key = _load_e2b_api_key()

    executor = E2BExecutor(api_key=api_key)
    if not executor.available():
        return ExecutionResult(status="error", exit_code=1, error="E2B not available (missing SDK or E2B_API_KEY)")

    # Use a single shell invocation so compound commands work reliably.
    commands = _build_smoke_commands(allow_network=body.allow_network)
    import shlex
    script = "set -e\n" + "\n".join(commands) + "\n"
    result = executor.run(workdir=workdir, commands=[f"/bin/sh -lc {shlex.quote(script)}"], timeout_sec=body.timeout_sec)
    if result.logs:
        for line in result.logs.splitlines():
            logger.log(run_id, "info", line, source="stdout")
    return result


def _run_openai_ci_smoke_blocking(run_id: str, workdir: Path, body: SmokeRequest) -> ExecutionResult:
    logger = get_execution_logger()
    if body.allow_network:
        logger.log(run_id, "warning", "Code Interpreter ignores allow_network; network is restricted.", source="system")

    model = _resolve_ci_model(body.model)
    executor = OpenAICodeInterpreterExecutor(model=model)
    commands = _build_smoke_commands(allow_network=False)
    return executor.run(workdir=workdir, commands=commands, run_id=run_id, logger=logger, timeout_sec=body.timeout_sec)


def _run_docker_smoke_blocking(run_id: str, workdir: Path, body: SmokeRequest) -> ExecutionResult:
    logger = get_execution_logger()

    try:
        import docker  # type: ignore
        from docker.errors import DockerException, APIError  # type: ignore
    except Exception:
        return ExecutionResult(status="error", exit_code=1, error="Docker SDK not installed. Install `docker` python package.")

    client = None
    try:
        client = docker.from_env()
    except DockerException as e:
        return ExecutionResult(status="error", exit_code=1, error=f"Docker not available: {e}")

    commands = _build_smoke_commands(allow_network=body.allow_network)
    script = "set -e\n" + "\n".join(commands) + "\n"
    command = ["/bin/sh", "-lc", script]

    start = time.time()
    container = None
    waiter_done = threading.Event()
    result: Dict[str, Any] = {"exit_code": 1, "timed_out": False, "error": None}

    cache_dir = Path(tempfile.gettempdir()) / "paperbot_cache" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        binds = {
            str(workdir): {"bind": "/workspace", "mode": "rw"},  # rw needed for compileall to write .pyc
            str(cache_dir): {"bind": "/cache", "mode": "rw"},
        }

        container = client.containers.run(
            body.docker_image,
            command=command,
            working_dir="/workspace",
            detach=True,
            network_disabled=not body.allow_network,
            volumes=binds,
            environment={"PIP_CACHE_DIR": "/cache/pip"},
        )

        def waiter() -> None:
            try:
                res = container.wait(timeout=body.timeout_sec)  # type: ignore[union-attr]
                result["exit_code"] = int(res.get("StatusCode", 1))
            except Exception as e:
                result["timed_out"] = True
                result["error"] = str(e)
                try:
                    container.kill()  # type: ignore[union-attr]
                except Exception:
                    pass
            finally:
                waiter_done.set()

        threading.Thread(target=waiter, daemon=True).start()

        try:
            for raw in container.logs(stream=True, follow=True):  # type: ignore[union-attr]
                line = raw.decode(errors="ignore").rstrip("\n")
                if line:
                    logger.log(run_id, "info", line, source="stdout")
                if waiter_done.is_set():
                    # container may still flush trailing logs; keep draining
                    continue
        except APIError as e:
            result["error"] = str(e)

        waiter_done.wait(timeout=5)
        duration = time.time() - start

        if result["timed_out"]:
            return ExecutionResult(
                status="failed",
                exit_code=int(result.get("exit_code") or 1),
                error=f"Timeout after {body.timeout_sec}s",
                duration_sec=duration,
            )

        exit_code = result.get("exit_code")
        if exit_code is None:
            exit_code = 1
        exit_code = int(exit_code)
        status = "success" if exit_code == 0 else "failed"
        return ExecutionResult(status=status, exit_code=exit_code, duration_sec=duration, error=result.get("error"))
    finally:
        if container is not None:
            try:
                container.remove(force=True)  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            if client is not None:
                client.close()
        except Exception:
            pass


def _run_e2b_step_blocking(
    run_id: str, step_name: str, workdir: Path, commands: List[str], body: StepRequest
) -> ExecutionResult:
    """Generic E2B step executor."""
    logger = get_execution_logger()

    api_key = _load_e2b_api_key()
    executor = E2BExecutor(api_key=api_key)
    if not executor.available():
        return ExecutionResult(status="error", exit_code=1, error="E2B not available (missing SDK or E2B_API_KEY)")

    # Build script with explicit cd to /home/user where files are uploaded
    script = "set -e\ncd /home/user\n" + "\n".join(commands) + "\n"
    # Use /bin/bash without -l to avoid login shell issues with 'source'
    result = executor.run(workdir=workdir, commands=[f"/bin/bash -c {shlex.quote(script)}"], timeout_sec=body.timeout_sec)
    if result.logs:
        for line in result.logs.splitlines():
            logger.log(run_id, "info", line, source="stdout")
    return result


def _run_openai_ci_step_blocking(
    run_id: str, step_name: str, workdir: Path, commands: List[str], body: StepRequest
) -> ExecutionResult:
    logger = get_execution_logger()
    if body.allow_network:
        logger.log(run_id, "warning", "Code Interpreter ignores allow_network; network is restricted.", source="system")

    model = _resolve_ci_model(body.model)
    executor = OpenAICodeInterpreterExecutor(model=model)
    return executor.run(workdir=workdir, commands=commands, run_id=run_id, logger=logger, timeout_sec=body.timeout_sec)


def _run_docker_step_blocking(
    run_id: str, step_name: str, workdir: Path, commands: List[str], body: StepRequest
) -> ExecutionResult:
    """Generic Docker step executor."""
    logger = get_execution_logger()

    try:
        import docker  # type: ignore
        from docker.errors import DockerException, APIError  # type: ignore
    except Exception:
        return ExecutionResult(status="error", exit_code=1, error="Docker SDK not installed. Install `docker` python package.")

    client = None
    try:
        client = docker.from_env()
    except DockerException as e:
        return ExecutionResult(status="error", exit_code=1, error=f"Docker not available: {e}")

    script = "set -e\n" + "\n".join(commands) + "\n"
    command = ["/bin/sh", "-lc", script]

    start = time.time()
    container = None
    waiter_done = threading.Event()
    result: Dict[str, Any] = {"exit_code": 1, "timed_out": False, "error": None}

    cache_dir = Path(tempfile.gettempdir()) / "paperbot_cache" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        binds = {
            str(workdir): {"bind": "/workspace", "mode": "rw"},  # rw for steps that write (train, eval)
            str(cache_dir): {"bind": "/cache", "mode": "rw"},
        }

        container = client.containers.run(
            body.docker_image,
            command=command,
            working_dir="/workspace",
            detach=True,
            network_disabled=not body.allow_network,
            volumes=binds,
            environment={"PIP_CACHE_DIR": "/cache/pip"},
        )

        def waiter() -> None:
            try:
                res = container.wait(timeout=body.timeout_sec)  # type: ignore[union-attr]
                result["exit_code"] = int(res.get("StatusCode", 1))
            except Exception as e:
                result["timed_out"] = True
                result["error"] = str(e)
                try:
                    container.kill()  # type: ignore[union-attr]
                except Exception:
                    pass
            finally:
                waiter_done.set()

        threading.Thread(target=waiter, daemon=True).start()

        try:
            for raw in container.logs(stream=True, follow=True):  # type: ignore[union-attr]
                line = raw.decode(errors="ignore").rstrip("\n")
                if line:
                    logger.log(run_id, "info", line, source="stdout")
                if waiter_done.is_set():
                    continue
        except APIError as e:
            result["error"] = str(e)

        waiter_done.wait(timeout=5)
        duration = time.time() - start

        if result["timed_out"]:
            return ExecutionResult(
                status="failed",
                exit_code=int(result.get("exit_code") or 1),
                error=f"Timeout after {body.timeout_sec}s",
                duration_sec=duration,
            )

        exit_code = result.get("exit_code")
        if exit_code is None:
            exit_code = 1
        exit_code = int(exit_code)
        status = "success" if exit_code == 0 else "failed"
        return ExecutionResult(status=status, exit_code=exit_code, duration_sec=duration, error=result.get("error"))
    finally:
        if container is not None:
            try:
                container.remove(force=True)  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            if client is not None:
                client.close()
        except Exception:
            pass
