"""
Runbook API Routes

Minimal endpoints to support DeepCode Studio Runbook steps:
- Start a smoke run (docker/e2b)
- Query run status

Logs are emitted via ExecutionLogger and can be streamed from:
  GET /api/sandbox/runs/{run_id}/logs/stream
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from paperbot.application.collaboration.message_schema import new_run_id
from paperbot.infrastructure.logging.execution_logger import get_execution_logger
from paperbot.infrastructure.monitoring.resource_monitor import get_resource_monitor
from paperbot.infrastructure.stores.models import AgentRunModel, Base, RunbookStepModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider
from paperbot.repro.e2b_executor import E2BExecutor
from paperbot.repro.execution_result import ExecutionResult

router = APIRouter()

ExecutorType = Literal["docker", "e2b"]

_provider = SessionProvider()
Base.metadata.create_all(_provider.engine)


def _allowed_workdir(workdir: Path) -> bool:
    allowed_prefixes = [Path(tempfile.gettempdir()).resolve()]
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


class SmokeRequest(BaseModel):
    project_dir: str = Field(..., description="Project directory on the API host (typically /tmp/... from gen-code)")
    executor: ExecutorType = Field("docker", description="Execution backend: docker (local) or e2b (remote)")
    allow_network: bool = Field(False, description="Allow network (pip install, downloads). Docker disables network by default.")
    timeout_sec: int = Field(300, ge=10, le=3600)
    docker_image: str = Field("python:3.10-slim", description="Docker image to use when executor=docker")


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

    target = (root / path).resolve()
    root_resolved = root.resolve()
    if not (target == root_resolved or str(target).startswith(str(root_resolved) + os.sep)):
        raise HTTPException(status_code=400, detail="invalid path")
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

    target = (root / body.path).resolve()
    root_resolved = root.resolve()
    if not (target == root_resolved or str(target).startswith(str(root_resolved) + os.sep)):
        raise HTTPException(status_code=400, detail="invalid path")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")

    return {"ok": True, "path": body.path}


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
        else:
            result = await loop.run_in_executor(None, _run_e2b_smoke_blocking, run_id, workdir, body)

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


def _build_smoke_commands(*, allow_network: bool) -> List[str]:
    commands: List[str] = []
    commands.append("python -V")
    commands.append("python -m pip --version || true")
    if allow_network:
        commands.append("test -f requirements.txt && python -m pip install -r requirements.txt || true")
    commands.append("python -m compileall -q .")
    return commands


def _run_e2b_smoke_blocking(run_id: str, workdir: Path, body: SmokeRequest) -> ExecutionResult:
    logger = get_execution_logger()

    executor = E2BExecutor()
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
            str(workdir): {"bind": "/workspace", "mode": "ro"},
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

        exit_code = int(result.get("exit_code") or 1)
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
