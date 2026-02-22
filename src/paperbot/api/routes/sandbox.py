"""
Sandbox Management API Routes

Provides endpoints for:
- Task queue management (view/cancel/retry)
- Run logs streaming
- Resource metrics streaming
- System status
- Execution backend settings (E2B, Docker)
"""

from __future__ import annotations

import os
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()

# Path for persisted settings (user-specific)
SETTINGS_FILE = Path.home() / ".paperbot" / "sandbox_settings.json"


# --- Request/Response Models ---


class CancelResponse(BaseModel):
    """Response for job cancellation"""

    status: str
    job_id: str
    message: str = ""


class RetryResponse(BaseModel):
    """Response for job retry"""

    status: str
    old_job_id: str
    new_job_id: Optional[str] = None
    message: str = ""


# --- Queue Management ---


@router.get("/sandbox/queue")
async def get_queue_status(
    http_request: Request,
    completed_limit: int = Query(20, ge=1, le=100, description="Max completed jobs to return"),
):
    """
    Get current queue status.

    Returns pending, running, and recently completed jobs.
    """
    try:
        from paperbot.infrastructure.queue.job_manager import JobManager

        manager = JobManager()
        await manager.connect()
        try:
            status = await manager.get_queue_status(completed_limit=completed_limit)
            return status.to_dict()
        finally:
            await manager.close()
    except ImportError:
        return {
            "error": "Redis/ARQ not configured",
            "pending": [],
            "running": [],
            "completed": [],
            "stats": {},
        }
    except Exception as e:
        return {"error": str(e), "pending": [], "running": [], "completed": [], "stats": {}}


@router.get("/sandbox/jobs/{job_id}")
async def get_job_info(job_id: str, http_request: Request):
    """Get information about a specific job."""
    try:
        from paperbot.infrastructure.queue.job_manager import JobManager

        manager = JobManager()
        await manager.connect()
        try:
            info = await manager.get_job_info(job_id)
            if info is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return info.to_dict()
        finally:
            await manager.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sandbox/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, http_request: Request) -> CancelResponse:
    """
    Cancel a pending job.

    Note: Running jobs cannot be cancelled.
    """
    try:
        from paperbot.infrastructure.queue.job_manager import JobManager

        manager = JobManager()
        await manager.connect()
        try:
            success = await manager.cancel_job(job_id)
            if success:
                return CancelResponse(
                    status="cancelled", job_id=job_id, message="Job cancelled successfully"
                )
            else:
                return CancelResponse(
                    status="failed",
                    job_id=job_id,
                    message="Cannot cancel job (may be running or completed)",
                )
        finally:
            await manager.close()
    except Exception as e:
        return CancelResponse(status="error", job_id=job_id, message=str(e))


@router.post("/sandbox/jobs/{job_id}/retry")
async def retry_job(job_id: str, http_request: Request) -> RetryResponse:
    """Retry a failed or completed job."""
    try:
        from paperbot.infrastructure.queue.job_manager import JobManager

        manager = JobManager()
        await manager.connect()
        try:
            new_job_id = await manager.retry_job(job_id)
            if new_job_id:
                return RetryResponse(
                    status="enqueued",
                    old_job_id=job_id,
                    new_job_id=new_job_id,
                    message="Job re-enqueued successfully",
                )
            else:
                return RetryResponse(
                    status="failed",
                    old_job_id=job_id,
                    message="Could not retry job (original job not found)",
                )
        finally:
            await manager.close()
    except Exception as e:
        return RetryResponse(status="error", old_job_id=job_id, message=str(e))


# --- Log Streaming ---


async def _log_stream_generator(run_id: str):
    """Generate SSE events for log streaming."""
    from paperbot.infrastructure.logging.execution_logger import get_execution_logger

    logger = get_execution_logger()

    async for entry in logger.stream_logs(run_id):
        yield StreamEvent(
            type="log",
            data=entry.to_dict(),
        )

    yield StreamEvent(type="done", message="Log stream ended")


@router.get("/sandbox/runs/{run_id}/logs/stream")
async def stream_logs(run_id: str, http_request: Request):
    """
    Stream logs for a run in real-time (SSE).

    Returns Server-Sent Events with log entries.
    """
    return StreamingResponse(
        wrap_generator(_log_stream_generator(run_id), workflow="sandbox_logs"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sandbox/runs/{run_id}/logs")
async def get_logs(
    run_id: str,
    http_request: Request,
    level: Optional[str] = Query(None, description="Filter by log level"),
    limit: int = Query(500, ge=1, le=5000, description="Max logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Get historical logs for a run."""
    try:
        from paperbot.infrastructure.logging.execution_logger import get_execution_logger

        logger = get_execution_logger()
        logs = logger.get_logs_dict(run_id, level=level, limit=limit, offset=offset)
        return {"run_id": run_id, "logs": logs, "count": len(logs)}
    except Exception as e:
        return {"run_id": run_id, "logs": [], "error": str(e)}


# --- Resource Metrics ---


async def _metrics_stream_generator(run_id: str):
    """Generate SSE events for metrics streaming."""
    from paperbot.infrastructure.monitoring.resource_monitor import get_resource_monitor

    monitor = get_resource_monitor()

    async for metrics in monitor.stream_metrics(run_id):
        yield StreamEvent(
            type="metrics",
            data=metrics.to_dict(),
        )

    yield StreamEvent(type="done", message="Metrics stream ended")


@router.get("/sandbox/runs/{run_id}/metrics/stream")
async def stream_metrics(run_id: str, http_request: Request):
    """
    Stream resource metrics for a run in real-time (SSE).

    Returns Server-Sent Events with CPU/memory metrics.
    """
    return StreamingResponse(
        wrap_generator(_metrics_stream_generator(run_id), workflow="sandbox_metrics"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sandbox/runs/{run_id}/metrics")
async def get_metrics(run_id: str, http_request: Request):
    """Get current resource metrics for a run."""
    try:
        from paperbot.infrastructure.monitoring.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()
        metrics = monitor.get_metrics(run_id)
        if metrics:
            return metrics.to_dict()
        else:
            return {"run_id": run_id, "status": "not_found"}
    except Exception as e:
        return {"run_id": run_id, "error": str(e)}


@router.get("/sandbox/runs/{run_id}/metrics/history")
async def get_metrics_history(
    run_id: str,
    http_request: Request,
    limit: int = Query(100, ge=1, le=1000, description="Max metrics to return"),
):
    """Get historical resource metrics for a run."""
    try:
        from paperbot.infrastructure.monitoring.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()
        history = monitor.get_metrics_history(run_id, limit=limit)
        return {
            "run_id": run_id,
            "metrics": [m.to_dict() for m in history],
            "count": len(history),
        }
    except Exception as e:
        return {"run_id": run_id, "metrics": [], "error": str(e)}


# --- System Status ---


@router.get("/sandbox/status")
async def get_system_status(http_request: Request):
    """
    Get overall sandbox system status.

    Returns status for Redis/queue.
    """
    try:
        from paperbot.infrastructure.monitoring.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()
        status = await monitor.get_system_status()
        return status.to_dict()
    except Exception as e:
        return {
            "queue": {"status": "unknown", "error": str(e)},
        }


# --- Settings Management ---


class E2BSettingsRequest(BaseModel):
    """Request body for saving E2B API key"""
    api_key: str


class E2BSettingsResponse(BaseModel):
    """Response for E2B settings"""
    configured: bool
    masked_key: Optional[str] = None


def _load_settings() -> Dict[str, Any]:
    """Load settings from file."""
    import json
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to file."""
    import json
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


@router.get("/sandbox/settings/e2b")
async def get_e2b_settings(http_request: Request) -> E2BSettingsResponse:
    """
    Get E2B configuration status.

    Returns whether E2B is configured and a masked version of the key.
    """
    # Check environment variable first
    env_key = os.getenv("E2B_API_KEY")
    if env_key:
        return E2BSettingsResponse(
            configured=True,
            masked_key=f"{env_key[:4]}...{env_key[-4:]}" if len(env_key) > 8 else "****"
        )

    # Check persisted settings
    settings = _load_settings()
    api_key = settings.get("e2b_api_key")
    if api_key:
        return E2BSettingsResponse(
            configured=True,
            masked_key=f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        )

    return E2BSettingsResponse(configured=False)


@router.post("/sandbox/settings/e2b")
async def save_e2b_settings(body: E2BSettingsRequest, http_request: Request):
    """
    Save E2B API key.

    Persists the key to ~/.paperbot/sandbox_settings.json and sets it in the environment.
    """
    settings = _load_settings()
    settings["e2b_api_key"] = body.api_key
    _save_settings(settings)

    # Also set in environment for current session
    os.environ["E2B_API_KEY"] = body.api_key

    return {
        "status": "saved",
        "message": "E2B API key saved successfully",
        "masked_key": f"{body.api_key[:4]}...{body.api_key[-4:]}" if len(body.api_key) > 8 else "****"
    }


@router.delete("/sandbox/settings/e2b")
async def delete_e2b_settings(http_request: Request):
    """Remove saved E2B API key."""
    settings = _load_settings()
    if "e2b_api_key" in settings:
        del settings["e2b_api_key"]
        _save_settings(settings)

    # Remove from environment
    if "E2B_API_KEY" in os.environ:
        del os.environ["E2B_API_KEY"]

    return {"status": "deleted", "message": "E2B API key removed"}


# --- Docker Management ---


@router.post("/sandbox/docker/start")
async def start_docker(http_request: Request):
    """
    Attempt to start Docker Desktop (macOS/Windows only).

    Returns status of the start attempt.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            subprocess.Popen(
                ["open", "-a", "Docker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {
                "status": "starting",
                "message": "Docker Desktop is starting. Please wait a moment.",
                "platform": "macOS",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start Docker: {e}",
                "platform": "macOS",
            }

    elif system == "Windows":
        try:
            # Try common Docker Desktop paths
            docker_paths = [
                r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
                r"C:\Program Files (x86)\Docker\Docker\Docker Desktop.exe",
            ]
            for path in docker_paths:
                if Path(path).exists():
                    subprocess.Popen(
                        [path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return {
                        "status": "starting",
                        "message": "Docker Desktop is starting. Please wait a moment.",
                        "platform": "Windows",
                    }
            return {
                "status": "not_found",
                "message": "Docker Desktop not found. Please install it from docker.com",
                "platform": "Windows",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start Docker: {e}",
                "platform": "Windows",
            }

    elif system == "Linux":
        # On Linux, Docker daemon is typically managed by systemd
        try:
            result = subprocess.run(
                ["systemctl", "start", "docker"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return {
                    "status": "starting",
                    "message": "Docker daemon is starting.",
                    "platform": "Linux",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to start Docker: {result.stderr}. Try 'sudo systemctl start docker'",
                    "platform": "Linux",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start Docker: {e}. Try 'sudo systemctl start docker'",
                "platform": "Linux",
            }

    return {
        "status": "unsupported",
        "message": f"Unsupported platform: {system}",
        "platform": system,
    }
