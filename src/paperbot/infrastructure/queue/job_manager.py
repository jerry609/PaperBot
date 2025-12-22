"""
Job Manager - Wraps ARQ operations for queue management.

Provides:
- Queue status inspection
- Job cancellation
- Job retry
- Paper2Code submission
"""

from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from arq.connections import ArqRedis, RedisSettings, create_pool
from arq.jobs import Job, JobStatus

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id


def _redis_settings() -> RedisSettings:
    return RedisSettings(
        host=os.getenv("PAPERBOT_REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("PAPERBOT_REDIS_PORT", "6379")),
        database=int(os.getenv("PAPERBOT_REDIS_DB", "0")),
        password=os.getenv("PAPERBOT_REDIS_PASSWORD") or None,
    )


@dataclass
class JobInfo:
    """Job information structure"""
    job_id: str
    function: str
    status: str  # pending, queued, in_progress, complete, not_found
    enqueue_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    finish_time: Optional[datetime] = None
    result: Any = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "function": self.function,
            "status": self.status,
            "enqueue_time": self.enqueue_time.isoformat() if self.enqueue_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
            "result": self.result,
            "args": list(self.args),
            "kwargs": self.kwargs,
        }


@dataclass
class QueueStatus:
    """Queue status structure"""
    pending: List[JobInfo] = field(default_factory=list)
    running: List[JobInfo] = field(default_factory=list)
    completed: List[JobInfo] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending": [j.to_dict() for j in self.pending],
            "running": [j.to_dict() for j in self.running],
            "completed": [j.to_dict() for j in self.completed],
            "stats": self.stats,
        }


class JobManager:
    """
    Job Manager - Encapsulates ARQ operations.

    Usage:
        manager = JobManager()
        await manager.connect()
        status = await manager.get_queue_status()
        await manager.close()
    """

    def __init__(self, redis_settings: Optional[RedisSettings] = None):
        self.redis_settings = redis_settings or _redis_settings()
        self._pool: Optional[ArqRedis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._pool is None:
            self._pool = await create_pool(self.redis_settings)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _ensure_connected(self) -> ArqRedis:
        """Ensure connection is established."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore

    async def get_job_info(self, job_id: str) -> Optional[JobInfo]:
        """Get information about a specific job."""
        pool = await self._ensure_connected()
        job = Job(job_id, pool)

        status = await job.status()
        if status == JobStatus.not_found:
            return None

        info = await job.info()
        if info is None:
            return JobInfo(job_id=job_id, function="unknown", status=status.value)

        return JobInfo(
            job_id=job_id,
            function=info.function,
            status=status.value,
            enqueue_time=info.enqueue_time,
            start_time=info.start_time,
            finish_time=info.finish_time,
            result=info.result,
            args=info.args,
            kwargs=info.kwargs,
        )

    async def get_queue_status(self, completed_limit: int = 20) -> QueueStatus:
        """
        Get current queue status.

        Note: ARQ doesn't provide a built-in way to list all jobs.
        This implementation uses Redis SCAN to find job keys.
        """
        pool = await self._ensure_connected()

        pending: List[JobInfo] = []
        running: List[JobInfo] = []
        completed: List[JobInfo] = []

        # Scan for job keys in Redis
        # ARQ stores jobs with keys like "arq:job:{job_id}"
        cursor = 0
        job_ids: List[str] = []

        try:
            while True:
                cursor, keys = await pool.scan(cursor, match="arq:job:*", count=100)
                for key in keys:
                    # Extract job_id from key
                    key_str = key.decode() if isinstance(key, bytes) else key
                    if key_str.startswith("arq:job:"):
                        job_id = key_str[8:]  # Remove "arq:job:" prefix
                        job_ids.append(job_id)
                if cursor == 0:
                    break
        except Exception:
            # If scan fails, return empty status
            pass

        # Get info for each job
        for job_id in job_ids[:100]:  # Limit to prevent overload
            info = await self.get_job_info(job_id)
            if info is None:
                continue

            if info.status in ("pending", "queued", "deferred"):
                pending.append(info)
            elif info.status == "in_progress":
                running.append(info)
            elif info.status == "complete":
                if len(completed) < completed_limit:
                    completed.append(info)

        # Sort by time
        pending.sort(key=lambda j: j.enqueue_time or datetime.min.replace(tzinfo=timezone.utc))
        running.sort(key=lambda j: j.start_time or datetime.min.replace(tzinfo=timezone.utc))
        completed.sort(key=lambda j: j.finish_time or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        return QueueStatus(
            pending=pending,
            running=running,
            completed=completed[:completed_limit],
            stats={
                "total_pending": len(pending),
                "total_running": len(running),
                "redis_connected": True,
            },
        )

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Note: ARQ doesn't support true job cancellation for running jobs.
        This aborts pending jobs only.
        """
        pool = await self._ensure_connected()
        job = Job(job_id, pool)

        status = await job.status()
        if status == JobStatus.not_found:
            return False

        if status in (JobStatus.queued, JobStatus.deferred):
            # Abort pending job
            await job.abort()
            return True

        # Cannot cancel running or completed jobs
        return False

    async def retry_job(self, job_id: str) -> Optional[str]:
        """
        Retry a job by re-enqueuing with same args.

        Returns new job_id if successful, None otherwise.
        """
        pool = await self._ensure_connected()
        job = Job(job_id, pool)

        info = await job.info()
        if info is None:
            return None

        # Re-enqueue with same function and args
        new_job = await pool.enqueue_job(info.function, *info.args, **info.kwargs)
        return new_job.job_id if new_job else None

    async def submit_paper2code(
        self,
        paper_url: Optional[str] = None,
        paper_id: Optional[str] = None,
        executor: str = "e2b",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Submit a Paper2Code job.

        Returns dict with job_id, run_id, trace_id.
        """
        pool = await self._ensure_connected()

        run_id = new_run_id()
        trace_id = new_trace_id()

        job_kwargs = {
            "paper_url": paper_url,
            "paper_id": paper_id,
            "executor": executor,
            "options": options or {},
            "run_id": run_id,
            "trace_id": trace_id,
        }

        job = await pool.enqueue_job("paper2code_job", **job_kwargs)

        return {
            "job_id": job.job_id if job else "",
            "run_id": run_id,
            "trace_id": trace_id,
        }

    async def submit_track_scholar(
        self,
        scholar_id: str,
        dry_run: bool = False,
        offline: bool = False,
    ) -> Dict[str, str]:
        """
        Submit a track_scholar job.

        Returns dict with job_id.
        """
        pool = await self._ensure_connected()

        job = await pool.enqueue_job(
            "track_scholar_job",
            scholar_id,
            dry_run=dry_run,
            offline=offline,
        )

        return {"job_id": job.job_id if job else ""}

    async def submit_analyze_paper(
        self,
        paper: Dict[str, Any],
        scholar_name: str = "",
    ) -> Dict[str, str]:
        """
        Submit an analyze_paper job.

        Returns dict with job_id.
        """
        pool = await self._ensure_connected()

        job = await pool.enqueue_job(
            "analyze_paper_job",
            paper,
            scholar_name=scholar_name,
        )

        return {"job_id": job.job_id if job else ""}
