"""
Resource Monitor - Tracks CPU, memory, and execution time for sandbox runs.

Provides:
- Real-time resource metrics streaming
- Metrics persistence to database
- System status aggregation
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy import select, desc

from paperbot.infrastructure.stores.models import ResourceMetricModel, Base
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


@dataclass
class ResourceMetrics:
    """Resource metrics structure"""
    run_id: str
    ts: datetime
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_limit_mb: float = 4096.0
    elapsed_seconds: float = 0.0
    timeout_seconds: float = 600.0
    status: str = "running"  # running, completed, failed, timeout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ts": self.ts.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_limit_mb": self.memory_limit_mb,
            "elapsed_seconds": self.elapsed_seconds,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status,
            "memory_percent": (self.memory_mb / self.memory_limit_mb * 100) if self.memory_limit_mb > 0 else 0,
            "time_percent": (self.elapsed_seconds / self.timeout_seconds * 100) if self.timeout_seconds > 0 else 0,
        }

    def to_sse(self) -> str:
        """Convert to SSE format"""
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class SystemStatus:
    """Overall system status"""
    e2b: Dict[str, Any] = field(default_factory=dict)
    docker: Dict[str, Any] = field(default_factory=dict)
    queue: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "e2b": self.e2b,
            "docker": self.docker,
            "queue": self.queue,
        }


class ResourceMonitor:
    """
    Resource Monitor - Tracks sandbox resource usage.

    Features:
    - Persists metrics to SQLite database
    - Streams metrics in real-time
    - Aggregates system-wide status
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)

        # In-memory tracking for active runs
        self._active_runs: Dict[str, Dict[str, Any]] = {}

    def start_run(
        self,
        run_id: str,
        timeout_seconds: float = 600.0,
        memory_limit_mb: float = 4096.0,
    ) -> None:
        """Start tracking a run."""
        self._active_runs[run_id] = {
            "start_time": datetime.now(timezone.utc),
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": memory_limit_mb,
            "status": "running",
        }

    def stop_run(self, run_id: str, status: str = "completed") -> None:
        """Stop tracking a run."""
        if run_id in self._active_runs:
            self._active_runs[run_id]["status"] = status

    def record_metrics(
        self,
        run_id: str,
        cpu_percent: float,
        memory_mb: float,
    ) -> None:
        """Record a metrics snapshot."""
        run_info = self._active_runs.get(run_id, {})
        start_time = run_info.get("start_time", datetime.now(timezone.utc))
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        try:
            with self._provider.session() as session:
                model = ResourceMetricModel(
                    run_id=run_id,
                    ts=datetime.now(timezone.utc),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_limit_mb=run_info.get("memory_limit_mb", 4096.0),
                )
                session.add(model)
                session.commit()
        except Exception:
            pass  # Best-effort persistence

    def get_metrics(self, run_id: str) -> Optional[ResourceMetrics]:
        """Get current metrics for a run."""
        run_info = self._active_runs.get(run_id)

        # If not actively tracked, try to get from DB
        if run_info is None:
            with self._provider.session() as session:
                stmt = (
                    select(ResourceMetricModel)
                    .where(ResourceMetricModel.run_id == run_id)
                    .order_by(desc(ResourceMetricModel.ts))
                    .limit(1)
                )
                row = session.execute(stmt).scalar_one_or_none()
                if row:
                    return ResourceMetrics(
                        run_id=run_id,
                        ts=row.ts,
                        cpu_percent=row.cpu_percent,
                        memory_mb=row.memory_mb,
                        memory_limit_mb=row.memory_limit_mb,
                        status="completed",
                    )
            return None

        # Calculate current metrics
        start_time = run_info.get("start_time", datetime.now(timezone.utc))
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        timeout = run_info.get("timeout_seconds", 600.0)
        memory_limit = run_info.get("memory_limit_mb", 4096.0)

        # Get latest metrics from DB
        cpu_percent = 0.0
        memory_mb = 0.0

        with self._provider.session() as session:
            stmt = (
                select(ResourceMetricModel)
                .where(ResourceMetricModel.run_id == run_id)
                .order_by(desc(ResourceMetricModel.ts))
                .limit(1)
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row:
                cpu_percent = row.cpu_percent
                memory_mb = row.memory_mb

        return ResourceMetrics(
            run_id=run_id,
            ts=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_limit_mb=memory_limit,
            elapsed_seconds=elapsed,
            timeout_seconds=timeout,
            status=run_info.get("status", "running"),
        )

    async def stream_metrics(
        self,
        run_id: str,
        interval: float = 1.0,
    ) -> AsyncGenerator[ResourceMetrics, None]:
        """
        Stream metrics for a run at regular intervals.

        Yields ResourceMetrics objects every `interval` seconds.
        Stops when the run is no longer active.
        """
        while run_id in self._active_runs and self._active_runs[run_id].get("status") == "running":
            metrics = self.get_metrics(run_id)
            if metrics:
                yield metrics

                # Check for timeout
                if metrics.elapsed_seconds >= metrics.timeout_seconds:
                    self.stop_run(run_id, "timeout")
                    yield metrics
                    break

            await asyncio.sleep(interval)

        # Yield final metrics
        metrics = self.get_metrics(run_id)
        if metrics:
            yield metrics

    def get_metrics_history(
        self,
        run_id: str,
        limit: int = 100,
    ) -> List[ResourceMetrics]:
        """Get historical metrics for a run."""
        with self._provider.session() as session:
            stmt = (
                select(ResourceMetricModel)
                .where(ResourceMetricModel.run_id == run_id)
                .order_by(desc(ResourceMetricModel.ts))
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars())

            # Reverse to get chronological order
            rows.reverse()

            return [
                ResourceMetrics(
                    run_id=row.run_id,
                    ts=row.ts,
                    cpu_percent=row.cpu_percent,
                    memory_mb=row.memory_mb,
                    memory_limit_mb=row.memory_limit_mb,
                )
                for row in rows
            ]

    async def get_system_status(self) -> SystemStatus:
        """
        Get overall system status.

        Checks E2B, Docker, and Redis availability.
        """
        e2b_status = await self._check_e2b_status()
        docker_status = await self._check_docker_status()
        queue_status = await self._check_queue_status()

        return SystemStatus(
            e2b=e2b_status,
            docker=docker_status,
            queue=queue_status,
        )

    async def _check_e2b_status(self) -> Dict[str, Any]:
        """Check E2B sandbox availability."""
        has_key = bool(os.getenv("E2B_API_KEY"))
        return {
            "status": "available" if has_key else "not_configured",
            "api_key_set": has_key,
            "sandboxes_active": len([
                r for r in self._active_runs.values()
                if r.get("executor") == "e2b" and r.get("status") == "running"
            ]),
        }

    async def _check_docker_status(self) -> Dict[str, Any]:
        """Check Docker availability."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            containers = len(client.containers.list())
            return {
                "status": "healthy",
                "containers_active": containers,
            }
        except Exception:
            return {
                "status": "unavailable",
                "containers_active": 0,
            }

    async def _check_queue_status(self) -> Dict[str, Any]:
        """Check Redis/ARQ queue availability."""
        try:
            from arq.connections import create_pool
            from paperbot.infrastructure.queue.job_manager import _redis_settings

            pool = await create_pool(_redis_settings())
            await pool.ping()
            await pool.close()

            return {
                "redis_connected": True,
                "status": "healthy",
            }
        except Exception:
            return {
                "redis_connected": False,
                "status": "unavailable",
            }

    def close(self) -> None:
        """Close database connection."""
        try:
            self._provider.engine.dispose()
        except Exception:
            pass


# Global singleton (optional)
_global_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create global ResourceMonitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor
