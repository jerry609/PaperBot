"""
Execution Logger - Captures and streams sandbox stdout/stderr.

Provides:
- Real-time log streaming
- Log persistence to database
- Log level filtering
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from sqlalchemy import select, asc

from paperbot.infrastructure.stores.models import ExecutionLogModel, Base
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


@dataclass
class LogEntry:
    """Log entry structure"""
    ts: datetime
    level: str  # debug, info, warning, error
    message: str
    source: str  # stdout, stderr, executor, system
    run_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

    def to_sse(self) -> str:
        """Convert to SSE format"""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class ExecutionLogger:
    """
    Execution Logger - Captures and streams sandbox logs.

    Features:
    - Persists logs to SQLite database
    - Streams logs in real-time via async generator
    - Supports log level filtering
    - Provides callback interface for executor integration
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)

        # In-memory buffer for real-time streaming
        self._buffers: Dict[str, asyncio.Queue[LogEntry]] = {}
        self._active_runs: Dict[str, bool] = {}

    def get_log_callback(self, run_id: str) -> Callable[[str, str, str], None]:
        """
        Get a callback function for executor integration.

        Usage:
            logger = ExecutionLogger()
            callback = logger.get_log_callback(run_id)
            executor = SomeExecutor(log_callback=callback)
        """
        def callback(level: str, message: str, source: str = "executor") -> None:
            entry = LogEntry(
                ts=datetime.now(timezone.utc),
                level=level,
                message=message,
                source=source,
                run_id=run_id,
            )
            self.append(entry)

        return callback

    def start_run(self, run_id: str) -> None:
        """Start capturing logs for a run."""
        self._buffers[run_id] = asyncio.Queue()
        self._active_runs[run_id] = True

    def stop_run(self, run_id: str) -> None:
        """Stop capturing logs for a run."""
        self._active_runs[run_id] = False
        # Signal end of stream
        if run_id in self._buffers:
            try:
                self._buffers[run_id].put_nowait(None)  # type: ignore
            except Exception:
                pass

    def append(self, entry: LogEntry) -> None:
        """Append a log entry (persists to DB and buffers for streaming)."""
        # Persist to database
        try:
            with self._provider.session() as session:
                model = ExecutionLogModel(
                    run_id=entry.run_id,
                    ts=entry.ts,
                    level=entry.level,
                    message=entry.message,
                    source=entry.source,
                )
                session.add(model)
                session.commit()
        except Exception:
            pass  # Best-effort persistence

        # Buffer for real-time streaming
        if entry.run_id in self._buffers:
            try:
                self._buffers[entry.run_id].put_nowait(entry)
            except Exception:
                pass

    def log(self, run_id: str, level: str, message: str, source: str = "system") -> None:
        """Convenience method to log a message."""
        entry = LogEntry(
            ts=datetime.now(timezone.utc),
            level=level,
            message=message,
            source=source,
            run_id=run_id,
        )
        self.append(entry)

    async def stream_logs(self, run_id: str, timeout: float = 0.5) -> AsyncGenerator[LogEntry, None]:
        """
        Stream logs for a run in real-time.

        Yields LogEntry objects as they arrive.
        Stops when the run is stopped or timeout occurs with no new logs.
        """
        # Ensure buffer exists
        if run_id not in self._buffers:
            self._buffers[run_id] = asyncio.Queue()
            self._active_runs[run_id] = True

        queue = self._buffers[run_id]

        while self._active_runs.get(run_id, False) or not queue.empty():
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=timeout)
                if entry is None:  # End signal
                    break
                yield entry
            except asyncio.TimeoutError:
                # Check if run is still active
                if not self._active_runs.get(run_id, False) and queue.empty():
                    break
                continue

    def get_logs(
        self,
        run_id: str,
        level: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[LogEntry]:
        """Get historical logs from database."""
        with self._provider.session() as session:
            stmt = select(ExecutionLogModel).where(ExecutionLogModel.run_id == run_id)

            if level:
                stmt = stmt.where(ExecutionLogModel.level == level)

            stmt = stmt.order_by(asc(ExecutionLogModel.ts)).offset(offset).limit(limit)

            rows = session.execute(stmt).scalars()
            return [
                LogEntry(
                    ts=row.ts,
                    level=row.level,
                    message=row.message,
                    source=row.source,
                    run_id=row.run_id,
                )
                for row in rows
            ]

    def get_logs_dict(
        self,
        run_id: str,
        level: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get historical logs as dictionaries."""
        return [log.to_dict() for log in self.get_logs(run_id, level, limit, offset)]

    def close(self) -> None:
        """Close database connection."""
        try:
            self._provider.engine.dispose()
        except Exception:
            pass


# Global singleton (optional)
_global_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """Get or create global ExecutionLogger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ExecutionLogger()
    return _global_logger
