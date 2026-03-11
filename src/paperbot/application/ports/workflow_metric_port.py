"""WorkflowMetricPort — workflow quality metric persistence interface."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class WorkflowMetricPort(Protocol):
    """Abstract interface for workflow metric operations."""

    def record_metric(
        self,
        *,
        workflow: str,
        stage: str = "",
        status: str = "completed",
        track_id: Optional[int] = None,
        coverage: Optional[float] = None,
        latency_ms: Optional[int] = None,
        detail_json: Optional[str] = None,
    ) -> None: ...

    def summarize(
        self, *, days: int = 7, workflow: Optional[str] = None
    ) -> Dict[str, Any]: ...
