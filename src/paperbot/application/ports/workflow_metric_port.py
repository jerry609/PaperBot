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
        claim_count: int = 0,
        evidence_count: int = 0,
        elapsed_ms: float = 0.0,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    def summarize(
        self,
        *,
        days: int = 7,
        workflow: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> Dict[str, Any]: ...
