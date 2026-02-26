"""ReproContextPort — interface for P2C context pack persistence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ReproContextPort(Protocol):
    """Abstract interface for ReproContextPack read/write operations."""

    def save(
        self,
        *,
        pack_id: str,
        user_id: str,
        paper_id: str,
        depth: str,
        pack_data: Dict[str, Any],
        paper_title: Optional[str] = None,
        project_id: Optional[str] = None,
        objective: Optional[str] = None,
        confidence_overall: float = 0.0,
        warning_count: int = 0,
    ) -> str:
        """Persist a new context pack. Returns the pack_id."""
        ...

    def update_status(
        self,
        pack_id: str,
        *,
        status: str,
        pack_data: Optional[Dict[str, Any]] = None,
        confidence_overall: Optional[float] = None,
        warning_count: Optional[int] = None,
        objective: Optional[str] = None,
    ) -> None:
        """Update status (and optionally pack_json) of an existing pack."""
        ...

    def get(self, pack_id: str) -> Optional[Dict[str, Any]]:
        """Return the full context pack dict, or None if not found / soft-deleted."""
        ...

    def list_by_user(
        self,
        *,
        user_id: str,
        paper_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return (summary_list, total_count) for the given user."""
        ...

    def soft_delete(self, pack_id: str) -> bool:
        """Mark a pack as deleted. Returns True if the row was found and updated."""
        ...

    def save_stage_result(
        self,
        *,
        pack_id: str,
        stage_name: str,
        status: str,
        result_data: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        duration_ms: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Persist an intermediate stage result for debugging / audit."""
        ...
