"""Read-only protocol for research track context aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ResearchTrackReadPort(Protocol):
    """Track-scoped read operations needed by track context services."""

    def get_track(self, *, user_id: str, track_id: int) -> Optional[Dict[str, Any]]: ...

    def get_active_track(self, *, user_id: str) -> Optional[Dict[str, Any]]: ...

    def list_tasks(
        self,
        *,
        user_id: str,
        track_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]: ...

    def list_milestones(
        self,
        *,
        user_id: str,
        track_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]: ...

    def list_effective_paper_feedback(
        self,
        *,
        user_id: str,
        track_id: int,
        limit: int = 200,
    ) -> List[Dict[str, Any]]: ...

    def list_saved_papers(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        collection_id: Optional[int] = None,
        limit: int = 200,
        sort_by: str = "saved_at",
    ) -> List[Dict[str, Any]]: ...

    def summarize_eval(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        days: int = 30,
        limit: int = 2000,
    ) -> Dict[str, Any]: ...
