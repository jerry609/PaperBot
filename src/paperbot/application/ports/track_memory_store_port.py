"""Protocol for track-scoped memory operations used by the application layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TrackMemoryStorePort(Protocol):
    """Subset of memory-store operations needed for track memory orchestration."""

    def list_memories(
        self,
        *,
        user_id: str,
        limit: int = 100,
        kind: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        include_pending: bool = False,
        include_deleted: bool = False,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def get_items_by_ids(
        self,
        *,
        user_id: str,
        item_ids: List[int],
    ) -> List[Dict[str, Any]]: ...

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 8,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        min_score: float = 0.0,
        candidate_multiplier: int = 4,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        half_life_days: float = 30.0,
    ) -> List[Dict[str, Any]]: ...

    def soft_delete_by_scope(
        self,
        *,
        user_id: str,
        scope_type: str,
        scope_id: Optional[str],
        actor_id: str = "system",
        reason: str = "",
    ) -> int: ...

    def bulk_update_items(
        self,
        *,
        user_id: str,
        item_ids: List[int],
        actor_id: str = "system",
        status: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...
