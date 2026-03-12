"""MemoryPort — memory item read/write interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class MemoryPort(Protocol):
    """Abstract interface for memory item operations."""

    def add_memories(
        self,
        *,
        user_id: str,
        memories: list,
        source_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        status: Optional[str] = None,
        actor_id: str = "system",
    ) -> Tuple[int, int, List[Any]]: ...

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

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]: ...

    def soft_delete_item(
        self,
        *,
        user_id: str,
        item_id: int,
        actor_id: str = "system",
        reason: str = "",
    ) -> bool: ...
