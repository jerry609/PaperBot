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
        workspace_id: str = "default",
        scope_type: str = "global",
        scope_id: str = "",
        source_type: str = "chat",
        source_name: str = "",
    ) -> Tuple[int, int, List[Dict[str, Any]]]: ...

    def list_memories(
        self,
        *,
        user_id: str,
        workspace_id: str = "default",
        limit: int = 50,
        offset: int = 0,
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
