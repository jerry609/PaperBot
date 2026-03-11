"""AuthorPort — author record read/write interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class AuthorPort(Protocol):
    """Abstract interface for author operations."""

    def upsert_author(
        self,
        *,
        name: str,
        author_id: Optional[str] = None,
        slug: Optional[str] = None,
        h_index: Optional[int] = None,
        citation_count: Optional[int] = None,
    ) -> Dict[str, Any]: ...

    def get_author(self, author_id: int) -> Optional[Dict[str, Any]]: ...

    def list_authors(
        self, *, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]: ...
