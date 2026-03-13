"""RegistryPort — canonical paper CRUD interface."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from paperbot.domain.harvest import HarvestedPaper


@runtime_checkable
class RegistryPort(Protocol):
    """Abstract interface for the canonical paper registry."""

    def upsert_paper(
        self, *, paper: Dict[str, Any], source_hint: Optional[str] = None
    ) -> Dict[str, Any]: ...

    def upsert_many(
        self,
        *,
        papers: Iterable[Dict[str, Any]],
        source_hint: Optional[str] = None,
        seen_at: Optional[datetime] = None,
    ) -> Dict[str, int]: ...

    def upsert_papers_batch(self, papers: List[HarvestedPaper]) -> Tuple[int, int]: ...
