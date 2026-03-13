"""Ports and value objects for document intelligence capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class DocumentSection:
    """Logical document section before chunking."""

    name: str
    heading: str
    content: str
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentContent:
    """Fetched source content for a canonical paper."""

    paper_id: int
    title: str
    source_type: str
    sections: List[DocumentSection]
    locator_url: Optional[str] = None
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """Indexed chunk persisted for later retrieval."""

    paper_id: int
    section: str
    heading: str
    content: str
    chunk_index: int
    token_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceHit:
    """Retrieved evidence chunk for a query."""

    paper_id: int
    chunk_id: int
    chunk_index: int
    paper_title: str
    section: str
    heading: str
    snippet: str
    score: float
    source_type: str
    locator_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentFetcherPort(Protocol):
    """Fetch canonical paper content ready for indexing."""

    def fetch_document(self, *, paper_id: int) -> Optional[DocumentContent]: ...


@runtime_checkable
class DocumentIndexerPort(Protocol):
    """Transform fetched content into retrieval chunks."""

    def index_document(self, *, document: DocumentContent) -> List[DocumentChunk]: ...


@runtime_checkable
class EvidenceRetrieverPort(Protocol):
    """Retrieve indexed evidence for a user query."""

    def retrieve_evidence(
        self,
        *,
        query: str,
        paper_ids: Optional[Sequence[int]] = None,
        limit: int = 6,
    ) -> List[EvidenceHit]: ...


@runtime_checkable
class AnalysisRuntimePort(Protocol):
    """Run deeper analysis on already indexed evidence."""

    async def analyze(
        self,
        *,
        query: str,
        paper_ids: Sequence[int],
        limit: int = 6,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class RankerPort(Protocol):
    """Optional re-ranker for retrieved candidates or evidence."""

    def rank(
        self,
        *,
        query: str,
        items: Sequence[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]: ...
