"""Port for building a grounded wiki concept snapshot from stored research data."""

from __future__ import annotations

from typing import List, Optional, Protocol, TypedDict, runtime_checkable


class PaperGroundingRecord(TypedDict):
    title: str
    abstract: str
    keywords: List[str]
    fields_of_study: List[str]
    citation_count: int
    year: Optional[int]


class TrackGroundingRecord(TypedDict):
    name: str
    description: str
    keywords: List[str]
    methods: List[str]


class GroundingSnapshot(TypedDict):
    papers: List[PaperGroundingRecord]
    tracks: List[TrackGroundingRecord]


@runtime_checkable
class WikiConceptPort(Protocol):
    """Read-only interface for wiki grounding inputs."""

    def load_grounding_snapshot(
        self,
        *,
        user_id: str,
        paper_limit: int = 250,
        track_limit: int = 100,
    ) -> GroundingSnapshot: ...
