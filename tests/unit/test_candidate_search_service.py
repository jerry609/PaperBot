from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

import pytest

from paperbot.application.services.candidate_search import (
    ingest_candidate_papers,
    search_candidate_papers,
    search_result_to_candidate_dicts,
)
from paperbot.application.services.paper_search_service import SearchResult
from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper import PaperCandidate


@dataclass
class _FakeSearchService:
    calls: list[dict]

    async def search(self, query: str, **kwargs) -> SearchResult:
        self.calls.append({"query": query, **kwargs})
        paper = PaperCandidate(
            title="Candidate Paper",
            abstract="long enough abstract to look academic",
            identities=[PaperIdentity("semantic_scholar", "s2-1")],
        )
        return SearchResult(papers=[paper], provenance={}, total_raw=1, duplicates_removed=0)


class _RegistryWithLookup:
    def get_paper_by_source_id_any(self, source_id: str) -> Optional[SimpleNamespace]:
        if source_id == "s2-1":
            return SimpleNamespace(id=42)
        return None


class _RegistryWithSeenAt:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert_many(self, **kwargs):
        self.calls.append(kwargs)
        papers = list(kwargs.get("papers") or [])
        return {"total": len(papers), "created": len(papers), "updated": 0}


class _RegistryWithoutSeenAt:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert_many(self, *, papers, source_hint=None, seen_at=None):
        payload = {"papers": list(papers), "source_hint": source_hint, "seen_at": seen_at}
        self.calls.append(payload)
        return {"total": len(payload["papers"]), "created": len(payload["papers"]), "updated": 0}


@pytest.mark.asyncio
async def test_search_candidate_papers_defaults_to_non_persist() -> None:
    service = _FakeSearchService(calls=[])

    result = await search_candidate_papers(
        service,
        query="transformer",
        sources=["semantic_scholar"],
        max_results=7,
        year_from=2022,
        year_to=2024,
    )

    assert len(result.papers) == 1
    assert service.calls == [
        {
            "query": "transformer",
            "sources": ["semantic_scholar"],
            "max_results": 7,
            "year_from": 2022,
            "year_to": 2024,
            "persist": False,
            "source_weights": None,
        }
    ]


def test_search_result_to_candidate_dicts_prefers_existing_registry_id() -> None:
    search_result = SearchResult(
        papers=[
            PaperCandidate(
                title="Candidate Paper",
                abstract="long enough abstract to look academic",
                identities=[PaperIdentity("semantic_scholar", "s2-1")],
            )
        ],
        provenance={},
        total_raw=1,
        duplicates_removed=0,
    )

    rows = search_result_to_candidate_dicts(search_result, registry=_RegistryWithLookup())

    assert rows[0]["paper_id"] == "42"
    assert rows[0]["canonical_paper_id"] == 42


def test_ingest_candidate_papers_passes_seen_at_when_supported() -> None:
    registry = _RegistryWithSeenAt()
    seen_at = datetime(2026, 3, 12, tzinfo=timezone.utc)

    payload = ingest_candidate_papers(
        papers=[{"title": "Candidate Paper"}],
        registry=registry,
        source_hint="papers_cool",
        seen_at=seen_at,
    )

    assert payload["total"] == 1
    assert registry.calls[0]["seen_at"] == seen_at


def test_ingest_candidate_papers_passes_seen_at_through_registry_contract() -> None:
    registry = _RegistryWithoutSeenAt()
    seen_at = datetime(2026, 3, 12, tzinfo=timezone.utc)

    payload = ingest_candidate_papers(
        papers=[{"title": "Candidate Paper"}],
        registry=registry,
        source_hint="papers_cool",
        seen_at=seen_at,
    )

    assert payload["total"] == 1
    assert registry.calls == [
        {
            "papers": [{"title": "Candidate Paper"}],
            "source_hint": "papers_cool",
            "seen_at": seen_at,
        }
    ]
