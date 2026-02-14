from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from paperbot.application.services.paper_search_service import PaperSearchService
from paperbot.domain.paper import PaperCandidate


@dataclass
class _FakeAdapter:
    source_name: str
    papers: list[PaperCandidate]

    async def search(
        self,
        query: str,
        *,
        max_results: int = 30,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> list[PaperCandidate]:
        return list(self.papers)[:max_results]

    async def close(self) -> None:
        return None


@dataclass
class _FakeRegistry:
    sync_flags: list[bool]

    def upsert_paper(self, *, paper: dict, source_hint: str, sync_authors: bool = True) -> dict:
        self.sync_flags.append(bool(sync_authors))
        return {"id": 1}


@pytest.mark.asyncio
async def test_rrf_fusion_prefers_cross_source_consensus() -> None:
    shared = PaperCandidate(title="Attention is all you need")
    unique_s2 = PaperCandidate(title="Semantic-only paper")
    unique_oa = PaperCandidate(title="OpenAlex-only paper")

    service = PaperSearchService(
        adapters={
            "semantic_scholar": _FakeAdapter("semantic_scholar", [shared, unique_s2]),
            "openalex": _FakeAdapter("openalex", [shared, unique_oa]),
        }
    )

    result = await service.search(
        "attention", sources=["semantic_scholar", "openalex"], persist=False
    )

    assert len(result.papers) == 3
    assert result.papers[0].title == "Attention is all you need"
    assert result.papers[0].retrieval_score > result.papers[1].retrieval_score


@pytest.mark.asyncio
async def test_rrf_merges_duplicates_and_tracks_retrieval_sources() -> None:
    same_a = PaperCandidate(title="Duplicate Title", abstract="A")
    same_b = PaperCandidate(title="Duplicate Title", abstract="B", citation_count=10)

    service = PaperSearchService(
        adapters={
            "semantic_scholar": _FakeAdapter("semantic_scholar", [same_a]),
            "arxiv": _FakeAdapter("arxiv", [same_b]),
        }
    )

    result = await service.search("duplicate", sources=["semantic_scholar", "arxiv"], persist=False)

    assert result.total_raw == 2
    assert result.duplicates_removed == 1
    assert len(result.papers) == 1

    merged = result.papers[0]
    assert merged.title == "Duplicate Title"
    assert set(merged.retrieval_sources) == {"semantic_scholar", "arxiv"}
    assert merged.retrieval_score > 0


@pytest.mark.asyncio
async def test_persist_search_results_disables_author_sync_for_latency() -> None:
    registry = _FakeRegistry(sync_flags=[])
    service = PaperSearchService(
        adapters={"semantic_scholar": _FakeAdapter("semantic_scholar", [PaperCandidate(title="P1")])},
        registry=registry,
    )

    result = await service.search("p1", sources=["semantic_scholar"], persist=True)

    assert len(result.papers) == 1
    assert registry.sync_flags == [False]
