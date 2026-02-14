from __future__ import annotations

import pytest

from paperbot.application.services.paper_search_service import SearchResult
from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig
from paperbot.domain.paper import PaperCandidate


class _FakeResearchStore:
    def __init__(self) -> None:
        self.track = {"id": 1, "name": "NLP", "keywords": ["transformer"]}

    def get_active_track(self, *, user_id: str):
        return self.track

    def get_track(self, *, user_id: str, track_id: int):
        return self.track

    def list_tasks(self, *, user_id: str, track_id: int, limit: int):
        return []

    def list_milestones(self, *, user_id: str, track_id: int, limit: int):
        return []

    def list_paper_feedback_ids(self, *, user_id: str, track_id: int, action: str):
        return set()

    def list_paper_feedback(self, *, user_id: str, track_id: int, limit: int):
        return []

    def create_context_run(self, **kwargs):
        return {"id": 1}


class _FakeMemoryStore:
    def list_memories(self, **kwargs):
        return []

    def touch_usage(self, **kwargs):
        return None

    def search_memories(self, **kwargs):
        return []


class _FakeTrackRouter:
    def suggest_track(self, **kwargs):
        return None


class _CaptureSearchService:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        paper = PaperCandidate(
            title="Transformer Paper",
            abstract="This abstract is long enough to pass the academic filter.",
            year=2022,
            citation_count=5,
            canonical_id=1,
        )
        return SearchResult(papers=[paper], provenance={}, total_raw=1, duplicates_removed=0)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_context_engine_passes_year_filters_to_search_service():
    search_service = _CaptureSearchService()
    engine = ContextEngine(
        research_store=_FakeResearchStore(),
        memory_store=_FakeMemoryStore(),
        paper_store=None,
        search_service=search_service,
        track_router=_FakeTrackRouter(),
        config=ContextEngineConfig(
            personalized=False,
            paper_limit=5,
            year_from=2020,
            year_to=2024,
            search_sources=["semantic_scholar"],
        ),
    )

    await engine.build_context_pack(user_id="u1", query="transformer", track_id=1)

    assert len(search_service.calls) == 1
    call = search_service.calls[0]
    assert call["year_from"] == 2020
    assert call["year_to"] == 2024
