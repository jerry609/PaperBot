from __future__ import annotations

import pytest

from paperbot.application.services.paper_search_service import SearchResult
from paperbot.context_engine import engine as engine_module
from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig
from paperbot.domain.paper import PaperCandidate


class _FakeResearchStore:
    def __init__(self) -> None:
        self.track = {"id": 1, "name": "NLP", "keywords": ["transformer"]}

    def get_active_track(self, *, user_id: str):
        return self.track

    def get_track(self, *, user_id: str, track_id: int):
        if int(track_id) == 1:
            return self.track
        return None

    def list_tasks(self, *, user_id: str, track_id: int, limit: int):
        return []

    def list_milestones(self, *, user_id: str, track_id: int, limit: int):
        return []

    def list_paper_feedback_ids(self, *, user_id: str, track_id: int, action: str):
        if action == "save":
            return {"1"}
        return set()

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


class _FakeSearchService:
    async def search(self, *args, **kwargs):
        paper = PaperCandidate(
            title="Transformer Paper",
            abstract="This abstract is long enough to pass the academic filter.",
            year=2024,
            citation_count=10,
            canonical_id=1,
        )
        return SearchResult(papers=[paper], provenance={}, total_raw=1, duplicates_removed=0)

    async def close(self) -> None:
        return None


class _FakeAnchorService:
    def get_followed_paper_anchor_scores(
        self, *, user_id: str, track_id: int, paper_ids: list[int]
    ):
        return {1: 1.0}


@pytest.mark.asyncio
async def test_personalized_mode_applies_saved_and_anchor_boosts(monkeypatch):
    monkeypatch.setattr(engine_module, "_get_anchor_service", lambda: _FakeAnchorService())

    engine = ContextEngine(
        research_store=_FakeResearchStore(),
        memory_store=_FakeMemoryStore(),
        paper_store=None,
        search_service=_FakeSearchService(),
        track_router=_FakeTrackRouter(),
        config=ContextEngineConfig(personalized=True, paper_limit=5),
    )

    pack = await engine.build_context_pack(user_id="u1", query="transformer", track_id=1)
    score = float(pack["paper_recommendation_scores"]["1"])

    assert score > 0.44  # includes +0.25 saved boost and +0.20 anchor boost


@pytest.mark.asyncio
async def test_global_mode_disables_personalization_boosts(monkeypatch):
    monkeypatch.setattr(engine_module, "_get_anchor_service", lambda: _FakeAnchorService())

    engine = ContextEngine(
        research_store=_FakeResearchStore(),
        memory_store=_FakeMemoryStore(),
        paper_store=None,
        search_service=_FakeSearchService(),
        track_router=_FakeTrackRouter(),
        config=ContextEngineConfig(personalized=False, paper_limit=5),
    )

    pack = await engine.build_context_pack(user_id="u1", query="transformer", track_id=1)
    score = float(pack["paper_recommendation_scores"]["1"])

    assert score < 0.30
