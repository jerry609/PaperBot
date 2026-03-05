"""Tests for #165 — context layered loading."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig


def _make_engine(*, memory_store=None, research_store=None):
    """Create a ContextEngine with faked stores."""
    rs = research_store or MagicMock()
    ms = memory_store or MagicMock()

    # Default return values for research_store methods.
    rs.get_active_track.return_value = None
    rs.get_track.return_value = None
    rs.list_tracks.return_value = []
    rs.list_tasks.return_value = []
    rs.list_milestones.return_value = []
    rs.list_paper_feedback_ids.return_value = set()
    rs.list_paper_feedback.return_value = []
    rs.create_context_run.return_value = {"id": 1}

    # Default return values for memory_store methods.
    ms.list_memories.return_value = [
        {"id": 1, "content": "pref1", "confidence": 0.9, "created_at": "2025-01-01T00:00:00+00:00", "use_count": 0},
    ]
    ms.search_memories.return_value = []
    ms.search_memories_batch.return_value = {}
    ms.touch_usage.return_value = None

    config = ContextEngineConfig(offline=True, paper_limit=0)
    engine = ContextEngine(
        research_store=rs,
        memory_store=ms,
        config=config,
        track_router=MagicMock(),
    )
    return engine, rs, ms


class TestLayer0Cache:
    def test_cache_returns_same_result_on_second_call(self):
        engine, rs, ms = _make_engine()
        prefs1 = engine._load_layer0_profile("u1")
        prefs2 = engine._load_layer0_profile("u1")
        assert prefs1 == prefs2
        # list_memories should only be called once due to cache.
        assert ms.list_memories.call_count == 1

    def test_cache_expires_after_ttl(self):
        engine, rs, ms = _make_engine()
        engine._layer0_ttl = 0.01  # 10ms TTL for test
        engine._load_layer0_profile("u1")
        time.sleep(0.02)
        engine._load_layer0_profile("u1")
        # Should have been called twice (cache expired).
        assert ms.list_memories.call_count == 2


class TestLayer1Track:
    def test_returns_tasks_and_milestones_when_track_present(self):
        engine, rs, ms = _make_engine()
        rs.list_tasks.return_value = [{"id": 1, "title": "task1", "status": "todo"}]
        rs.list_milestones.return_value = [{"id": 1, "name": "m1"}]
        track = {"id": 42, "name": "ML Research"}
        result = engine._load_layer1_track("u1", track)
        assert len(result["tasks"]) == 1
        assert len(result["milestones"]) == 1

    def test_returns_empty_when_no_track(self):
        engine, rs, ms = _make_engine()
        result = engine._load_layer1_track("u1", None)
        assert result["tasks"] == []
        assert result["milestones"] == []


class TestLayer3Paper:
    def test_skips_when_no_paper_id(self):
        engine, rs, ms = _make_engine()
        result = engine._load_layer3_paper("u1", None)
        assert result == []
        ms.list_memories.assert_not_called()

    def test_loads_when_paper_id_present(self):
        engine, rs, ms = _make_engine()
        ms.list_memories.return_value = [{"id": 10, "content": "paper note"}]
        result = engine._load_layer3_paper("u1", "p123")
        assert len(result) == 1
        ms.list_memories.assert_called_once()


class TestBuildContextPackLayers:
    @pytest.mark.asyncio
    async def test_context_layers_in_return(self):
        engine, rs, ms = _make_engine()
        result = await engine.build_context_pack(
            user_id="u1",
            query="transformers",
        )
        assert "context_layers" in result
        layers = result["context_layers"]
        assert "layer0_profile_tokens" in layers
        assert "layer1_track_tokens" in layers
        assert "layer2_query_tokens" in layers
        assert "layer3_paper_tokens" in layers

    @pytest.mark.asyncio
    async def test_no_paper_id_layer3_is_zero(self):
        engine, rs, ms = _make_engine()
        result = await engine.build_context_pack(
            user_id="u1",
            query="transformers",
            paper_id=None,
        )
        assert result["context_layers"]["layer3_paper_tokens"] == 0
        assert result["paper_memories"] == []

    @pytest.mark.asyncio
    async def test_backward_compatible_keys(self):
        """All original return keys must still be present."""
        engine, rs, ms = _make_engine()
        result = await engine.build_context_pack(
            user_id="u1",
            query="test",
        )
        expected_keys = {
            "user_id",
            "context_run_id",
            "routing",
            "user_prefs",
            "active_track",
            "progress_state",
            "relevant_memories",
            "cross_track_memories",
            "paper_memories",
            "paper_recommendations",
            "paper_recommendation_scores",
            "paper_recommendation_reasons",
            "context_layers",
        }
        assert expected_keys.issubset(set(result.keys()))
