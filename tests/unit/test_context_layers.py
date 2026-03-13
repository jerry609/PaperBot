"""Tests for #165 — context layered loading."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from paperbot.application.ports.document_intelligence_port import EvidenceHit
from paperbot.context_engine import engine as engine_module
from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate


def _make_engine(
    *,
    memory_store=None,
    research_store=None,
    config=None,
    evidence_retriever=None,
    query_grounder=None,
):
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
    if hasattr(ms.list_memories, "return_value"):
        ms.list_memories.return_value = [
            {
                "id": 1,
                "content": "pref1",
                "confidence": 0.9,
                "created_at": "2025-01-01T00:00:00+00:00",
                "use_count": 0,
            },
        ]
    if hasattr(ms.search_memories, "return_value"):
        ms.search_memories.return_value = []
    if hasattr(ms.search_memories_batch, "return_value"):
        ms.search_memories_batch.return_value = {}
    if hasattr(ms.touch_usage, "return_value"):
        ms.touch_usage.return_value = None

    config = config or ContextEngineConfig(offline=True, paper_limit=0)
    engine = ContextEngine(
        research_store=rs,
        memory_store=ms,
        evidence_retriever=evidence_retriever,
        config=config,
        query_grounder=query_grounder,
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

    def test_layer0_cache_is_user_scoped(self):
        """Cache must not leak data across different user_ids."""
        engine, rs, ms = _make_engine()
        u1_prefs = [{"id": 1, "content": "u1-pref"}]
        u2_prefs = [{"id": 2, "content": "u2-pref"}]

        ms.list_memories.side_effect = lambda **kw: (
            u1_prefs if kw.get("user_id") == "u1" else u2_prefs
        )

        result_u1 = engine._load_layer0_profile("u1")
        result_u2 = engine._load_layer0_profile("u2")

        assert result_u1 == u1_prefs
        assert result_u2 == u2_prefs
        assert ms.list_memories.call_count == 2

        # Subsequent cached calls must still return correct user data.
        assert engine._load_layer0_profile("u1") == u1_prefs
        assert engine._load_layer0_profile("u2") == u2_prefs
        # No additional calls — both served from per-user cache.
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
    async def test_context_pack_includes_evidence_hits_when_retriever_available(self):
        class _FakeEvidenceRetriever:
            def retrieve_evidence(self, *, query, paper_ids=None, limit=6):
                return [
                    EvidenceHit(
                        paper_id=1,
                        chunk_id=10,
                        chunk_index=0,
                        paper_title="Transformer Paper",
                        section="abstract",
                        heading="Abstract",
                        snippet="Transformer retrieval evidence",
                        score=0.9,
                        source_type="paper_metadata",
                        locator_url="https://example.com/paper",
                        metadata={},
                    )
                ]

        async def _fake_search_candidate_papers(*args, **kwargs):
            return object()

        def _fake_search_result_to_candidate_dicts(*args, **kwargs):
            return [
                {
                    "paper_id": "1",
                    "canonical_id": 1,
                    "title": "Transformer Paper",
                    "abstract": (
                        "Transformer retrieval evidence is described in enough detail "
                        "to pass the academic paper filter."
                    ),
                    "year": 2026,
                    "citation_count": 5,
                    "url": "https://example.com/paper",
                }
            ]

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                engine_module, "search_candidate_papers", _fake_search_candidate_papers
            )
            monkeypatch.setattr(
                engine_module,
                "search_result_to_candidate_dicts",
                _fake_search_result_to_candidate_dicts,
            )
            engine, rs, ms = _make_engine(
                evidence_retriever=_FakeEvidenceRetriever(),
                config=ContextEngineConfig(offline=False, paper_limit=2, evidence_limit=3),
            )
            engine.search_service = MagicMock()

            result = await engine.build_context_pack(
                user_id="u1",
                query="transformer retrieval",
            )
        finally:
            monkeypatch.undo()

        assert len(result["evidence_hits"]) == 1
        assert result["evidence_hits"][0]["paper_id"] == 1
        assert result["routing"]["evidence_hit_count"] == 1

    @pytest.mark.asyncio
    async def test_context_pack_uses_grounded_query_for_search_and_routing(self):
        captured_queries: list[str] = []

        class _FakeGrounder:
            def ground_query(self, *, user_id: str, query: str, limit: int = 3):
                return type(
                    "_Grounded",
                    (),
                    {
                        "canonical_query": "retrieval augmented generation latency",
                        "concepts": [object()],
                        "to_dict": lambda self: {
                            "original_query": query,
                            "canonical_query": "retrieval augmented generation latency",
                            "search_queries": [
                                query,
                                "retrieval augmented generation latency",
                            ],
                            "concepts": [{"id": "rag"}],
                        },
                    },
                )()

        async def _fake_search_candidate_papers(service, *, query, **kwargs):
            captured_queries.append(query)
            return object()

        def _fake_search_result_to_candidate_dicts(*args, **kwargs):
            return [
                {
                    "paper_id": "1",
                    "canonical_id": 1,
                    "title": "RAG Latency Study",
                    "abstract": (
                        "Retrieval-augmented generation latency is analyzed with enough detail "
                        "to pass the academic paper filter."
                    ),
                    "year": 2026,
                    "citation_count": 4,
                    "url": "https://example.com/rag-latency",
                }
            ]

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                engine_module, "search_candidate_papers", _fake_search_candidate_papers
            )
            monkeypatch.setattr(
                engine_module,
                "search_result_to_candidate_dicts",
                _fake_search_result_to_candidate_dicts,
            )
            engine, rs, ms = _make_engine(
                query_grounder=_FakeGrounder(),
                config=ContextEngineConfig(offline=False, paper_limit=1),
            )
            engine.search_service = MagicMock()
            rs.get_active_track.return_value = {"id": 7, "name": "RAG Systems"}

            result = await engine.build_context_pack(
                user_id="u1",
                query="rag latency",
            )
        finally:
            monkeypatch.undo()

        assert captured_queries == ["retrieval augmented generation latency"]
        assert result["routing"]["resolved_query"] == "retrieval augmented generation latency"
        assert result["routing"]["routing_query"] == (
            "rag latency retrieval augmented generation latency"
        )
        assert result["routing"]["query_grounding"]["concepts"] == [{"id": "rag"}]
        assert engine.track_router.suggest_track.call_args.kwargs["query"] == (
            "rag latency retrieval augmented generation latency"
        )

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

    @pytest.mark.asyncio
    async def test_cross_track_batch_hits_do_not_manual_touch_usage(self):
        engine, rs, ms = _make_engine()
        rs.get_active_track.return_value = {"id": 1, "name": "main"}
        rs.list_tracks.return_value = [
            {"id": 1, "name": "main"},
            {"id": 2, "name": "other"},
        ]
        ms.search_memories_batch.return_value = {
            "2": [
                {"id": 201, "content": "cross hit", "scope_id": "2"},
            ]
        }

        await engine.build_context_pack(
            user_id="u1",
            query="transformers",
            include_cross_track=True,
        )

        touch_calls = [call.kwargs for call in ms.touch_usage.call_args_list]
        assert not any(201 in kwargs.get("item_ids", []) for kwargs in touch_calls)

    @pytest.mark.asyncio
    async def test_relevant_memory_search_chain_touches_usage_once(self, tmp_path):
        store = SqlAlchemyMemoryStore(
            db_url=f"sqlite:///{tmp_path / 'context_memory_usage.db'}",
            auto_create_schema=True,
        )
        _, _, rows = store.add_memories(
            user_id="u1",
            memories=[
                MemoryCandidate(
                    kind="fact",
                    content="transformer retrieval memory",
                    confidence=0.9,
                    scope_type="track",
                    scope_id="1",
                    tags=["transformer"],
                    evidence={},
                )
            ],
            actor_id="test",
        )

        engine, rs, _ = _make_engine(memory_store=store)
        rs.get_track.return_value = {"id": 1, "name": "main"}

        await engine.build_context_pack(
            user_id="u1",
            query="transformer",
            track_id=1,
        )

        stored = store.get_items_by_ids(user_id="u1", item_ids=[rows[0].id])
        assert len(stored) == 1
        assert stored[0]["use_count"] == 1
        assert stored[0]["last_used_at"] is not None

    @pytest.mark.asyncio
    async def test_context_token_guard_trims_when_budget_exceeded(self):
        cfg = ContextEngineConfig(
            offline=True,
            paper_limit=0,
            context_token_budget=100,
        )
        engine, rs, ms = _make_engine(config=cfg)
        rs.get_active_track.return_value = {"id": 1, "name": "main"}
        ms.search_memories.return_value = [
            {"id": 101, "content": "hit1"},
            {"id": 102, "content": "hit2"},
        ]

        result = await engine.build_context_pack(
            user_id="u1",
            query="transformers",
        )

        total_tokens = sum(result["context_layers"].values())
        assert total_tokens <= 100
        assert result["routing"].get("token_guard", {}).get("enabled") is True
