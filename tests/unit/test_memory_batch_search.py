"""Tests for #164 — cross-track batch search."""
from __future__ import annotations

import pytest

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate


def _store_with_data(tmp_path):
    """Create a store with memories in multiple scopes."""
    db_url = f"sqlite:///{tmp_path / 'batch.db'}"
    store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

    # Track 1 memories
    store.add_memories(
        user_id="u1",
        memories=[
            MemoryCandidate(
                kind="fact",
                content="transformer architecture is great",
                confidence=0.9,
                tags=[],
                evidence={},
            ),
            MemoryCandidate(
                kind="goal",
                content="study attention mechanisms",
                confidence=0.8,
                tags=[],
                evidence={},
            ),
        ],
        scope_type="track",
        scope_id="t1",
        actor_id="test",
    )

    # Track 2 memories
    store.add_memories(
        user_id="u1",
        memories=[
            MemoryCandidate(
                kind="fact",
                content="transformer models for vision tasks",
                confidence=0.85,
                tags=[],
                evidence={},
            ),
        ],
        scope_type="track",
        scope_id="t2",
        actor_id="test",
    )

    # Track 3 — no matching content
    store.add_memories(
        user_id="u1",
        memories=[
            MemoryCandidate(
                kind="fact",
                content="database indexing strategies",
                confidence=0.7,
                tags=[],
                evidence={},
            ),
        ],
        scope_type="track",
        scope_id="t3",
        actor_id="test",
    )

    return store


class TestSearchMemoriesBatch:
    def test_returns_grouped_results(self, tmp_path):
        store = _store_with_data(tmp_path)
        results = store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1", "t2", "t3"],
            scope_type="track",
        )
        assert set(results.keys()) == {"t1", "t2", "t3"}
        assert len(results["t1"]) >= 1
        assert len(results["t2"]) >= 1
        # t3 has no transformer content
        assert len(results["t3"]) == 0

    def test_empty_scope_ids_returns_empty(self, tmp_path):
        store = _store_with_data(tmp_path)
        results = store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=[],
        )
        assert results == {}

    def test_single_scope_matches_search_memories(self, tmp_path):
        store = _store_with_data(tmp_path)
        batch = store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1"],
            scope_type="track",
            limit_per_scope=8,
        )
        single = store.search_memories(
            user_id="u1",
            query="transformer",
            limit=8,
            scope_type="track",
            scope_id="t1",
        )
        # Both should find the same items (ids match).
        batch_ids = {r["id"] for r in batch.get("t1", [])}
        single_ids = {r["id"] for r in single}
        assert batch_ids == single_ids

    def test_respects_limit_per_scope(self, tmp_path):
        store = _store_with_data(tmp_path)
        results = store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1"],
            scope_type="track",
            limit_per_scope=1,
        )
        assert len(results["t1"]) <= 1

    def test_empty_query_returns_empty_lists(self, tmp_path):
        store = _store_with_data(tmp_path)
        results = store.search_memories_batch(
            user_id="u1",
            query="",
            scope_ids=["t1", "t2"],
        )
        assert results == {"t1": [], "t2": []}

    def test_min_score_filters_low_relevance_hits(self, tmp_path):
        store = _store_with_data(tmp_path)
        results = store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1", "t2"],
            min_score=0.95,
        )
        # With default weights and fresh memories, decay_score stays below 0.95.
        assert results == {"t1": [], "t2": []}

    def test_candidate_multiplier_keeps_expected_scope_hits(self, tmp_path):
        store = _store_with_data(tmp_path)
        results = store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1", "t2", "t3"],
            candidate_multiplier=6,
        )
        assert len(results["t1"]) >= 1
        assert len(results["t2"]) >= 1

    def test_batch_passes_scope_ids_to_fts_phase(self, tmp_path, monkeypatch):
        store = _store_with_data(tmp_path)
        captured = {}

        def _fake_fts5(*, scope_ids=None, **_kwargs):
            captured["scope_ids"] = scope_ids
            return []

        monkeypatch.setattr(store, "_search_fts5", _fake_fts5)
        monkeypatch.setattr(store, "_get_embedding_provider", lambda: None)

        store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1", "t2"],
        )

        assert captured["scope_ids"] == ["t1", "t2"]

    def test_batch_passes_scope_ids_to_vector_phase(self, tmp_path, monkeypatch):
        store = _store_with_data(tmp_path)
        captured = {}

        class _FakeProvider:
            def embed(self, _text):
                return [0.1, 0.2, 0.3]

        def _fake_vec(*, scope_ids=None, **_kwargs):
            captured["scope_ids"] = scope_ids
            return []

        monkeypatch.setattr(store, "_get_embedding_provider", lambda: _FakeProvider())
        monkeypatch.setattr(store, "_search_vec", _fake_vec)
        monkeypatch.setattr(store, "_search_fts5", lambda **_kwargs: [])

        store.search_memories_batch(
            user_id="u1",
            query="transformer",
            scope_ids=["t1", "t2"],
        )

        assert captured["scope_ids"] == ["t1", "t2"]
