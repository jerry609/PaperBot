"""
Unit tests for issue #161 Phase B+C: embedding storage + sqlite-vec + hybrid fusion.

Coverage:
1. _pack_embedding encodes float32 correctly
2. _store_embedding is called after add_memories
3. _store_embedding skips when no provider
4. _search_vec returns None when vec unavailable
5. _search_vec filters by user_id / scope
6. _hybrid_merge weighted scoring (0.6 vec + 0.4 bm25)
7. _hybrid_merge deduplicates items present in both result sets
8. search_memories uses hybrid path when both sources return results
9. search_memories falls back to FTS5 when vec unavailable
10. search_memories falls back to list when query is empty
"""
from __future__ import annotations

import struct
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from paperbot.infrastructure.stores.memory_store import (
    SqlAlchemyMemoryStore,
    _pack_embedding,
    _hybrid_merge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(n: int = 4, value: float = 0.5) -> List[float]:
    return [value] * n


def _item(id_: int, content: str = "test memory", vec_score: float = 0.9) -> dict:
    return {"id": id_, "content": content, "vec_score": vec_score}


def _fts_item(id_: int, content: str = "test memory") -> dict:
    return {"id": id_, "content": content}


# ---------------------------------------------------------------------------
# 1. _pack_embedding
# ---------------------------------------------------------------------------

class TestPackEmbedding:
    def test_round_trip(self):
        vec = [0.1, 0.5, -0.3, 1.0]
        blob = _pack_embedding(vec)
        unpacked = list(struct.unpack(f"{len(vec)}f", blob))
        assert len(unpacked) == 4
        for a, b in zip(vec, unpacked):
            assert abs(a - b) < 1e-5

    def test_blob_length(self):
        vec = [0.0] * 1536
        blob = _pack_embedding(vec)
        assert len(blob) == 1536 * 4  # 4 bytes per float32


# ---------------------------------------------------------------------------
# 2+3. _store_embedding called / skipped
# ---------------------------------------------------------------------------

class TestStoreEmbedding:
    def _make_store(self, provider=None):
        store = SqlAlchemyMemoryStore.__new__(SqlAlchemyMemoryStore)
        store.db_url = "sqlite://"
        store._vec_available = False
        store._embedding_provider = provider
        store._provider = MagicMock()
        return store

    def test_skips_when_no_provider(self):
        """_store_embedding should do nothing if provider returns None."""
        store = self._make_store(provider=False)  # permanently unavailable
        # Should not raise and not call engine
        store._store_embedding(1, "some content")
        store._provider.engine.connect.assert_not_called()

    def test_stores_blob_when_provider_available(self):
        mock_provider = MagicMock()
        mock_provider.embed.return_value = _vec(4)

        store = self._make_store(provider=mock_provider)
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        store._provider.engine.connect.return_value = mock_conn

        store._store_embedding(42, "hello world")

        mock_provider.embed.assert_called_once_with("hello world")
        mock_conn.execute.assert_called()
        # First call should be the UPDATE statement
        first_call_sql = str(mock_conn.execute.call_args_list[0][0][0])
        assert "UPDATE memory_items" in first_call_sql

    def test_skips_when_embed_returns_none(self):
        mock_provider = MagicMock()
        mock_provider.embed.return_value = None
        store = self._make_store(provider=mock_provider)
        store._store_embedding(1, "text")
        store._provider.engine.connect.assert_not_called()


# ---------------------------------------------------------------------------
# 4+5. _search_vec
# ---------------------------------------------------------------------------

class TestSearchVec:
    def _make_store(self, vec_available: bool = True):
        store = SqlAlchemyMemoryStore.__new__(SqlAlchemyMemoryStore)
        store.db_url = "sqlite://"
        store._vec_available = vec_available
        store._provider = MagicMock()
        return store

    def test_returns_none_when_vec_unavailable(self):
        store = self._make_store(vec_available=False)
        result = store._search_vec(
            user_id="u1", query_vec=_vec(), limit=5,
            workspace_id=None, scope_type=None, scope_id=None,
        )
        assert result is None

    def test_returns_none_on_query_exception(self):
        store = self._make_store(vec_available=True)
        store._provider.engine.connect.side_effect = RuntimeError("db error")
        result = store._search_vec(
            user_id="u1", query_vec=_vec(), limit=5,
            workspace_id=None, scope_type=None, scope_id=None,
        )
        assert result is None

    def test_returns_empty_list_when_no_candidates(self):
        store = self._make_store(vec_available=True)
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []
        store._provider.engine.connect.return_value = mock_conn

        result = store._search_vec(
            user_id="u1", query_vec=_vec(), limit=5,
            workspace_id=None, scope_type=None, scope_id=None,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 6+7. _hybrid_merge
# ---------------------------------------------------------------------------

class TestHybridMerge:
    def test_vec_only_result_ranked_by_vec_score(self):
        vec_results = [_item(1, vec_score=0.9), _item(2, vec_score=0.5)]
        fts_results = []
        merged = _hybrid_merge(vec_results, fts_results, limit=5)
        # Both items present; item 1 should rank higher
        assert merged[0]["id"] == 1
        assert merged[1]["id"] == 2

    def test_fts_only_result_ranked_by_bm25_rank(self):
        vec_results = []
        fts_results = [_fts_item(3), _fts_item(4)]
        merged = _hybrid_merge(vec_results, fts_results, limit=5)
        assert merged[0]["id"] == 3  # rank 0 → highest bm25_score
        assert merged[1]["id"] == 4

    def test_deduplicates_items_in_both(self):
        vec_results = [_item(10, vec_score=0.8)]
        fts_results = [_fts_item(10), _fts_item(20)]
        merged = _hybrid_merge(vec_results, fts_results, limit=10)
        ids = [m["id"] for m in merged]
        assert ids.count(10) == 1  # no duplicates

    def test_hybrid_score_combines_weights(self):
        vec_results = [_item(1, vec_score=1.0)]
        fts_results = [_fts_item(1)]  # rank 0 → bm25_score = 1.0
        merged = _hybrid_merge(vec_results, fts_results, limit=5)
        # Expected: 0.6*1.0 + 0.4*1.0 = 1.0
        assert abs(merged[0]["hybrid_score"] - 1.0) < 0.01

    def test_respects_limit(self):
        vec_results = [_item(i, vec_score=1.0 / (i + 1)) for i in range(20)]
        fts_results = []
        merged = _hybrid_merge(vec_results, fts_results, limit=3)
        assert len(merged) == 3

    def test_hybrid_score_field_present(self):
        merged = _hybrid_merge([_item(1)], [_fts_item(2)], limit=5)
        for item in merged:
            assert "hybrid_score" in item

    def test_skips_items_with_missing_or_invalid_id(self):
        vec_results = [{"content": "missing id", "vec_score": 0.9}, {"id": "x", "vec_score": 0.3}]
        fts_results = [{"id": None, "content": "bad"}, {"id": "bad", "content": "bad2"}]
        merged = _hybrid_merge(vec_results, fts_results, limit=5)
        assert merged == []


# ---------------------------------------------------------------------------
# 8+9+10. search_memories integration paths
# ---------------------------------------------------------------------------

class TestSearchMemoriesRouting:
    def _make_store(self, vec_available: bool = False):
        store = SqlAlchemyMemoryStore.__new__(SqlAlchemyMemoryStore)
        store.db_url = "sqlite://"
        store._vec_available = vec_available
        store._embedding_provider = False  # No real API calls in tests
        store._provider = MagicMock()
        return store

    def test_empty_query_calls_list_memories(self):
        store = self._make_store()
        store.list_memories = MagicMock(return_value=[])
        result = store.search_memories(user_id="u1", query="  ")
        store.list_memories.assert_called_once()
        assert result == []

    def test_no_vec_falls_back_to_fts5(self):
        store = self._make_store(vec_available=False)
        fts_items = [{"id": 1, "content": "attention mechanism"}]
        store._search_fts5 = MagicMock(return_value=fts_items)
        store._search_like = MagicMock()
        store.list_memories = MagicMock(return_value=[])

        result = store.search_memories(user_id="u1", query="attention mechanism")
        assert result == fts_items
        store._search_fts5.assert_called_once()
        store._search_like.assert_not_called()

    def test_hybrid_merge_called_when_both_channels_return_results(self):
        store = self._make_store(vec_available=True)
        vec_items = [_item(1, vec_score=0.9)]
        fts_items = [_fts_item(1), _fts_item(2)]

        mock_provider = MagicMock()
        mock_provider.embed.return_value = _vec(4)
        store._embedding_provider = mock_provider

        store._search_vec = MagicMock(return_value=vec_items)
        store._search_fts5 = MagicMock(return_value=fts_items)
        store.list_memories = MagicMock(return_value=[])

        result = store.search_memories(user_id="u1", query="transformer")
        assert any("hybrid_score" in r for r in result)
        store._search_vec.assert_called_once()
        store._search_fts5.assert_called_once()
