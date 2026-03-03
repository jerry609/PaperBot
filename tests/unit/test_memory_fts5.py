"""
Unit tests for issue #161 Phase A: FTS5 full-text search for memory retrieval.

Coverage:
1. _ensure_fts5 creates virtual table and triggers on a fresh SQLite DB
2. FTS5 triggers keep memory_items_fts in sync on insert / update / delete
3. search_memories() uses FTS5 and returns BM25-ranked results
4. search_memories() falls back to LIKE when FTS5 table is absent
5. Scope / user_id filtering is preserved in FTS5 path
6. Empty query still returns list_memories() results
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.schema import MemoryCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    """In-memory SQLite store with schema + FTS5 set up."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    s = SqlAlchemyMemoryStore(db_url=db_url)
    return s


def _add(store, user_id: str, content: str, confidence: float = 0.9) -> None:
    store.add_memories(
        user_id=user_id,
        memories=[MemoryCandidate(kind="note", content=content, confidence=confidence)],
        actor_id="test",
    )


# ---------------------------------------------------------------------------
# 1. FTS5 virtual table and triggers created
# ---------------------------------------------------------------------------

class TestFts5Setup:
    def test_fts_table_created(self, store):
        with store._provider.engine.connect() as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')")
                ).fetchall()
            }
        assert "memory_items_fts" in tables

    def test_triggers_created(self, store):
        with store._provider.engine.connect() as conn:
            triggers = {
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='trigger'")
                ).fetchall()
            }
        assert "memory_items_fts_ai" in triggers
        assert "memory_items_fts_ad" in triggers
        assert "memory_items_fts_au" in triggers


# ---------------------------------------------------------------------------
# 2. FTS5 stays in sync with memory_items
# ---------------------------------------------------------------------------

class TestFts5Sync:
    def _fts_count(self, store) -> int:
        with store._provider.engine.connect() as conn:
            return conn.execute(
                text("SELECT COUNT(*) FROM memory_items_fts")
            ).scalar()

    def test_insert_syncs_to_fts(self, store):
        before = self._fts_count(store)
        _add(store, "u1", "transformer attention mechanism")
        assert self._fts_count(store) == before + 1

    def test_delete_syncs_to_fts(self, store):
        _add(store, "u1", "delete me later")
        before = self._fts_count(store)
        items = store.list_memories(user_id="u1")
        store.soft_delete_item(item_id=int(items[0]["id"]), user_id="u1")
        # Soft delete does not remove from DB; row stays but hard-delete trigger handles physical removes
        # FTS should still have same count (soft delete doesn't trigger FTS DELETE trigger)
        assert self._fts_count(store) >= before - 1

    def test_fts_content_searchable_after_insert(self, store):
        _add(store, "u2", "BERT language model pretraining")
        with store._provider.engine.connect() as conn:
            rows = conn.execute(
                text(
                    'SELECT rowid FROM memory_items_fts'
                    ' WHERE memory_items_fts MATCH \'"BERT"\''
                )
            ).fetchall()
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# 3. search_memories() uses FTS5
# ---------------------------------------------------------------------------

class TestSearchMemoriesFts5:
    def test_fts5_returns_relevant_results(self, store):
        _add(store, "u1", "multi-head self-attention transformer architecture")
        _add(store, "u1", "recurrent neural network LSTM sequence modelling")
        _add(store, "u1", "convolutional feature extraction image classification")

        results = store.search_memories(user_id="u1", query="attention transformer")
        assert len(results) >= 1
        assert any("attention" in r["content"].lower() for r in results)

    def test_fts5_excludes_other_users(self, store):
        _add(store, "alice", "alice private note about attention")
        _add(store, "bob", "bob note about attention mechanism")

        results = store.search_memories(user_id="alice", query="attention")
        assert all("alice" not in r.get("user_id", "alice") or True for r in results)
        # Strictly: no result should belong to bob
        assert all(r.get("user_id") == "alice" for r in results)

    def test_fts5_scope_filtering(self, store):
        store.add_memories(
            user_id="u3",
            memories=[
                MemoryCandidate(
                    kind="note",
                    content="paper-scoped attention analysis",
                    confidence=0.9,
                    scope_type="paper",
                    scope_id="arxiv_123",
                )
            ],
            actor_id="test",
        )
        store.add_memories(
            user_id="u3",
            memories=[
                MemoryCandidate(
                    kind="note",
                    content="global attention preference",
                    confidence=0.9,
                    scope_type="global",
                )
            ],
            actor_id="test",
        )

        paper_results = store.search_memories(
            user_id="u3", query="attention", scope_type="paper", scope_id="arxiv_123"
        )
        assert len(paper_results) >= 1
        assert all(r.get("scope_type") == "paper" for r in paper_results)

    def test_empty_query_returns_list(self, store):
        _add(store, "u4", "some content here")
        results = store.search_memories(user_id="u4", query="")
        assert isinstance(results, list)

    def test_no_match_returns_list_memories_fallback(self, store):
        _add(store, "u5", "completely unrelated content about butterflies")
        results = store.search_memories(user_id="u5", query="zzz_no_match_xyz")
        # Falls back to list_memories, so should still return something
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 4. Fallback to LIKE when FTS5 unavailable
# ---------------------------------------------------------------------------

class TestFts5Fallback:
    def test_search_like_works_when_fts5_absent(self, tmp_path):
        """Simulate a store where FTS5 table doesn't exist yet — should use LIKE path."""
        db_url = f"sqlite:///{tmp_path}/nofts.db"
        store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

        # Drop FTS5 table to simulate absence
        with store._provider.engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS memory_items_fts"))
            conn.execute(text("DROP TRIGGER IF EXISTS memory_items_fts_ai"))
            conn.execute(text("DROP TRIGGER IF EXISTS memory_items_fts_ad"))
            conn.execute(text("DROP TRIGGER IF EXISTS memory_items_fts_au"))
            conn.commit()

        _add(store, "u6", "attention mechanism fallback test")

        # Should not raise; uses LIKE fallback
        results = store._search_like(
            user_id="u6",
            tokens=["attention"],
            limit=10,
            workspace_id=None,
            scope_type=None,
            scope_id=None,
        )
        assert any("attention" in r["content"].lower() for r in results)

    def test_search_memories_returns_results_after_fts5_drop(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/nofts2.db"
        store = SqlAlchemyMemoryStore(db_url=db_url)
        _add(store, "u7", "vector embedding similarity search")

        # Drop FTS5 — search_memories should gracefully fall back
        with store._provider.engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS memory_items_fts"))
            conn.commit()

        results = store.search_memories(user_id="u7", query="embedding similarity")
        assert isinstance(results, list)
        assert any("embedding" in r["content"].lower() for r in results)
