"""Tests for #163 — memory decay mechanism."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paperbot.infrastructure.stores.memory_store import (
    SqlAlchemyMemoryStore,
    _apply_decay_ranking,
    _decay_score,
)
from paperbot.memory.schema import MemoryCandidate


def _make_item(
    *,
    confidence: float = 0.8,
    created_at: str | None = None,
    use_count: int = 0,
) -> dict:
    now = datetime.now(timezone.utc)
    if created_at is None:
        created_at = now.isoformat()
    return {
        "id": 1,
        "confidence": confidence,
        "created_at": created_at,
        "use_count": use_count,
    }


class TestDecayScore:
    def test_fresh_high_confidence_scores_high(self):
        now = datetime.now(timezone.utc)
        item = _make_item(confidence=0.9, created_at=now.isoformat(), use_count=5)
        score = _decay_score(item, now=now)
        # relevance=0.9*0.7 + recency=1.0*0.2 + usage=0.5*0.1 = 0.63+0.2+0.05 = 0.88
        assert abs(score - 0.88) < 0.01

    def test_old_item_has_lower_recency(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=180)
        item = _make_item(confidence=0.9, created_at=old.isoformat(), use_count=0)
        score = _decay_score(item, now=now)
        # recency = exp(-180/90) = exp(-2) ≈ 0.135
        # total ≈ 0.63 + 0.135*0.2 + 0 ≈ 0.657
        assert score < 0.70

    def test_high_usage_boosts_score(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=180)
        low_use = _make_item(confidence=0.5, created_at=old.isoformat(), use_count=0)
        high_use = _make_item(confidence=0.5, created_at=old.isoformat(), use_count=10)
        score_low = _decay_score(low_use, now=now)
        score_high = _decay_score(high_use, now=now)
        assert score_high > score_low

    def test_use_count_caps_at_ten(self):
        now = datetime.now(timezone.utc)
        item10 = _make_item(use_count=10)
        item20 = _make_item(use_count=20)
        s10 = _decay_score(item10, now=now)
        s20 = _decay_score(item20, now=now)
        assert abs(s10 - s20) < 0.001  # both capped at 1.0


class TestApplyDecayRanking:
    def test_reranks_by_decay_score(self):
        now = datetime.now(timezone.utc)
        fresh = _make_item(confidence=0.9, created_at=now.isoformat(), use_count=5)
        fresh["id"] = 1
        old = _make_item(
            confidence=0.3,
            created_at=(now - timedelta(days=300)).isoformat(),
            use_count=0,
        )
        old["id"] = 2
        results = _apply_decay_ranking([old, fresh], now=now)
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert "decay_score" in results[0]

    def test_empty_list_returns_empty(self):
        assert _apply_decay_ranking([]) == []


class TestExpiresAtDefault:
    def test_add_memories_sets_expires_at(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'decay.db'}"
        store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

        candidate = MemoryCandidate(
            kind="fact",
            content="test memory",
            confidence=0.9,
            tags=[],
            evidence={},
        )
        created, skipped, rows = store.add_memories(
            user_id="u1",
            memories=[candidate],
            actor_id="test",
        )
        assert created == 1
        assert rows[0].expires_at is not None
        # Verify expires_at is roughly 365 days from created_at.
        delta = rows[0].expires_at - rows[0].created_at
        assert 364 <= delta.days <= 366

    def test_expired_memories_excluded_from_list(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'decay2.db'}"
        store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

        candidate = MemoryCandidate(
            kind="fact",
            content="old memory",
            confidence=0.9,
            tags=[],
            evidence={},
        )
        _, _, rows = store.add_memories(
            user_id="u1",
            memories=[candidate],
            actor_id="test",
        )
        # Manually set expires_at to the past.
        from sqlalchemy import text as sa_text

        with store._provider.engine.connect() as conn:
            conn.execute(
                sa_text(
                    "UPDATE memory_items SET expires_at = :ea WHERE id = :rid"
                ),
                {"ea": datetime(2020, 1, 1, tzinfo=timezone.utc), "rid": rows[0].id},
            )
            conn.commit()

        results = store.list_memories(user_id="u1", limit=10)
        assert len(results) == 0


class TestSearchAutoTouchUsage:
    def test_search_updates_last_used_at(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'decay3.db'}"
        store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

        candidate = MemoryCandidate(
            kind="fact",
            content="machine learning transformer architecture",
            confidence=0.9,
            tags=["ml"],
            evidence={},
        )
        store.add_memories(user_id="u1", memories=[candidate], actor_id="test")

        # Search should auto-update usage.
        hits = store.search_memories(user_id="u1", query="transformer")
        assert len(hits) >= 1

        # Verify use_count was incremented.
        items = store.list_memories(user_id="u1", limit=10)
        assert items[0]["use_count"] >= 1
        assert items[0]["last_used_at"] is not None
