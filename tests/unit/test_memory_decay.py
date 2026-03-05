"""Tests for #163 — memory decay mechanism."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import exp, log

import pytest

from paperbot.infrastructure.stores.memory_store import (
    SqlAlchemyMemoryStore,
    _apply_decay_ranking,
    _decay_score,
    _is_evergreen_memory,
    _to_decay_lambda,
)
from paperbot.memory.schema import MemoryCandidate


def _make_item(
    *,
    confidence: float = 0.8,
    created_at: str | None = None,
    use_count: int = 0,
    scope_type: str = "track",
    kind: str = "fact",
) -> dict:
    now = datetime.now(timezone.utc)
    if created_at is None:
        created_at = now.isoformat()
    return {
        "id": 1,
        "confidence": confidence,
        "created_at": created_at,
        "use_count": use_count,
        "scope_type": scope_type,
        "kind": kind,
    }


class TestToDecayLambda:
    """Verify decay lambda follows ln(2)/halfLifeDays (aligned with OpenClaw)."""

    def test_lambda_formula(self):
        assert abs(_to_decay_lambda(30) - log(2) / 30) < 1e-12

    def test_half_life_produces_exact_half(self):
        lam = _to_decay_lambda(30)
        assert abs(exp(-lam * 30) - 0.5) < 1e-12

    def test_zero_half_life_returns_zero(self):
        assert _to_decay_lambda(0) == 0.0

    def test_negative_half_life_returns_zero(self):
        assert _to_decay_lambda(-10) == 0.0


class TestEvergreenMemory:
    """Evergreen memories (global scope, preference kind) skip recency decay."""

    def test_global_scope_is_evergreen(self):
        assert _is_evergreen_memory({"scope_type": "global", "kind": "fact"})

    def test_preference_kind_is_evergreen(self):
        assert _is_evergreen_memory({"scope_type": "track", "kind": "preference"})

    def test_track_fact_is_not_evergreen(self):
        assert not _is_evergreen_memory({"scope_type": "track", "kind": "fact"})


class TestDecayScore:
    def test_fresh_high_confidence_scores_high(self):
        now = datetime.now(timezone.utc)
        item = _make_item(confidence=0.9, created_at=now.isoformat(), use_count=5)
        score = _decay_score(item, now=now)
        # relevance=0.9*0.7 + recency=1.0*0.2 + usage=0.5*0.1 = 0.63+0.2+0.05 = 0.88
        assert abs(score - 0.88) < 0.01

    def test_old_item_has_lower_recency(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        item = _make_item(confidence=0.9, created_at=old.isoformat(), use_count=0)
        score = _decay_score(item, now=now)
        # λ = ln2/30, recency = exp(-λ*60) = exp(-2*ln2) = 0.25
        # total = 0.63 + 0.25*0.2 + 0 = 0.68
        assert score < 0.70

    def test_at_half_life_recency_is_half(self):
        """At exactly half_life_days, recency multiplier should be ~0.5."""
        now = datetime.now(timezone.utc)
        item = _make_item(
            confidence=0.0,
            created_at=(now - timedelta(days=30)).isoformat(),
            use_count=0,
        )
        score = _decay_score(item, now=now, half_life_days=30)
        # relevance=0, recency=0.5*0.2=0.1, usage=0
        assert abs(score - 0.1) < 0.01

    def test_high_usage_boosts_score(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
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

    def test_evergreen_memory_ignores_age(self):
        """Global/preference memories should get recency=1.0 regardless of age."""
        now = datetime.now(timezone.utc)
        very_old = now - timedelta(days=3650)
        global_item = _make_item(
            confidence=0.8,
            created_at=very_old.isoformat(),
            use_count=0,
            scope_type="global",
        )
        track_item = _make_item(
            confidence=0.8,
            created_at=very_old.isoformat(),
            use_count=0,
            scope_type="track",
        )
        global_score = _decay_score(global_item, now=now)
        track_score = _decay_score(track_item, now=now)
        # Global should score higher because it's immune to decay.
        assert global_score > track_score
        # Global recency component is full 0.2.
        assert abs(global_score - (0.8 * 0.7 + 1.0 * 0.2 + 0.0)) < 0.01


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
