from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from paperbot.application.services.anchor_service import (
    AnchorService,
    _collapse_effective_feedback_actions,
)
from paperbot.infrastructure.stores.author_store import AuthorStore
from paperbot.infrastructure.stores.models import (
    AuthorModel,
    Base,
    PaperFeedbackModel,
    ResearchTrackModel,
    UserAnchorScoreModel,
)
from paperbot.infrastructure.stores.paper_store import PaperStore
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider


def _seed_track(db_url: str) -> int:
    provider = SessionProvider(db_url)
    Base.metadata.create_all(provider.engine)
    with provider.session() as session:
        track = ResearchTrackModel(
            user_id="anchor-user",
            name="LLM Systems",
            description="",
            keywords_json=json.dumps(["attention", "transformer"]),
            venues_json="[]",
            methods_json="[]",
            is_active=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(track)
        session.commit()
        session.refresh(track)
        return int(track.id)


def test_anchor_service_discovers_and_scores_authors(tmp_path: Path):
    user_id = "anchor-user"
    db_url = f"sqlite:///{tmp_path / 'anchor-service.db'}"
    paper_store = PaperStore(db_url=db_url)
    author_store = AuthorStore(db_url=db_url)
    provider = SessionProvider(db_url)

    track_id = _seed_track(db_url)

    p1 = paper_store.upsert_paper(
        paper={
            "title": "Attention Is All You Need",
            "abstract": "Transformer architecture for sequence modeling.",
            "paper_id": "1706.03762",
            "url": "https://arxiv.org/abs/1706.03762",
            "year": 2017,
            "citation_count": 5000,
            "authors": ["Alice Smith"],
        },
        source_hint="arxiv",
    )
    p2 = paper_store.upsert_paper(
        paper={
            "title": "Scaling Transformer Inference",
            "abstract": "Efficient attention serving system.",
            "paper_id": "2401.00011",
            "url": "https://arxiv.org/abs/2401.00011",
            "year": 2025,
            "citation_count": 120,
            "authors": ["Alice Smith"],
        },
        source_hint="arxiv",
    )
    p3 = paper_store.upsert_paper(
        paper={
            "title": "Database Joins for HTAP",
            "abstract": "Index design for mixed workloads.",
            "paper_id": "2401.00022",
            "url": "https://arxiv.org/abs/2401.00022",
            "year": 2025,
            "citation_count": 900,
            "authors": ["Bob Lee"],
        },
        source_hint="arxiv",
    )

    p4 = paper_store.upsert_paper(
        paper={
            "title": "Collaborative Transformer Systems",
            "abstract": "attention systems co-design.",
            "paper_id": "2402.00033",
            "url": "https://arxiv.org/abs/2402.00033",
            "year": 2025,
            "citation_count": 600,
            "authors": ["Alice Smith", "Carol Chen"],
        },
        source_hint="arxiv",
    )

    author_store.replace_paper_authors(paper_id=int(p1["id"]), authors=["Alice Smith"])
    author_store.replace_paper_authors(paper_id=int(p2["id"]), authors=["Alice Smith"])
    author_store.replace_paper_authors(paper_id=int(p3["id"]), authors=["Bob Lee"])
    author_store.replace_paper_authors(
        paper_id=int(p4["id"]),
        authors=["Alice Smith", "Carol Chen"],
    )

    with provider.session() as session:
        session.add(
            PaperFeedbackModel(
                user_id=user_id,
                track_id=track_id,
                paper_id=str(p2["id"]),
                paper_ref_id=int(p2["id"]),
                canonical_paper_id=int(p2["id"]),
                action="like",
                weight=1.0,
                ts=datetime.now(timezone.utc),
                metadata_json="{}",
            )
        )
        session.commit()

    service = AnchorService(db_url=db_url)
    anchors = service.discover(track_id=track_id, user_id=user_id, limit=5, window_years=15)

    assert len(anchors) >= 2
    assert anchors[0]["name"] == "Alice Smith"
    assert anchors[0]["anchor_level"] in {"core", "active", "emerging"}
    assert anchors[0]["anchor_score"] >= anchors[1]["anchor_score"]
    assert anchors[0]["relevance_score"] > 0
    assert anchors[0]["score_breakdown"]["total"] == anchors[0]["anchor_score"]
    assert anchors[0]["score_breakdown"]["network"] > 0
    assert anchors[0]["evidence_status"] == "ok"
    assert anchors[0]["evidence_papers"]

    global_mode = service.discover(
        track_id=track_id,
        user_id=None,
        limit=5,
        window_years=15,
        personalized=False,
    )
    assert global_mode[0]["score_breakdown"]["personalization"] == 0
    assert anchors[0]["anchor_score"] >= global_mode[0]["anchor_score"]

    with provider.session() as session:
        score_rows = session.execute(select(UserAnchorScoreModel)).scalars().all()
    assert score_rows


def test_anchor_service_raises_for_unknown_track(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'anchor-track-missing.db'}"
    service = AnchorService(db_url=db_url)
    with pytest.raises(ValueError, match="track not found"):
        service.discover(track_id=999, user_id="anchor-user")


def test_collapse_effective_feedback_actions_ignores_toggled_off_state() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        PaperFeedbackModel(
            user_id="u1",
            track_id=1,
            paper_id="42",
            paper_ref_id=42,
            canonical_paper_id=42,
            action="unlike",
            weight=0.0,
            ts=now,
            metadata_json="{}",
        ),
        PaperFeedbackModel(
            user_id="u1",
            track_id=1,
            paper_id="42",
            paper_ref_id=42,
            canonical_paper_id=42,
            action="like",
            weight=0.0,
            ts=now,
            metadata_json="{}",
        ),
        PaperFeedbackModel(
            user_id="u1",
            track_id=1,
            paper_id="84",
            paper_ref_id=84,
            canonical_paper_id=84,
            action="dislike",
            weight=0.0,
            ts=now,
            metadata_json="{}",
        ),
    ]

    assert _collapse_effective_feedback_actions(rows) == ["dislike"]


def test_recompute_author_network_scores_updates_metadata(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'anchor-network-recompute.db'}"
    paper_store = PaperStore(db_url=db_url)
    author_store = AuthorStore(db_url=db_url)
    provider = SessionProvider(db_url)

    p1 = paper_store.upsert_paper(
        paper={
            "title": "Joint Work A",
            "paper_id": "2501.10001",
            "url": "https://arxiv.org/abs/2501.10001",
            "year": 2025,
            "citation_count": 50,
            "authors": ["A", "B"],
        },
        source_hint="arxiv",
    )
    p2 = paper_store.upsert_paper(
        paper={
            "title": "Joint Work B",
            "paper_id": "2501.10002",
            "url": "https://arxiv.org/abs/2501.10002",
            "year": 2025,
            "citation_count": 80,
            "authors": ["B", "C"],
        },
        source_hint="arxiv",
    )

    author_store.replace_paper_authors(paper_id=int(p1["id"]), authors=["A", "B"])
    author_store.replace_paper_authors(paper_id=int(p2["id"]), authors=["B", "C"])

    service = AnchorService(db_url=db_url)
    stats = service.recompute_author_network_scores(window_years=5)
    assert stats["authors"] >= 3
    assert stats["updated"] >= 3

    with provider.session() as session:
        rows = session.execute(select(AuthorModel)).scalars().all()
        found = 0
        for row in rows:
            metadata = json.loads(row.metadata_json or "{}")
            if "network_score" in metadata:
                found += 1
                assert metadata["network_score"] >= 0
        assert found >= 3


def test_cleared_feedback_does_not_contribute_to_anchor_personalization() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        PaperFeedbackModel(
            user_id="anchor-user",
            track_id=1,
            paper_id="paper-1",
            paper_ref_id=1,
            canonical_paper_id=1,
            action="unlike",
            weight=0.0,
            ts=now,
            metadata_json="{}",
        ),
        PaperFeedbackModel(
            user_id="anchor-user",
            track_id=1,
            paper_id="paper-1",
            paper_ref_id=1,
            canonical_paper_id=1,
            action="like",
            weight=0.0,
            ts=now,
            metadata_json="{}",
        ),
    ]

    assert _collapse_effective_feedback_actions(rows) == []
