from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from paperbot.infrastructure.stores.models import (
    Base,
    PaperFeedbackModel,
    PaperModel,
    ResearchTrackModel,
)
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _insert_paper(session, *, title: str, keywords: list[str], paper_id: int | None = None) -> int:
    now = datetime.now(timezone.utc)
    paper = PaperModel(
        title=title,
        abstract=f"Abstract about {title}",
        title_hash=_sha256(title.lower().strip()),
        keywords_json=json.dumps(keywords),
        fields_of_study_json="[]",
        authors_json='["Author A"]',
        primary_source="test",
        sources_json='["test"]',
        citation_count=10,
        created_at=now,
        updated_at=now,
    )
    session.add(paper)
    session.flush()
    return int(paper.id)


def test_saved_papers_rank_higher_than_skipped(tmp_path: Path):
    """Saved papers should rank higher than skipped papers in the track feed."""
    db_url = f"sqlite:///{tmp_path / 'feed-ranking.db'}"
    store = SqlAlchemyResearchStore(db_url=db_url)

    user_id = "test-user"
    track = store.create_track(
        user_id=user_id,
        name="ML Research",
        keywords=["machine learning"],
        activate=True,
    )
    track_id = int(track["id"])

    now = datetime.now(timezone.utc)
    with store._provider.session() as session:
        saved_pid = _insert_paper(
            session,
            title="Saved Paper on Machine Learning Optimization",
            keywords=["machine learning", "optimization"],
        )
        skipped_pid = _insert_paper(
            session,
            title="Skipped Paper on Machine Learning Inference",
            keywords=["machine learning", "inference"],
        )
        neutral_pid = _insert_paper(
            session,
            title="Neutral Paper on Machine Learning Training",
            keywords=["machine learning", "training"],
        )

        # Add "save" feedback for saved_pid
        session.add(
            PaperFeedbackModel(
                user_id=user_id,
                track_id=track_id,
                paper_id=str(saved_pid),
                paper_ref_id=saved_pid,
                canonical_paper_id=saved_pid,
                action="save",
                weight=0.0,
                ts=now,
                metadata_json="{}",
            )
        )
        # Add "skip" feedback for skipped_pid
        session.add(
            PaperFeedbackModel(
                user_id=user_id,
                track_id=track_id,
                paper_id=str(skipped_pid),
                paper_ref_id=skipped_pid,
                canonical_paper_id=skipped_pid,
                action="skip",
                weight=0.0,
                ts=now,
                metadata_json="{}",
            )
        )
        session.commit()

    result = store.list_track_feed(user_id=user_id, track_id=track_id, limit=10, offset=0)
    items = result["items"]

    assert len(items) >= 2, f"Expected at least 2 items, got {len(items)}"

    scores_by_pid = {}
    for item in items:
        pid = int(item["paper"]["id"])
        scores_by_pid[pid] = item["feed_score"]

    assert saved_pid in scores_by_pid, "Saved paper should appear in feed"
    assert skipped_pid in scores_by_pid, "Skipped paper should appear in feed"

    assert scores_by_pid[saved_pid] > scores_by_pid[skipped_pid], (
        f"Saved paper score ({scores_by_pid[saved_pid]}) should be higher "
        f"than skipped paper score ({scores_by_pid[skipped_pid]})"
    )


def test_track_feed_restores_saved_state_after_preference_is_cleared(tmp_path: Path):
    """Clearing a preference should preserve an independent save state."""
    db_url = f"sqlite:///{tmp_path / 'feed-effective-state.db'}"
    store = SqlAlchemyResearchStore(db_url=db_url)

    user_id = "test-user"
    track = store.create_track(
        user_id=user_id,
        name="ML Research",
        keywords=["machine learning"],
        activate=True,
    )
    track_id = int(track["id"])

    with store._provider.session() as session:
        paper_id = _insert_paper(
            session,
            title="Effective Feedback State Paper",
            keywords=["machine learning", "optimization"],
        )
        session.commit()

    store.add_paper_feedback(
        user_id=user_id,
        track_id=track_id,
        paper_id=str(paper_id),
        action="save",
    )
    store.add_paper_feedback(
        user_id=user_id,
        track_id=track_id,
        paper_id=str(paper_id),
        action="like",
    )
    store.add_paper_feedback(
        user_id=user_id,
        track_id=track_id,
        paper_id=str(paper_id),
        action="unlike",
    )

    result = store.list_track_feed(user_id=user_id, track_id=track_id, limit=10, offset=0)
    item = next(row for row in result["items"] if int(row["paper"]["id"]) == paper_id)

    assert item["is_saved"] is True
    assert item["is_liked"] is False
    assert item["is_disliked"] is False
    assert item["latest_feedback_action"] == "save"


def test_track_feed_keeps_save_and_like_as_independent_flags(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'feed-flags.db'}"
    store = SqlAlchemyResearchStore(db_url=db_url)

    user_id = "stateful-user"
    track = store.create_track(
        user_id=user_id,
        name="Agents",
        keywords=["agent"],
        activate=True,
    )
    track_id = int(track["id"])

    with store._provider.session() as session:
        paper_id = _insert_paper(
            session,
            title="Agent Planning Systems",
            keywords=["agent", "planning"],
        )
        session.commit()

    store.add_paper_feedback(
        user_id=user_id,
        track_id=track_id,
        paper_id=str(paper_id),
        action="save",
        metadata={"title": "Agent Planning Systems"},
    )
    store.add_paper_feedback(
        user_id=user_id,
        track_id=track_id,
        paper_id=str(paper_id),
        action="like",
        metadata={"title": "Agent Planning Systems"},
    )

    first = store.list_track_feed(user_id=user_id, track_id=track_id, limit=10, offset=0)
    item = next(row for row in first["items"] if int(row["paper"]["id"]) == paper_id)
    assert item["is_saved"] is True
    assert item["is_liked"] is True
    assert item["is_disliked"] is False
    assert item["latest_feedback_action"] == "like"

    store.add_paper_feedback(
        user_id=user_id,
        track_id=track_id,
        paper_id=str(paper_id),
        action="unlike",
        metadata={"title": "Agent Planning Systems"},
    )

    second = store.list_track_feed(user_id=user_id, track_id=track_id, limit=10, offset=0)
    item = next(row for row in second["items"] if int(row["paper"]["id"]) == paper_id)
    assert item["is_saved"] is True
    assert item["is_liked"] is False
    assert item["is_disliked"] is False
    assert item["latest_feedback_action"] == "save"
