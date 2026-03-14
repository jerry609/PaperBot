from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from paperbot.domain.identity import PaperIdentity
from paperbot.infrastructure.stores.models import PaperJudgeScoreModel, PaperReadingStatusModel
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.infrastructure.stores.identity_store import IdentityStore


def _judged_report():
    return {
        "title": "Daily",
        "date": "2026-02-10",
        "generated_at": "2026-02-10T00:00:00+00:00",
        "source": "papers.cool",
        "sources": ["papers_cool"],
        "queries": [
            {
                "raw_query": "ICL压缩",
                "normalized_query": "icl compression",
                "top_items": [
                    {
                        "title": "UniICL",
                        "url": "https://arxiv.org/abs/2501.12345",
                        "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
                        "authors": ["A"],
                        "snippet": "compress context",
                        "judge": {
                            "overall": 4.2,
                            "recommendation": "must_read",
                            "one_line_summary": "good",
                            "judge_model": "fake",
                            "judge_cost_tier": 1,
                            "relevance": {"score": 5},
                            "novelty": {"score": 4},
                            "rigor": {"score": 4},
                            "impact": {"score": 4},
                            "clarity": {"score": 4},
                        },
                    }
                ],
            }
        ],
        "global_top": [],
    }


def test_upsert_judge_scores_from_report_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "judge-registry.db"
    store = SqlAlchemyPaperStore(db_url=f"sqlite:///{db_path}")

    report = _judged_report()
    first = store.upsert_judge_scores_from_report(report)
    second = store.upsert_judge_scores_from_report(report)

    assert first == {"total": 1, "created": 1, "updated": 0}
    assert second == {"total": 1, "created": 0, "updated": 1}

    with store._provider.session() as session:
        rows = session.execute(select(PaperJudgeScoreModel)).scalars().all()
        assert len(rows) == 1
        assert rows[0].query == "icl compression"
        assert float(rows[0].overall) == 4.2


def test_feedback_links_to_paper_registry_row(tmp_path: Path):
    db_path = tmp_path / "feedback-link.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "UniICL",
            "url": "https://arxiv.org/abs/2501.12345",
            "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
        }
    )

    track = research_store.create_track(user_id="judge-user", name="t1", activate=True)
    feedback = research_store.add_paper_feedback(
        user_id="judge-user",
        track_id=int(track["id"]),
        paper_id="https://arxiv.org/abs/2501.12345",
        action="save",
        metadata={"url": "https://arxiv.org/abs/2501.12345", "title": "UniICL"},
    )

    assert feedback is not None
    assert feedback["paper_ref_id"] == int(paper["id"])


def test_saved_list_and_detail_from_research_store(tmp_path: Path):
    db_path = tmp_path / "saved-detail.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "UniICL",
            "url": "https://arxiv.org/abs/2501.12345",
            "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
        }
    )

    track = research_store.create_track(user_id="u1", name="track-u1", activate=True)
    feedback = research_store.add_paper_feedback(
        user_id="u1",
        track_id=int(track["id"]),
        paper_id=str(paper["id"]),
        action="save",
        metadata={"title": "UniICL"},
    )
    assert feedback and feedback["paper_ref_id"] == int(paper["id"])

    status = research_store.set_paper_reading_status(
        user_id="u1",
        paper_id=str(paper["id"]),
        status="read",
        mark_saved=True,
    )
    assert status is not None
    assert status["status"] == "read"

    saved = research_store.list_saved_papers(user_id="u1", limit=10)
    assert len(saved) == 1
    assert saved[0]["paper"]["title"] == "UniICL"

    detail = research_store.get_paper_detail(user_id="u1", paper_id=str(paper["id"]))
    assert detail is not None
    assert detail["paper"]["title"] == "UniICL"
    assert detail["reading_status"]["status"] == "read"


def test_unsave_removes_saved_state_from_library(tmp_path: Path):
    db_path = tmp_path / "saved-unsave.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "Toggle Save Paper",
            "url": "https://example.com/toggle-save",
            "pdf_url": "https://example.com/toggle-save.pdf",
        }
    )

    track = research_store.create_track(user_id="u-save", name="track-save", activate=True)
    research_store.add_paper_feedback(
        user_id="u-save",
        track_id=int(track["id"]),
        paper_id=str(paper["id"]),
        action="save",
        metadata={"title": "Toggle Save Paper"},
    )

    assert len(research_store.list_saved_papers(user_id="u-save", limit=10)) == 1

    research_store.add_paper_feedback(
        user_id="u-save",
        track_id=int(track["id"]),
        paper_id=str(paper["id"]),
        action="unsave",
        metadata={"title": "Toggle Save Paper"},
    )

    assert research_store.list_saved_papers(user_id="u-save", limit=10) == []

    with research_store._provider.session() as session:
        status = session.execute(
            select(PaperReadingStatusModel).where(
                PaperReadingStatusModel.user_id == "u-save",
                PaperReadingStatusModel.paper_id == int(paper["id"]),
            )
        ).scalar_one_or_none()

    assert status is not None
    assert status.saved_at is None


def test_feedback_ids_follow_effective_toggle_state(tmp_path: Path):
    db_path = tmp_path / "feedback-effective-state.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "Toggle Reaction Paper",
            "url": "https://example.com/toggle-reaction",
            "pdf_url": "https://example.com/toggle-reaction.pdf",
        }
    )

    track = research_store.create_track(user_id="u-react", name="track-react", activate=True)
    track_id = int(track["id"])
    paper_id = str(paper["id"])

    research_store.add_paper_feedback(
        user_id="u-react",
        track_id=track_id,
        paper_id=paper_id,
        action="like",
        metadata={"title": "Toggle Reaction Paper"},
    )
    assert research_store.list_paper_feedback_ids(
        user_id="u-react", track_id=track_id, action="like"
    ) == {paper_id}

    research_store.add_paper_feedback(
        user_id="u-react",
        track_id=track_id,
        paper_id=paper_id,
        action="unlike",
        metadata={"title": "Toggle Reaction Paper"},
    )
    assert (
        research_store.list_paper_feedback_ids(user_id="u-react", track_id=track_id, action="like")
        == set()
    )

    research_store.add_paper_feedback(
        user_id="u-react",
        track_id=track_id,
        paper_id=paper_id,
        action="dislike",
        metadata={"title": "Toggle Reaction Paper"},
    )
    assert research_store.list_paper_feedback_ids(
        user_id="u-react", track_id=track_id, action="dislike"
    ) == {paper_id}
    assert (
        research_store.list_paper_feedback_ids(user_id="u-react", track_id=track_id, action="like")
        == set()
    )

    research_store.add_paper_feedback(
        user_id="u-react",
        track_id=track_id,
        paper_id=paper_id,
        action="undislike",
        metadata={"title": "Toggle Reaction Paper"},
    )
    assert (
        research_store.list_paper_feedback_ids(
            user_id="u-react", track_id=track_id, action="dislike"
        )
        == set()
    )


def test_feedback_resolves_via_identity_store_mapping(tmp_path: Path):
    db_path = tmp_path / "identity-feedback.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)
    identity_store = IdentityStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "CrossSource Paper",
            "url": "https://example.com/p/abc",
            "pdf_url": "https://example.com/p/abc.pdf",
        }
    )

    identity_store.upsert_identity(
        paper_id=int(paper["id"]),
        identity=PaperIdentity(source="papers_cool", external_id="pc:abc123"),
    )

    track = research_store.create_track(user_id="u2", name="track-u2", activate=True)
    feedback = research_store.add_paper_feedback(
        user_id="u2",
        track_id=int(track["id"]),
        paper_id="pc:abc123",
        action="save",
        metadata={},
    )

    assert feedback is not None
    assert feedback["paper_ref_id"] == int(paper["id"])


def test_get_latest_judge_scores_returns_latest_row_per_paper(tmp_path: Path):
    db_path = tmp_path / "judge-latest.db"
    store = SqlAlchemyPaperStore(db_url=f"sqlite:///{db_path}")

    p1 = store.upsert_paper(
        paper={
            "title": "P1",
            "url": "https://example.com/p1",
            "pdf_url": "https://example.com/p1.pdf",
        }
    )
    p2 = store.upsert_paper(
        paper={
            "title": "P2",
            "url": "https://example.com/p2",
            "pdf_url": "https://example.com/p2.pdf",
        }
    )

    now = datetime.now(timezone.utc)
    with store._provider.session() as session:
        session.add(
            PaperJudgeScoreModel(
                paper_id=int(p1["id"]),
                query="q-old",
                overall=3.1,
                recommendation="worth_reading",
                one_line_summary="old",
                judge_model="m1",
                judge_cost_tier=1,
                scored_at=now - timedelta(days=1),
            )
        )
        session.add(
            PaperJudgeScoreModel(
                paper_id=int(p1["id"]),
                query="q-new",
                overall=4.7,
                recommendation="must_read",
                one_line_summary="new",
                judge_model="m2",
                judge_cost_tier=1,
                scored_at=now,
            )
        )
        session.add(
            PaperJudgeScoreModel(
                paper_id=int(p2["id"]),
                query="q-only",
                overall=4.0,
                recommendation="worth_reading",
                one_line_summary="only",
                judge_model="m3",
                judge_cost_tier=1,
                scored_at=now,
            )
        )
        session.commit()

    latest = store.get_latest_judge_scores([int(p1["id"]), int(p2["id"]), -1])

    assert set(latest.keys()) == {int(p1["id"]), int(p2["id"])}
    assert latest[int(p1["id"])]["overall"] == 4.7
    assert latest[int(p1["id"])]["one_line_summary"] == "new"
    assert latest[int(p2["id"])]["judge_model"] == "m3"
