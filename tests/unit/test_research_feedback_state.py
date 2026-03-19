from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from paperbot.api import main as api_main
from paperbot.api.auth.dependencies import get_required_user_id
from paperbot.api.routes import research as research_route
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore
from paperbot.infrastructure.stores.models import PaperJudgeScoreModel
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore


def _prepare_feedback_state_db(tmp_path: Path) -> tuple[SqlAlchemyResearchStore, dict, dict]:
    db_path = tmp_path / "feedback-state.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "Transformer Alignment in Practice",
            "abstract": "transformer alignment retrieval analysis",
            "url": "https://example.com/paper",
        }
    )
    track = research_store.create_track(
        user_id="u-feedback",
        name="feedback-track",
        keywords=["transformer"],
        activate=True,
    )
    return research_store, paper, track


def test_feedback_route_returns_effective_action_after_toggle(tmp_path, monkeypatch):
    store, paper, track = _prepare_feedback_state_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", store)

    api_main.app.dependency_overrides[get_required_user_id] = lambda: "u-feedback"
    try:
        with TestClient(api_main.app) as client:
            liked = client.post(
                "/api/research/papers/feedback",
                json={
                    "user_id": "u-feedback",
                    "track_id": int(track["id"]),
                    "paper_id": str(paper["id"]),
                    "action": "like",
                },
            )
            cleared = client.post(
                "/api/research/papers/feedback",
                json={
                    "user_id": "u-feedback",
                    "track_id": int(track["id"]),
                    "paper_id": str(paper["id"]),
                    "action": "unlike",
                },
            )
    finally:
        api_main.app.dependency_overrides.pop(get_required_user_id, None)

    assert liked.status_code == 200
    assert liked.json()["current_action"] == "like"

    assert cleared.status_code == 200
    assert cleared.json()["current_action"] is None


def test_like_keeps_saved_state_while_updating_effective_preference_ids(tmp_path: Path):
    store, paper, track = _prepare_feedback_state_db(tmp_path)
    track_id = int(track["id"])
    paper_id = str(paper["id"])

    store.add_paper_feedback(
        user_id="u-feedback",
        track_id=track_id,
        paper_id=paper_id,
        action="save",
        metadata={"title": paper["title"]},
    )
    saved_before = store.list_saved_papers(user_id="u-feedback", track_id=track_id)
    assert len(saved_before) == 1

    store.add_paper_feedback(
        user_id="u-feedback",
        track_id=track_id,
        paper_id=paper_id,
        action="like",
        metadata={},
    )

    saved_after = store.list_saved_papers(user_id="u-feedback", track_id=track_id)
    assert len(saved_after) == 1
    assert store.list_paper_feedback_ids(
        user_id="u-feedback",
        track_id=track_id,
        action="save",
    ) == {paper_id}
    assert store.list_paper_feedback_ids(
        user_id="u-feedback",
        track_id=track_id,
        action="like",
    ) == {paper_id}


def test_unsave_clears_feed_saved_flags(tmp_path: Path):
    store, paper, track = _prepare_feedback_state_db(tmp_path)
    track_id = int(track["id"])
    paper_id = int(paper["id"])

    store.add_paper_feedback(
        user_id="u-feedback",
        track_id=track_id,
        paper_id=str(paper_id),
        action="save",
        metadata={"title": paper["title"]},
    )
    store.add_paper_feedback(
        user_id="u-feedback",
        track_id=track_id,
        paper_id=str(paper_id),
        action="unsave",
        metadata={},
    )

    feed = store.list_track_feed(user_id="u-feedback", track_id=track_id, limit=10, offset=0)
    item = next(row for row in feed["items"] if int(row["paper"]["id"]) == paper_id)

    assert item["latest_feedback_action"] is None
    assert item["is_saved"] is False


def test_feedback_route_schedules_obsidian_export_for_save_and_unsave(tmp_path: Path, monkeypatch):
    store, paper, track = _prepare_feedback_state_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", store)

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        research_route,
        "_schedule_obsidian_export",
        lambda background_tasks, *, user_id, track_id, for_tracks=False: captured.append(
            {"user_id": user_id, "track_id": track_id, "for_tracks": for_tracks}
        ),
    )

    api_main.app.dependency_overrides[get_required_user_id] = lambda: "u-feedback"
    try:
        with TestClient(api_main.app) as client:
            saved = client.post(
                "/api/research/papers/feedback",
                json={
                    "user_id": "u-feedback",
                    "track_id": int(track["id"]),
                    "paper_id": str(paper["id"]),
                    "action": "save",
                },
            )
            unsaved = client.post(
                "/api/research/papers/feedback",
                json={
                    "user_id": "u-feedback",
                    "track_id": int(track["id"]),
                    "paper_id": str(paper["id"]),
                    "action": "unsave",
                },
            )
    finally:
        api_main.app.dependency_overrides.pop(get_required_user_id, None)

    assert saved.status_code == 200
    assert unsaved.status_code == 200
    assert captured == [
        {"user_id": "u-feedback", "track_id": int(track["id"]), "for_tracks": False},
        {"user_id": "u-feedback", "track_id": int(track["id"]), "for_tracks": False},
    ]


def test_feedback_route_schedules_obsidian_export_using_persisted_track_owner(monkeypatch):
    class _FakeResearchStore:
        def get_active_track(self, *, user_id: str):
            assert user_id == "spoofed-request-user"
            return None

        def add_paper_feedback(self, **kwargs):
            assert kwargs["user_id"] == "spoofed-request-user"
            assert kwargs["track_id"] == 9
            return {"id": 1, "track_id": 9}

        def _normalize_feedback_action(self, action: str) -> str:
            return action

        def _effective_feedback_action(self, action: str):
            return action

        def get_track_by_id(self, *, track_id: int):
            assert track_id == 9
            return {"id": 9, "user_id": "persisted-owner", "name": "Secure Track"}

    monkeypatch.setattr(research_route, "_research_store", _FakeResearchStore())

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        research_route,
        "_schedule_obsidian_export",
        lambda background_tasks, *, user_id, track_id, for_tracks=False: captured.append(
            {"user_id": user_id, "track_id": track_id, "for_tracks": for_tracks}
        ),
    )

    api_main.app.dependency_overrides[get_required_user_id] = lambda: "spoofed-request-user"
    try:
        with TestClient(api_main.app) as client:
            response = client.post(
                "/api/research/papers/feedback",
                json={
                    "user_id": "spoofed-request-user",
                    "track_id": 9,
                    "paper_id": "paper-1",
                    "action": "save",
                },
            )
    finally:
        api_main.app.dependency_overrides.pop(get_required_user_id, None)

    assert response.status_code == 200
    assert captured == [
        {"user_id": "persisted-owner", "track_id": 9, "for_tracks": False},
    ]


def test_global_feedback_without_active_track_is_accepted(monkeypatch):
    class _FakeResearchStore:
        def get_active_track(self, *, user_id: str):
            assert user_id == "u-feedback"
            return None

        def add_paper_feedback(self, **kwargs):
            assert kwargs["user_id"] == "u-feedback"
            assert kwargs["track_id"] is None
            assert kwargs["paper_id"] == "paper-1"
            return {"id": 1, "track_id": None, "paper_id": "paper-1", "action": "save"}

        def _normalize_feedback_action(self, action: str) -> str:
            return action

        def _effective_feedback_action(self, action: str):
            return action

    monkeypatch.setattr(research_route, "_research_store", _FakeResearchStore())

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        research_route,
        "_schedule_obsidian_export_for_track",
        lambda *args, **kwargs: captured.append({"called": True}),
    )

    api_main.app.dependency_overrides[get_required_user_id] = lambda: "u-feedback"
    try:
        with TestClient(api_main.app) as client:
            response = client.post(
                "/api/research/papers/feedback",
                json={
                    "paper_id": "paper-1",
                    "action": "save",
                    "paper_title": "Global Save",
                },
            )
    finally:
        api_main.app.dependency_overrides.pop(get_required_user_id, None)

    assert response.status_code == 200
    assert response.json()["current_action"] == "save"
    assert captured == []


def test_store_allows_global_feedback_without_track(tmp_path: Path):
    store, paper, _ = _prepare_feedback_state_db(tmp_path)
    paper_id = str(paper["id"])

    feedback = store.add_paper_feedback(
        user_id="u-feedback",
        track_id=None,
        paper_id=paper_id,
        action="save",
        metadata={"title": paper["title"]},
    )

    assert feedback is not None
    assert feedback["track_id"] is None

    saved = store.list_saved_papers(user_id="u-feedback", limit=10)
    assert len(saved) == 1
    assert str(saved[0]["paper"]["id"]) == paper_id


def test_store_falls_back_when_legacy_schema_still_requires_track(tmp_path: Path):
    store, paper, _ = _prepare_feedback_state_db(tmp_path)
    paper_id = str(paper["id"])

    with store._provider.engine.begin() as conn:
        conn.execute(text("DROP TABLE paper_feedback"))
        conn.execute(text("""
                CREATE TABLE paper_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(64),
                    track_id INTEGER NOT NULL,
                    paper_id VARCHAR(64),
                    paper_ref_id INTEGER,
                    action VARCHAR(32),
                    canonical_paper_id INTEGER,
                    weight FLOAT,
                    ts DATETIME,
                    metadata_json TEXT,
                    FOREIGN KEY(track_id) REFERENCES research_tracks (id),
                    FOREIGN KEY(paper_ref_id) REFERENCES papers (id),
                    FOREIGN KEY(canonical_paper_id) REFERENCES papers (id)
                )
                """))

    feedback = store.add_paper_feedback(
        user_id="u-feedback",
        track_id=None,
        paper_id=paper_id,
        action="save",
        metadata={"title": paper["title"]},
    )

    assert feedback is not None
    assert feedback["track_id"] is not None

    visible_tracks = store.list_tracks(user_id="u-feedback")
    assert all(track["name"] != "__paperbot_legacy_global_feedback__" for track in visible_tracks)

    saved = store.list_saved_papers(user_id="u-feedback", limit=10)
    assert len(saved) == 1
    assert str(saved[0]["paper"]["id"]) == paper_id


def test_saved_papers_include_provenance_and_workflow_summary(tmp_path: Path):
    store, paper, track = _prepare_feedback_state_db(tmp_path)
    track_id = int(track["id"])
    paper_id = int(paper["id"])

    store.add_paper_feedback(
        user_id="u-feedback",
        track_id=track_id,
        paper_id=str(paper_id),
        action="save",
        metadata={
            "title": paper["title"],
            "import_source": "daily_brief",
        },
    )

    with store._provider.session() as session:
        session.add(
            PaperJudgeScoreModel(
                paper_id=paper_id,
                query="daily-brief",
                overall=4.6,
                relevance=4.5,
                novelty=4.0,
                rigor=4.4,
                impact=4.7,
                clarity=4.8,
                recommendation="must_read",
                one_line_summary="High-signal paper for the active daily brief.",
                judge_model="test-model",
                judge_cost_tier=1,
                scored_at=datetime.now(timezone.utc),
                metadata_json="{}",
            )
        )
        session.commit()

    saved = store.list_saved_papers(user_id="u-feedback", track_id=track_id, limit=10)

    assert len(saved) == 1
    assert saved[0]["track_id"] == track_id
    assert saved[0]["latest_judge"]["one_line_summary"] == (
        "High-signal paper for the active daily brief."
    )
    assert saved[0]["provenance"]["primary"] == "daily_brief"
    assert "Daily Brief" in saved[0]["provenance"]["labels"]
    assert "Workflow reviewed" in saved[0]["provenance"]["labels"]
    assert saved[0]["provenance"]["is_manual"] is False
    assert saved[0]["provenance"]["is_workflow"] is True
