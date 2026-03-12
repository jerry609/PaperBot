from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import research as research_route
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore
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

    assert response.status_code == 200
    assert captured == [
        {"user_id": "persisted-owner", "track_id": 9, "for_tracks": False},
    ]
