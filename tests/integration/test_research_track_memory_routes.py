from __future__ import annotations

from fastapi.testclient import TestClient

from paperbot.api.main import app
from paperbot.api.routes import research as research_route
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.memory.schema import MemoryCandidate


def _prepare_memory_routes(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'track-memory.db'}"
    research_store = SqlAlchemyResearchStore(db_url=db_url, auto_create_schema=True)
    memory_store = SqlAlchemyMemoryStore(db_url=db_url)

    active_track = research_store.create_track(user_id="u-memory", name="Active Track", activate=True)
    other_track = research_store.create_track(user_id="u-memory", name="Other Track", activate=False)

    memory_store.add_memories(
        user_id="u-memory",
        memories=[
            MemoryCandidate(
                kind="note",
                content="Pending memory for active track",
                confidence=0.4,
                tags=["active"],
                scope_type="track",
                scope_id=str(active_track["id"]),
                status="pending",
            ),
            MemoryCandidate(
                kind="note",
                content="Pending memory for other track",
                confidence=0.4,
                tags=["other"],
                scope_type="track",
                scope_id=str(other_track["id"]),
                status="pending",
            ),
        ],
    )

    return research_store, memory_store, int(active_track["id"])


def test_track_memory_inbox_uses_active_track_scope(tmp_path, monkeypatch):
    research_store, memory_store, active_track_id = _prepare_memory_routes(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_memory_store", memory_store)

    with TestClient(app) as client:
        response = client.get("/api/research/memory/inbox", params={"user_id": "u-memory"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["scope_id"] == str(active_track_id)
    assert payload["items"][0]["content"] == "Pending memory for active track"


def test_track_memory_inbox_returns_404_without_active_track(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'track-memory-empty.db'}"
    research_store = SqlAlchemyResearchStore(db_url=db_url, auto_create_schema=True)
    memory_store = SqlAlchemyMemoryStore(db_url=db_url)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_memory_store", memory_store)

    with TestClient(app) as client:
        response = client.get("/api/research/memory/inbox", params={"user_id": "missing-user"})

    assert response.status_code == 404
    assert response.json()["detail"] == "No active track for user"


def test_clear_track_memory_returns_404_for_missing_or_inaccessible_track(tmp_path, monkeypatch):
    research_store, memory_store, active_track_id = _prepare_memory_routes(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_memory_store", memory_store)

    with TestClient(app) as client:
        missing = client.post(
            "/api/research/tracks/99999/memory/clear",
            params={"user_id": "u-memory", "confirm": True},
        )
        wrong_user = client.post(
            f"/api/research/tracks/{active_track_id}/memory/clear",
            params={"user_id": "other-user", "confirm": True},
        )

    assert missing.status_code == 404
    assert wrong_user.status_code == 404
