from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from paperbot.api.main import app
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore


@pytest.fixture
def client_with_store(tmp_path, monkeypatch):
    """Create test client with isolated database."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqlAlchemyResearchStore(db_url=db_url, auto_create_schema=True)

    # Monkeypatch the global store in the routes module
    import paperbot.api.routes.research as research_module

    monkeypatch.setattr(research_module, "_research_store", store)

    return TestClient(app), store


def test_patch_track_success(client_with_store):
    """Test PATCH returns 200 with updated track."""
    client, store = client_with_store

    track = store.create_track(user_id="test", name="Original", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test", json={"name": "Updated"}
    )

    assert response.status_code == 200
    assert response.json()["track"]["name"] == "Updated"


def test_patch_track_not_found(client_with_store):
    """Test PATCH returns 404 for non-existent track."""
    client, _ = client_with_store

    response = client.patch(
        "/api/research/tracks/99999?user_id=test", json={"name": "Test"}
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_patch_track_duplicate_name(client_with_store):
    """Test PATCH returns 409 when name conflicts."""
    client, store = client_with_store

    store.create_track(user_id="test", name="Track A", activate=False)
    track_b = store.create_track(user_id="test", name="Track B", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track_b['id']}?user_id=test", json={"name": "Track A"}
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


def test_patch_track_no_fields(client_with_store):
    """Test PATCH returns 400 when no fields provided."""
    client, store = client_with_store

    track = store.create_track(user_id="test", name="Test", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test", json={}
    )

    assert response.status_code == 400
    assert "no fields" in response.json()["detail"].lower()


def test_patch_track_partial_update(client_with_store):
    """Test PATCH updates only specified fields."""
    client, store = client_with_store

    track = store.create_track(
        user_id="test",
        name="Original Name",
        description="Original Description",
        keywords=["keyword1", "keyword2"],
        activate=False,
    )

    # Update only name
    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test",
        json={"name": "New Name"},
    )

    assert response.status_code == 200
    result = response.json()["track"]
    assert result["name"] == "New Name"
    # Other fields should remain unchanged
    assert result["description"] == "Original Description"
    assert result["keywords"] == ["keyword1", "keyword2"]


def test_patch_track_update_description(client_with_store):
    """Test PATCH can update description."""
    client, store = client_with_store

    track = store.create_track(user_id="test", name="Test Track", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test",
        json={"description": "New description"},
    )

    assert response.status_code == 200
    assert response.json()["track"]["description"] == "New description"


def test_patch_track_update_keywords(client_with_store):
    """Test PATCH can update keywords list."""
    client, store = client_with_store

    track = store.create_track(user_id="test", name="Test Track", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test",
        json={"keywords": ["ml", "nlp", "transformers"]},
    )

    assert response.status_code == 200
    assert response.json()["track"]["keywords"] == ["ml", "nlp", "transformers"]


def test_patch_track_update_multiple_fields(client_with_store):
    """Test PATCH can update multiple fields at once."""
    client, store = client_with_store

    track = store.create_track(user_id="test", name="Original", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test",
        json={
            "name": "Updated Name",
            "description": "Updated Description",
            "keywords": ["new", "keywords"],
        },
    )

    assert response.status_code == 200
    result = response.json()["track"]
    assert result["name"] == "Updated Name"
    assert result["description"] == "Updated Description"
    assert result["keywords"] == ["new", "keywords"]


def test_patch_track_wrong_user(client_with_store):
    """Test PATCH returns 404 for track owned by different user."""
    client, store = client_with_store

    track = store.create_track(user_id="user1", name="Test Track", activate=False)

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=user2",
        json={"name": "New Name"},
    )

    assert response.status_code == 404


def test_patch_track_default_user(client_with_store):
    """Test PATCH uses default user_id when not specified."""
    client, store = client_with_store

    track = store.create_track(user_id="default", name="Test Track", activate=False)

    # Call without user_id parameter (should default to "default")
    response = client.patch(
        f"/api/research/tracks/{track['id']}",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    assert response.json()["track"]["name"] == "Updated Name"


def test_create_track_schedules_obsidian_export(client_with_store, monkeypatch):
    """Track creation should trigger the Obsidian export hook for MOC/note bootstrap."""
    client, _ = client_with_store

    import paperbot.api.routes.research as research_module

    captured: list[dict[str, object]] = []

    monkeypatch.setattr(research_module, "_schedule_embedding_precompute", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        research_module,
        "_schedule_obsidian_export",
        lambda background_tasks, *, user_id, track_id, for_tracks=False: captured.append(
            {"user_id": user_id, "track_id": track_id, "for_tracks": for_tracks}
        ),
    )

    response = client.post(
        "/api/research/tracks",
        json={
            "user_id": "test",
            "name": "Obsidian Sync Track",
            "keywords": ["obsidian", "knowledge-base"],
            "activate": False,
        },
    )

    assert response.status_code == 200
    track_id = int(response.json()["track"]["id"])
    assert captured == [{"user_id": "test", "track_id": track_id, "for_tracks": True}]


def test_patch_track_schedules_obsidian_export(client_with_store, monkeypatch):
    """Track updates should refresh the exported track snapshot."""
    client, store = client_with_store

    import paperbot.api.routes.research as research_module

    track = store.create_track(user_id="test", name="Original", activate=False)
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(research_module, "_schedule_embedding_precompute", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        research_module,
        "_schedule_obsidian_export",
        lambda background_tasks, *, user_id, track_id, for_tracks=False: captured.append(
            {"user_id": user_id, "track_id": track_id, "for_tracks": for_tracks}
        ),
    )

    response = client.patch(
        f"/api/research/tracks/{track['id']}?user_id=test",
        json={"keywords": ["obsidian", "moc"]},
    )

    assert response.status_code == 200
    assert captured == [{"user_id": "test", "track_id": int(track["id"]), "for_tracks": True}]


def test_create_track_schedules_obsidian_export_using_persisted_track_owner(monkeypatch):
    import paperbot.api.routes.research as research_module

    class _FakeResearchStore:
        def create_track(self, **kwargs):
            return {
                "id": 41,
                "user_id": "persisted-owner",
                "name": kwargs["name"],
                "description": kwargs.get("description", ""),
                "keywords": kwargs.get("keywords") or [],
                "methods": kwargs.get("methods") or [],
                "venues": kwargs.get("venues") or [],
                "is_active": False,
            }

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(research_module, "_research_store", _FakeResearchStore())
    monkeypatch.setattr(
        research_module,
        "_schedule_embedding_precompute",
        lambda background_tasks, *, user_id, track_ids: captured.append(
            {"kind": "embedding", "user_id": user_id, "track_ids": track_ids}
        ),
    )
    monkeypatch.setattr(
        research_module,
        "_schedule_obsidian_export",
        lambda background_tasks, *, user_id, track_id, for_tracks=False: captured.append(
            {"kind": "obsidian", "user_id": user_id, "track_id": track_id, "for_tracks": for_tracks}
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/research/tracks",
            json={
                "user_id": "spoofed-request-user",
                "name": "Obsidian Sync Track",
                "keywords": ["obsidian"],
                "activate": False,
            },
        )

    assert response.status_code == 200
    assert captured == [
        {"kind": "embedding", "user_id": "persisted-owner", "track_ids": [41]},
        {"kind": "obsidian", "user_id": "persisted-owner", "track_id": 41, "for_tracks": True},
    ]
