from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import embedding_settings as embedding_settings_route
from paperbot.infrastructure.stores.embedding_endpoint_store import EmbeddingEndpointStore


def test_embedding_settings_route_exposes_environment_fallback(tmp_path: Path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'embedding-settings-env.db'}"
    store = EmbeddingEndpointStore(db_url=db_url)
    monkeypatch.setattr(embedding_settings_route, "_store", store)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env-embeddings.example/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    with TestClient(api_main.app) as client:
        resp = client.get("/api/embedding-settings")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["effective_source"] == "environment"
    assert payload["environment"]["api_key_present"] is True
    assert payload["environment"]["base_url"] == "https://env-embeddings.example/v1"


def test_embedding_settings_route_save_and_remote_test(tmp_path: Path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'embedding-settings-save.db'}"
    store = EmbeddingEndpointStore(db_url=db_url)
    monkeypatch.setattr(embedding_settings_route, "_store", store)

    class _FakeEmbeddingProvider:
        def __init__(self, config) -> None:
            self.config = config

        def embed(self, text: str):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embedding_settings_route, "OpenAIEmbeddingProvider", _FakeEmbeddingProvider)

    with TestClient(api_main.app) as client:
        saved = client.patch(
            "/api/embedding-settings",
            json={
                "enabled": True,
                "provider": "openai",
                "base_url": "https://embed.example/v1",
                "api_key_env": "PAPERBOT_EMBEDDING_API_KEY",
                "api_key": "sk-embedding-123",
                "model": "text-embedding-3-small",
            },
        )
        assert saved.status_code == 200
        assert saved.json()["item"]["enabled"] is True
        assert saved.json()["effective_source"] == "settings"

        tested = client.post(
            "/api/embedding-settings/test",
            json={
                "enabled": True,
                "provider": "openai",
                "base_url": "https://embed.example/v1",
                "api_key_env": "PAPERBOT_EMBEDDING_API_KEY",
                "api_key": "sk-embedding-123",
                "model": "text-embedding-3-small",
                "remote": True,
            },
        )

    assert tested.status_code == 200
    payload = tested.json()
    assert payload["success"] is True
    assert payload["vector_dim"] == 3
