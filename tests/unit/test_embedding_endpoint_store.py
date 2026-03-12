from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.stores.embedding_endpoint_store import EmbeddingEndpointStore


def test_embedding_endpoint_store_round_trip(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'embedding-endpoint.db'}"
    store = EmbeddingEndpointStore(db_url=db_url)

    saved = store.upsert_config(
        payload={
            "enabled": True,
            "provider": "openai",
            "base_url": "https://embeddings.example/v1",
            "api_key_env": "PAPERBOT_EMBEDDING_API_KEY",
            "api_key": "sk-embedding-1234567890",
            "model": "text-embedding-3-small",
        }
    )

    assert saved["enabled"] is True
    assert saved["provider"] == "openai"
    assert saved["base_url"] == "https://embeddings.example/v1"
    assert saved["api_key"].startswith("***")

    raw = store.get_config(include_secrets=True)
    assert raw is not None
    assert raw["api_key"] == "sk-embedding-1234567890"


def test_embedding_endpoint_store_masked_write_keeps_secret(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'embedding-endpoint-mask.db'}"
    store = EmbeddingEndpointStore(db_url=db_url)

    created = store.upsert_config(
        payload={
            "enabled": True,
            "provider": "openai",
            "api_key": "sk-live-embedding-secret",
            "model": "text-embedding-3-small",
        }
    )

    updated = store.upsert_config(
        payload={
            "enabled": True,
            "api_key": created["api_key"],
            "model": "text-embedding-3-large",
        }
    )

    assert updated["model"] == "text-embedding-3-large"
    raw = store.get_config(include_secrets=True)
    assert raw is not None
    assert raw["api_key"] == "sk-live-embedding-secret"
