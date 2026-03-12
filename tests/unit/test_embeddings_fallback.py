from __future__ import annotations

from paperbot.context_engine.embeddings import (
    EmbeddingConfig,
    HashEmbeddingProvider,
    try_build_default_embedding_provider,
)


def test_hash_fallback_provider_available_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAPERBOT_EMBEDDING_PROVIDER_CHAIN", "openai,hash,none")

    provider = try_build_default_embedding_provider()
    assert provider is not None

    vec = provider.embed("transformer attention")
    assert vec is not None
    assert len(vec) >= 64


def test_none_chain_disables_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAPERBOT_EMBEDDING_PROVIDER_CHAIN", "none")

    provider = try_build_default_embedding_provider()
    assert provider is None


def test_embedding_config_prefers_dedicated_embedding_envs(monkeypatch):
    monkeypatch.setenv("PAPERBOT_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("OPENAI_API_KEY", "chat-key")
    monkeypatch.setenv("PAPERBOT_EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("PAPERBOT_EMBEDDING_MODEL", "text-embedding-custom")

    config = EmbeddingConfig()

    assert config.resolve_api_key() == "embedding-key"
    assert config.resolve_base_url() == "https://embed.example/v1"
    assert config.resolve_model() == "text-embedding-custom"


def test_embedding_config_falls_back_to_openai_envs(monkeypatch):
    monkeypatch.delenv("PAPERBOT_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("PAPERBOT_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("PAPERBOT_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "chat-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-fallback")

    config = EmbeddingConfig()

    assert config.resolve_api_key() == "chat-key"
    assert config.resolve_base_url() == "https://chat.example/v1"
    assert config.resolve_model() == "text-embedding-fallback"


def test_hash_provider_tokenizes_japanese_kana_and_kanji() -> None:
    provider = HashEmbeddingProvider()

    tokens = provider._token_rx.findall("機械学習のテスト")

    assert tokens == ["機", "械", "学", "習", "の", "テ", "ス", "ト"]


def test_hash_provider_tokenizes_hangul_characters() -> None:
    provider = HashEmbeddingProvider()

    tokens = provider._token_rx.findall("한국어테스트")

    assert tokens == list("한국어테스트")
