from __future__ import annotations

from paperbot.context_engine.embeddings import try_build_default_embedding_provider


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
