from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    model: Optional[str] = None
    model_env: str = "PAPERBOT_EMBEDDING_MODEL"
    api_key_env: str = "PAPERBOT_EMBEDDING_API_KEY"
    base_url_env: str = "PAPERBOT_EMBEDDING_BASE_URL"
    timeout_seconds: float = 30.0

    def resolve_model(self) -> str:
        explicit = str(self.model or "").strip()
        if explicit:
            return explicit
        env_value = _first_nonempty_env(self.model_env, "OPENAI_EMBEDDING_MODEL")
        if env_value:
            return env_value
        return "text-embedding-3-small"

    def resolve_api_key(self) -> str:
        return _first_nonempty_env(self.api_key_env, "OPENAI_API_KEY")

    def resolve_base_url(self) -> str:
        return _first_nonempty_env(self.base_url_env, "OPENAI_BASE_URL")


def _first_nonempty_env(*names: str) -> str:
    for name in names:
        value = os.getenv(str(name or ""), "").strip()
        if value:
            return value
    return ""


class EmbeddingProvider:
    def embed(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        api_key = self.config.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing embedding API key env: "
                f"{self.config.api_key_env} (fallback: OPENAI_API_KEY)"
            )

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai library not installed") from e

        client_kwargs = {"api_key": api_key}
        base_url = self.config.resolve_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

    def embed(self, text: str) -> Optional[List[float]]:
        s = (text or "").strip()
        if not s:
            return None
        # Guardrail: keep request bounded.
        if len(s) > 24000:
            s = s[:24000]
        resp = self._client.embeddings.create(model=self.config.resolve_model(), input=s)
        if not getattr(resp, "data", None):
            return None
        vec = getattr(resp.data[0], "embedding", None)
        if not isinstance(vec, list):
            return None
        return [float(x) for x in vec]


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local fallback provider based on token hashing."""

    def __init__(self, dim: int = 1536):
        self.dim = max(64, int(dim))
        self._token_rx = re.compile(
            r"[A-Za-z0-9_+-]+|"
            r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff"
            r"\u1100-\u11ff\uac00-\ud7af\uff66-\uff9f]"
        )

    def embed(self, text: str) -> Optional[List[float]]:
        s = (text or "").strip()
        if not s:
            return None

        vec = [0.0] * self.dim
        tokens = self._token_rx.findall(s.lower())
        if not tokens:
            tokens = [s[:64].lower()]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            return vec
        return [v / norm for v in vec]


def _provider_chain_from_env() -> List[str]:
    raw = os.getenv("PAPERBOT_EMBEDDING_PROVIDER_CHAIN", "openai,none")
    chain = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return chain or ["openai", "none"]


def try_build_default_embedding_provider(
    *, config: Optional[EmbeddingConfig] = None
) -> Optional[EmbeddingProvider]:
    """
    Best-effort embedding provider.

    Provider chain is configurable via PAPERBOT_EMBEDDING_PROVIDER_CHAIN,
    e.g. "openai,hash,none".
    """
    chain = _provider_chain_from_env()
    for provider in chain:
        if provider == "none":
            return None
        if provider == "openai":
            try:
                return OpenAIEmbeddingProvider(config=config)
            except Exception:
                continue
        if provider == "hash":
            try:
                return HashEmbeddingProvider()
            except Exception:
                continue
    return None
