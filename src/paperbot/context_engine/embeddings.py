from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    timeout_seconds: float = 30.0


class EmbeddingProvider:
    def embed(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        api_key = os.getenv(self.config.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env: {self.config.api_key_env}")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai library not installed") from e

        client_kwargs = {"api_key": api_key}
        base_url = os.getenv(self.config.base_url_env)
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
        resp = self._client.embeddings.create(model=self.config.model, input=s)
        if not getattr(resp, "data", None):
            return None
        vec = getattr(resp.data[0], "embedding", None)
        if not isinstance(vec, list):
            return None
        return [float(x) for x in vec]


def try_build_default_embedding_provider(
    *, config: Optional[EmbeddingConfig] = None
) -> Optional[EmbeddingProvider]:
    """
    Best-effort embedding provider.

    Returns None if openai is unavailable or no API key is configured.
    """
    try:
        return OpenAIEmbeddingProvider(config=config)
    except Exception:
        return None
