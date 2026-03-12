from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from paperbot.context_engine.embeddings import EmbeddingConfig, OpenAIEmbeddingProvider
from paperbot.infrastructure.stores.embedding_endpoint_store import EmbeddingEndpointStore

router = APIRouter()

_store: Optional[EmbeddingEndpointStore] = None


def _get_store() -> EmbeddingEndpointStore:
    global _store
    if _store is None:
        _store = EmbeddingEndpointStore()
    return _store


def _environment_snapshot() -> Dict[str, Any]:
    config = EmbeddingConfig()
    api_key_value = config.resolve_api_key()
    base_url_value = config.resolve_base_url()
    return {
        "provider": "openai",
        "api_key_env": config.resolve_api_key_source(),
        "api_key_present": bool(api_key_value),
        "base_url_env": config.resolve_base_url_source(),
        "base_url": base_url_value or None,
        "model_env": config.resolve_model_source(),
        "model": config.resolve_model(),
    }


def _default_item() -> Dict[str, Any]:
    return {
        "provider": "openai",
        "base_url": None,
        "api_key_env": "PAPERBOT_EMBEDDING_API_KEY",
        "api_key": "",
        "api_key_present": False,
        "key_source": "",
        "model": "text-embedding-3-small",
        "enabled": False,
        "updated_at": None,
    }


def _effective_source(item: Dict[str, Any], environment: Dict[str, Any]) -> str:
    if bool(item.get("enabled")):
        return "settings"
    if bool(environment.get("api_key_present")):
        return "environment"
    return "none"


def _build_provider_config(payload: Optional[Dict[str, Any]]) -> Optional[EmbeddingConfig]:
    config_payload = dict(payload or {})
    enabled = config_payload.get("enabled")
    if enabled is False:
        return None

    api_key = str(config_payload.get("api_key") or "").strip() or None
    api_key_env = str(config_payload.get("api_key_env") or "PAPERBOT_EMBEDDING_API_KEY").strip()
    base_url = str(config_payload.get("base_url") or "").strip() or None
    model = str(config_payload.get("model") or "").strip() or None
    return EmbeddingConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_key_env=api_key_env or "PAPERBOT_EMBEDDING_API_KEY",
    )


class EmbeddingSettingsUpdateRequest(BaseModel):
    enabled: bool = False
    provider: str = Field(default="openai", min_length=1, max_length=32)
    base_url: Optional[str] = None
    api_key_env: str = "PAPERBOT_EMBEDDING_API_KEY"
    api_key: Optional[str] = None
    model: str = Field(default="text-embedding-3-small", min_length=1, max_length=128)


class EmbeddingSettingsTestRequest(BaseModel):
    enabled: Optional[bool] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    remote: bool = True
    sample_text: str = Field(default="embedding connectivity check", max_length=500)


class EmbeddingSettingsResponse(BaseModel):
    item: Dict[str, Any]
    environment: Dict[str, Any]
    effective_source: str


class EmbeddingSettingsTestResponse(BaseModel):
    ok: bool
    success: bool
    source: str
    provider: str
    model: str
    vector_dim: Optional[int] = None
    latency_ms: int
    message: str


@router.get("/embedding-settings", response_model=EmbeddingSettingsResponse)
def get_embedding_settings():
    item = _get_store().get_config() or _default_item()
    environment = _environment_snapshot()
    return EmbeddingSettingsResponse(
        item=item,
        environment=environment,
        effective_source=_effective_source(item, environment),
    )


@router.patch("/embedding-settings", response_model=EmbeddingSettingsResponse)
def update_embedding_settings(req: EmbeddingSettingsUpdateRequest):
    try:
        item = _get_store().upsert_config(payload=req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    environment = _environment_snapshot()
    return EmbeddingSettingsResponse(
        item=item,
        environment=environment,
        effective_source=_effective_source(item, environment),
    )


@router.post("/embedding-settings/test", response_model=EmbeddingSettingsTestResponse)
def test_embedding_settings(req: EmbeddingSettingsTestRequest):
    current = _get_store().get_config(include_secrets=True)
    payload: Dict[str, Any]
    if req.model_dump(exclude_none=True):
        payload = {**(current or {}), **req.model_dump(exclude_none=True)}
    else:
        payload = current or {}

    config = _build_provider_config(payload)
    source = "settings" if config is not None else "environment"
    if config is None:
        config = EmbeddingConfig()

    t0 = time.monotonic()
    try:
        provider = OpenAIEmbeddingProvider(config=config)
        vector_dim = None
        if req.remote:
            vector = provider.embed(req.sample_text)
            vector_dim = len(vector) if vector else 0

        latency_ms = int((time.monotonic() - t0) * 1000)
        return EmbeddingSettingsTestResponse(
            ok=True,
            success=True,
            source=source,
            provider="openai",
            model=config.resolve_model(),
            vector_dim=vector_dim,
            latency_ms=latency_ms,
            message=(
                f"Embedding endpoint ready ({vector_dim} dims)."
                if vector_dim is not None
                else "Embedding provider initialized successfully."
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"embedding test failed: {exc}") from exc
