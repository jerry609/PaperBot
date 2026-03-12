from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from paperbot.infrastructure.llm.router import ModelConfig, ModelRouter, RouterConfig
from paperbot.infrastructure.stores.llm_usage_store import LLMUsageStore
from paperbot.infrastructure.stores.model_endpoint_store import ModelEndpointStore

router = APIRouter()

_store: Optional[ModelEndpointStore] = None
_usage_store: Optional[LLMUsageStore] = None


def _get_store() -> ModelEndpointStore:
    global _store
    if _store is None:
        _store = ModelEndpointStore()
    return _store


def _get_usage_store() -> LLMUsageStore:
    global _usage_store
    if _usage_store is None:
        _usage_store = LLMUsageStore()
    return _usage_store

_ALLOWED_VENDORS = ["openai_compatible", "openai", "anthropic", "ollama"]
_ALLOWED_TASK_TYPES = [
    "default",
    "extraction",
    "summary",
    "analysis",
    "reasoning",
    "code",
    "review",
    "chat",
]


class ModelEndpointCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    vendor: str = "openai_compatible"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    task_types: List[str] = Field(default_factory=list)
    enabled: bool = True
    is_default: bool = False


class ModelEndpointUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    vendor: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    models: Optional[List[str]] = None
    task_types: Optional[List[str]] = None
    enabled: Optional[bool] = None
    is_default: Optional[bool] = None


class ModelEndpointListResponse(BaseModel):
    items: List[Dict[str, Any]]


class ModelEndpointResponse(BaseModel):
    item: Dict[str, Any]


class EndpointTestRequest(BaseModel):
    remote: bool = False


class EndpointTestResponse(BaseModel):
    ok: bool
    success: bool
    endpoint_id: int
    provider: Dict[str, Any]
    api_key_present: bool
    latency_ms: int
    message: str
    error: Optional[str] = None


class EndpointCapabilitiesResponse(BaseModel):
    vendors: List[str]
    task_types: List[str]


class EndpointActivateResponse(BaseModel):
    item: Dict[str, Any]


class LLMUsageSummaryResponse(BaseModel):
    summary: Dict[str, Any]


def _build_model_config(endpoint: Dict[str, Any]) -> ModelConfig:
    models = [str(x).strip() for x in (endpoint.get("models") or []) if str(x).strip()]
    if not models:
        raise ValueError("endpoint has no models")

    vendor = str(endpoint.get("vendor") or "openai_compatible").strip().lower()
    provider = ModelRouter._normalize_provider(vendor)
    return ModelConfig(
        provider=provider,
        model=models[0],
        api_key_env=str(endpoint.get("api_key_env") or "OPENAI_API_KEY"),
        api_key=(str(endpoint.get("api_key") or "").strip() or None),
        base_url=(str(endpoint.get("base_url") or "").strip() or None),
    )


@router.get("/model-endpoints", response_model=ModelEndpointListResponse)
def list_model_endpoints(enabled_only: bool = False):
    rows = _get_store().list_endpoints(enabled_only=enabled_only)
    return ModelEndpointListResponse(items=rows)


@router.get("/model-endpoints/capabilities", response_model=EndpointCapabilitiesResponse)
def get_model_endpoint_capabilities():
    return EndpointCapabilitiesResponse(vendors=_ALLOWED_VENDORS, task_types=_ALLOWED_TASK_TYPES)


@router.post("/model-endpoints", response_model=ModelEndpointResponse)
def create_model_endpoint(req: ModelEndpointCreateRequest):
    try:
        row = _get_store().upsert_endpoint(payload=req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelEndpointResponse(item=row)


@router.patch("/model-endpoints/{endpoint_id}", response_model=ModelEndpointResponse)
def update_model_endpoint(endpoint_id: int, req: ModelEndpointUpdateRequest):
    existing = _get_store().get_endpoint(endpoint_id)
    if not existing:
        raise HTTPException(status_code=404, detail="model endpoint not found")

    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        row = _get_store().upsert_endpoint(payload=payload, endpoint_id=endpoint_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelEndpointResponse(item=row)


@router.delete("/model-endpoints/{endpoint_id}")
def delete_model_endpoint(endpoint_id: int):
    ok = _get_store().delete_endpoint(endpoint_id)
    if not ok:
        raise HTTPException(status_code=404, detail="model endpoint not found")
    return {"ok": True}


@router.post("/model-endpoints/{endpoint_id}/activate", response_model=EndpointActivateResponse)
def activate_model_endpoint(endpoint_id: int):
    row = _get_store().activate_endpoint(endpoint_id)
    if not row:
        raise HTTPException(status_code=404, detail="model endpoint not found")
    return EndpointActivateResponse(item=row)


@router.get("/model-endpoints/usage", response_model=LLMUsageSummaryResponse)
def get_llm_usage_summary(days: int = 7):
    window = max(1, min(int(days), 90))
    summary = _get_usage_store().summarize(days=window)
    return LLMUsageSummaryResponse(summary=summary)


@router.post("/model-endpoints/{endpoint_id}/test", response_model=EndpointTestResponse)
def test_model_endpoint(endpoint_id: int, req: EndpointTestRequest):
    endpoint = _get_store().get_endpoint(endpoint_id, include_secrets=True)
    if not endpoint:
        raise HTTPException(status_code=404, detail="model endpoint not found")

    api_key_env = str(endpoint.get("api_key_env") or "OPENAI_API_KEY")
    api_key_present = bool(str(endpoint.get("api_key") or "").strip()) or bool(
        os.getenv(api_key_env)
    )

    t0 = time.monotonic()
    try:
        cfg = _build_model_config(endpoint)
        router = ModelRouter(RouterConfig(models={"__test__": cfg}, fallback_model="__test__"))
        provider = router.get_provider("default")
        info = provider.info

        message = "Provider initialized successfully."
        if req.remote:
            if not api_key_present:
                raise ValueError(f"missing API key env: {api_key_env}")
            pong = provider.invoke_simple(
                "You are a connection checker.",
                "Reply with a single word: OK",
                max_tokens=16,
                temperature=0,
            )
            message = f"Remote check success: {(pong or '').strip()[:80]}"

        latency_ms = int((time.monotonic() - t0) * 1000)
        return EndpointTestResponse(
            ok=True,
            success=True,
            endpoint_id=endpoint_id,
            provider={
                "provider_name": info.provider_name,
                "model_name": info.model_name,
                "api_base": info.api_base,
            },
            api_key_present=api_key_present,
            latency_ms=latency_ms,
            message=message,
            error=None,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        raise HTTPException(status_code=400, detail=f"test failed: {exc}") from exc
