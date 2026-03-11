from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from paperbot.api.error_handling import install_api_error_handling
from paperbot.api.middleware import install_api_auth, install_cors, install_rate_limiting
from paperbot.api.routes.studio_chat import get_mode_flags


def _make_secured_app() -> FastAPI:
    app = FastAPI()
    install_api_error_handling(app)
    install_api_auth(app)
    install_rate_limiting(app)
    install_cors(app)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/api/chat")
    async def chat():
        return {"ok": True}

    @app.get("/api/papers/search")
    async def paper_search():
        return {"ok": True}

    return app


def test_code_mode_uses_safe_permission_mode():
    flags = get_mode_flags("Code")
    assert flags == ["--permission-mode", "acceptEdits"]
    assert "--dangerously-skip-permissions" not in flags


async def _request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_local_requests_bypass_api_key(monkeypatch):
    monkeypatch.setenv("PAPERBOT_API_KEY", "secret")
    app = _make_secured_app()

    response = await _request(
        app,
        "GET",
        "/api/chat",
        headers={"X-Forwarded-For": "127.0.0.1"},
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_remote_requests_require_api_key(monkeypatch):
    monkeypatch.setenv("PAPERBOT_API_KEY", "secret")
    app = _make_secured_app()

    response = await _request(
        app,
        "GET",
        "/api/chat",
        headers={"X-Forwarded-For": "203.0.113.10"},
    )

    assert response.status_code == 401
    payload = response.json()
    assert payload["detail"] == "Unauthorized"
    assert payload["trace_id"]


@pytest.mark.asyncio
async def test_remote_requests_accept_valid_bearer_token(monkeypatch):
    monkeypatch.setenv("PAPERBOT_API_KEY", "secret")
    app = _make_secured_app()

    response = await _request(
        app,
        "GET",
        "/api/chat",
        headers={
            "X-Forwarded-For": "203.0.113.10",
            "Authorization": "Bearer secret",
        },
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_llm_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("PAPERBOT_API_KEY", "secret")
    monkeypatch.setenv("PAPERBOT_RATE_LIMIT_LLM_PER_MINUTE", "2")
    app = _make_secured_app()
    headers = {
        "X-Forwarded-For": "203.0.113.10",
        "Authorization": "Bearer secret",
    }

    assert (await _request(app, "GET", "/api/chat", headers=headers)).status_code == 200
    assert (await _request(app, "GET", "/api/chat", headers=headers)).status_code == 200

    response = await _request(app, "GET", "/api/chat", headers=headers)

    assert response.status_code == 429
    payload = response.json()
    assert payload["detail"] == "Rate limit exceeded for llm requests"
    assert payload["trace_id"]
    assert int(response.headers["Retry-After"]) >= 59


@pytest.mark.asyncio
async def test_cors_only_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("PAPERBOT_CORS_ORIGINS", "https://allowed.example")
    app = _make_secured_app()

    allowed = await _request(
        app,
        "OPTIONS",
        "/api/chat",
        headers={
            "Origin": "https://allowed.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    blocked = await _request(
        app,
        "OPTIONS",
        "/api/chat",
        headers={
            "Origin": "https://blocked.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.headers.get("access-control-allow-origin") == "https://allowed.example"
    assert blocked.headers.get("access-control-allow-origin") is None
