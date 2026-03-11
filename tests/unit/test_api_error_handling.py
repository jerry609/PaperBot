from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from paperbot.api.error_handling import (
    GENERIC_STREAM_ERROR_MESSAGE,
    install_api_error_handling,
)
from paperbot.api.routes.chat import ChatMessage, ChatRequest, chat_stream


def _make_guarded_app() -> FastAPI:
    app = FastAPI()
    install_api_error_handling(app)
    return app


async def _request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_unhandled_exception_is_redacted_and_traced(monkeypatch):
    monkeypatch.delenv("PAPERBOT_DEBUG", raising=False)
    app = _make_guarded_app()

    @app.get("/boom")
    async def boom():
        raise RuntimeError("postgres://db-user:secret@localhost/paperbot")

    response = await _request(app, "GET", "/boom")

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Internal server error"
    assert payload["trace_id"]
    assert response.headers["X-Trace-Id"] == payload["trace_id"]
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_unhandled_exception_returns_debug_detail_when_enabled(monkeypatch):
    monkeypatch.setenv("PAPERBOT_DEBUG", "true")
    app = _make_guarded_app()

    @app.get("/boom")
    async def boom():
        raise RuntimeError("db_url=sqlite:///tmp/test.db")

    response = await _request(app, "GET", "/boom")

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "db_url=sqlite:///tmp/test.db"
    assert payload["trace_id"]


@pytest.mark.asyncio
async def test_request_size_limit_rejects_large_payloads(monkeypatch):
    monkeypatch.setenv("PAPERBOT_MAX_REQUEST_BYTES", "16")
    app = _make_guarded_app()

    @app.post("/echo")
    async def echo(payload: dict):
        return payload

    response = await _request(
        app,
        "POST",
        "/echo",
        content='{"message":"this is definitely too large"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    payload = response.json()
    assert payload["detail"] == "Request body too large (max 16 bytes)"
    assert payload["trace_id"]


def test_chat_request_trims_history_to_recent_entries(monkeypatch):
    monkeypatch.setenv("PAPERBOT_CHAT_MAX_HISTORY", "3")
    request = ChatRequest(
        message="latest",
        history=[
            ChatMessage(role="user", content="m1"),
            ChatMessage(role="assistant", content="m2"),
            ChatMessage(role="user", content="m3"),
            ChatMessage(role="assistant", content="m4"),
            ChatMessage(role="user", content="m5"),
        ],
    )

    assert [message.content for message in request.history] == ["m3", "m4", "m5"]


@pytest.mark.asyncio
async def test_chat_stream_redacts_internal_errors(monkeypatch):
    import paperbot.infrastructure.llm as llm_module

    class BrokenModelRouter:
        @classmethod
        def from_env(cls):
            raise RuntimeError("OPENAI_API_KEY=super-secret")

    monkeypatch.setattr(llm_module, "ModelRouter", BrokenModelRouter)

    events = [
        event
        async for event in chat_stream(
            ChatRequest(message="hello"),
            trace_id="trace-test-251",
        )
    ]

    assert events[-1].type == "error"
    assert events[-1].message == GENERIC_STREAM_ERROR_MESSAGE
    assert events[-1].data["trace_id"] == "trace-test-251"
    assert "super-secret" not in str(events[-1].data)


@pytest.mark.asyncio
async def test_jobs_route_redacts_internal_errors(monkeypatch):
    import paperbot.api.routes.jobs as jobs_route

    async def broken_create_pool(_settings):
        raise RuntimeError("redis://paperbot:super-secret@localhost:6379/0")

    monkeypatch.delenv("PAPERBOT_DEBUG", raising=False)
    monkeypatch.setattr(jobs_route, "_redis_settings", lambda: object())
    monkeypatch.setattr(jobs_route, "create_pool", broken_create_pool)

    app = _make_guarded_app()
    app.include_router(jobs_route.router, prefix="/api")
    response = await _request(
        app,
        "POST",
        "/api/jobs/track-scholar",
        json={"scholar_id": "s2:123", "dry_run": True, "offline": False},
    )

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Internal server error"
    assert payload["trace_id"]
    assert "super-secret" not in response.text


@pytest.mark.asyncio
async def test_sandbox_route_redacts_internal_errors(monkeypatch):
    import paperbot.api.routes.sandbox as sandbox_route
    import paperbot.infrastructure.logging.execution_logger as execution_logger_module

    class BrokenExecutionLogger:
        def get_logs_dict(self, *_args, **_kwargs):
            raise RuntimeError("sqlite:///paperbot?token=super-secret")

    monkeypatch.delenv("PAPERBOT_DEBUG", raising=False)
    monkeypatch.setattr(
        execution_logger_module,
        "get_execution_logger",
        lambda: BrokenExecutionLogger(),
    )

    app = _make_guarded_app()
    app.include_router(sandbox_route.router, prefix="/api")
    response = await _request(app, "GET", "/api/sandbox/runs/run-251/logs")

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Internal server error"
    assert payload["trace_id"]
    assert payload["run_id"] == "run-251"
    assert "super-secret" not in response.text
