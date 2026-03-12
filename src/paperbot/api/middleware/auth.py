from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from paperbot.api.error_handling import json_error_response

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient", "testserver"}
_PUBLIC_PATHS = {"/health", "/openapi.json"}
_PUBLIC_PATH_PREFIXES = ("/docs", "/redoc")
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


def resolve_cors_origins() -> list[str]:
    raw = os.getenv("PAPERBOT_CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)

    origins: list[str] = []
    for entry in raw.split(","):
        value = entry.strip()
        if value and value != "*" and value not in origins:
            origins.append(value)
    return origins or list(_DEFAULT_CORS_ORIGINS)


def get_request_host(request: Request) -> str:
    client = request.client.host if request.client else ""
    return str(client or "").strip().lower()


def is_local_request(request: Request) -> bool:
    host = get_request_host(request)
    if not host:
        return False
    if host in _LOCAL_HOSTS:
        return True
    if host.startswith("127.") or host.startswith("::ffff:127."):
        return True
    return False


def is_public_path(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES)


def _extract_bearer_token(request: Request) -> Optional[str]:
    raw_header = request.headers.get("authorization", "").strip()
    if not raw_header:
        return None
    scheme, _, token = raw_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def resolve_api_key() -> str:
    return os.getenv("PAPERBOT_API_KEY", "").strip()


class APIKeyAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.method.upper() == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if is_public_path(request.url.path) or is_local_request(request):
            await self.app(scope, receive, send)
            return

        expected_api_key = resolve_api_key()
        bearer_token = _extract_bearer_token(request)
        if not expected_api_key or bearer_token != expected_api_key:
            response = json_error_response(
                request,
                status_code=401,
                detail="Unauthorized",
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def install_api_auth(app: FastAPI) -> None:
    app.add_middleware(APIKeyAuthMiddleware)


def install_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolve_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "X-Trace-Id"],
    )
