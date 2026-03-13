from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from fastapi import FastAPI, Request
from starlette.types import ASGIApp, Receive, Scope, Send

from paperbot.api.error_handling import json_error_response
from paperbot.api.middleware.auth import get_request_host, is_local_request, is_public_path

_LLM_PATH_PREFIXES = (
    "/api/chat",
    "/api/analyze",
    "/api/review",
    "/api/gen-code",
    "/api/studio/chat",
)


def _env_limit(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class RateLimitPolicy:
    bucket: str
    limit: int
    window_seconds: int = 60


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, *, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        now = time.monotonic()
        earliest = now - window_seconds
        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] <= earliest:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])))
                return False, retry_after
            bucket.append(now)
        return True, 0


def resolve_rate_limit_policy(request: Request) -> Optional[RateLimitPolicy]:
    if request.method.upper() == "OPTIONS":
        return None

    path = request.url.path
    if is_public_path(path):
        return None
    if any(path.startswith(prefix) for prefix in _LLM_PATH_PREFIXES):
        return RateLimitPolicy(
            bucket="llm",
            limit=_env_limit("PAPERBOT_RATE_LIMIT_LLM_PER_MINUTE", 10),
        )
    if "/search" in path:
        return RateLimitPolicy(
            bucket="search",
            limit=_env_limit("PAPERBOT_RATE_LIMIT_SEARCH_PER_MINUTE", 30),
        )
    return RateLimitPolicy(
        bucket="default",
        limit=_env_limit("PAPERBOT_RATE_LIMIT_DEFAULT_PER_MINUTE", 60),
    )


class RequestRateLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._limiter = InMemoryRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if is_local_request(request):
            await self.app(scope, receive, send)
            return

        policy = resolve_rate_limit_policy(request)
        if policy is None:
            await self.app(scope, receive, send)
            return

        request_key = f"{policy.bucket}:{get_request_host(request) or 'anonymous'}"
        allowed, retry_after = self._limiter.allow(
            key=request_key,
            limit=policy.limit,
            window_seconds=policy.window_seconds,
        )
        if not allowed:
            response = json_error_response(
                request,
                status_code=429,
                detail=f"Rate limit exceeded for {policy.bucket} requests",
            )
            response.headers["Retry-After"] = str(retry_after)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def install_rate_limiting(app: FastAPI) -> None:
    app.add_middleware(RequestRateLimitMiddleware)
