from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class RequestPolicy:
    timeout_s: float = 20.0
    max_retries: int = 3
    base_backoff_s: float = 0.8
    max_backoff_s: float = 10.0
    rate_limit_per_sec: float = 2.0  # simple global throttle
    user_agent: str = "PaperBot/2.0"


class AsyncRequestLayer:
    """
    Minimal async request layer with retry + backoff + simple rate limiting.

    Phase B: used as the single place to implement network reliability policies.
    """

    def __init__(self, policy: Optional[RequestPolicy] = None):
        self.policy = policy or RequestPolicy()
        self._lock = asyncio.Lock()
        self._last_request_ts = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.policy.timeout_s,
                follow_redirects=True,
            )
        return self._client

    async def _throttle(self) -> None:
        if self.policy.rate_limit_per_sec <= 0:
            return
        min_interval = 1.0 / self.policy.rate_limit_per_sec
        async with self._lock:
            now = time.time()
            wait = (self._last_request_ts + min_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.time()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        await self._throttle()
        hdrs = {"User-Agent": self.policy.user_agent}
        if headers:
            hdrs.update(headers)

        client = await self._get_client()
        last_err: Optional[Exception] = None
        for attempt in range(self.policy.max_retries + 1):
            try:
                resp = await client.request(method, url, headers=hdrs, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt >= self.policy.max_retries:
                        resp.raise_for_status()
                    retry_after = resp.headers.get("Retry-After")
                    await resp.aread()
                    delay = self._retry_delay(attempt, retry_after)
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                return resp
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
                last_err = e
                if attempt >= self.policy.max_retries:
                    break
                await asyncio.sleep(self._retry_delay(attempt))
        raise last_err or RuntimeError("request failed")

    def _retry_delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), self.policy.max_backoff_s))
            except (TypeError, ValueError):
                pass

        delay = min(
            self.policy.max_backoff_s,
            self.policy.base_backoff_s * (2**attempt),
        )
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.0, delay + jitter)

    async def get_bytes(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        response = await self._request("GET", url, headers=headers, params=params)
        return response.content

    async def get_text(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = await self.get_bytes(url, headers=headers, params=params)
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode(errors="replace")

    async def get_json(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        response = await self._request("GET", url, headers=headers, params=params)
        return response.json()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncRequestLayer":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

