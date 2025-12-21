from __future__ import annotations

import asyncio
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

    async def get_bytes(self, url: str, *, headers: Optional[Dict[str, str]] = None) -> bytes:
        await self._throttle()
        hdrs = {"User-Agent": self.policy.user_agent}
        if headers:
            hdrs.update(headers)

        last_err: Optional[Exception] = None
        for attempt in range(self.policy.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.policy.timeout_s, follow_redirects=True) as client:
                    resp = await client.get(url, headers=hdrs)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                last_err = e
                if attempt >= self.policy.max_retries:
                    break
                backoff = min(self.policy.max_backoff_s, self.policy.base_backoff_s * (2**attempt))
                await asyncio.sleep(backoff)
        raise last_err or RuntimeError("request failed")

    async def get_text(self, url: str, *, headers: Optional[Dict[str, str]] = None) -> str:
        data = await self.get_bytes(url, headers=headers)
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode(errors="replace")


