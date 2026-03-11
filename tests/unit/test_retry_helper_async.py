from __future__ import annotations

import pytest

from paperbot.utils.retry_helper import RetryConfig, async_retry


@pytest.mark.asyncio
async def test_async_retry_eventually_succeeds():
    attempts = {"count": 0}

    @async_retry(
        RetryConfig(
            max_retries=2,
            initial_delay=0.0,
            backoff_factor=1.0,
            max_delay=0.0,
            retry_on_exceptions=(RuntimeError,),
        )
    )
    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_async_retry_raises_when_exhausted():
    attempts = {"count": 0}

    @async_retry(
        RetryConfig(
            max_retries=1,
            initial_delay=0.0,
            backoff_factor=1.0,
            max_delay=0.0,
            retry_on_exceptions=(RuntimeError,),
        )
    )
    async def always_fail():
        attempts["count"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await always_fail()

    assert attempts["count"] == 2
