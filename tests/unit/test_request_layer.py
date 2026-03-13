from __future__ import annotations

import httpx
import pytest

from paperbot.infrastructure.crawling.request_layer import AsyncRequestLayer, RequestPolicy


def _response(status_code: int, payload: dict, url: str = "https://example.com/api") -> httpx.Response:
    request = httpx.Request("GET", url)
    return httpx.Response(status_code, json=payload, request=request)


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.closed = False

    async def request(self, method, url, headers=None, params=None):
        self.calls.append({"method": method, "url": url, "params": params})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_request_layer_retries_transient_transport_error(monkeypatch):
    client = _FakeClient(
        [
            httpx.ConnectTimeout("timeout"),
            _response(200, {"ok": True}),
        ]
    )
    layer = AsyncRequestLayer(
        RequestPolicy(timeout_s=1.0, max_retries=2, base_backoff_s=0.0, max_backoff_s=0.0)
    )
    
    async def _fake_get_client():
        return client

    monkeypatch.setattr(layer, "_get_client", _fake_get_client)

    payload = await layer.get_json("https://example.com/api", params={"q": "paperbot"})

    assert payload == {"ok": True}
    assert len(client.calls) == 2
    assert client.calls[0]["params"] == {"q": "paperbot"}


@pytest.mark.asyncio
async def test_request_layer_retries_429_response(monkeypatch):
    client = _FakeClient(
        [
            _response(429, {"error": "slow down"}),
            _response(200, {"items": [1, 2, 3]}),
        ]
    )
    layer = AsyncRequestLayer(
        RequestPolicy(timeout_s=1.0, max_retries=2, base_backoff_s=0.0, max_backoff_s=0.0)
    )
    
    async def _fake_get_client():
        return client

    monkeypatch.setattr(layer, "_get_client", _fake_get_client)

    payload = await layer.get_json("https://example.com/api")

    assert payload["items"] == [1, 2, 3]
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_request_layer_raises_after_retry_budget(monkeypatch):
    client = _FakeClient([httpx.ReadTimeout("timeout"), httpx.ReadTimeout("timeout")])
    layer = AsyncRequestLayer(
        RequestPolicy(timeout_s=1.0, max_retries=1, base_backoff_s=0.0, max_backoff_s=0.0)
    )
    
    async def _fake_get_client():
        return client

    monkeypatch.setattr(layer, "_get_client", _fake_get_client)

    with pytest.raises(httpx.ReadTimeout):
        await layer.get_text("https://example.com/api")

    assert len(client.calls) == 2
