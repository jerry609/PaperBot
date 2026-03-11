from __future__ import annotations

import pytest

from paperbot.infrastructure.connectors.openalex_connector import OpenAlexConnector


class _FakeRequestLayer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get_json(self, url, *, headers=None, params=None):
        self.calls.append({"url": url, "params": params})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_resolve_work_by_doi():
    request_layer = _FakeRequestLayer(
        [{"results": [{"id": "https://openalex.org/W123", "title": "A"}]}]
    )
    connector = OpenAlexConnector(request_layer=request_layer)

    row = await connector.resolve_work(seed_type="doi", seed_id="10.1000/xyz.1")

    assert row is not None
    assert row["id"] == "https://openalex.org/W123"
    assert request_layer.calls[0]["params"]["filter"].startswith("doi:")


@pytest.mark.asyncio
async def test_related_and_references_load_by_batched_ids():
    request_layer = _FakeRequestLayer(
        [
            {
                "results": [
                    {"id": "https://openalex.org/W2", "title": "Related"},
                    {"id": "https://openalex.org/W3", "title": "Ref"},
                ]
            },
            {"results": [{"id": "https://openalex.org/W3", "title": "Ref"}]},
        ]
    )
    connector = OpenAlexConnector(request_layer=request_layer)
    seed = {
        "id": "https://openalex.org/W1",
        "title": "Seed",
        "related_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "referenced_works": ["https://openalex.org/W3"],
    }

    related = await connector.get_related_works(seed, limit=5)
    refs = await connector.get_referenced_works(seed, limit=5)

    assert [row["title"] for row in related] == ["Related", "Ref"]
    assert [row["title"] for row in refs] == ["Ref"]
    assert "openalex_id:W2|W3" in request_layer.calls[0]["params"]["filter"]


@pytest.mark.asyncio
async def test_get_citing_works():
    request_layer = _FakeRequestLayer(
        [{"results": [{"id": "https://openalex.org/W9", "title": "Citing"}]}]
    )
    connector = OpenAlexConnector(request_layer=request_layer)

    rows = await connector.get_citing_works(
        {"cited_by_api_url": "https://api.openalex.org/works?filter=cited-by:W1"},
        limit=10,
    )

    assert len(rows) == 1
    assert rows[0]["title"] == "Citing"
