from __future__ import annotations

import pytest

from paperbot.agents.mixins.semantic_scholar import SemanticScholarMixin
from paperbot.agents.scholar_tracking.semantic_scholar_agent import SemanticScholarAgent


class _DummyS2Mixin(SemanticScholarMixin):
    pass


class _FakeSemanticScholarClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        _FakeSemanticScholarClient.instances.append(self)

    async def search_papers(self, query, limit=10, fields=None):
        return [{"title": "Paper", "query": query, "limit": limit, "fields": fields}]

    async def get_paper(self, paper_id, fields=None):
        return {"paperId": paper_id, "title": "Detail", "fields": fields}

    async def get_author(self, author_id, fields=None):
        return {"authorId": author_id, "name": "Author", "fields": fields}

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_semantic_scholar_mixin_uses_shared_client(monkeypatch):
    _FakeSemanticScholarClient.instances.clear()
    monkeypatch.setattr(
        "paperbot.agents.mixins.semantic_scholar.SemanticScholarClient",
        _FakeSemanticScholarClient,
    )

    agent = _DummyS2Mixin()
    agent.init_s2_client({"semantic_scholar_api_key": "key"})

    search_rows = await agent.search_semantic_scholar("transformers", limit=5)
    detail = await agent.get_paper_details("paper-1")
    await agent.close_s2_client()

    assert len(_FakeSemanticScholarClient.instances) == 1
    assert search_rows[0]["query"] == "transformers"
    assert detail["paperId"] == "paper-1"
    assert _FakeSemanticScholarClient.instances[0].kwargs["api_key"] == "key"
    assert _FakeSemanticScholarClient.instances[0].closed is True


@pytest.mark.asyncio
async def test_semantic_scholar_agent_uses_unified_client(monkeypatch):
    _FakeSemanticScholarClient.instances.clear()
    monkeypatch.setattr(
        "paperbot.agents.scholar_tracking.semantic_scholar_agent.SemanticScholarClient",
        _FakeSemanticScholarClient,
    )

    agent = SemanticScholarAgent({"api": {"semantic_scholar": {"api_key": "key", "timeout": 9}}})
    client = await agent._get_client()
    author = await agent.fetch_author_info("a-1")
    await agent.close()

    assert isinstance(client, _FakeSemanticScholarClient)
    assert author["authorId"] == "a-1"
    assert client.kwargs["api_key"] == "key"
    assert client.kwargs["timeout"] == 9
    assert client.closed is True
