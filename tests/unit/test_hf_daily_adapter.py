from __future__ import annotations

import pytest

from paperbot.infrastructure.adapters.hf_daily_adapter import HFDailyAdapter
from paperbot.infrastructure.connectors.hf_daily_papers_connector import HFDailyPaperRecord


class _FakeConnector:
    def __init__(self):
        self.search_calls = []
        self.trending_calls = []

    def search(self, *, query: str, max_results: int, page_size: int, max_pages: int):
        self.search_calls.append(
            {
                "query": query,
                "max_results": max_results,
                "page_size": page_size,
                "max_pages": max_pages,
            }
        )
        return [self._record("2602.12345", "Search Result")]

    def get_daily(self, *, limit: int):
        return [self._record("2602.99999", "Daily Result")]

    def get_trending(self, *, mode: str, limit: int):
        self.trending_calls.append({"mode": mode, "limit": limit})
        return [self._record("2602.88888", "Trending Result")]

    @staticmethod
    def _record(paper_id: str, title: str) -> HFDailyPaperRecord:
        return HFDailyPaperRecord(
            paper_id=paper_id,
            title=title,
            summary="Summary",
            published_at="2026-02-08T00:00:00.000Z",
            submitted_on_daily_at="2026-02-10T00:00:00.000Z",
            authors=["Alice"],
            ai_keywords=["kv cache"],
            upvotes=10,
            paper_url=f"https://huggingface.co/papers/{paper_id}",
            external_url=f"https://arxiv.org/abs/{paper_id}",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
        )


@pytest.mark.asyncio
async def test_hf_daily_adapter_exposes_daily_and_trending_methods():
    connector = _FakeConnector()
    adapter = HFDailyAdapter(connector=connector)

    daily = await adapter.get_daily(limit=5)
    trending = await adapter.get_trending(mode="rising", limit=3)

    assert len(daily) == 1
    assert daily[0].title == "Daily Result"
    assert len(trending) == 1
    assert connector.trending_calls == [{"mode": "rising", "limit": 3}]


@pytest.mark.asyncio
async def test_hf_daily_adapter_adds_arxiv_identity_for_dedup():
    connector = _FakeConnector()
    adapter = HFDailyAdapter(connector=connector)

    papers = await adapter.search("kv cache", max_results=5)
    identities = {(item.source, item.external_id) for item in papers[0].identities}

    assert ("hf_daily", "2602.12345") in identities
    assert ("arxiv", "2602.12345") in identities
