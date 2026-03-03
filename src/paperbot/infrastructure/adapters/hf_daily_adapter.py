"""Hugging Face Daily Papers SearchPort adapter."""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper import PaperCandidate
from paperbot.infrastructure.connectors.hf_daily_papers_connector import (
    HFDailyPapersConnector,
)

_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([^/?#]+)")


class HFDailyAdapter:
    """SearchPort implementation wrapping HFDailyPapersConnector."""

    def __init__(self, connector: Optional[HFDailyPapersConnector] = None):
        self._connector = connector or HFDailyPapersConnector()

    @property
    def source_name(self) -> str:
        return "hf_daily"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 30,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[PaperCandidate]:
        max_pages = max(1, min(10, (max_results + 99) // 100 + 2))
        records = await asyncio.to_thread(
            self._connector.search,
            query=query,
            max_results=max_results,
            page_size=100,
            max_pages=max_pages,
        )
        return [self._to_candidate(r) for r in records]

    async def get_daily(self, *, limit: int = 100) -> List[PaperCandidate]:
        records = await asyncio.to_thread(self._connector.get_daily, limit=limit)
        return [self._to_candidate(r) for r in records]

    async def get_trending(
        self,
        *,
        mode: str = "hot",
        limit: int = 30,
    ) -> List[PaperCandidate]:
        records = await asyncio.to_thread(self._connector.get_trending, mode=mode, limit=limit)
        return [self._to_candidate(r) for r in records]

    @staticmethod
    def _to_candidate(r) -> PaperCandidate:
        identities = [PaperIdentity("hf_daily", r.paper_id)] if r.paper_id else []
        arxiv_id = _extract_arxiv_id(r.external_url)
        if arxiv_id:
            identities.append(PaperIdentity("arxiv", arxiv_id))
        return PaperCandidate(
            title=r.title,
            abstract=r.summary[:2000] if r.summary else "",
            authors=r.authors,
            url=r.paper_url,
            pdf_url=r.pdf_url,
            venue="Hugging Face Daily Papers",
            publication_date=r.submitted_on_daily_at or r.published_at,
            keywords=r.ai_keywords,
            identities=identities,
        )

    async def close(self) -> None:
        pass


def _extract_arxiv_id(url: str) -> str:
    text = str(url or "").strip()
    match = _ARXIV_ABS_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip()
