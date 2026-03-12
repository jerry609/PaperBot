"""arXiv SearchPort adapter."""

from __future__ import annotations

from typing import List, Optional

from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper import PaperCandidate
from paperbot.infrastructure.connectors.arxiv_connector import ArxivConnector


class ArxivSearchAdapter:
    """SearchPort implementation wrapping ArxivConnector."""

    def __init__(self, connector: Optional[ArxivConnector] = None):
        self._connector = connector or ArxivConnector()

    @property
    def source_name(self) -> str:
        return "arxiv"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 30,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[PaperCandidate]:
        records = await self._connector.search(query=query, max_results=max_results)
        return [self._to_candidate(r) for r in records]

    @staticmethod
    def _to_candidate(r) -> PaperCandidate:
        identities = [PaperIdentity("arxiv", r.arxiv_id)] if r.arxiv_id else []
        return PaperCandidate(
            title=r.title,
            abstract=r.summary[:2000] if r.summary else "",
            authors=r.authors,
            url=r.abs_url,
            pdf_url=r.pdf_url,
            publication_date=r.published,
            identities=identities,
        )

    async def close(self) -> None:
        await self._connector.close()
