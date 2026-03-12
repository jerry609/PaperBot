"""papers.cool SearchPort adapter."""

from __future__ import annotations

from typing import List, Optional

from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper import PaperCandidate
from paperbot.infrastructure.connectors.paperscool_connector import PapersCoolConnector


class PapersCoolAdapter:
    """SearchPort implementation wrapping PapersCoolConnector."""

    def __init__(self, connector: Optional[PapersCoolConnector] = None):
        self._connector = connector or PapersCoolConnector()

    @property
    def source_name(self) -> str:
        return "papers_cool"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 30,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[PaperCandidate]:
        records = await self._connector.search(
            branch="arxiv",
            query=query,
            highlight=True,
            show=max_results,
        )
        return [self._to_candidate(r) for r in records]

    @staticmethod
    def _to_candidate(r) -> PaperCandidate:
        identities = [PaperIdentity("papers_cool", r.paper_id)] if r.paper_id else []
        return PaperCandidate(
            title=r.title,
            abstract=r.snippet[:2000] if r.snippet else "",
            authors=r.authors,
            url=r.url,
            pdf_url=r.pdf_url,
            venue=r.subject_or_venue,
            publication_date=r.published_at,
            keywords=r.keywords,
            identities=identities,
        )

    async def close(self) -> None:
        await self._connector.close()
