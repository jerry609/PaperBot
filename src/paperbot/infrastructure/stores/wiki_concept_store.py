from __future__ import annotations

from typing import List, Optional

from paperbot.application.ports.wiki_concept_port import (
    GroundingSnapshot,
    PaperGroundingRecord,
    TrackGroundingRecord,
    WikiConceptPort,
)
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.infrastructure.stores.sqlalchemy_db import get_db_url


class WikiConceptStore(WikiConceptPort):
    """Read model for grounding wiki concepts from stored papers and tracks."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self._research_store = SqlAlchemyResearchStore(db_url=self.db_url)

    def load_grounding_snapshot(
        self,
        *,
        user_id: str,
        paper_limit: int = 250,
        track_limit: int = 100,
    ) -> GroundingSnapshot:
        saved_rows = self._research_store.list_saved_papers(
            user_id=user_id,
            limit=max(1, paper_limit),
            sort_by="saved_at",
        )
        track_rows = self._research_store.list_tracks(
            user_id=user_id,
            include_archived=False,
            limit=max(1, track_limit),
        )
        papers: List[PaperGroundingRecord] = [
            {
                "title": str((row.get("paper") or {}).get("title") or "").strip(),
                "abstract": str((row.get("paper") or {}).get("abstract") or "").strip(),
                "keywords": list((row.get("paper") or {}).get("keywords") or []),
                "fields_of_study": list((row.get("paper") or {}).get("fields_of_study") or []),
                "citation_count": int((row.get("paper") or {}).get("citation_count") or 0),
                "year": (
                    int((row.get("paper") or {}).get("year"))
                    if (row.get("paper") or {}).get("year") is not None
                    else None
                ),
            }
            for row in saved_rows
            if isinstance(row, dict) and isinstance(row.get("paper"), dict)
        ]
        tracks: List[TrackGroundingRecord] = [
            {
                "name": str(row.get("name") or "").strip(),
                "description": str(row.get("description") or "").strip(),
                "keywords": [
                    str(value).strip() for value in row.get("keywords") or [] if str(value).strip()
                ],
                "methods": [
                    str(value).strip() for value in row.get("methods") or [] if str(value).strip()
                ],
            }
            for row in track_rows
            if isinstance(row, dict)
        ]
        return {"papers": papers, "tracks": tracks}
