from __future__ import annotations

import json
from typing import List, Optional

from sqlalchemy import desc, or_, select

from paperbot.application.ports.wiki_concept_port import (
    GroundingSnapshot,
    PaperGroundingRecord,
    TrackGroundingRecord,
    WikiConceptPort,
)
from paperbot.infrastructure.stores.models import PaperModel, ResearchTrackModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _load_list(raw: str) -> List[str]:
    try:
        values = json.loads(raw or "[]")
        if isinstance(values, list):
            return [str(value).strip() for value in values if str(value).strip()]
    except Exception:
        return []
    return []


class WikiConceptStore(WikiConceptPort):
    """Read model for grounding wiki concepts from stored papers and tracks."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)

    def load_grounding_snapshot(
        self,
        *,
        user_id: str,
        paper_limit: int = 250,
        track_limit: int = 100,
    ) -> GroundingSnapshot:
        with self._provider.session() as session:
            paper_rows = (
                session.execute(
                    select(PaperModel)
                    .where(
                        PaperModel.deleted_at.is_(None),
                        or_(
                            PaperModel.title != "",
                            PaperModel.abstract != "",
                        ),
                    )
                    .order_by(
                        desc(PaperModel.citation_count),
                        desc(PaperModel.updated_at),
                        desc(PaperModel.created_at),
                    )
                    .limit(max(1, paper_limit))
                )
                .scalars()
                .all()
            )
            track_rows = (
                session.execute(
                    select(ResearchTrackModel)
                    .where(
                        ResearchTrackModel.user_id == user_id,
                        ResearchTrackModel.archived_at.is_(None),
                    )
                    .order_by(
                        desc(ResearchTrackModel.is_active),
                        desc(ResearchTrackModel.updated_at),
                    )
                    .limit(max(1, track_limit))
                )
                .scalars()
                .all()
            )

        papers: List[PaperGroundingRecord] = [
            {
                "title": str(row.title or "").strip(),
                "abstract": str(row.abstract or "").strip(),
                "keywords": row.get_keywords(),
                "fields_of_study": row.get_fields_of_study(),
                "citation_count": int(row.citation_count or 0),
                "year": int(row.year) if row.year is not None else None,
            }
            for row in paper_rows
        ]
        tracks: List[TrackGroundingRecord] = [
            {
                "name": str(row.name or "").strip(),
                "description": str(row.description or "").strip(),
                "keywords": _load_list(row.keywords_json),
                "methods": _load_list(row.methods_json),
            }
            for row in track_rows
        ]
        return {"papers": papers, "tracks": tracks}
