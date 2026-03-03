"""SQLAlchemy store for ReproCodeExperienceModel (issue #162)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from paperbot.infrastructure.stores.models import Base, ReproCodeExperienceModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url

_VALID_TYPES = {"success_pattern", "failure_reason", "verified_structure"}


class ReproExperienceStore:
    """CRUD store for persisted code generation experiences."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)

    def add(
        self,
        *,
        pattern_type: str,
        content: str,
        paper_id: Optional[str] = None,
        pack_id: Optional[str] = None,
        code_snippet: Optional[str] = None,
    ) -> ReproCodeExperienceModel:
        """Persist one experience record. Returns the saved row."""
        if pattern_type not in _VALID_TYPES:
            raise ValueError(f"pattern_type must be one of {_VALID_TYPES}")
        now = datetime.now(timezone.utc)
        row = ReproCodeExperienceModel(
            pack_id=pack_id,
            paper_id=paper_id,
            pattern_type=pattern_type,
            content=(content or "").strip(),
            code_snippet=code_snippet,
            created_at=now,
        )
        with self._provider.session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def get_by_paper_id(
        self,
        paper_id: str,
        *,
        pattern_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences for a specific paper, newest first."""
        with self._provider.session() as session:
            stmt = (
                select(ReproCodeExperienceModel)
                .where(ReproCodeExperienceModel.paper_id == paper_id)
            )
            if pattern_type:
                stmt = stmt.where(ReproCodeExperienceModel.pattern_type == pattern_type)
            stmt = stmt.order_by(ReproCodeExperienceModel.created_at.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._to_dict(r) for r in rows]

    def get_by_pack_id(
        self,
        pack_id: str,
        *,
        pattern_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences for a specific P2C pack, newest first."""
        with self._provider.session() as session:
            stmt = (
                select(ReproCodeExperienceModel)
                .where(ReproCodeExperienceModel.pack_id == pack_id)
            )
            if pattern_type:
                stmt = stmt.where(ReproCodeExperienceModel.pattern_type == pattern_type)
            stmt = stmt.order_by(ReproCodeExperienceModel.created_at.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._to_dict(r) for r in rows]

    @staticmethod
    def _to_dict(r: ReproCodeExperienceModel) -> Dict[str, Any]:
        return {
            "id": r.id,
            "pack_id": r.pack_id,
            "paper_id": r.paper_id,
            "pattern_type": r.pattern_type,
            "content": r.content,
            "code_snippet": r.code_snippet,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
