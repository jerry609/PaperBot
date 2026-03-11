"""Identity store — CRUD for paper_identifiers table."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from paperbot.domain.identity import PaperIdentity
from paperbot.infrastructure.stores.models import Base, PaperIdentifierModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IdentityStore:
    """CRUD for the paper_identifiers mapping table."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    # --- writes ---

    def upsert_identity(self, paper_id: int, identity: PaperIdentity) -> bool:
        """Insert if not exists. Returns True if created, False if already present."""
        if not identity:
            return False
        now = _utcnow()
        with self._provider.session() as session:
            existing = session.execute(
                select(PaperIdentifierModel).where(
                    PaperIdentifierModel.source == identity.source,
                    PaperIdentifierModel.external_id == identity.external_id,
                )
            ).scalar_one_or_none()
            if existing is not None:
                if existing.paper_id != paper_id:
                    existing.paper_id = paper_id
                    session.commit()
                return False
            row = PaperIdentifierModel(
                paper_id=paper_id,
                source=identity.source,
                external_id=identity.external_id,
                created_at=now,
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return False
            return True

    def upsert_identifiers(
        self, paper_id: int, identities: List[PaperIdentity]
    ) -> Dict[str, int]:
        created = 0
        for ident in identities:
            if self.upsert_identity(paper_id, ident):
                created += 1
        return {"total": len(identities), "created": created}

    # --- reads ---

    def resolve(self, source: str, external_id: str) -> Optional[int]:
        """Resolve (source, external_id) → papers.id. O(1) index lookup."""
        with self._provider.session() as session:
            row = session.execute(
                select(PaperIdentifierModel).where(
                    PaperIdentifierModel.source == source,
                    PaperIdentifierModel.external_id == external_id,
                )
            ).scalar_one_or_none()
            return int(row.paper_id) if row else None

    def resolve_any(self, external_id: str) -> Optional[int]:
        """Resolve an external_id across all sources."""
        with self._provider.session() as session:
            row = session.execute(
                select(PaperIdentifierModel).where(
                    PaperIdentifierModel.external_id == external_id,
                )
            ).scalar_one_or_none()
            return int(row.paper_id) if row else None

    def list_identities(self, paper_id: int) -> List[PaperIdentity]:
        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(PaperIdentifierModel).where(
                        PaperIdentifierModel.paper_id == paper_id
                    )
                )
                .scalars()
                .all()
            )
            return [PaperIdentity(source=r.source, external_id=r.external_id) for r in rows]

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass
