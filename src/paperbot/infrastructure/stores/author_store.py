from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import delete, func, select

from paperbot.application.ports.author_port import AuthorPort
from paperbot.infrastructure.stores.models import AuthorModel, Base, PaperAuthorModel, PaperModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9\s_-]+", "", normalized).strip().lower()
    slug = re.sub(r"[-\s_]+", "-", normalized).strip("-")
    return slug or "unknown-author"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except Exception:
        return default


class AuthorStore(AuthorPort):
    """CRUD helpers for `authors` and `paper_authors` tables."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    def upsert_author(
        self,
        *,
        name: str,
        author_id: Optional[str] = None,
        slug: Optional[str] = None,
        h_index: Optional[int] = None,
        citation_count: Optional[int] = None,
        paper_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("author name is required")

        resolved_slug = str(slug or "").strip() or _slugify(normalized_name)
        resolved_author_id = str(author_id or "").strip() or f"name:{resolved_slug}"
        now = _utcnow()

        with self._provider.session() as session:
            row = (
                session.execute(
                    select(AuthorModel).where(AuthorModel.author_id == resolved_author_id).limit(1)
                )
                .scalars()
                .first()
            )
            if row is None:
                row = (
                    session.execute(
                        select(AuthorModel).where(AuthorModel.slug == resolved_slug).limit(1)
                    )
                    .scalars()
                    .first()
                )
            if row is None:
                row = (
                    session.execute(
                        select(AuthorModel)
                        .where(func.lower(AuthorModel.name) == normalized_name.lower())
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )

            if row is None:
                row = AuthorModel(
                    author_id=resolved_author_id,
                    name=normalized_name,
                    slug=resolved_slug,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)

            row.author_id = row.author_id or resolved_author_id
            row.name = normalized_name
            row.slug = resolved_slug
            if h_index is not None:
                row.h_index = _safe_int(h_index)
            if citation_count is not None:
                row.citation_count = _safe_int(citation_count)
            if paper_count is not None:
                row.paper_count = _safe_int(paper_count)
            if metadata is not None:
                row.metadata_json = json.dumps(metadata, ensure_ascii=False)
            row.updated_at = now

            session.commit()
            session.refresh(row)
            return self._author_to_dict(row)

    def get_author(self, author_id: int) -> Optional[dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(AuthorModel).where(AuthorModel.id == int(author_id)).limit(1)
            ).scalar_one_or_none()
            return self._author_to_dict(row) if row else None

    def list_authors(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(AuthorModel)
                    .order_by(func.lower(AuthorModel.name), AuthorModel.id)
                    .offset(max(0, int(offset)))
                    .limit(max(1, int(limit)))
                )
                .scalars()
                .all()
            )
            return [self._author_to_dict(row) for row in rows]

    def replace_paper_authors(
        self, *, paper_id: int, authors: Iterable[Any]
    ) -> list[dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            paper = session.execute(
                select(PaperModel).where(PaperModel.id == int(paper_id))
            ).scalar_one_or_none()
            if paper is None:
                raise ValueError(f"paper not found: {paper_id}")

            session.execute(
                delete(PaperAuthorModel).where(PaperAuthorModel.paper_id == int(paper_id))
            )

            # TODO: N+1 query — batch-lookup existing authors before the loop
            #  to avoid per-author DB queries (PR #112 review).
            for order, raw_author in enumerate(authors or []):
                author_name = ""
                resolved_author_id: Optional[str] = None
                resolved_slug: Optional[str] = None
                is_corresponding = False

                if isinstance(raw_author, dict):
                    author_name = str(raw_author.get("name") or "").strip()
                    resolved_author_id = str(raw_author.get("author_id") or "").strip() or None
                    resolved_slug = str(raw_author.get("slug") or "").strip() or None
                    is_corresponding = bool(raw_author.get("is_corresponding"))
                else:
                    author_name = str(raw_author or "").strip()

                if not author_name:
                    continue

                author_row = self._get_or_create_author_row(
                    session,
                    name=author_name,
                    author_id=resolved_author_id,
                    slug=resolved_slug,
                    now=now,
                )

                session.add(
                    PaperAuthorModel(
                        paper_id=int(paper_id),
                        author_id=int(author_row.id),
                        author_order=order,
                        is_corresponding=is_corresponding,
                        created_at=now,
                    )
                )

            session.commit()

        return self.get_paper_authors(paper_id=int(paper_id))

    def get_paper_authors(self, *, paper_id: int) -> list[dict[str, Any]]:
        with self._provider.session() as session:
            rows = session.execute(
                select(PaperAuthorModel, AuthorModel)
                .join(AuthorModel, AuthorModel.id == PaperAuthorModel.author_id)
                .where(PaperAuthorModel.paper_id == int(paper_id))
                .order_by(PaperAuthorModel.author_order.asc(), PaperAuthorModel.id.asc())
            ).all()

            payload: list[dict[str, Any]] = []
            for link, author in rows:
                payload.append(
                    {
                        "author_id": int(author.id),
                        "author_ref": author.author_id,
                        "name": author.name,
                        "slug": author.slug,
                        "author_order": int(link.author_order or 0),
                        "is_corresponding": bool(link.is_corresponding),
                    }
                )
            return payload

    @staticmethod
    def _get_or_create_author_row(
        session,
        *,
        name: str,
        author_id: Optional[str],
        slug: Optional[str],
        now: datetime,
    ) -> AuthorModel:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("author name is required")

        resolved_slug = str(slug or "").strip() or _slugify(normalized_name)
        resolved_author_id = str(author_id or "").strip() or f"name:{resolved_slug}"

        row = (
            session.execute(
                select(AuthorModel).where(AuthorModel.author_id == resolved_author_id).limit(1)
            )
            .scalars()
            .first()
        )
        if row is None:
            row = (
                session.execute(
                    select(AuthorModel).where(AuthorModel.slug == resolved_slug).limit(1)
                )
                .scalars()
                .first()
            )
        if row is None:
            row = (
                session.execute(
                    select(AuthorModel)
                    .where(func.lower(AuthorModel.name) == normalized_name.lower())
                    .limit(1)
                )
                .scalars()
                .first()
            )

        if row is None:
            row = AuthorModel(
                author_id=resolved_author_id,
                name=normalized_name,
                slug=resolved_slug,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return row

        row.name = normalized_name
        row.slug = resolved_slug
        row.updated_at = now
        session.flush()
        return row

    @staticmethod
    def _author_to_dict(row: AuthorModel) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        try:
            metadata = json.loads(row.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}

        return {
            "id": int(row.id),
            "author_id": row.author_id,
            "name": row.name,
            "slug": row.slug,
            "h_index": int(row.h_index or 0),
            "citation_count": int(row.citation_count or 0),
            "paper_count": int(row.paper_count or 0),
            "anchor_score": float(row.anchor_score or 0.0),
            "anchor_level": row.anchor_level or "background",
            "metadata": metadata,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
