from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select, desc, or_
from sqlalchemy.exc import IntegrityError

from paperbot.infrastructure.stores.models import Base, MemoryItemModel, MemorySourceModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.memory.schema import MemoryCandidate


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SqlAlchemyMemoryStore:
    """
    SQLite-backed long-term memory store.

    Notes:
    - Uses a unique constraint (user_id, content_hash) to deduplicate extracted memories.
    - Stores provenance via MemorySourceModel rows.
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)

    def upsert_source(
        self,
        *,
        user_id: str,
        platform: str,
        filename: str,
        raw_bytes: bytes,
        message_count: int,
        conversation_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemorySourceModel:
        sha256 = _sha256_bytes(raw_bytes)
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            existing = session.execute(
                select(MemorySourceModel).where(
                    MemorySourceModel.user_id == user_id,
                    MemorySourceModel.sha256 == sha256,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return existing

            src = MemorySourceModel(
                user_id=user_id,
                platform=platform or "unknown",
                filename=filename or "",
                sha256=sha256,
                ingested_at=now,
                message_count=int(message_count or 0),
                conversation_count=int(conversation_count or 0),
            )
            src.metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
            session.add(src)
            session.commit()
            session.refresh(src)
            return src

    def add_memories(
        self, *, user_id: str, memories: Iterable[MemoryCandidate], source_id: Optional[int] = None
    ) -> Tuple[int, int, List[MemoryItemModel]]:
        """
        Returns: (created_count, skipped_count, created_rows)
        """
        now = datetime.now(timezone.utc)
        created = 0
        skipped = 0
        created_rows: List[MemoryItemModel] = []
        with self._provider.session() as session:
            for m in memories:
                content = (m.content or "").strip()
                if not content:
                    skipped += 1
                    continue
                content_hash = _sha256_text(f"{m.kind}:{content}")

                row = MemoryItemModel(
                    user_id=user_id,
                    kind=m.kind,
                    content=content,
                    content_hash=content_hash,
                    confidence=float(m.confidence),
                    created_at=now,
                    updated_at=now,
                    source_id=source_id,
                )
                row.tags_json = json.dumps(list(m.tags or []), ensure_ascii=False)
                row.evidence_json = json.dumps(dict(m.evidence or {}), ensure_ascii=False)
                session.add(row)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    skipped += 1
                    continue

                session.refresh(row)
                created += 1
                created_rows.append(row)

        return created, skipped, created_rows

    def list_memories(self, *, user_id: str, limit: int = 100, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            stmt = select(MemoryItemModel).where(MemoryItemModel.user_id == user_id)
            if kind:
                stmt = stmt.where(MemoryItemModel.kind == kind)
            stmt = stmt.order_by(desc(MemoryItemModel.updated_at)).limit(int(limit))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def search_memories(self, *, user_id: str, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        tokens = [t.strip() for t in (query or "").split() if t.strip()]
        if not tokens:
            return self.list_memories(user_id=user_id, limit=limit)

        with self._provider.session() as session:
            stmt = select(MemoryItemModel).where(MemoryItemModel.user_id == user_id)
            # Coarse filter in SQL
            ors = [MemoryItemModel.content.contains(t) for t in tokens[:8]]
            if ors:
                stmt = stmt.where(or_(*ors))
            rows = session.execute(stmt.limit(250)).scalars().all()

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for r in rows:
            d = self._row_to_dict(r)
            score = 0
            content = (d.get("content") or "").lower()
            tags = " ".join(d.get("tags") or []).lower()
            for t in tokens:
                tl = t.lower()
                if tl in content:
                    score += 2
                if tl in tags:
                    score += 1
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[: int(limit)] if s > 0] or self.list_memories(user_id=user_id, limit=limit)

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass

    @staticmethod
    def _row_to_dict(r: MemoryItemModel) -> Dict[str, Any]:
        try:
            tags = json.loads(r.tags_json or "[]")
            if not isinstance(tags, list):
                tags = []
        except Exception:
            tags = []
        try:
            evidence = json.loads(r.evidence_json or "{}")
            if not isinstance(evidence, dict):
                evidence = {}
        except Exception:
            evidence = {}

        return {
            "id": r.id,
            "user_id": r.user_id,
            "kind": r.kind,
            "content": r.content,
            "confidence": r.confidence,
            "tags": [str(t) for t in tags if str(t).strip()],
            "evidence": evidence,
            "source_id": r.source_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }

