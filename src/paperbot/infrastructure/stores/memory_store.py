from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select, desc, or_, text
from sqlalchemy.exc import IntegrityError

from paperbot.infrastructure.stores.models import Base, MemoryAuditLogModel, MemoryItemModel, MemorySourceModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.memory.schema import MemoryCandidate


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_EMAIL_RX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_PHONE_RX = re.compile(r"\b(\+?\d[\d -]{7,}\d)\b")


def _estimate_pii_risk(text_value: str) -> int:
    s = (text_value or "").strip()
    if not s:
        return 0
    if "[REDACTED_" in s:
        return 1
    if _EMAIL_RX.search(s) or _PHONE_RX.search(s):
        return 2
    return 0


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
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        """
        Best-effort schema creation + lightweight SQLite column upgrades.

        Notes:
        - `create_all()` doesn't add columns to existing tables.
        - For local dev (default SQLite), we apply additive `ALTER TABLE ADD COLUMN` migrations.
        """
        Base.metadata.create_all(self._provider.engine)

        if not str(self.db_url).startswith("sqlite:"):
            return

        desired_columns: Dict[str, str] = {
            "workspace_id": "VARCHAR(64)",
            "scope_type": "VARCHAR(16) DEFAULT 'global'",
            "scope_id": "VARCHAR(64)",
            "status": "VARCHAR(16) DEFAULT 'approved'",
            "supersedes_id": "INTEGER",
            "expires_at": "DATETIME",
            "last_used_at": "DATETIME",
            "use_count": "INTEGER DEFAULT 0",
            "pii_risk": "INTEGER DEFAULT 0",
            "deleted_at": "DATETIME",
            "deleted_reason": "TEXT DEFAULT ''",
        }

        with self._provider.engine.connect() as conn:
            try:
                rows = conn.execute(text("PRAGMA table_info(memory_items)")).fetchall()
            except Exception:
                return
            existing = {r[1] for r in rows}  # (cid, name, type, notnull, dflt_value, pk)
            for col, ddl in desired_columns.items():
                if col in existing:
                    continue
                try:
                    conn.execute(text(f"ALTER TABLE memory_items ADD COLUMN {col} {ddl}"))
                except Exception:
                    pass
            try:
                conn.commit()
            except Exception:
                pass

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
        self,
        *,
        user_id: str,
        memories: Iterable[MemoryCandidate],
        source_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        status: Optional[str] = None,
        actor_id: str = "system",
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
                effective_scope_type = (getattr(m, "scope_type", None) or scope_type or "global").strip() or "global"
                effective_scope_id = getattr(m, "scope_id", None)
                if effective_scope_id is None:
                    effective_scope_id = scope_id

                # Dedup within a user's scope boundary (prevents cross-track pollution/dedup collisions).
                content_hash = _sha256_text(
                    f"{effective_scope_type}:{effective_scope_id or ''}:{m.kind}:{content}"
                )

                effective_status = getattr(m, "status", None) or status
                if not effective_status:
                    effective_status = "approved" if float(m.confidence) >= 0.60 else "pending"
                pii_risk = _estimate_pii_risk(content)
                if pii_risk >= 2 and effective_status == "approved":
                    effective_status = "pending"

                row = MemoryItemModel(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    scope_type=effective_scope_type,
                    scope_id=effective_scope_id,
                    kind=m.kind,
                    content=content,
                    content_hash=content_hash,
                    confidence=float(m.confidence),
                    status=effective_status,
                    pii_risk=pii_risk,
                    created_at=now,
                    updated_at=now,
                    source_id=source_id,
                )
                row.tags_json = json.dumps(list(m.tags or []), ensure_ascii=False)
                row.evidence_json = json.dumps(dict(m.evidence or {}), ensure_ascii=False)
                session.add(row)
                try:
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    skipped += 1
                    continue

                session.add(
                    MemoryAuditLogModel(
                        ts=now,
                        actor_id=actor_id,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        action="create",
                        item_id=row.id,
                        source_id=source_id,
                        detail_json=json.dumps(
                            {
                                "kind": row.kind,
                                "confidence": row.confidence,
                                "status": row.status,
                                "scope_type": getattr(row, "scope_type", "global"),
                                "scope_id": getattr(row, "scope_id", None),
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()
                session.refresh(row)
                created += 1
                created_rows.append(row)

        return created, skipped, created_rows

    def list_memories(
        self,
        *,
        user_id: str,
        limit: int = 100,
        kind: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        include_pending: bool = False,
        include_deleted: bool = False,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            stmt = select(MemoryItemModel).where(MemoryItemModel.user_id == user_id)
            if workspace_id is not None:
                stmt = stmt.where(MemoryItemModel.workspace_id == workspace_id)
            if scope_type is not None:
                if scope_type == "global":
                    stmt = stmt.where(or_(MemoryItemModel.scope_type == scope_type, MemoryItemModel.scope_type.is_(None)))
                else:
                    stmt = stmt.where(MemoryItemModel.scope_type == scope_type)
            if scope_id is not None:
                stmt = stmt.where(MemoryItemModel.scope_id == scope_id)
            if kind:
                stmt = stmt.where(MemoryItemModel.kind == kind)
            if not include_deleted:
                stmt = stmt.where(MemoryItemModel.deleted_at.is_(None))
            if status is not None:
                stmt = stmt.where(MemoryItemModel.status == status)
            elif not include_pending:
                stmt = stmt.where(MemoryItemModel.status == "approved")
            stmt = stmt.where(or_(MemoryItemModel.expires_at.is_(None), MemoryItemModel.expires_at > now))
            stmt = stmt.order_by(desc(MemoryItemModel.updated_at)).limit(int(limit))
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def get_items_by_ids(
        self,
        *,
        user_id: str,
        item_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """Get memory items by their IDs for a specific user."""
        if not item_ids:
            return []
        with self._provider.session() as session:
            stmt = (
                select(MemoryItemModel)
                .where(MemoryItemModel.user_id == user_id)
                .where(MemoryItemModel.id.in_(item_ids))
            )
            rows = session.execute(stmt).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 8,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        tokens = [t.strip() for t in (query or "").split() if t.strip()]
        if not tokens:
            return self.list_memories(
                user_id=user_id,
                limit=limit,
                workspace_id=workspace_id,
                scope_type=scope_type,
                scope_id=scope_id,
            )

        with self._provider.session() as session:
            stmt = select(MemoryItemModel).where(MemoryItemModel.user_id == user_id)
            if workspace_id is not None:
                stmt = stmt.where(MemoryItemModel.workspace_id == workspace_id)
            if scope_type is not None:
                if scope_type == "global":
                    stmt = stmt.where(or_(MemoryItemModel.scope_type == scope_type, MemoryItemModel.scope_type.is_(None)))
                else:
                    stmt = stmt.where(MemoryItemModel.scope_type == scope_type)
            if scope_id is not None:
                stmt = stmt.where(MemoryItemModel.scope_id == scope_id)
            now = datetime.now(timezone.utc)
            stmt = stmt.where(MemoryItemModel.deleted_at.is_(None))
            stmt = stmt.where(MemoryItemModel.status == "approved")
            stmt = stmt.where(or_(MemoryItemModel.expires_at.is_(None), MemoryItemModel.expires_at > now))
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
            d["score"] = score
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[: int(limit)] if s > 0] or self.list_memories(
            user_id=user_id,
            limit=limit,
            workspace_id=workspace_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    def touch_usage(self, *, item_ids: List[int], actor_id: str = "system") -> None:
        """
        Update last_used_at/use_count for retrieved items (best-effort).
        """
        if not item_ids:
            return
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            rows = session.execute(select(MemoryItemModel).where(MemoryItemModel.id.in_(item_ids))).scalars().all()
            for r in rows:
                r.last_used_at = now
                r.use_count = int(getattr(r, "use_count", 0) or 0) + 1
                session.add(
                    MemoryAuditLogModel(
                        ts=now,
                        actor_id=actor_id,
                        user_id=r.user_id,
                        workspace_id=r.workspace_id,
                        action="use",
                        item_id=r.id,
                        source_id=r.source_id,
                        detail_json=json.dumps({}, ensure_ascii=False),
                    )
                )
            session.commit()

    def update_item(
        self,
        *,
        user_id: str,
        item_id: int,
        actor_id: str = "system",
        content: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = session.execute(
                select(MemoryItemModel).where(MemoryItemModel.id == item_id, MemoryItemModel.user_id == user_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            if workspace_id is not None:
                row.workspace_id = workspace_id
            if scope_type is not None:
                row.scope_type = scope_type.strip() or "global"
            if scope_id is not None:
                row.scope_id = scope_id
            if kind is not None:
                row.kind = kind
            if content is not None:
                row.content = content.strip()
                row.content_hash = _sha256_text(
                    f"{(getattr(row, 'scope_type', None) or 'global')}:{(getattr(row, 'scope_id', None) or '')}:{row.kind}:{row.content}"
                )
                row.pii_risk = _estimate_pii_risk(row.content)
            elif scope_type is not None or scope_id is not None or kind is not None:
                # Scope/kind change requires re-hash even if content stays the same.
                row.content_hash = _sha256_text(
                    f"{(getattr(row, 'scope_type', None) or 'global')}:{(getattr(row, 'scope_id', None) or '')}:{row.kind}:{row.content}"
                )
            if tags is not None:
                row.tags_json = json.dumps([t for t in tags if str(t).strip()], ensure_ascii=False)
            if status is not None:
                row.status = status
            row.updated_at = now
            session.add(
                MemoryAuditLogModel(
                    ts=now,
                    actor_id=actor_id,
                    user_id=user_id,
                    workspace_id=row.workspace_id,
                    action="update",
                    item_id=row.id,
                    source_id=row.source_id,
                    detail_json=json.dumps({"status": row.status}, ensure_ascii=False),
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return None
            session.refresh(row)
            return self._row_to_dict(row)

    def soft_delete_item(self, *, user_id: str, item_id: int, actor_id: str = "system", reason: str = "") -> bool:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = session.execute(
                select(MemoryItemModel).where(MemoryItemModel.id == item_id, MemoryItemModel.user_id == user_id)
            ).scalar_one_or_none()
            if row is None:
                return False
            row.deleted_at = now
            row.deleted_reason = reason or ""
            row.updated_at = now
            session.add(
                MemoryAuditLogModel(
                    ts=now,
                    actor_id=actor_id,
                    user_id=user_id,
                    workspace_id=row.workspace_id,
                    action="delete",
                    item_id=row.id,
                    source_id=row.source_id,
                    detail_json=json.dumps({"reason": reason or ""}, ensure_ascii=False),
                )
            )
            session.commit()
            return True

    def soft_delete_by_scope(
        self,
        *,
        user_id: str,
        scope_type: str,
        scope_id: Optional[str],
        actor_id: str = "system",
        reason: str = "",
    ) -> int:
        """
        Soft-delete all memory items in a scope.

        Returns number of rows affected (best-effort; 0 if none found).
        """
        now = datetime.now(timezone.utc)
        scope_type = (scope_type or "global").strip() or "global"
        with self._provider.session() as session:
            stmt = select(MemoryItemModel).where(
                MemoryItemModel.user_id == user_id,
                MemoryItemModel.deleted_at.is_(None),
                MemoryItemModel.scope_type == scope_type,
            )
            if scope_id is not None:
                stmt = stmt.where(MemoryItemModel.scope_id == scope_id)
            rows = session.execute(stmt).scalars().all()
            if not rows:
                return 0
            for row in rows:
                row.deleted_at = now
                row.deleted_reason = reason or ""
                row.updated_at = now
                session.add(row)
            session.add(
                MemoryAuditLogModel(
                    ts=now,
                    actor_id=actor_id,
                    user_id=user_id,
                    workspace_id=None,
                    action="delete_scope",
                    item_id=None,
                    source_id=None,
                    detail_json=json.dumps(
                        {"scope_type": scope_type, "scope_id": scope_id, "reason": reason or ""}, ensure_ascii=False
                    ),
                )
            )
            session.commit()
            return len(rows)

    def bulk_update_items(
        self,
        *,
        user_id: str,
        item_ids: List[int],
        actor_id: str = "system",
        status: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bulk update status/scope for multiple items.

        Notes:
        - Recomputes content_hash when scope changes (to preserve scope-aware dedup semantics).
        - Returns updated items (best-effort; empty list if none).
        """
        if not item_ids:
            return []
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            rows = session.execute(
                select(MemoryItemModel).where(
                    MemoryItemModel.user_id == user_id,
                    MemoryItemModel.id.in_(item_ids),
                )
            ).scalars().all()
            if not rows:
                return []

            updated_ids: List[int] = []
            for row in rows:
                if status is not None:
                    row.status = status
                if scope_type is not None:
                    row.scope_type = scope_type.strip() or "global"
                if scope_id is not None:
                    row.scope_id = scope_id
                if scope_type is not None or scope_id is not None:
                    row.content_hash = _sha256_text(
                        f"{(getattr(row, 'scope_type', None) or 'global')}:{(getattr(row, 'scope_id', None) or '')}:{row.kind}:{row.content}"
                    )
                row.updated_at = now
                updated_ids.append(int(row.id))
                session.add(row)
                session.add(
                    MemoryAuditLogModel(
                        ts=now,
                        actor_id=actor_id,
                        user_id=user_id,
                        workspace_id=row.workspace_id,
                        action="bulk_update",
                        item_id=row.id,
                        source_id=row.source_id,
                        detail_json=json.dumps(
                            {"status": status, "scope_type": scope_type, "scope_id": scope_id}, ensure_ascii=False
                        ),
                    )
                )

            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return []

            refreshed = session.execute(
                select(MemoryItemModel).where(MemoryItemModel.user_id == user_id, MemoryItemModel.id.in_(updated_ids))
            ).scalars().all()
            return [self._row_to_dict(r) for r in refreshed]

    def hard_delete_item(self, *, user_id: str, item_id: int, actor_id: str = "system") -> bool:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = session.execute(
                select(MemoryItemModel).where(MemoryItemModel.id == item_id, MemoryItemModel.user_id == user_id)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.add(
                MemoryAuditLogModel(
                    ts=now,
                    actor_id=actor_id,
                    user_id=user_id,
                    workspace_id=row.workspace_id,
                    action="hard_delete",
                    item_id=row.id,
                    source_id=row.source_id,
                    detail_json=json.dumps({}, ensure_ascii=False),
                )
            )
            session.delete(row)
            session.commit()
            return True

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
            "workspace_id": getattr(r, "workspace_id", None),
            "scope_type": getattr(r, "scope_type", None) or "global",
            "scope_id": getattr(r, "scope_id", None),
            "kind": r.kind,
            "content": r.content,
            "confidence": r.confidence,
            "status": getattr(r, "status", "approved"),
            "supersedes_id": getattr(r, "supersedes_id", None),
            "expires_at": (r.expires_at.isoformat() if getattr(r, "expires_at", None) else None),
            "last_used_at": (r.last_used_at.isoformat() if getattr(r, "last_used_at", None) else None),
            "use_count": int(getattr(r, "use_count", 0) or 0),
            "pii_risk": int(getattr(r, "pii_risk", 0) or 0),
            "deleted_at": (r.deleted_at.isoformat() if getattr(r, "deleted_at", None) else None),
            "tags": [str(t) for t in tags if str(t).strip()],
            "evidence": evidence,
            "source_id": r.source_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
