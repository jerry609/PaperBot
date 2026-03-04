from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import logging
import re
import struct
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select, desc, or_, text
from sqlalchemy.exc import IntegrityError

from paperbot.infrastructure.stores.models import Base, MemoryAuditLogModel, MemoryItemModel, MemorySourceModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.memory.schema import MemoryCandidate

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536  # text-embedding-3-small default dimension
_FTS_TOKEN_RX = re.compile(r"[A-Za-z0-9_+-]+")


def _pack_embedding(vec: List[float]) -> bytes:
    """Pack a float32 vector into a byte blob for sqlite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _hybrid_merge(
    vec_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
    *,
    limit: int,
    vec_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """Merge vector and BM25 results with weighted scoring (Phase C).

    Scoring: final_score = 0.6 × vec_score + 0.4 × rank_score
    where rank_score = 1 / (1 + rank_position) for BM25 results.
    """
    scores: Dict[int, float] = {}
    items: Dict[int, Dict[str, Any]] = {}

    for rank, item in enumerate(vec_results):
        raw_id = item.get("id")
        if raw_id is None:
            logger.warning("Skipping vec result without id: %r", item)
            continue
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            logger.warning("Skipping vec result with invalid id=%r: %r", raw_id, item)
            continue
        vec_score = float(item.get("vec_score", 1.0 / (1.0 + rank)))
        scores[item_id] = scores.get(item_id, 0.0) + vec_weight * vec_score
        items[item_id] = item

    for rank, item in enumerate(fts_results):
        raw_id = item.get("id")
        if raw_id is None:
            logger.warning("Skipping FTS result without id: %r", item)
            continue
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            logger.warning("Skipping FTS result with invalid id=%r: %r", raw_id, item)
            continue
        bm25_score = 1.0 / (1.0 + rank)
        scores[item_id] = scores.get(item_id, 0.0) + bm25_weight * bm25_score
        if item_id not in items:
            items[item_id] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for item_id, score in ranked[:limit]:
        entry = items[item_id].copy()
        entry["hybrid_score"] = round(score, 4)
        result.append(entry)
    return result


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

    def __init__(
        self,
        db_url: Optional[str] = None,
        *,
        auto_create_schema: bool = True,
        embedding_provider=None,
    ):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        # embedding_provider: None = lazy-init, False = permanently unavailable, else provider
        self._embedding_provider = embedding_provider
        # Bounded pool prevents unbounded daemon thread growth on heavy writes.
        self._embed_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="memory-embed",
        )
        atexit.register(self._shutdown_embed_executor)
        self._vec_available = False
        if str(self.db_url).startswith("sqlite:"):
            self._try_enable_vec_extension()
        if auto_create_schema:
            self._ensure_schema()

    def _try_enable_vec_extension(self) -> None:
        """Register sqlite-vec extension loader on every new SQLAlchemy connection."""
        try:
            import sqlite_vec  # type: ignore  # noqa: PLC0415
            from sqlalchemy import event

            @event.listens_for(self._provider.engine, "connect")
            def _load_vec(dbapi_conn, _):
                dbapi_conn.enable_load_extension(True)
                sqlite_vec.load(dbapi_conn)
                dbapi_conn.enable_load_extension(False)

            # Probe: confirm the extension is loadable right now.
            with self._provider.engine.connect() as conn:
                conn.execute(text("SELECT vec_version()"))
            self._vec_available = True
            logger.debug("sqlite-vec loaded, vector search enabled")
        except Exception:
            logger.debug("sqlite-vec unavailable, vector search disabled")

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
            "embedding": "BLOB",
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
            self._ensure_fts5(conn)
            if self._vec_available:
                self._ensure_vec_table(conn)
            try:
                conn.commit()
            except Exception:
                pass

    @staticmethod
    def _ensure_fts5(conn) -> None:
        """Create FTS5 virtual table + sync triggers if they don't exist (SQLite only)."""
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')")
                ).fetchall()
            }
            if "memory_items_fts" not in tables:
                conn.execute(
                    text(
                        "CREATE VIRTUAL TABLE memory_items_fts"
                        " USING fts5(content, tokenize='porter ascii')"
                    )
                )
                # Back-fill existing approved rows.
                conn.execute(
                    text(
                        "INSERT INTO memory_items_fts(rowid, content)"
                        " SELECT id, content FROM memory_items WHERE deleted_at IS NULL"
                    )
                )

            triggers = {
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='trigger'")
                ).fetchall()
            }
            if "memory_items_fts_ai" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER memory_items_fts_ai"
                        " AFTER INSERT ON memory_items BEGIN"
                        "   INSERT INTO memory_items_fts(rowid, content)"
                        "   VALUES (new.id, new.content);"
                        " END"
                    )
                )
            if "memory_items_fts_ad" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER memory_items_fts_ad"
                        " AFTER DELETE ON memory_items BEGIN"
                        "   DELETE FROM memory_items_fts WHERE rowid = old.id;"
                        " END"
                    )
                )
            if "memory_items_fts_au" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER memory_items_fts_au"
                        " AFTER UPDATE OF content ON memory_items BEGIN"
                        "   DELETE FROM memory_items_fts WHERE rowid = old.id;"
                        "   INSERT INTO memory_items_fts(rowid, content)"
                        "   VALUES (new.id, new.content);"
                        " END"
                    )
                )
        except Exception:
            pass  # FTS5 not available or already set up — degrade silently

    @staticmethod
    def _ensure_vec_table(conn, dim: int = _EMBEDDING_DIM) -> None:
        """Create vec_items sqlite-vec virtual table + sync triggers if absent."""
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')")
                ).fetchall()
            }
            if "vec_items" not in tables:
                conn.execute(
                    text(f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{dim}])")
                )
                # Back-fill existing embeddings.
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO vec_items(rowid, embedding)"
                        " SELECT id, embedding FROM memory_items"
                        " WHERE embedding IS NOT NULL AND deleted_at IS NULL"
                    )
                )
            triggers = {
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='trigger'")
                ).fetchall()
            }
            if "memory_items_vec_ai" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER memory_items_vec_ai"
                        " AFTER INSERT ON memory_items"
                        " WHEN new.embedding IS NOT NULL BEGIN"
                        "   INSERT OR REPLACE INTO vec_items(rowid, embedding)"
                        "   VALUES (new.id, new.embedding);"
                        " END"
                    )
                )
            if "memory_items_vec_au" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER memory_items_vec_au"
                        " AFTER UPDATE OF embedding ON memory_items BEGIN"
                        "   DELETE FROM vec_items WHERE rowid = old.id;"
                        "   INSERT OR IGNORE INTO vec_items(rowid, embedding)"
                        "   SELECT new.id, new.embedding WHERE new.embedding IS NOT NULL;"
                        " END"
                    )
                )
            if "memory_items_vec_ad" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER memory_items_vec_ad"
                        " AFTER DELETE ON memory_items BEGIN"
                        "   DELETE FROM vec_items WHERE rowid = old.id;"
                        " END"
                    )
                )
        except Exception:
            pass  # sqlite-vec not available — degrade silently

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
                # Generate and store embedding asynchronously (best-effort).
                self._schedule_embedding(row.id, content)

        return created, skipped, created_rows

    def _schedule_embedding(self, row_id: int, content: str) -> None:
        try:
            self._embed_executor.submit(self._store_embedding, row_id, content)
        except RuntimeError:
            # Executor may already be shutting down during process teardown.
            logger.debug("Embedding executor unavailable; skipping row_id=%s", row_id)

    def _get_embedding_provider(self):
        """Lazy-initialise embedding provider; returns None if unavailable."""
        if self._embedding_provider is False:
            return None
        if self._embedding_provider is not None:
            return self._embedding_provider
        try:
            from paperbot.context_engine.embeddings import (  # noqa: PLC0415
                try_build_default_embedding_provider,
            )

            provider = try_build_default_embedding_provider()
            self._embedding_provider = provider if provider is not None else False
            return provider
        except Exception:
            self._embedding_provider = False
            return None

    def _store_embedding(self, row_id: int, content: str) -> None:
        """Generate embedding for *content* and persist it (best-effort)."""
        provider = self._get_embedding_provider()
        if provider is None:
            return
        try:
            vec = provider.embed(content)
            if vec is None:
                return
            blob = _pack_embedding(vec)
            with self._provider.engine.connect() as conn:
                conn.execute(
                    text("UPDATE memory_items SET embedding = :blob WHERE id = :rid"),
                    {"blob": blob, "rid": row_id},
                )
                # The memory_items_vec_au trigger keeps vec_items in sync; no
                # manual DELETE/INSERT needed here.
                conn.commit()
        except Exception:  # noqa: BLE001 — non-critical, degrade gracefully
            logger.warning(
                "Failed to store embedding for memory item %d", row_id, exc_info=True
            )

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

        _fallback = lambda: self.list_memories(  # noqa: E731
            user_id=user_id, limit=limit, workspace_id=workspace_id,
            scope_type=scope_type, scope_id=scope_id,
        )
        _scope = dict(
            user_id=user_id, limit=limit,
            workspace_id=workspace_id, scope_type=scope_type, scope_id=scope_id,
        )

        # --- Phase B+C: vector search + hybrid fusion ---
        vec_results: Optional[List[Dict[str, Any]]] = None
        provider = self._get_embedding_provider()
        if provider is not None:
            try:
                query_vec = provider.embed(query[:500])
                if query_vec is not None:
                    vec_results = self._search_vec(query_vec=query_vec, **_scope)
            except Exception:  # noqa: BLE001
                pass

        # --- Phase A: FTS5 BM25 search ---
        fts_results = self._search_fts5(tokens=tokens, **_scope)

        # Hybrid fusion when both channels return results.
        if vec_results is not None and fts_results is not None:
            merged = _hybrid_merge(vec_results, fts_results, limit=limit)
            return merged or _fallback()

        if vec_results is not None:
            return vec_results or _fallback()

        if fts_results is not None:
            return fts_results or _fallback()

        return self._search_like(tokens=tokens, **_scope)

    def _search_fts5(
        self,
        *,
        user_id: str,
        tokens: List[str],
        limit: int,
        workspace_id: Optional[str],
        scope_type: Optional[str],
        scope_id: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        FTS5 BM25 search. Returns a list on success, None if FTS5 is unavailable.
        Results are already filtered by user_id / scope / status.
        """
        if not str(self.db_url).startswith("sqlite:"):
            return None  # FTS5 is SQLite-only

        # Build a safe FTS5 query: wrap each token in double quotes to treat as
        # phrase tokens, joined with AND (FTS5 default when space-separated).
        def _escape_fts(token: str) -> str:
            safe_parts = _FTS_TOKEN_RX.findall(token or "")
            clean = " ".join(safe_parts).strip()
            if not clean:
                return ""
            return '"' + clean.replace('"', '""') + '"'

        escaped_tokens = [_escape_fts(t) for t in tokens[:8]]
        escaped_tokens = [t for t in escaped_tokens if t]
        if not escaped_tokens:
            return []
        fts_query = " ".join(escaped_tokens)

        try:
            with self._provider.engine.connect() as conn:
                # Join with memory_items so user_id filtering and LIMIT both
                # apply only to the current user's data (multi-tenant isolation).
                fts_rows = conn.execute(
                    text(
                        "SELECT f.rowid FROM memory_items_fts f"
                        " JOIN memory_items m ON m.id = f.rowid"
                        " WHERE f MATCH :q"
                        " AND m.user_id = :uid"
                        " ORDER BY rank LIMIT 250"
                    ),
                    {"q": fts_query, "uid": user_id},
                ).fetchall()
        except Exception:
            return None  # FTS5 table not available yet

        if not fts_rows:
            return []

        candidate_ids = [r[0] for r in fts_rows]
        # Preserve FTS5 BM25 rank order via a mapping.
        rank_map = {rid: idx for idx, rid in enumerate(candidate_ids)}

        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            stmt = (
                select(MemoryItemModel)
                .where(MemoryItemModel.id.in_(candidate_ids))
                .where(MemoryItemModel.user_id == user_id)
                .where(MemoryItemModel.deleted_at.is_(None))
                .where(MemoryItemModel.status == "approved")
                .where(
                    or_(
                        MemoryItemModel.expires_at.is_(None),
                        MemoryItemModel.expires_at > now,
                    )
                )
            )
            if workspace_id is not None:
                stmt = stmt.where(MemoryItemModel.workspace_id == workspace_id)
            if scope_type is not None:
                if scope_type == "global":
                    stmt = stmt.where(
                        or_(
                            MemoryItemModel.scope_type == scope_type,
                            MemoryItemModel.scope_type.is_(None),
                        )
                    )
                else:
                    stmt = stmt.where(MemoryItemModel.scope_type == scope_type)
            if scope_id is not None:
                stmt = stmt.where(MemoryItemModel.scope_id == scope_id)

            rows = session.execute(stmt).scalars().all()

        results = [self._row_to_dict(r) for r in rows]
        # Sort by FTS5 BM25 rank (lower rank index = better match).
        results.sort(key=lambda d: rank_map.get(int(d.get("id", 0)), 9999))
        return results[: int(limit)]

    def _search_like(
        self,
        *,
        user_id: str,
        tokens: List[str],
        limit: int,
        workspace_id: Optional[str],
        scope_type: Optional[str],
        scope_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Legacy LIKE + token-overlap scoring fallback."""
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            stmt = select(MemoryItemModel).where(MemoryItemModel.user_id == user_id)
            if workspace_id is not None:
                stmt = stmt.where(MemoryItemModel.workspace_id == workspace_id)
            if scope_type is not None:
                if scope_type == "global":
                    stmt = stmt.where(
                        or_(
                            MemoryItemModel.scope_type == scope_type,
                            MemoryItemModel.scope_type.is_(None),
                        )
                    )
                else:
                    stmt = stmt.where(MemoryItemModel.scope_type == scope_type)
            if scope_id is not None:
                stmt = stmt.where(MemoryItemModel.scope_id == scope_id)
            stmt = stmt.where(MemoryItemModel.deleted_at.is_(None))
            stmt = stmt.where(MemoryItemModel.status == "approved")
            stmt = stmt.where(
                or_(MemoryItemModel.expires_at.is_(None), MemoryItemModel.expires_at > now)
            )
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

    def _search_vec(
        self,
        *,
        user_id: str,
        query_vec: List[float],
        limit: int,
        workspace_id: Optional[str],
        scope_type: Optional[str],
        scope_id: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """sqlite-vec ANN search. Returns None if vec is unavailable or errors."""
        if not self._vec_available:
            return None
        query_blob = _pack_embedding(query_vec)
        try:
            with self._provider.engine.connect() as conn:
                # Join with memory_items so user_id filtering and LIMIT both
                # apply only to the current user's data (multi-tenant isolation).
                rows = conn.execute(
                    text(
                        "SELECT v.rowid, v.distance FROM vec_items v"
                        " JOIN memory_items m ON m.id = v.rowid"
                        " WHERE v.embedding MATCH :blob"
                        " AND m.user_id = :uid"
                        " ORDER BY v.distance LIMIT :k"
                    ),
                    {"blob": query_blob, "k": limit * 5, "uid": user_id},
                ).fetchall()
        except Exception:  # noqa: BLE001 — vec table may not exist yet
            return None

        if not rows:
            return []

        candidate_ids = [r[0] for r in rows]
        distance_map = {int(r[0]): float(r[1]) for r in rows}

        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            stmt = (
                select(MemoryItemModel)
                .where(MemoryItemModel.id.in_(candidate_ids))
                .where(MemoryItemModel.user_id == user_id)
                .where(MemoryItemModel.deleted_at.is_(None))
                .where(MemoryItemModel.status == "approved")
                .where(
                    or_(
                        MemoryItemModel.expires_at.is_(None),
                        MemoryItemModel.expires_at > now,
                    )
                )
            )
            if workspace_id is not None:
                stmt = stmt.where(MemoryItemModel.workspace_id == workspace_id)
            if scope_type is not None:
                if scope_type == "global":
                    stmt = stmt.where(
                        or_(
                            MemoryItemModel.scope_type == scope_type,
                            MemoryItemModel.scope_type.is_(None),
                        )
                    )
                else:
                    stmt = stmt.where(MemoryItemModel.scope_type == scope_type)
            if scope_id is not None:
                stmt = stmt.where(MemoryItemModel.scope_id == scope_id)
            db_rows = session.execute(stmt).scalars().all()

        results = []
        for r in db_rows:
            d = self._row_to_dict(r)
            dist = distance_map.get(int(d.get("id", 0)), 1e9)
            d["vec_distance"] = dist
            d["vec_score"] = 1.0 / (1.0 + dist)
            results.append(d)
        results.sort(key=lambda x: x["vec_distance"])
        return results[:limit]

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
        self._shutdown_embed_executor()
        try:
            self._provider.engine.dispose()
        except Exception:
            pass

    def _shutdown_embed_executor(self) -> None:
        executor = getattr(self, "_embed_executor", None)
        if executor is None:
            return
        try:
            executor.shutdown(wait=True, cancel_futures=False)
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
