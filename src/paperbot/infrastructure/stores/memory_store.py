from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import logging
import re
import struct
from datetime import datetime, timedelta, timezone
from math import exp, log
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import bindparam, select, desc, or_, text
from sqlalchemy.exc import IntegrityError

from paperbot.application.ports.memory_port import MemoryPort
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


def _to_decay_lambda(half_life_days: float) -> float:
    """Convert half-life in days to exponential decay lambda.

    Follows the standard formula: λ = ln(2) / halfLifeDays
    such that the multiplier is exactly 0.5 at the half-life point.
    (Aligned with OpenClaw's temporal-decay.ts:toDecayLambda)
    """
    if not (half_life_days > 0 and half_life_days == half_life_days):
        return 0.0
    return log(2) / half_life_days


def _is_evergreen_memory(item: Dict[str, Any]) -> bool:
    """Check if a memory is 'evergreen' and should not be decayed.

    Global preferences and user-preference memories represent stable knowledge
    that should not lose relevance over time.
    (Inspired by OpenClaw's isEvergreenMemoryPath)
    """
    scope_type = str(item.get("scope_type", "")).lower()
    kind = str(item.get("kind", "")).lower()
    return scope_type == "global" or kind == "preference"


def _memory_relevance_score(item: Dict[str, Any]) -> float:
    """Pick the strongest available retrieval relevance score for decay ranking."""
    for field_name in ("hybrid_score", "vec_score", "confidence"):
        raw_score = item.get(field_name)
        if raw_score is not None:
            return float(raw_score)
    return 0.5


def _decay_score(
    item: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = 30.0,
    relevance_weight: float = 0.7,
    recency_weight: float = 0.2,
    usage_weight: float = 0.1,
) -> float:
    """Compute a combined decay-aware score for a memory item.

    Score = relevance × 0.7 + recency × 0.2 + usage × 0.1
    where recency = exp(-λ × age_days), λ = ln(2) / half_life_days
    and   usage   = min(use_count / 10, 1.0)

    Evergreen memories (global scope, preference kind) skip recency decay
    and receive a recency score of 1.0.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    relevance = _memory_relevance_score(item)

    # Evergreen memories are immune to time-based decay.
    if _is_evergreen_memory(item):
        recency_score = 1.0
    else:
        created_str = item.get("created_at")
        if isinstance(created_str, str):
            try:
                created = datetime.fromisoformat(created_str)
            except (ValueError, TypeError):
                created = now
        elif isinstance(created_str, datetime):
            created = created_str
        else:
            created = now

        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        decay_lambda = _to_decay_lambda(half_life_days)
        recency_score = exp(-decay_lambda * age_days) if decay_lambda > 0 else 1.0

    use_count = int(item.get("use_count", 0) or 0)
    usage_score = min(use_count / 10.0, 1.0)

    return (
        relevance_weight * relevance
        + recency_weight * recency_score
        + usage_weight * usage_score
    )


def _apply_decay_ranking(
    results: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = 30.0,
) -> List[Dict[str, Any]]:
    """Re-rank search results by decay-aware score (descending)."""
    if not results:
        return results
    if now is None:
        now = datetime.now(timezone.utc)
    for item in results:
        item["decay_score"] = _decay_score(
            item,
            now=now,
            half_life_days=half_life_days,
        )
    results.sort(key=lambda x: x.get("decay_score", 0.0), reverse=True)
    return results


def _tokenize_for_mmr(text_value: str) -> set[str]:
    return {t.lower() for t in _FTS_TOKEN_RX.findall(text_value or "") if t.strip()}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0


def _mmr_relevance_score(item: Dict[str, Any]) -> float:
    for key in ("decay_score", "hybrid_score", "vec_score", "score", "confidence"):
        raw = item.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _apply_mmr_rerank(
    results: List[Dict[str, Any]],
    *,
    lambda_value: float = 0.7,
) -> List[Dict[str, Any]]:
    if len(results) <= 1:
        return list(results)

    lam = max(0.0, min(1.0, float(lambda_value)))
    if lam >= 1.0:
        return sorted(results, key=_mmr_relevance_score, reverse=True)

    scored = sorted(results, key=_mmr_relevance_score, reverse=True)
    token_cache = {
        id(item): _tokenize_for_mmr(str(item.get("content") or item.get("snippet") or ""))
        for item in scored
    }

    selected: List[Dict[str, Any]] = []
    remaining = list(scored)
    while remaining:
        best_item = None
        best_score = float("-inf")
        for cand in remaining:
            relevance = _mmr_relevance_score(cand)
            cand_tokens = token_cache.get(id(cand), set())
            max_sim = 0.0
            for picked in selected:
                sim = _jaccard_similarity(
                    cand_tokens,
                    token_cache.get(id(picked), set()),
                )
                if sim > max_sim:
                    max_sim = sim
            mmr_score = lam * relevance - (1.0 - lam) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = cand
        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)

    return selected


class SqlAlchemyMemoryStore(MemoryPort):
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
        Lightweight schema bootstrap for SQLite (dev/test) + column upgrades.

        Notes:
        - For production (PostgreSQL), schema is managed exclusively by Alembic.
        - For SQLite (dev/test), we create tables and apply additive ALTER TABLE
          ADD COLUMN migrations for new columns not yet in Alembic.
        """
        if not str(self.db_url).startswith("sqlite:"):
            return

        self._provider.ensure_tables(Base.metadata)

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
                    expires_at=now + timedelta(days=365),
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
        min_score: float = 0.0,
        candidate_multiplier: int = 4,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        half_life_days: float = 30.0,
    ) -> List[Dict[str, Any]]:
        candidate_multiplier = max(1, int(candidate_multiplier or 1))
        candidate_limit = min(250, max(int(limit), int(limit) * candidate_multiplier))
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
            user_id=user_id, limit=candidate_limit, workspace_id=workspace_id,
            scope_type=scope_type, scope_id=scope_id,
        )
        _scope = dict(
            user_id=user_id, limit=candidate_limit,
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
            results = _hybrid_merge(vec_results, fts_results, limit=limit)
            results = results or _fallback()
        elif vec_results is not None:
            results = vec_results or _fallback()
        elif fts_results is not None:
            results = fts_results or _fallback()
        else:
            results = self._search_like(tokens=tokens, **_scope)

        # Apply decay-aware re-ranking.
        results = _apply_decay_ranking(results, half_life_days=half_life_days)
        if mmr_enabled:
            results = _apply_mmr_rerank(results, lambda_value=mmr_lambda)
        threshold = max(0.0, float(min_score or 0.0))
        if threshold > 0:
            results = [r for r in results if float(r.get("decay_score", 0.0)) >= threshold]
        results = results[:limit]

        # Auto-touch usage for search hits.
        hit_ids = [int(r["id"]) for r in results if r.get("id")]
        if hit_ids:
            try:
                self.touch_usage(item_ids=hit_ids, actor_id="search")
            except Exception:  # noqa: BLE001 — best-effort
                pass

        return results

    def search_memories_batch(
        self,
        *,
        user_id: str,
        query: str,
        scope_ids: List[str],
        scope_type: str = "track",
        limit_per_scope: int = 4,
        workspace_id: Optional[str] = None,
        min_score: float = 0.0,
        candidate_multiplier: int = 4,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        half_life_days: float = 30.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search memories across multiple scopes in a single query.

        Returns a dict keyed by scope_id, each value a list of matching memories.
        """
        if not scope_ids:
            return {}

        tokens = [t.strip() for t in (query or "").split() if t.strip()]
        if not tokens:
            return {sid: [] for sid in scope_ids}

        normalized_scope_ids = list(dict.fromkeys([str(sid) for sid in scope_ids if str(sid)]))
        allowed_scope_ids = set(normalized_scope_ids)
        candidate_multiplier = max(1, int(candidate_multiplier or 1))
        global_limit = min(
            500,
            max(
                int(limit_per_scope) * len(allowed_scope_ids),
                int(limit_per_scope) * candidate_multiplier * len(allowed_scope_ids),
            ),
        )

        _scope = dict(
            user_id=user_id,
            limit=max(1, global_limit),
            workspace_id=workspace_id,
            scope_type=scope_type,
            scope_id=None,
        )

        vec_results: Optional[List[Dict[str, Any]]] = None
        provider = self._get_embedding_provider()
        if provider is not None:
            try:
                query_vec = provider.embed(query[:500])
                if query_vec is not None:
                    vec_results = self._search_vec(
                        query_vec=query_vec,
                        scope_ids=normalized_scope_ids,
                        **_scope,
                    )
            except Exception:  # noqa: BLE001
                pass

        fts_results = self._search_fts5(tokens=tokens, scope_ids=normalized_scope_ids, **_scope)

        if vec_results is not None and fts_results is not None:
            results = _hybrid_merge(vec_results, fts_results, limit=max(1, global_limit))
        elif vec_results is not None:
            results = vec_results
        elif fts_results is not None:
            results = fts_results
        else:
            now = datetime.now(timezone.utc)
            with self._provider.session() as session:
                stmt = (
                    select(MemoryItemModel)
                    .where(MemoryItemModel.user_id == user_id)
                    .where(MemoryItemModel.scope_type == scope_type)
                    .where(MemoryItemModel.scope_id.in_(normalized_scope_ids))
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
                ors = [MemoryItemModel.content.contains(t) for t in tokens[:8]]
                if ors:
                    stmt = stmt.where(or_(*ors))
                rows = session.execute(stmt.limit(max(1, global_limit))).scalars().all()

            results = []
            for row in rows:
                item = self._row_to_dict(row)
                content = (item.get("content") or "").lower()
                item["score"] = sum(2 for t in tokens if t.lower() in content)
                results.append(item)
            results.sort(key=lambda x: x.get("score", 0), reverse=True)

        filtered = []
        for item in results:
            sid = str(item.get("scope_id") or "")
            if sid in allowed_scope_ids:
                filtered.append(item)

        filtered = _apply_decay_ranking(filtered, half_life_days=half_life_days)
        threshold = max(0.0, float(min_score or 0.0))
        if threshold > 0:
            filtered = [r for r in filtered if float(r.get("decay_score", 0.0)) >= threshold]

        grouped: Dict[str, List[Dict[str, Any]]] = {sid: [] for sid in normalized_scope_ids}
        for item in filtered:
            sid = str(item.get("scope_id") or "")
            if sid in grouped:
                grouped[sid].append(item)

        for sid in grouped:
            if mmr_enabled:
                grouped[sid] = _apply_mmr_rerank(grouped[sid], lambda_value=mmr_lambda)
            grouped[sid] = grouped[sid][:limit_per_scope]

        hit_ids = sorted(
            {
                int(item["id"])
                for hits in grouped.values()
                for item in hits
                if item.get("id")
            }
        )
        if hit_ids:
            try:
                self.touch_usage(item_ids=hit_ids, actor_id="search")
            except Exception:  # noqa: BLE001 — best-effort
                pass

        return grouped

    def _search_fts5(
        self,
        *,
        user_id: str,
        tokens: List[str],
        limit: int,
        workspace_id: Optional[str],
        scope_type: Optional[str],
        scope_id: Optional[str],
        scope_ids: Optional[List[str]] = None,
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
                where_parts = [
                    "f MATCH :q",
                    "m.user_id = :uid",
                ]
                params: Dict[str, Any] = {"q": fts_query, "uid": user_id}
                use_scope_ids = [sid for sid in (scope_ids or []) if sid]
                use_scope_ids = list(dict.fromkeys([str(sid) for sid in use_scope_ids]))
                if use_scope_ids:
                    where_parts.append("m.scope_id IN :scope_ids")
                if scope_type is not None and scope_type != "global":
                    where_parts.append("m.scope_type = :scope_type")
                    params["scope_type"] = scope_type

                sql = (
                    "SELECT f.rowid FROM memory_items_fts f"
                    " JOIN memory_items m ON m.id = f.rowid"
                    " WHERE " + " AND ".join(where_parts) +
                    " ORDER BY rank LIMIT 250"
                )
                stmt = text(sql)
                if use_scope_ids:
                    stmt = stmt.bindparams(bindparam("scope_ids", expanding=True))
                    params["scope_ids"] = use_scope_ids

                # Join with memory_items so user_id filtering and LIMIT both
                # apply only to the current user's data (multi-tenant isolation).
                fts_rows = conn.execute(stmt, params).fetchall()
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
            if scope_ids:
                use_scope_ids = [str(sid) for sid in scope_ids if sid]
                if use_scope_ids:
                    stmt = stmt.where(MemoryItemModel.scope_id.in_(use_scope_ids))

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
        scope_ids: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """sqlite-vec ANN search. Returns None if vec is unavailable or errors."""
        if not self._vec_available:
            return None
        query_blob = _pack_embedding(query_vec)
        try:
            with self._provider.engine.connect() as conn:
                where_parts = [
                    "v.embedding MATCH :blob",
                    "m.user_id = :uid",
                ]
                params: Dict[str, Any] = {
                    "blob": query_blob,
                    "k": limit * 5,
                    "uid": user_id,
                }
                use_scope_ids = [sid for sid in (scope_ids or []) if sid]
                use_scope_ids = list(dict.fromkeys([str(sid) for sid in use_scope_ids]))
                if use_scope_ids:
                    where_parts.append("m.scope_id IN :scope_ids")
                if scope_type is not None and scope_type != "global":
                    where_parts.append("m.scope_type = :scope_type")
                    params["scope_type"] = scope_type

                sql = (
                    "SELECT v.rowid, v.distance FROM vec_items v"
                    " JOIN memory_items m ON m.id = v.rowid"
                    " WHERE " + " AND ".join(where_parts) +
                    " ORDER BY v.distance LIMIT :k"
                )
                stmt = text(sql)
                if use_scope_ids:
                    stmt = stmt.bindparams(bindparam("scope_ids", expanding=True))
                    params["scope_ids"] = use_scope_ids

                # Join with memory_items so user_id filtering and LIMIT both
                # apply only to the current user's data (multi-tenant isolation).
                rows = conn.execute(stmt, params).fetchall()
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
            if scope_ids:
                use_scope_ids = [str(sid) for sid in scope_ids if sid]
                if use_scope_ids:
                    stmt = stmt.where(MemoryItemModel.scope_id.in_(use_scope_ids))
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
