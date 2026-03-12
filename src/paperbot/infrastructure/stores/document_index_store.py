from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import bindparam, delete, or_, select, text

from paperbot.application.ports.document_intelligence_port import (
    DocumentChunk,
    EvidenceHit,
    EvidenceRetrieverPort,
)
from paperbot.context_engine.embeddings import EmbeddingProvider
from paperbot.infrastructure.stores.models import (
    Base,
    DocumentAssetModel,
    DocumentChunkModel,
    DocumentIndexJobModel,
    PaperModel,
)
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url

_FTS_TOKEN_RX = re.compile(r"[A-Za-z0-9_+-]+")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_paper_ids(paper_ids: Optional[Sequence[int]]) -> List[int]:
    if not paper_ids:
        return []
    normalized: List[int] = []
    seen: set[int] = set()
    for paper_id in paper_ids:
        try:
            value = int(paper_id)
        except (TypeError, ValueError):
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _serialize_embedding(embedding: Optional[Sequence[float]]) -> Optional[str]:
    if not embedding:
        return None
    try:
        return json.dumps([float(value) for value in embedding], separators=(",", ":"))
    except Exception:
        return None


def _deserialize_embedding(raw_value: Optional[str]) -> Optional[List[float]]:
    if not raw_value:
        return None
    try:
        payload = json.loads(raw_value)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    values: List[float] = []
    for item in payload:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return None
    return values or None


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += float(left_value) * float(right_value)
        left_norm += float(left_value) * float(left_value)
        right_norm += float(right_value) * float(right_value)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _snippet(text_value: str, *, limit: int = 320) -> str:
    clean = " ".join((text_value or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


class DocumentIndexStore(EvidenceRetrieverPort):
    """Persistence and retrieval for document intelligence chunks."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        *,
        auto_create_schema: bool = True,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        self.embedding_provider = embedding_provider
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)
        if str(self.db_url).startswith("sqlite:"):
            with self._provider.engine.connect() as conn:
                self._ensure_fts5(conn)
                try:
                    conn.commit()
                except Exception:
                    pass

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass

    @staticmethod
    def _ensure_fts5(conn) -> None:
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')")
                ).fetchall()
            }
            if "document_chunks_fts" not in tables:
                conn.execute(
                    text(
                        "CREATE VIRTUAL TABLE document_chunks_fts"
                        " USING fts5(heading, content, tokenize='porter ascii')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO document_chunks_fts(rowid, heading, content)"
                        " SELECT id, heading, content FROM document_chunks"
                    )
                )

            triggers = {
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='trigger'")
                ).fetchall()
            }
            if "document_chunks_fts_ai" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER document_chunks_fts_ai"
                        " AFTER INSERT ON document_chunks BEGIN"
                        "   INSERT INTO document_chunks_fts(rowid, heading, content)"
                        "   VALUES (new.id, new.heading, new.content);"
                        " END"
                    )
                )
            if "document_chunks_fts_ad" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER document_chunks_fts_ad"
                        " AFTER DELETE ON document_chunks BEGIN"
                        "   DELETE FROM document_chunks_fts WHERE rowid = old.id;"
                        " END"
                    )
                )
            if "document_chunks_fts_au" not in triggers:
                conn.execute(
                    text(
                        "CREATE TRIGGER document_chunks_fts_au"
                        " AFTER UPDATE OF heading, content ON document_chunks BEGIN"
                        "   DELETE FROM document_chunks_fts WHERE rowid = old.id;"
                        "   INSERT INTO document_chunks_fts(rowid, heading, content)"
                        "   VALUES (new.id, new.heading, new.content);"
                        " END"
                    )
                )
        except Exception:
            return

    def enqueue_jobs(self, *, paper_ids: Sequence[int], trigger_source: str) -> Dict[str, int]:
        normalized_ids = _normalize_paper_ids(paper_ids)
        if not normalized_ids:
            return {"total": 0, "queued": 0, "skipped": 0}

        now = _utcnow()
        queued = 0
        skipped = 0
        with self._provider.session() as session:
            existing_rows = session.execute(
                select(DocumentIndexJobModel.paper_id)
                .where(DocumentIndexJobModel.paper_id.in_(normalized_ids))
                .where(DocumentIndexJobModel.status.in_(("queued", "running")))
            ).all()
            blocked_ids = {int(row[0]) for row in existing_rows}

            for paper_id in normalized_ids:
                if paper_id in blocked_ids:
                    skipped += 1
                    continue
                session.add(
                    DocumentIndexJobModel(
                        paper_id=paper_id,
                        trigger_source=str(trigger_source or "manual"),
                        status="queued",
                        chunk_count=0,
                        attempt_count=0,
                        enqueued_at=now,
                        updated_at=now,
                    )
                )
                queued += 1

            session.commit()

        return {"total": len(normalized_ids), "queued": queued, "skipped": skipped}

    def claim_pending_jobs(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        claimed: List[Dict[str, Any]] = []
        now = _utcnow()
        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(DocumentIndexJobModel)
                    .where(DocumentIndexJobModel.status == "queued")
                    .order_by(DocumentIndexJobModel.enqueued_at, DocumentIndexJobModel.id)
                    .limit(max(1, int(limit)))
                )
                .scalars()
                .all()
            )
            for row in rows:
                row.status = "running"
                row.attempt_count = int(row.attempt_count or 0) + 1
                row.started_at = now
                row.updated_at = now
                claimed.append(
                    {
                        "id": int(row.id),
                        "paper_id": int(row.paper_id),
                        "trigger_source": row.trigger_source,
                    }
                )
            session.commit()
        return claimed

    def complete_job(self, *, job_id: int, asset_id: int, chunk_count: int) -> None:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.get(DocumentIndexJobModel, int(job_id))
            if row is None:
                return
            row.asset_id = int(asset_id)
            row.chunk_count = max(0, int(chunk_count))
            row.status = "completed"
            row.error = None
            row.finished_at = now
            row.updated_at = now
            session.commit()

    def fail_job(self, *, job_id: int, error: str) -> None:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.get(DocumentIndexJobModel, int(job_id))
            if row is None:
                return
            row.status = "failed"
            row.error = (error or "")[:2000]
            row.finished_at = now
            row.updated_at = now
            session.commit()

    def upsert_asset(
        self,
        *,
        paper_id: int,
        source_type: str,
        title: str,
        locator_url: Optional[str],
        checksum: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(DocumentAssetModel).where(
                    DocumentAssetModel.paper_id == int(paper_id),
                    DocumentAssetModel.source_type == str(source_type or "paper_metadata"),
                )
            ).scalar_one_or_none()
            if row is None:
                row = DocumentAssetModel(
                    paper_id=int(paper_id),
                    source_type=str(source_type or "paper_metadata"),
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)

            row.title = str(title or "")
            row.locator_url = str(locator_url or "").strip() or None
            row.checksum = str(checksum or "")
            row.updated_at = now
            row.set_metadata(metadata or {})
            session.commit()
            session.refresh(row)
            return {
                "id": int(row.id),
                "paper_id": int(row.paper_id),
                "source_type": row.source_type,
                "locator_url": row.locator_url,
            }

    def replace_chunks(
        self,
        *,
        asset_id: int,
        paper_id: int,
        chunks: Sequence[DocumentChunk],
    ) -> int:
        now = _utcnow()
        with self._provider.session() as session:
            session.execute(
                delete(DocumentChunkModel).where(DocumentChunkModel.asset_id == int(asset_id))
            )
            inserted = 0
            for chunk in chunks:
                content = str(chunk.content or "").strip()
                if not content:
                    continue
                row = DocumentChunkModel(
                    paper_id=int(paper_id),
                    asset_id=int(asset_id),
                    chunk_index=int(chunk.chunk_index),
                    section=str(chunk.section or ""),
                    heading=str(chunk.heading or ""),
                    content=content,
                    content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    token_count=max(0, int(chunk.token_count or 0)),
                    embedding_json=_serialize_embedding(chunk.embedding),
                    created_at=now,
                    updated_at=now,
                )
                row.set_metadata(chunk.metadata or {})
                session.add(row)
                inserted += 1
            session.commit()
        return inserted

    def retrieve_evidence(
        self,
        *,
        query: str,
        paper_ids: Optional[Sequence[int]] = None,
        limit: int = 6,
    ) -> List[EvidenceHit]:
        max_hits = max(1, int(limit))
        scoped_paper_ids = _normalize_paper_ids(paper_ids)
        if paper_ids is not None and not scoped_paper_ids:
            return []

        fts_candidate_ids = self._search_chunk_ids_with_fts(
            query=query,
            paper_ids=scoped_paper_ids or None,
            limit=max(max_hits * 8, 24),
        )
        if fts_candidate_ids is None:
            fts_candidate_ids = self._search_chunk_ids_with_like(
                query=query,
                paper_ids=scoped_paper_ids or None,
                limit=max(max_hits * 8, 24),
            )
        query_embedding = None
        if self.embedding_provider is not None:
            try:
                query_embedding = self.embedding_provider.embed(query[:500])
            except Exception:
                query_embedding = None

        embedding_ranked: List[Dict[str, Any]] = []
        if query_embedding:
            embedding_ranked = self._search_chunks_with_embedding(
                query_embedding=query_embedding,
                paper_ids=scoped_paper_ids or None,
                limit=max(max_hits * 8, 24),
            )

        candidate_ids: List[int] = []
        seen_ids: set[int] = set()
        for chunk_id in fts_candidate_ids or []:
            chunk_int = int(chunk_id)
            if chunk_int in seen_ids:
                continue
            seen_ids.add(chunk_int)
            candidate_ids.append(chunk_int)
        for row in embedding_ranked:
            chunk_int = int(row["chunk_id"])
            if chunk_int in seen_ids:
                continue
            seen_ids.add(chunk_int)
            candidate_ids.append(chunk_int)
        if not candidate_ids:
            return []

        with self._provider.session() as session:
            rows = session.execute(
                select(DocumentChunkModel, DocumentAssetModel, PaperModel)
                .join(DocumentAssetModel, DocumentAssetModel.id == DocumentChunkModel.asset_id)
                .join(PaperModel, PaperModel.id == DocumentChunkModel.paper_id)
                .where(DocumentChunkModel.id.in_(candidate_ids))
            ).all()

        hits: List[EvidenceHit] = []
        order_map = {chunk_id: index for index, chunk_id in enumerate(candidate_ids)}
        fts_rank_map = {
            int(chunk_id): index for index, chunk_id in enumerate(fts_candidate_ids or [])
        }
        embedding_score_map = {
            int(row["chunk_id"]): float(row.get("score", 0.0)) for row in embedding_ranked
        }
        for chunk_row, asset_row, paper_row in sorted(
            rows,
            key=lambda row: order_map.get(int(row[0].id), len(order_map) + 1),
        ):
            chunk_id = int(chunk_row.id)
            score = 0.0
            if chunk_id in fts_rank_map:
                score += 1.0 / (1.0 + float(fts_rank_map[chunk_id]))
            if chunk_id in embedding_score_map:
                score += 0.5 * max(0.0, float(embedding_score_map[chunk_id]))
            hits.append(
                EvidenceHit(
                    paper_id=int(chunk_row.paper_id),
                    chunk_id=chunk_id,
                    chunk_index=int(chunk_row.chunk_index),
                    paper_title=str(paper_row.title or ""),
                    section=str(chunk_row.section or ""),
                    heading=str(chunk_row.heading or ""),
                    snippet=_snippet(chunk_row.content),
                    score=round(float(score), 6),
                    source_type=str(asset_row.source_type or "paper_metadata"),
                    locator_url=asset_row.locator_url,
                    metadata=chunk_row.get_metadata(),
                )
            )

        hits.sort(key=lambda hit: float(hit.score), reverse=True)
        return hits[:max_hits]

    def list_chunks(self, *, paper_ids: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
        scoped_paper_ids = _normalize_paper_ids(paper_ids)
        stmt = (
            select(DocumentChunkModel, DocumentAssetModel, PaperModel)
            .join(DocumentAssetModel, DocumentAssetModel.id == DocumentChunkModel.asset_id)
            .join(PaperModel, PaperModel.id == DocumentChunkModel.paper_id)
            .order_by(DocumentChunkModel.paper_id, DocumentChunkModel.chunk_index)
        )
        if scoped_paper_ids:
            stmt = stmt.where(DocumentChunkModel.paper_id.in_(scoped_paper_ids))

        with self._provider.session() as session:
            rows = session.execute(stmt).all()

        chunks: List[Dict[str, Any]] = []
        for chunk_row, asset_row, paper_row in rows:
            metadata = chunk_row.get_metadata()
            section_chunk_index = int(metadata.get("section_chunk_index") or 0)
            chunks.append(
                {
                    "chunk_id": int(chunk_row.id),
                    "paper_id": int(chunk_row.paper_id),
                    "paper_title": str(paper_row.title or ""),
                    "chunk_index": int(chunk_row.chunk_index),
                    "section": str(chunk_row.section or ""),
                    "heading": str(chunk_row.heading or ""),
                    "content": str(chunk_row.content or ""),
                    "source_type": str(asset_row.source_type or "paper_metadata"),
                    "locator_url": asset_row.locator_url,
                    "embedding": _deserialize_embedding(chunk_row.embedding_json),
                    "metadata": metadata,
                    "chunk_ref": f"{int(chunk_row.paper_id)}:{str(chunk_row.section or '')}:{section_chunk_index}",
                }
            )
        return chunks

    def _search_chunk_ids_with_fts(
        self,
        *,
        query: str,
        paper_ids: Optional[Sequence[int]],
        limit: int,
    ) -> Optional[List[int]]:
        if not str(self.db_url).startswith("sqlite:"):
            return None

        def _escape_token(token: str) -> str:
            safe_parts = _FTS_TOKEN_RX.findall(token or "")
            cleaned = " ".join(safe_parts).strip()
            if not cleaned:
                return ""
            return '"' + cleaned.replace('"', '""') + '"'

        escaped_tokens = [_escape_token(token) for token in query.split()[:8]]
        escaped_tokens = [token for token in escaped_tokens if token]
        if not escaped_tokens:
            return []

        params: Dict[str, Any] = {"query": " ".join(escaped_tokens), "limit": max(1, int(limit))}
        where_parts = ["document_chunks_fts MATCH :query"]
        statement = (
            "SELECT dc.id FROM document_chunks_fts"
            " JOIN document_chunks dc ON dc.id = document_chunks_fts.rowid"
        )
        if paper_ids:
            where_parts.append("dc.paper_id IN :paper_ids")
        sql = (
            statement
            + " WHERE "
            + " AND ".join(where_parts)
            + " ORDER BY bm25(document_chunks_fts) LIMIT :limit"
        )

        stmt = text(sql)
        if paper_ids:
            stmt = stmt.bindparams(bindparam("paper_ids", expanding=True))
            params["paper_ids"] = list(paper_ids)

        try:
            with self._provider.engine.connect() as conn:
                rows = conn.execute(stmt, params).fetchall()
        except Exception:
            return None
        return [int(row[0]) for row in rows]

    def _search_chunk_ids_with_like(
        self,
        *,
        query: str,
        paper_ids: Optional[Sequence[int]],
        limit: int,
    ) -> List[int]:
        tokens = [
            token.lower() for token in _FTS_TOKEN_RX.findall(query or "")[:8] if token.strip()
        ]
        if not tokens:
            return []

        clauses = []
        for token in tokens:
            pattern = f"%{token}%"
            clauses.append(DocumentChunkModel.content.ilike(pattern))
            clauses.append(DocumentChunkModel.heading.ilike(pattern))

        stmt = (
            select(DocumentChunkModel.id)
            .where(or_(*clauses))
            .order_by(
                DocumentChunkModel.updated_at.desc(),
                DocumentChunkModel.id.desc(),
            )
        )
        if paper_ids:
            stmt = stmt.where(DocumentChunkModel.paper_id.in_(paper_ids))
        stmt = stmt.limit(max(1, int(limit)))

        with self._provider.session() as session:
            rows = session.execute(stmt).all()
        return [int(row[0]) for row in rows]

    def _search_chunks_with_embedding(
        self,
        *,
        query_embedding: Sequence[float],
        paper_ids: Optional[Sequence[int]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for chunk in self.list_chunks(paper_ids=paper_ids):
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            ranked.append(
                {
                    **chunk,
                    "score": float(_cosine_similarity(query_embedding, embedding)),
                }
            )
        ranked.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return ranked[: max(1, int(limit))]
