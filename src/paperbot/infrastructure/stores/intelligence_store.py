from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select

from paperbot.infrastructure.stores.models import Base, IntelligenceEventModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _isoformat_datetime(value: Any) -> Optional[str]:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed else None


def _dump_list(values: Optional[List[str]]) -> str:
    cleaned = [str(value).strip() for value in (values or []) if str(value).strip()]
    return json.dumps(cleaned, ensure_ascii=False)


def _load_list(raw: str) -> List[str]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        pass
    return []


def _dump_dict(payload: Optional[Dict[str, Any]]) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return normalized.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _load_dict(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


class IntelligenceStore:
    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)

    def get_event(self, *, user_id: str, external_id: str) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(IntelligenceEventModel).where(
                    IntelligenceEventModel.user_id == user_id,
                    IntelligenceEventModel.external_id == external_id,
                )
            ).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    def upsert_event(
        self,
        *,
        user_id: str,
        external_id: str,
        source: str,
        source_label: str,
        kind: str,
        title: str,
        summary: str,
        url: str = "",
        repo_full_name: str = "",
        author_name: str = "",
        keyword_hits: Optional[List[str]] = None,
        author_matches: Optional[List[str]] = None,
        repo_matches: Optional[List[str]] = None,
        metric_name: str = "",
        metric_value: int = 0,
        metric_delta: int = 0,
        score: float = 0.0,
        published_at: Optional[datetime] = None,
        detected_at: Optional[datetime] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _parse_datetime(detected_at) or _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(IntelligenceEventModel).where(
                    IntelligenceEventModel.user_id == user_id,
                    IntelligenceEventModel.external_id == external_id,
                )
            ).scalar_one_or_none()
            if row is None:
                row = IntelligenceEventModel(
                    user_id=user_id,
                    external_id=external_id,
                    created_at=now,
                )

            row.source = (source or "unknown").strip()[:32]
            row.source_label = (source_label or row.source).strip()[:64]
            row.kind = (kind or "signal").strip()[:64]
            row.title = (title or "Untitled signal").strip()
            row.summary = (summary or "").strip()
            row.url = (url or "").strip()
            row.repo_full_name = (repo_full_name or "").strip()[:128]
            row.author_name = (author_name or "").strip()[:128]
            row.keyword_hits_json = _dump_list(keyword_hits)
            row.author_matches_json = _dump_list(author_matches)
            row.repo_matches_json = _dump_list(repo_matches)
            row.metric_name = (metric_name or "").strip()[:64]
            row.metric_value = int(metric_value or 0)
            row.metric_delta = int(metric_delta or 0)
            row.score = float(score or 0.0)
            row.published_at = _parse_datetime(published_at) or row.published_at or now
            row.detected_at = now
            row.updated_at = now
            row.payload_json = _dump_dict(payload)

            session.add(row)
            session.commit()
            session.refresh(row)
            return self._row_to_dict(row)

    def list_events(
        self,
        *,
        user_id: str,
        limit: int = 20,
        max_age_days: int = 14,
    ) -> List[Dict[str, Any]]:
        cutoff = _utcnow() - timedelta(days=max(1, int(max_age_days)))
        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(IntelligenceEventModel)
                    .where(
                        IntelligenceEventModel.user_id == user_id,
                        IntelligenceEventModel.detected_at >= cutoff,
                    )
                    .order_by(
                        desc(IntelligenceEventModel.score),
                        desc(IntelligenceEventModel.metric_delta),
                        desc(IntelligenceEventModel.published_at),
                        desc(IntelligenceEventModel.detected_at),
                    )
                    .limit(max(1, int(limit)))
                )
                .scalars()
                .all()
            )
            return [self._row_to_dict(row) for row in rows]

    def latest_detected_at(self, *, user_id: str) -> Optional[datetime]:
        with self._provider.session() as session:
            row = (
                session.execute(
                    select(IntelligenceEventModel)
                    .where(IntelligenceEventModel.user_id == user_id)
                    .order_by(desc(IntelligenceEventModel.detected_at))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            return _parse_datetime(row.detected_at) if row else None

    @staticmethod
    def _row_to_dict(row: IntelligenceEventModel) -> Dict[str, Any]:
        return {
            "id": row.id,
            "user_id": row.user_id,
            "external_id": row.external_id,
            "source": row.source,
            "source_label": row.source_label,
            "kind": row.kind,
            "title": row.title,
            "summary": row.summary,
            "url": row.url,
            "repo_full_name": row.repo_full_name,
            "author_name": row.author_name,
            "keyword_hits": _load_list(row.keyword_hits_json),
            "author_matches": _load_list(row.author_matches_json),
            "repo_matches": _load_list(row.repo_matches_json),
            "metric_name": row.metric_name,
            "metric_value": int(row.metric_value or 0),
            "metric_delta": int(row.metric_delta or 0),
            "score": float(row.score or 0.0),
            "published_at": _isoformat_datetime(row.published_at),
            "detected_at": _isoformat_datetime(row.detected_at),
            "created_at": _isoformat_datetime(row.created_at),
            "updated_at": _isoformat_datetime(row.updated_at),
            "payload": _load_dict(row.payload_json),
        }
