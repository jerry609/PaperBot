from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError

from paperbot.infrastructure.stores.models import (
    Base,
    PaperFeedbackModel,
    ResearchMilestoneModel,
    ResearchTaskModel,
    ResearchTrackEmbeddingModel,
    ResearchTrackModel,
)
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _dump_list(values: Optional[List[str]]) -> str:
    return json.dumps([str(v).strip() for v in (values or []) if str(v).strip()], ensure_ascii=False)


def _load_list(raw: str) -> List[str]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except Exception:
        pass
    return []


class SqlAlchemyResearchStore:
    """
    Track/progress store for personalized paper recommendation.

    Tables:
    - research_tracks, research_tasks, research_milestones
    - paper_feedback (like/dislike/save/...)
    - research_track_embeddings (optional embedding cache)
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)

    def create_track(
        self,
        *,
        user_id: str,
        name: str,
        description: str = "",
        keywords: Optional[List[str]] = None,
        venues: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        activate: bool = True,
    ) -> Dict[str, Any]:
        now = _utcnow()
        with self._provider.session() as session:
            track = ResearchTrackModel(
                user_id=user_id,
                name=(name or "").strip(),
                description=(description or "").strip(),
                keywords_json=_dump_list(keywords),
                venues_json=_dump_list(venues),
                methods_json=_dump_list(methods),
                is_active=1 if activate else 0,
                created_at=now,
                updated_at=now,
            )
            session.add(track)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                existing = session.execute(
                    select(ResearchTrackModel).where(
                        ResearchTrackModel.user_id == user_id, ResearchTrackModel.name == track.name
                    )
                ).scalar_one()
                if activate:
                    return self.activate_track(user_id=user_id, track_id=existing.id) or self._track_to_dict(existing)
                return self._track_to_dict(existing)

            if activate:
                session.execute(
                    ResearchTrackModel.__table__.update()
                    .where(ResearchTrackModel.user_id == user_id, ResearchTrackModel.id != track.id)
                    .values(is_active=0, updated_at=now)
                )

            session.commit()
            session.refresh(track)
            return self._track_to_dict(track)

    def list_tracks(self, *, user_id: str, include_archived: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            stmt = select(ResearchTrackModel).where(ResearchTrackModel.user_id == user_id)
            if not include_archived:
                stmt = stmt.where(ResearchTrackModel.archived_at.is_(None))
            stmt = stmt.order_by(desc(ResearchTrackModel.is_active), desc(ResearchTrackModel.updated_at)).limit(limit)
            tracks = session.execute(stmt).scalars().all()
            return [self._track_to_dict(t) for t in tracks]

    def get_track(self, *, user_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id)
            ).scalar_one_or_none()
            return self._track_to_dict(row) if row else None

    def get_active_track(self, *, user_id: str) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id,
                    ResearchTrackModel.is_active == 1,
                    ResearchTrackModel.archived_at.is_(None),
                )
            ).scalar_one_or_none()
            return self._track_to_dict(row) if row else None

    def activate_track(self, *, user_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            if row.archived_at is not None:
                row.archived_at = None
            row.is_active = 1
            row.updated_at = now
            session.add(row)
            session.execute(
                ResearchTrackModel.__table__.update()
                .where(ResearchTrackModel.user_id == user_id, ResearchTrackModel.id != track_id)
                .values(is_active=0, updated_at=now)
            )
            session.commit()
            session.refresh(row)
            return self._track_to_dict(row)

    def archive_track(self, *, user_id: str, track_id: int, archived: bool = True) -> bool:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            row.archived_at = now if archived else None
            if archived:
                row.is_active = 0
            row.updated_at = now
            session.add(row)
            session.commit()
            return True

    def add_task(
        self,
        *,
        user_id: str,
        track_id: int,
        title: str,
        status: str = "todo",
        priority: int = 0,
        paper_id: Optional[str] = None,
        paper_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                return None

            task = ResearchTaskModel(
                track_id=track_id,
                title=(title or "").strip(),
                status=(status or "todo").strip() or "todo",
                priority=int(priority or 0),
                paper_id=(paper_id.strip() if paper_id else None),
                paper_url=(paper_url.strip() if paper_url else None),
                metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
                created_at=now,
                updated_at=now,
                done_at=(now if status == "done" else None),
            )
            session.add(task)
            track.updated_at = now
            session.add(track)
            session.commit()
            session.refresh(task)
            return self._task_to_dict(task)

    def list_tasks(
        self,
        *,
        user_id: str,
        track_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                return []

            stmt = select(ResearchTaskModel).where(ResearchTaskModel.track_id == track_id)
            if status:
                stmt = stmt.where(ResearchTaskModel.status == status)
            stmt = stmt.order_by(desc(ResearchTaskModel.priority), desc(ResearchTaskModel.updated_at)).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._task_to_dict(r) for r in rows]

    def add_milestone(
        self,
        *,
        user_id: str,
        track_id: int,
        name: str,
        status: str = "todo",
        notes: str = "",
        due_at: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                return None

            ms = ResearchMilestoneModel(
                track_id=track_id,
                name=(name or "").strip(),
                status=(status or "todo").strip() or "todo",
                notes=(notes or "").strip(),
                due_at=due_at,
                created_at=now,
                updated_at=now,
            )
            session.add(ms)
            track.updated_at = now
            session.add(track)
            session.commit()
            session.refresh(ms)
            return self._milestone_to_dict(ms)

    def list_milestones(
        self,
        *,
        user_id: str,
        track_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                return []

            stmt = select(ResearchMilestoneModel).where(ResearchMilestoneModel.track_id == track_id)
            if status:
                stmt = stmt.where(ResearchMilestoneModel.status == status)
            stmt = stmt.order_by(desc(ResearchMilestoneModel.updated_at)).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._milestone_to_dict(r) for r in rows]

    def add_paper_feedback(
        self,
        *,
        user_id: str,
        track_id: int,
        paper_id: str,
        action: str,
        weight: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                return None

            row = PaperFeedbackModel(
                user_id=user_id,
                track_id=track_id,
                paper_id=(paper_id or "").strip(),
                action=(action or "").strip(),
                weight=float(weight or 0.0),
                ts=now,
                metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
            )
            session.add(row)
            track.updated_at = now
            session.add(track)
            session.commit()
            session.refresh(row)
            return self._feedback_to_dict(row)

    def list_paper_feedback(
        self,
        *,
        user_id: str,
        track_id: int,
        action: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                return []

            stmt = select(PaperFeedbackModel).where(
                PaperFeedbackModel.user_id == user_id, PaperFeedbackModel.track_id == track_id
            )
            if action:
                stmt = stmt.where(PaperFeedbackModel.action == action)
            stmt = stmt.order_by(desc(PaperFeedbackModel.ts)).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._feedback_to_dict(r) for r in rows]

    def list_paper_feedback_ids(
        self,
        *,
        user_id: str,
        track_id: int,
        action: str,
        limit: int = 500,
    ) -> set[str]:
        ids: set[str] = set()
        for row in self.list_paper_feedback(user_id=user_id, track_id=track_id, action=action, limit=limit):
            pid = str(row.get("paper_id") or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def get_track_embedding(self, *, track_id: int, model: str) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackEmbeddingModel).where(
                    ResearchTrackEmbeddingModel.track_id == track_id,
                    ResearchTrackEmbeddingModel.model == model,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            try:
                vec = json.loads(row.embedding_json or "[]")
                if not isinstance(vec, list):
                    vec = []
            except Exception:
                vec = []
            return {
                "track_id": row.track_id,
                "model": row.model,
                "text_hash": row.text_hash,
                "embedding": [float(x) for x in vec],
                "dim": int(row.dim or len(vec) or 0),
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    def upsert_track_embedding(
        self,
        *,
        track_id: int,
        model: str,
        profile_text: str,
        embedding: List[float],
    ) -> Dict[str, Any]:
        now = _utcnow()
        text_hash = _sha256_text(profile_text)
        vec = [float(x) for x in (embedding or [])]
        dim = int(len(vec))
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackEmbeddingModel).where(
                    ResearchTrackEmbeddingModel.track_id == track_id,
                    ResearchTrackEmbeddingModel.model == model,
                )
            ).scalar_one_or_none()
            if row is None:
                row = ResearchTrackEmbeddingModel(
                    track_id=track_id,
                    model=model,
                    text_hash=text_hash,
                    embedding_json=json.dumps(vec, ensure_ascii=False),
                    dim=dim,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.text_hash = text_hash
                row.embedding_json = json.dumps(vec, ensure_ascii=False)
                row.dim = dim
                row.updated_at = now
                session.add(row)
            session.commit()
            session.refresh(row)
            return {
                "track_id": row.track_id,
                "model": row.model,
                "text_hash": row.text_hash,
                "dim": int(row.dim or dim),
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass

    @staticmethod
    def _track_to_dict(t: ResearchTrackModel) -> Dict[str, Any]:
        return {
            "id": t.id,
            "user_id": t.user_id,
            "name": t.name,
            "description": t.description,
            "keywords": _load_list(t.keywords_json),
            "venues": _load_list(t.venues_json),
            "methods": _load_list(t.methods_json),
            "is_active": bool(int(t.is_active or 0)),
            "archived_at": t.archived_at.isoformat() if t.archived_at else None,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }

    @staticmethod
    def _task_to_dict(t: ResearchTaskModel) -> Dict[str, Any]:
        try:
            metadata = json.loads(t.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
        return {
            "id": t.id,
            "track_id": t.track_id,
            "title": t.title,
            "status": t.status,
            "priority": int(t.priority or 0),
            "paper_id": t.paper_id,
            "paper_url": t.paper_url,
            "metadata": metadata,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            "done_at": t.done_at.isoformat() if t.done_at else None,
        }

    @staticmethod
    def _milestone_to_dict(m: ResearchMilestoneModel) -> Dict[str, Any]:
        return {
            "id": m.id,
            "track_id": m.track_id,
            "name": m.name,
            "status": m.status,
            "notes": m.notes,
            "due_at": m.due_at.isoformat() if m.due_at else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }

    @staticmethod
    def _feedback_to_dict(f: PaperFeedbackModel) -> Dict[str, Any]:
        try:
            metadata = json.loads(f.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
        return {
            "id": f.id,
            "user_id": f.user_id,
            "track_id": f.track_id,
            "paper_id": f.paper_id,
            "action": f.action,
            "weight": float(f.weight or 0.0),
            "ts": f.ts.isoformat() if f.ts else None,
            "metadata": metadata,
        }

