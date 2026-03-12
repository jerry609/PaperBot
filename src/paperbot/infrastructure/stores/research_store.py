from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import desc, func, or_, select
from sqlalchemy.exc import IntegrityError

from paperbot.application.ports.feedback_port import FeedbackPort
from paperbot.application.services.identity_resolver import IdentityResolver
from paperbot.domain.paper_identity import normalize_arxiv_id, normalize_doi
from paperbot.infrastructure.stores.models import (
    Base,
    PaperCollectionItemModel,
    PaperCollectionModel,
    PaperFeedbackModel,
    PaperImpressionModel,
    PaperJudgeScoreModel,
    PaperModel,
    PaperReadingStatusModel,
    PaperRepoModel,
    ResearchContextRunModel,
    ResearchMilestoneModel,
    ResearchTaskModel,
    ResearchTrackEmbeddingModel,
    ResearchTrackModel,
)
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.utils.logging_config import LogFiles, Logger


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _dump_list(values: Optional[List[str]]) -> str:
    return json.dumps(
        [str(v).strip() for v in (values or []) if str(v).strip()], ensure_ascii=False
    )


def _load_list(raw: str) -> List[str]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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

_FEEDBACK_ACTION_ALIASES: Dict[str, str] = {
    "not_relevant": "dislike",
    "not-relevant": "dislike",
    "not related": "dislike",
}
_FEEDBACK_GROUP_BY_ACTION: Dict[str, str] = {
    "save": "save_state",
    "unsave": "save_state",
    "like": "preference_state",
    "unlike": "preference_state",
    "dislike": "preference_state",
    "undislike": "preference_state",
    "skip": "preference_state",
    "cite": "cite_state",
}
_FEEDBACK_EFFECTIVE_ACTIONS: Dict[str, Optional[str]] = {
    "save": "save",
    "unsave": None,
    "like": "like",
    "unlike": None,
    "dislike": "dislike",
    "undislike": None,
    "skip": "skip",
    "cite": "cite",
}
class SqlAlchemyResearchStore(FeedbackPort):
    """
    Track/progress store for personalized paper recommendation.

    Tables:
    - research_tracks, research_tasks, research_milestones
    - paper_feedback (like/dislike/save/...)
    - research_track_embeddings (optional embedding cache)
    - research_context_runs, paper_impressions (eval/replay)
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)
        self._identity_resolver = IdentityResolver(db_url=self.db_url)

    @staticmethod
    def _normalize_feedback_action(value: str) -> str:
        normalized = (value or "").strip().lower().replace(" ", "_")
        return _FEEDBACK_ACTION_ALIASES.get(normalized, normalized)

    @staticmethod
    def _feedback_group(action: str) -> str:
        return _FEEDBACK_GROUP_BY_ACTION.get(action, action)

    @staticmethod
    def _effective_feedback_action(action: str) -> Optional[str]:
        return _FEEDBACK_EFFECTIVE_ACTIONS.get(action, action or None)

    @staticmethod
    def _feedback_identity_key(
        *,
        paper_ref_id: Optional[int],
        canonical_paper_id: Optional[int],
        paper_id: str,
    ) -> str:
        resolved_ref_id = int(canonical_paper_id or paper_ref_id or 0)
        if resolved_ref_id > 0:
            return f"ref:{resolved_ref_id}"
        normalized_paper_id = str(paper_id or "").strip()
        return f"external:{normalized_paper_id}"

    @classmethod
    def _feedback_state_key(
        cls, row: PaperFeedbackModel, normalized_action: str
    ) -> tuple[str, str]:
        return (
            cls._feedback_identity_key(
                paper_ref_id=row.paper_ref_id,
                canonical_paper_id=row.canonical_paper_id,
                paper_id=row.paper_id,
            ),
            cls._feedback_group(normalized_action),
        )

    @classmethod
    def _collapse_effective_feedback_rows(
        cls, rows: Iterable[PaperFeedbackModel]
    ) -> List[Dict[str, Any]]:
        collapsed: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            normalized_action = cls._normalize_feedback_action(str(row.action or ""))
            state_key = cls._feedback_state_key(row, normalized_action)
            if state_key in seen:
                continue
            seen.add(state_key)

            effective_action = cls._effective_feedback_action(normalized_action)
            if effective_action is None:
                continue

            payload = cls._feedback_to_dict(row)
            payload["action"] = effective_action
            collapsed.append(payload)
        return collapsed

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
                    return self.activate_track(
                        user_id=user_id, track_id=existing.id
                    ) or self._track_to_dict(existing)
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

    def list_tracks(
        self, *, user_id: str, include_archived: bool = False, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            stmt = select(ResearchTrackModel).where(ResearchTrackModel.user_id == user_id)
            if not include_archived:
                stmt = stmt.where(ResearchTrackModel.archived_at.is_(None))
            stmt = stmt.order_by(
                desc(ResearchTrackModel.is_active), desc(ResearchTrackModel.updated_at)
            ).limit(limit)
            tracks = session.execute(stmt).scalars().all()
            return [self._track_to_dict(t) for t in tracks]

    def get_track(self, *, user_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            return self._track_to_dict(row) if row else None

    def get_track_by_id(self, *, track_id: int) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(ResearchTrackModel.id == track_id)
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

    def update_track(
        self,
        *,
        user_id: str,
        track_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        venues: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            if name is not None:
                row.name = name.strip()
            if description is not None:
                row.description = description.strip()
            if keywords is not None:
                row.keywords_json = _dump_list(keywords)
            if venues is not None:
                row.venues_json = _dump_list(venues)
            if methods is not None:
                row.methods_json = _dump_list(methods)

            row.updated_at = now
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise
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
            stmt = stmt.order_by(
                desc(ResearchTaskModel.priority), desc(ResearchTaskModel.updated_at)
            ).limit(limit)
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
        Logger.info("Recording paper feedback", file=LogFiles.HARVEST)
        now = _utcnow()
        metadata = dict(metadata or {})
        normalized_action = self._normalize_feedback_action(action)
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id, ResearchTrackModel.id == track_id
                )
            ).scalar_one_or_none()
            if track is None:
                Logger.error("Track not found", file=LogFiles.HARVEST)
                return None

            resolved_paper_ref_id = self._resolve_paper_ref_id(
                session=session,
                paper_id=(paper_id or "").strip(),
                metadata=metadata,
            )
            Logger.info("Creating new feedback record", file=LogFiles.HARVEST)
            row = PaperFeedbackModel(
                user_id=user_id,
                track_id=track_id,
                paper_id=(paper_id or "").strip(),
                paper_ref_id=resolved_paper_ref_id,
                canonical_paper_id=resolved_paper_ref_id,  # dual-write
                action=normalized_action,
                weight=float(weight or 0.0),
                ts=now,
                metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
            )
            session.add(row)

            if resolved_paper_ref_id:
                if normalized_action == "save":
                    self._upsert_reading_status_row(
                        session=session,
                        user_id=user_id,
                        paper_ref_id=resolved_paper_ref_id,
                        status="unread",
                        mark_saved=True,
                        metadata=metadata,
                        now=now,
                    )
                elif normalized_action == "unsave":
                    existing_status = session.execute(
                        select(PaperReadingStatusModel).where(
                            PaperReadingStatusModel.user_id == user_id,
                            PaperReadingStatusModel.paper_id == int(resolved_paper_ref_id),
                        )
                    ).scalar_one_or_none()
                    self._upsert_reading_status_row(
                        session=session,
                        user_id=user_id,
                        paper_ref_id=resolved_paper_ref_id,
                        status=(
                            str(existing_status.status or "unread")
                            if existing_status is not None
                            else "unread"
                        ),
                        mark_saved=False,
                        metadata=metadata,
                        now=now,
                    )

            track.updated_at = now
            session.add(track)
            session.commit()
            session.refresh(row)
            Logger.info("Feedback record created successfully", file=LogFiles.HARVEST)
            return self._feedback_to_dict(row)

    def list_effective_paper_feedback(
        self,
        *,
        user_id: str,
        track_id: int,
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

            rows = (
                session.execute(
                    select(PaperFeedbackModel)
                    .where(
                        PaperFeedbackModel.user_id == user_id,
                        PaperFeedbackModel.track_id == track_id,
                    )
                    .order_by(desc(PaperFeedbackModel.ts), desc(PaperFeedbackModel.id))
                )
                .scalars()
                .all()
            )
            collapsed = self._collapse_effective_feedback_rows(rows)
            return collapsed[: max(1, int(limit))]

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
                stmt = stmt.where(
                    PaperFeedbackModel.action == self._normalize_feedback_action(action)
                )
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
        normalized_action = self._normalize_feedback_action(action)
        for row in self.list_effective_paper_feedback(
            user_id=user_id,
            track_id=track_id,
            limit=limit,
        ):
            if row.get("action") != normalized_action:
                continue
            pid = str(row.get("paper_id") or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def set_paper_reading_status(
        self,
        *,
        user_id: str,
        paper_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        mark_saved: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        metadata = dict(metadata or {})
        with self._provider.session() as session:
            paper_ref_id = self._resolve_paper_ref_id(
                session=session,
                paper_id=(paper_id or "").strip(),
                metadata=metadata,
            )
            if not paper_ref_id:
                return None

            row = self._upsert_reading_status_row(
                session=session,
                user_id=user_id,
                paper_ref_id=paper_ref_id,
                status=status,
                mark_saved=mark_saved,
                metadata=metadata,
                now=now,
            )
            session.commit()
            session.refresh(row)
            return self._reading_status_to_dict(row)

    def list_saved_papers(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        collection_id: Optional[int] = None,
        limit: int = 200,
        sort_by: str = "saved_at",
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            saved_at_by_paper: Dict[int, datetime] = {}
            saved_track_membership: Dict[int, set[int]] = defaultdict(set)

            status_rows = (
                session.execute(
                    select(PaperReadingStatusModel).where(
                        PaperReadingStatusModel.user_id == user_id,
                        PaperReadingStatusModel.saved_at.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
            for row in status_rows:
                if row.paper_id and row.saved_at:
                    saved_at_by_paper[int(row.paper_id)] = row.saved_at

            feedback_rows = (
                session.execute(
                    select(PaperFeedbackModel)
                    .where(
                        PaperFeedbackModel.user_id == user_id,
                        PaperFeedbackModel.action.in_(["save", "unsave"]),
                        PaperFeedbackModel.paper_ref_id.is_not(None),
                    )
                    .order_by(desc(PaperFeedbackModel.ts), desc(PaperFeedbackModel.id))
                )
                .scalars()
                .all()
            )
            latest_save_state: Dict[int, tuple[bool, Optional[datetime]]] = {}
            for row in feedback_rows:
                pid = int(row.paper_ref_id or 0)
                if pid <= 0:
                    continue
                normalized_action = self._normalize_feedback_action(str(row.action or ""))
                if normalized_action == "save":
                    saved_track_membership[pid].add(int(row.track_id or 0))
                if pid not in latest_save_state:
                    latest_save_state[pid] = (normalized_action == "save", row.ts)

            for pid, (is_saved, ts) in latest_save_state.items():
                if not is_saved:
                    saved_at_by_paper.pop(pid, None)
                    continue
                current = saved_at_by_paper.get(pid)
                if current is None or ((ts or _utcnow()) > current):
                    saved_at_by_paper[pid] = ts or _utcnow()

            if track_id is not None:
                scoped_paper_ids = [
                    pid
                    for pid in saved_at_by_paper.keys()
                    if int(track_id) in saved_track_membership.get(pid, set())
                ]
                paper_ids = scoped_paper_ids
            else:
                paper_ids = list(saved_at_by_paper.keys())
            if not paper_ids:
                return []

            if collection_id is not None:
                collection = session.execute(
                    select(PaperCollectionModel).where(
                        PaperCollectionModel.id == int(collection_id),
                        PaperCollectionModel.user_id == user_id,
                        PaperCollectionModel.archived_at.is_(None),
                    )
                ).scalar_one_or_none()
                if collection is None:
                    return []
                collection_paper_ids = {
                    int(row.paper_id)
                    for row in session.execute(
                        select(PaperCollectionItemModel).where(
                            PaperCollectionItemModel.collection_id == int(collection_id)
                        )
                    )
                    .scalars()
                    .all()
                }
                paper_ids = [pid for pid in paper_ids if pid in collection_paper_ids]
                if not paper_ids:
                    return []

            papers = (
                session.execute(select(PaperModel).where(PaperModel.id.in_(paper_ids)))
                .scalars()
                .all()
            )
            status_by_paper = {
                int(row.paper_id): row
                for row in session.execute(
                    select(PaperReadingStatusModel).where(
                        PaperReadingStatusModel.user_id == user_id,
                        PaperReadingStatusModel.paper_id.in_(paper_ids),
                    )
                )
                .scalars()
                .all()
            }

            latest_judge_by_paper: Dict[int, PaperJudgeScoreModel] = {}
            for pid in paper_ids:
                judge = session.execute(
                    select(PaperJudgeScoreModel)
                    .where(PaperJudgeScoreModel.paper_id == pid)
                    .order_by(desc(PaperJudgeScoreModel.scored_at), desc(PaperJudgeScoreModel.id))
                    .limit(1)
                ).scalar_one_or_none()
                if judge is not None:
                    latest_judge_by_paper[pid] = judge

            rows: List[Dict[str, Any]] = []
            for paper in papers:
                pid = int(paper.id)
                status_row = status_by_paper.get(pid)
                judge_row = latest_judge_by_paper.get(pid)
                rows.append(
                    {
                        "paper": self._paper_to_dict(paper),
                        "saved_at": (
                            saved_at_by_paper.get(pid).isoformat()
                            if saved_at_by_paper.get(pid)
                            else None
                        ),
                        "reading_status": (
                            self._reading_status_to_dict(status_row) if status_row else None
                        ),
                        "latest_judge": self._judge_score_to_dict(judge_row) if judge_row else None,
                    }
                )

            if sort_by == "judge_score":
                rows.sort(
                    key=lambda row: float(((row.get("latest_judge") or {}).get("overall") or 0.0)),
                    reverse=True,
                )
            elif sort_by == "published_at":
                rows.sort(
                    key=lambda row: str(((row.get("paper") or {}).get("published_at") or "")),
                    reverse=True,
                )
            else:
                rows.sort(key=lambda row: str(row.get("saved_at") or ""), reverse=True)

            return rows[: max(1, int(limit))]

    def list_track_feed(
        self,
        *,
        user_id: str,
        track_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(
                    ResearchTrackModel.user_id == user_id,
                    ResearchTrackModel.id == int(track_id),
                    ResearchTrackModel.archived_at.is_(None),
                )
            ).scalar_one_or_none()
            if track is None:
                return {"items": [], "total": 0}

            track_dict = self._track_to_dict(track)
            raw_terms = [
                *track_dict.get("keywords", []),
                *track_dict.get("methods", []),
                *track_dict.get("venues", []),
            ]
            terms = sorted({str(term).strip().lower() for term in raw_terms if str(term).strip()})

            stmt = select(PaperModel).where(PaperModel.deleted_at.is_(None))
            if terms:
                term_filters = []
                for term in terms:
                    like = f"%{term}%"
                    term_filters.extend(
                        [
                            func.lower(func.coalesce(PaperModel.title, "")).like(like),
                            func.lower(func.coalesce(PaperModel.abstract, "")).like(like),
                            func.lower(func.coalesce(PaperModel.venue, "")).like(like),
                            func.lower(func.coalesce(PaperModel.keywords_json, "")).like(like),
                            func.lower(func.coalesce(PaperModel.fields_of_study_json, "")).like(
                                like
                            ),
                        ]
                    )
                stmt = stmt.where(or_(*term_filters))

            feedback_rows = (
                session.execute(
                    select(PaperFeedbackModel)
                    .where(
                        PaperFeedbackModel.user_id == user_id,
                        PaperFeedbackModel.track_id == int(track_id),
                        PaperFeedbackModel.action.in_(
                            [
                                "save",
                                "unsave",
                                "like",
                                "unlike",
                                "dislike",
                                "undislike",
                                "skip",
                                "cite",
                            ]
                        ),
                    )
                    .order_by(desc(PaperFeedbackModel.ts), desc(PaperFeedbackModel.id))
                )
                .scalars()
                .all()
            )

            feedback_candidate_ids = {
                int(row.canonical_paper_id or row.paper_ref_id or 0)
                for row in feedback_rows
                if int(row.canonical_paper_id or row.paper_ref_id or 0) > 0
            }

            fetch_cap = max(200, (int(offset) + int(limit)) * 8)
            candidates = (
                session.execute(
                    stmt.order_by(desc(PaperModel.created_at), desc(PaperModel.id)).limit(fetch_cap)
                )
                .scalars()
                .all()
            )

            if feedback_candidate_ids:
                existing_ids = {int(p.id) for p in candidates}
                missing_ids = sorted(feedback_candidate_ids - existing_ids)
                if missing_ids:
                    extra = (
                        session.execute(
                            select(PaperModel)
                            .where(PaperModel.id.in_(missing_ids), PaperModel.deleted_at.is_(None))
                            .order_by(desc(PaperModel.created_at), desc(PaperModel.id))
                        )
                        .scalars()
                        .all()
                    )
                    candidates.extend(extra)

            if not candidates:
                return {"items": [], "total": 0}

            candidate_ids = [int(p.id) for p in candidates]

            feedback_summary_by_paper: Dict[int, Dict[str, int]] = {}
            feedback_state_by_paper: Dict[int, Dict[str, Any]] = {}
            resolved_feedback_groups: set[tuple[int, str]] = set()
            for row in feedback_rows:
                pid = int(row.canonical_paper_id or row.paper_ref_id or 0)
                if pid <= 0:
                    continue
                normalized_action = self._normalize_feedback_action(str(row.action or ""))
                if normalized_action:
                    action_counter = feedback_summary_by_paper.setdefault(pid, {})
                    action_counter[normalized_action] = action_counter.get(normalized_action, 0) + 1

                group = self._feedback_group(normalized_action)
                group_key = (pid, group)
                if group_key in resolved_feedback_groups:
                    continue
                resolved_feedback_groups.add(group_key)

                effective_action = self._effective_feedback_action(normalized_action)
                if effective_action is None:
                    continue

                state = feedback_state_by_paper.setdefault(
                    pid,
                    {
                        "is_saved": False,
                        "is_liked": False,
                        "is_disliked": False,
                        "latest_effective_action": None,
                        "latest_effective_ts": None,
                    },
                )

                if effective_action == "save":
                    state["is_saved"] = True
                elif effective_action == "like":
                    state["is_liked"] = True
                    state["is_disliked"] = False
                elif effective_action == "dislike":
                    state["is_disliked"] = True
                    state["is_liked"] = False

                event_ts = row.ts or _utcnow()
                latest_effective_ts = state.get("latest_effective_ts")
                if latest_effective_ts is None or event_ts >= latest_effective_ts:
                    state["latest_effective_action"] = effective_action
                    state["latest_effective_ts"] = event_ts

            status_by_paper = {
                int(row.paper_id): row
                for row in session.execute(
                    select(PaperReadingStatusModel).where(
                        PaperReadingStatusModel.user_id == user_id,
                        PaperReadingStatusModel.paper_id.in_(candidate_ids),
                    )
                )
                .scalars()
                .all()
            }

            latest_judge_by_paper: Dict[int, PaperJudgeScoreModel] = {}
            judge_rows = (
                session.execute(
                    select(PaperJudgeScoreModel)
                    .where(PaperJudgeScoreModel.paper_id.in_(candidate_ids))
                    .order_by(desc(PaperJudgeScoreModel.scored_at), desc(PaperJudgeScoreModel.id))
                )
                .scalars()
                .all()
            )
            for judge in judge_rows:
                pid = int(judge.paper_id or 0)
                if pid > 0 and pid not in latest_judge_by_paper:
                    latest_judge_by_paper[pid] = judge

            scored_rows: List[Dict[str, Any]] = []
            for paper in candidates:
                pid = int(paper.id)
                text_blob = " ".join(
                    [
                        str(paper.title or ""),
                        str(paper.abstract or ""),
                        str(paper.venue or ""),
                        " ".join(str(x) for x in (paper.get_keywords() or [])),
                        " ".join(str(x) for x in (paper.get_fields_of_study() or [])),
                    ]
                ).lower()

                matched_terms = [term for term in terms if term and term in text_blob]
                keyword_score = float(len(matched_terms))

                feedback_state = feedback_state_by_paper.get(pid, {})
                latest_feedback_action = (
                    str(feedback_state.get("latest_effective_action") or "").strip().lower()
                )
                feedback_boost = 0.0
                if bool(feedback_state.get("is_saved")):
                    feedback_boost += 3.0
                if bool(feedback_state.get("is_liked")):
                    feedback_boost += 2.0
                if bool(feedback_state.get("is_disliked")):
                    feedback_boost -= 4.0
                if latest_feedback_action == "skip":
                    feedback_boost -= 1.0

                citation_score = min(float(paper.citation_count or 0) / 200.0, 2.0)
                judge_row = latest_judge_by_paper.get(pid)
                judge_score = float(judge_row.overall or 0.0) if judge_row else 0.0

                if terms and keyword_score <= 0 and abs(feedback_boost) < 1e-6:
                    continue

                feed_score = (
                    keyword_score * 2.5 + feedback_boost + citation_score + judge_score * 0.3
                )
                scored_rows.append(
                    {
                        "paper": self._paper_to_dict(paper),
                        "latest_judge": (
                            self._judge_score_to_dict(latest_judge_by_paper[pid])
                            if pid in latest_judge_by_paper
                            else None
                        ),
                        "reading_status": (
                            self._reading_status_to_dict(status_by_paper[pid])
                            if pid in status_by_paper
                            else None
                        ),
                        "latest_feedback_action": latest_feedback_action or None,
                        "is_saved": bool(feedback_state.get("is_saved")),
                        "is_liked": bool(feedback_state.get("is_liked")),
                        "is_disliked": bool(feedback_state.get("is_disliked")),
                        "feedback_summary": feedback_summary_by_paper.get(pid, {}),
                        "matched_terms": matched_terms,
                        "keyword_score": keyword_score,
                        "feed_score": round(feed_score, 4),
                    }
                )

            scored_rows.sort(
                key=lambda row: (
                    float(row.get("feed_score") or 0.0),
                    float(((row.get("latest_judge") or {}).get("overall") or 0.0)),
                    str(((row.get("paper") or {}).get("created_at") or "")),
                ),
                reverse=True,
            )

            total = len(scored_rows)
            start = max(0, int(offset))
            end = start + max(1, int(limit))
            return {"items": scored_rows[start:end], "total": total}

    def create_collection(
        self,
        *,
        user_id: str,
        name: str,
        description: str = "",
        track_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        now = _utcnow()
        with self._provider.session() as session:
            if track_id is not None:
                track = session.execute(
                    select(ResearchTrackModel).where(
                        ResearchTrackModel.user_id == user_id,
                        ResearchTrackModel.id == int(track_id),
                    )
                ).scalar_one_or_none()
                if track is None:
                    raise ValueError("Track not found")

            row = PaperCollectionModel(
                user_id=user_id,
                track_id=int(track_id) if track_id is not None else None,
                name=str(name or "").strip(),
                description=str(description or "").strip(),
                created_at=now,
                updated_at=now,
                metadata_json="{}",
            )
            session.add(row)
            try:
                session.commit()
                session.refresh(row)
            except IntegrityError:
                session.rollback()
                existing = session.execute(
                    select(PaperCollectionModel).where(
                        PaperCollectionModel.user_id == user_id,
                        PaperCollectionModel.name == row.name,
                    )
                ).scalar_one()
                return self._collection_to_dict(existing, item_count=0)
            return self._collection_to_dict(row, item_count=0)

    def list_collections(
        self,
        *,
        user_id: str,
        include_archived: bool = False,
        track_id: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            stmt = select(PaperCollectionModel).where(PaperCollectionModel.user_id == user_id)
            if not include_archived:
                stmt = stmt.where(PaperCollectionModel.archived_at.is_(None))
            if track_id is not None:
                stmt = stmt.where(PaperCollectionModel.track_id == int(track_id))
            rows = (
                session.execute(
                    stmt.order_by(desc(PaperCollectionModel.updated_at)).limit(max(1, int(limit)))
                )
                .scalars()
                .all()
            )
            counts: Dict[int, int] = {
                int(collection_id): int(item_count)
                for collection_id, item_count in session.execute(
                    select(
                        PaperCollectionItemModel.collection_id,
                        func.count(PaperCollectionItemModel.id),
                    )
                    .where(
                        PaperCollectionItemModel.collection_id.in_(
                            [int(row.id) for row in rows] or [-1]
                        )
                    )
                    .group_by(PaperCollectionItemModel.collection_id)
                ).all()
            }
            return [
                self._collection_to_dict(row, item_count=counts.get(int(row.id), 0)) for row in rows
            ]

    def update_collection(
        self,
        *,
        user_id: str,
        collection_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        archived: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(PaperCollectionModel).where(
                    PaperCollectionModel.user_id == user_id,
                    PaperCollectionModel.id == int(collection_id),
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            if name is not None:
                row.name = str(name).strip()
            if description is not None:
                row.description = str(description).strip()
            if archived is not None:
                row.archived_at = now if bool(archived) else None
            row.updated_at = now
            session.add(row)
            try:
                session.commit()
                session.refresh(row)
            except IntegrityError:
                session.rollback()
                raise

            item_count = session.execute(
                select(func.count(PaperCollectionItemModel.id)).where(
                    PaperCollectionItemModel.collection_id == int(collection_id)
                )
            ).scalar_one()
            return self._collection_to_dict(row, item_count=int(item_count or 0))

    def list_collection_items(
        self,
        *,
        user_id: str,
        collection_id: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            collection = session.execute(
                select(PaperCollectionModel).where(
                    PaperCollectionModel.id == int(collection_id),
                    PaperCollectionModel.user_id == user_id,
                )
            ).scalar_one_or_none()
            if collection is None:
                return []

            rows = (
                session.execute(
                    select(PaperCollectionItemModel)
                    .where(PaperCollectionItemModel.collection_id == int(collection_id))
                    .order_by(desc(PaperCollectionItemModel.updated_at))
                    .limit(max(1, int(limit)))
                )
                .scalars()
                .all()
            )
            if not rows:
                return []
            paper_ids = [int(row.paper_id) for row in rows]
            paper_by_id = {
                int(p.id): p
                for p in session.execute(select(PaperModel).where(PaperModel.id.in_(paper_ids)))
                .scalars()
                .all()
            }
            result: List[Dict[str, Any]] = []
            for row in rows:
                paper = paper_by_id.get(int(row.paper_id))
                if paper is None:
                    continue
                result.append(
                    {
                        "id": int(row.id),
                        "collection_id": int(row.collection_id),
                        "paper_id": int(row.paper_id),
                        "paper": self._paper_to_dict(paper),
                        "note": row.note,
                        "tags": _load_list(row.tags_json),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    }
                )
            return result

    def upsert_collection_item(
        self,
        *,
        user_id: str,
        collection_id: int,
        paper_id: str,
        note: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            collection = session.execute(
                select(PaperCollectionModel).where(
                    PaperCollectionModel.id == int(collection_id),
                    PaperCollectionModel.user_id == user_id,
                    PaperCollectionModel.archived_at.is_(None),
                )
            ).scalar_one_or_none()
            if collection is None:
                return None

            resolved_paper_id = self._resolve_paper_ref_id(
                session=session,
                paper_id=str(paper_id or "").strip(),
                metadata={},
            )
            if resolved_paper_id is None:
                return None

            row = session.execute(
                select(PaperCollectionItemModel).where(
                    PaperCollectionItemModel.collection_id == int(collection_id),
                    PaperCollectionItemModel.paper_id == int(resolved_paper_id),
                )
            ).scalar_one_or_none()
            if row is None:
                row = PaperCollectionItemModel(
                    collection_id=int(collection_id),
                    paper_id=int(resolved_paper_id),
                    created_at=now,
                    updated_at=now,
                    note=str(note or "").strip(),
                    tags_json=_dump_list(tags),
                    metadata_json="{}",
                )
                session.add(row)
            else:
                if note is not None:
                    row.note = str(note).strip()
                if tags is not None:
                    row.tags_json = _dump_list(tags)
                row.updated_at = now
                session.add(row)

            collection.updated_at = now
            session.add(collection)
            session.commit()
            session.refresh(row)

            paper = session.execute(
                select(PaperModel).where(PaperModel.id == int(row.paper_id))
            ).scalar_one()
            return {
                "id": int(row.id),
                "collection_id": int(row.collection_id),
                "paper_id": int(row.paper_id),
                "paper": self._paper_to_dict(paper),
                "note": row.note,
                "tags": _load_list(row.tags_json),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    def remove_collection_item(
        self,
        *,
        user_id: str,
        collection_id: int,
        paper_id: str,
    ) -> bool:
        now = _utcnow()
        with self._provider.session() as session:
            collection = session.execute(
                select(PaperCollectionModel).where(
                    PaperCollectionModel.id == int(collection_id),
                    PaperCollectionModel.user_id == user_id,
                )
            ).scalar_one_or_none()
            if collection is None:
                return False

            resolved_paper_id = self._resolve_paper_ref_id(
                session=session,
                paper_id=str(paper_id or "").strip(),
                metadata={},
            )
            if resolved_paper_id is None:
                return False

            row = session.execute(
                select(PaperCollectionItemModel).where(
                    PaperCollectionItemModel.collection_id == int(collection_id),
                    PaperCollectionItemModel.paper_id == int(resolved_paper_id),
                )
            ).scalar_one_or_none()
            if row is None:
                return False

            session.delete(row)
            collection.updated_at = now
            session.add(collection)
            session.commit()
            return True

    def ingest_repo_enrichment_rows(
        self,
        *,
        rows: List[Dict[str, Any]],
        source: str = "paperscool_repo_enrich",
    ) -> Dict[str, int]:
        now = _utcnow()
        created = 0
        updated = 0
        skipped = 0
        unresolved = 0

        with self._provider.session() as session:
            for raw in rows or []:
                if not isinstance(raw, dict):
                    skipped += 1
                    continue

                github = raw.get("github") if isinstance(raw.get("github"), dict) else {}
                repo_url = str(raw.get("repo_url") or github.get("repo_url") or "").strip()
                if not repo_url:
                    skipped += 1
                    continue

                paper_meta = {
                    "title": raw.get("title"),
                    "paper_url": raw.get("paper_url"),
                    "url": raw.get("paper_url"),
                }
                paper_hint = str(raw.get("paper_id") or raw.get("paper_ref_id") or "").strip()
                paper_ref_id = self._resolve_paper_ref_id(
                    session=session,
                    paper_id=paper_hint,
                    metadata=paper_meta,
                )
                if not paper_ref_id:
                    unresolved += 1
                    continue

                was_created = self._upsert_paper_repo_row(
                    session=session,
                    paper_ref_id=int(paper_ref_id),
                    repo_row=raw,
                    source=source,
                    now=now,
                )
                if was_created is None:
                    skipped += 1
                elif was_created:
                    created += 1
                else:
                    updated += 1

            session.commit()

        return {
            "total": created + updated,
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "unresolved_paper": unresolved,
        }

    def list_paper_repos(self, *, paper_id: str) -> Optional[List[Dict[str, Any]]]:
        with self._provider.session() as session:
            paper_ref_id = self._resolve_paper_ref_id(
                session=session,
                paper_id=(paper_id or "").strip(),
                metadata={},
            )
            if not paper_ref_id:
                return None

            rows = (
                session.execute(
                    select(PaperRepoModel)
                    .where(PaperRepoModel.paper_id == int(paper_ref_id))
                    .order_by(desc(PaperRepoModel.stars), desc(PaperRepoModel.synced_at))
                )
                .scalars()
                .all()
            )
            return [self._repo_to_dict(row) for row in rows]

    def get_paper_detail(
        self, *, paper_id: str, user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            paper_ref_id = self._resolve_paper_ref_id(
                session=session,
                paper_id=(paper_id or "").strip(),
                metadata={},
            )
            if not paper_ref_id:
                return None

            paper = session.execute(
                select(PaperModel).where(PaperModel.id == int(paper_ref_id))
            ).scalar_one_or_none()
            if not paper:
                return None

            reading_status = session.execute(
                select(PaperReadingStatusModel).where(
                    PaperReadingStatusModel.user_id == user_id,
                    PaperReadingStatusModel.paper_id == int(paper_ref_id),
                )
            ).scalar_one_or_none()

            judge_scores = (
                session.execute(
                    select(PaperJudgeScoreModel)
                    .where(PaperJudgeScoreModel.paper_id == int(paper_ref_id))
                    .order_by(desc(PaperJudgeScoreModel.scored_at), desc(PaperJudgeScoreModel.id))
                )
                .scalars()
                .all()
            )

            feedback_rows = (
                session.execute(
                    select(PaperFeedbackModel)
                    .where(
                        PaperFeedbackModel.user_id == user_id,
                        PaperFeedbackModel.paper_ref_id == int(paper_ref_id),
                    )
                    .order_by(desc(PaperFeedbackModel.ts), desc(PaperFeedbackModel.id))
                    .limit(100)
                )
                .scalars()
                .all()
            )

            repo_rows = (
                session.execute(
                    select(PaperRepoModel)
                    .where(PaperRepoModel.paper_id == int(paper_ref_id))
                    .order_by(desc(PaperRepoModel.stars), desc(PaperRepoModel.synced_at))
                )
                .scalars()
                .all()
            )

            feedback_summary: Dict[str, int] = {}
            for row in feedback_rows:
                action = str(row.action or "")
                if not action:
                    continue
                feedback_summary[action] = feedback_summary.get(action, 0) + 1

            return {
                "paper": self._paper_to_dict(paper),
                "reading_status": (
                    self._reading_status_to_dict(reading_status) if reading_status else None
                ),
                "latest_judge": (
                    self._judge_score_to_dict(judge_scores[0]) if judge_scores else None
                ),
                "judge_scores": [self._judge_score_to_dict(row) for row in judge_scores],
                "repos": [self._repo_to_dict(row) for row in repo_rows],
                "feedback_summary": feedback_summary,
                "feedback_rows": [self._feedback_to_dict(row) for row in feedback_rows],
            }

    def create_context_run(
        self,
        *,
        user_id: str,
        track_id: Optional[int],
        query: str,
        merged_query: str,
        stage: str,
        exploration_ratio: float,
        diversity_strength: float,
        routing: Dict[str, Any],
        papers: List[Dict[str, Any]],
        paper_scores: Dict[str, float],
        paper_reasons: Dict[str, List[str]],
    ) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        with self._provider.session() as session:
            run = ResearchContextRunModel(
                user_id=user_id,
                track_id=int(track_id) if track_id is not None else None,
                query=(query or "").strip(),
                merged_query=(merged_query or "").strip(),
                stage=(stage or "auto").strip() or "auto",
                exploration_ratio=float(exploration_ratio or 0.0),
                diversity_strength=float(diversity_strength or 0.0),
                routing_json=json.dumps(routing or {}, ensure_ascii=False),
                created_at=now,
            )
            session.add(run)
            session.flush()

            for idx, p in enumerate(papers or []):
                pid = str(p.get("paper_id") or "").strip()
                if not pid:
                    continue
                reasons = paper_reasons.get(pid) or []
                session.add(
                    PaperImpressionModel(
                        run_id=int(run.id),
                        user_id=user_id,
                        track_id=int(track_id) if track_id is not None else None,
                        paper_id=pid,
                        rank=int(idx),
                        score=float(paper_scores.get(pid) or 0.0),
                        reasons_json=json.dumps(reasons, ensure_ascii=False),
                        created_at=now,
                    )
                )

            session.commit()
            session.refresh(run)
            return {
                "id": int(run.id),
                "user_id": run.user_id,
                "track_id": run.track_id,
                "stage": run.stage,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }

    def summarize_eval(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        days: int = 30,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        now = _utcnow()
        since = now.replace(microsecond=0) - timedelta(days=int(days or 30))

        with self._provider.session() as session:
            runs_stmt = select(ResearchContextRunModel).where(
                ResearchContextRunModel.user_id == user_id,
                ResearchContextRunModel.created_at >= since,
            )
            if track_id is not None:
                runs_stmt = runs_stmt.where(ResearchContextRunModel.track_id == int(track_id))
            runs = (
                session.execute(
                    runs_stmt.order_by(desc(ResearchContextRunModel.created_at)).limit(int(limit))
                )
                .scalars()
                .all()
            )
            run_ids = [int(r.id) for r in runs]

            impressions: List[PaperImpressionModel] = []
            if run_ids:
                imp_stmt = select(PaperImpressionModel).where(
                    PaperImpressionModel.run_id.in_(run_ids)
                )
                impressions = session.execute(imp_stmt).scalars().all()

            fb_stmt = select(PaperFeedbackModel).where(
                PaperFeedbackModel.user_id == user_id,
                PaperFeedbackModel.ts >= since,
            )
            if track_id is not None:
                fb_stmt = fb_stmt.where(PaperFeedbackModel.track_id == int(track_id))
            feedback = session.execute(fb_stmt).scalars().all()

        total_runs = len(runs)
        total_impressions = len(impressions)
        unique_papers = len({(int(i.track_id or 0), i.paper_id) for i in impressions})
        repeat_rate = 0.0
        if total_impressions > 0:
            repeat_rate = max(0.0, 1.0 - (unique_papers / float(total_impressions)))

        # Map most recent feedback per (track, paper) within the window.
        fb_by_paper: Dict[tuple[int, str], str] = {}
        linked_feedback = 0
        for f in feedback:
            key = (int(f.track_id or 0), str(f.paper_id or "").strip())
            if not key[1]:
                continue
            fb_by_paper[key] = str(f.action or "").strip()
            try:
                meta = json.loads(f.metadata_json or "{}")
                if isinstance(meta, dict) and meta.get("context_run_id"):
                    linked_feedback += 1
            except Exception:
                pass

        recommended_keys = {
            (int(i.track_id or 0), str(i.paper_id or "").strip())
            for i in impressions
            if str(i.paper_id or "").strip()
        }
        feedback_on_recommended: Dict[str, int] = {
            "like": 0,
            "save": 0,
            "dislike": 0,
            "skip": 0,
            "cite": 0,
            "other": 0,
        }
        for key in recommended_keys:
            action = fb_by_paper.get(key)
            if not action:
                continue
            if action in feedback_on_recommended:
                feedback_on_recommended[action] += 1
            else:
                feedback_on_recommended["other"] += 1

        denom = max(1, len(recommended_keys))
        return {
            "window_days": int(days or 30),
            "runs": total_runs,
            "impressions": total_impressions,
            "unique_recommended_papers": len(recommended_keys),
            "repeat_rate": float(repeat_rate),
            "feedback_on_recommended": feedback_on_recommended,
            "feedback_coverage": float(sum(feedback_on_recommended.values())) / float(denom),
            "linked_feedback_rows": int(linked_feedback),
        }

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
    def _collection_to_dict(c: PaperCollectionModel, *, item_count: int = 0) -> Dict[str, Any]:
        try:
            metadata = json.loads(c.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
        return {
            "id": int(c.id),
            "user_id": c.user_id,
            "track_id": int(c.track_id) if c.track_id is not None else None,
            "name": c.name,
            "description": c.description,
            "archived_at": c.archived_at.isoformat() if c.archived_at else None,
            "item_count": int(item_count),
            "metadata": metadata,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }

    @staticmethod
    def _normalize_reading_status(value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized in {"unread", "reading", "read", "archived"}:
            return normalized
        return "unread"

    def _upsert_reading_status_row(
        self,
        *,
        session,
        user_id: str,
        paper_ref_id: int,
        status: str,
        mark_saved: Optional[bool],
        metadata: Optional[Dict[str, Any]],
        now: datetime,
    ) -> PaperReadingStatusModel:
        row = session.execute(
            select(PaperReadingStatusModel).where(
                PaperReadingStatusModel.user_id == user_id,
                PaperReadingStatusModel.paper_id == int(paper_ref_id),
            )
        ).scalar_one_or_none()
        if row is None:
            row = PaperReadingStatusModel(
                user_id=user_id,
                paper_id=int(paper_ref_id),
                created_at=now,
                updated_at=now,
            )
            session.add(row)

        row.status = self._normalize_reading_status(status)
        row.updated_at = now

        if row.status == "read" and row.read_at is None:
            row.read_at = now

        if mark_saved is True:
            row.saved_at = row.saved_at or now
        elif mark_saved is False:
            row.saved_at = None

        if metadata is not None:
            row.metadata_json = json.dumps(metadata, ensure_ascii=False)

        session.add(row)
        return row

    @staticmethod
    def _paper_to_dict(p: PaperModel) -> Dict[str, Any]:
        metadata_raw = getattr(p, "metadata_json", "{}") or "{}"
        try:
            metadata = json.loads(metadata_raw)
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}

        published_at = None
        if getattr(p, "publication_date", None):
            published_at = str(getattr(p, "publication_date"))

        source = getattr(p, "primary_source", None) or getattr(p, "source", "")

        return {
            "id": int(p.id),
            "arxiv_id": p.arxiv_id,
            "doi": p.doi,
            "semantic_scholar_id": getattr(p, "semantic_scholar_id", None),
            "openalex_id": getattr(p, "openalex_id", None),
            "title": p.title,
            "authors": p.get_authors(),
            "abstract": p.abstract,
            "url": p.url,
            "external_url": p.url,
            "pdf_url": p.pdf_url,
            "source": source,
            "primary_source": source,
            "venue": p.venue,
            "year": getattr(p, "year", None),
            "publication_date": getattr(p, "publication_date", None),
            "published_at": published_at,
            "first_seen_at": (
                (getattr(p, "first_seen_at", None) or p.created_at).isoformat()
                if (getattr(p, "first_seen_at", None) or p.created_at)
                else None
            ),
            "keywords": p.get_keywords(),
            "fields_of_study": p.get_fields_of_study(),
            "sources": p.get_sources(),
            "citation_count": int(getattr(p, "citation_count", 0) or 0),
            "metadata": metadata,
        }

    @staticmethod
    def _judge_score_to_dict(row: PaperJudgeScoreModel) -> Dict[str, Any]:
        try:
            metadata = json.loads(row.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}

        return {
            "id": int(row.id),
            "paper_id": int(row.paper_id),
            "query": row.query,
            "overall": float(row.overall or 0.0),
            "relevance": float(row.relevance or 0.0),
            "novelty": float(row.novelty or 0.0),
            "rigor": float(row.rigor or 0.0),
            "impact": float(row.impact or 0.0),
            "clarity": float(row.clarity or 0.0),
            "recommendation": row.recommendation,
            "one_line_summary": row.one_line_summary,
            "judge_model": row.judge_model,
            "judge_cost_tier": row.judge_cost_tier,
            "scored_at": row.scored_at.isoformat() if row.scored_at else None,
            "metadata": metadata,
        }

    @staticmethod
    def _reading_status_to_dict(row: PaperReadingStatusModel) -> Dict[str, Any]:
        try:
            metadata = json.loads(row.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
        return {
            "id": int(row.id),
            "user_id": row.user_id,
            "paper_id": int(row.paper_id),
            "status": row.status,
            "saved_at": row.saved_at.isoformat() if row.saved_at else None,
            "read_at": row.read_at.isoformat() if row.read_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "metadata": metadata,
        }

    @staticmethod
    def _repo_to_dict(row: PaperRepoModel) -> Dict[str, Any]:
        try:
            metadata = json.loads(row.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}

        return {
            "id": int(row.id),
            "paper_id": int(row.paper_id),
            "repo_url": row.repo_url,
            "full_name": row.full_name,
            "description": row.description,
            "stars": int(row.stars or 0),
            "forks": int(row.forks or 0),
            "open_issues": int(row.open_issues or 0),
            "watchers": int(row.watchers or 0),
            "language": row.language,
            "license": row.license,
            "archived": bool(row.archived),
            "html_url": row.html_url,
            "topics": row.get_topics(),
            "updated_at_remote": (
                row.updated_at_remote.isoformat() if row.updated_at_remote else None
            ),
            "pushed_at_remote": row.pushed_at_remote.isoformat() if row.pushed_at_remote else None,
            "query": row.query,
            "source": row.source,
            "synced_at": row.synced_at.isoformat() if row.synced_at else None,
            "metadata": metadata,
        }

    def _upsert_paper_repo_row(
        self,
        *,
        session,
        paper_ref_id: int,
        repo_row: Dict[str, Any],
        source: str,
        now: datetime,
    ) -> Optional[bool]:
        github = repo_row.get("github") if isinstance(repo_row.get("github"), dict) else {}
        repo_url = str(repo_row.get("repo_url") or github.get("repo_url") or "").strip()
        if not repo_url:
            return None

        row = session.execute(
            select(PaperRepoModel).where(
                PaperRepoModel.paper_id == int(paper_ref_id),
                PaperRepoModel.repo_url == repo_url,
            )
        ).scalar_one_or_none()
        created = row is None
        if row is None:
            row = PaperRepoModel(
                paper_id=int(paper_ref_id),
                repo_url=repo_url,
                created_at=now,
                updated_at=now,
                synced_at=now,
            )
            session.add(row)

        row.full_name = str(github.get("full_name") or repo_row.get("full_name") or "").strip()
        row.description = str(github.get("description") or repo_row.get("description") or "")
        row.stars = _safe_int(github.get("stars") or repo_row.get("stars"), 0)
        row.forks = _safe_int(github.get("forks") or repo_row.get("forks"), 0)
        row.open_issues = _safe_int(github.get("open_issues") or repo_row.get("open_issues"), 0)
        row.watchers = _safe_int(github.get("watchers") or repo_row.get("watchers"), 0)
        row.language = str(github.get("language") or repo_row.get("language") or "").strip()
        row.license = str(github.get("license") or repo_row.get("license") or "").strip()
        row.archived = bool(github.get("archived") or repo_row.get("archived"))
        row.html_url = str(github.get("html_url") or repo_row.get("html_url") or repo_url).strip()
        row.updated_at_remote = _parse_datetime(
            github.get("updated_at") or repo_row.get("updated_at")
        )
        row.pushed_at_remote = _parse_datetime(github.get("pushed_at") or repo_row.get("pushed_at"))
        row.query = str(repo_row.get("query") or "").strip()
        row.source = (str(source or "").strip() or "paperscool_repo_enrich")[:32]

        topics = github.get("topics") or repo_row.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        row.set_topics([str(v) for v in topics if str(v).strip()])

        metadata = {
            "title": repo_row.get("title"),
            "paper_url": repo_row.get("paper_url"),
            "github": github,
        }
        row.metadata_json = json.dumps(metadata, ensure_ascii=False)
        row.synced_at = now
        row.updated_at = now

        session.add(row)
        return created

    def _resolve_paper_ref_id(
        self,
        *,
        session,
        paper_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[int]:
        pid = (paper_id or "").strip()
        hints = dict(metadata or {})

        # Main path: centralized identity resolver (paper_identifiers + normalized fallbacks).
        resolved = self._identity_resolver.resolve(pid, hints=hints)
        if resolved is not None:
            return int(resolved)

        Logger.info(
            "IdentityResolver miss; falling back to legacy paper_id resolution",
            file=LogFiles.HARVEST,
        )
        return self._resolve_paper_ref_id_legacy(session=session, paper_id=pid, metadata=hints)

    @staticmethod
    def _resolve_paper_ref_id_legacy(
        *,
        session,
        paper_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[int]:
        pid = (paper_id or "").strip()

        if pid.isdigit():
            row = session.execute(
                select(PaperModel).where(PaperModel.id == int(pid))
            ).scalar_one_or_none()
            if row is not None:
                return int(row.id)

        arxiv_id = normalize_arxiv_id(pid) if pid else None
        doi = normalize_doi(pid) if pid else None

        url_candidates = []
        for key in ("paper_url", "url", "external_url", "pdf_url"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                url_candidates.append(value.strip())
        if pid.startswith("http"):
            url_candidates.append(pid)

        if not arxiv_id:
            for candidate in url_candidates:
                arxiv_id = normalize_arxiv_id(candidate)
                if arxiv_id:
                    break
        if not doi:
            for candidate in url_candidates:
                doi = normalize_doi(candidate)
                if doi:
                    break

        if arxiv_id:
            row = session.execute(
                select(PaperModel).where(PaperModel.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            if row is not None:
                return int(row.id)

        if doi:
            row = session.execute(
                select(PaperModel).where(PaperModel.doi == doi)
            ).scalar_one_or_none()
            if row is not None:
                return int(row.id)

        # TODO: scalar_one_or_none() can raise MultipleResultsFound if
        #  multiple papers share the same URL or title. Switch to .first().
        if url_candidates:
            row = session.execute(
                select(PaperModel).where(
                    or_(
                        PaperModel.url.in_(url_candidates),
                        PaperModel.pdf_url.in_(url_candidates),
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                return int(row.id)

        title = str(metadata.get("title") or "").strip()
        if title:
            row = session.execute(
                select(PaperModel).where(func.lower(PaperModel.title) == title.lower())
            ).scalar_one_or_none()
            if row is not None:
                return int(row.id)

        return None

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
            "paper_ref_id": f.paper_ref_id,
            "canonical_paper_id": f.canonical_paper_id,
            "action": f.action,
            "weight": float(f.weight or 0.0),
            "ts": f.ts.isoformat() if f.ts else None,
            "metadata": metadata,
        }
