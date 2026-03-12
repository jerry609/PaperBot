"""Track-centric read model for research workspace surfaces."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from paperbot.application.ports.memory_port import MemoryPort
from paperbot.application.ports.research_track_read_port import ResearchTrackReadPort


@dataclass(frozen=True)
class TrackContextQuery:
    """Read limits for a track context snapshot."""

    task_limit: int = 5
    milestone_limit: int = 3
    feedback_limit: int = 25
    feedback_scan_limit: int = 200
    saved_preview_limit: int = 5
    saved_scan_limit: int = 200
    memory_scan_limit: int = 500
    top_tag_limit: int = 5
    eval_days: int = 30


@dataclass(frozen=True)
class TrackMemoryStats:
    """Basic memory summary for a research track."""

    total_items: int
    approved_items: int
    pending_items: int
    top_tags: List[str] = field(default_factory=list)
    latest_memory_at: Optional[str] = None


@dataclass(frozen=True)
class TrackFeedbackSummary:
    """Effective feedback summary for a research track."""

    total_items: int
    actions: Dict[str, int] = field(default_factory=dict)
    latest_feedback_at: Optional[str] = None
    recent_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TrackSavedPaperSummary:
    """Saved-paper summary for a research track."""

    total_items: int
    latest_saved_at: Optional[str] = None
    recent_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TrackContextSnapshot:
    """Track-centric aggregate used by API routes and web clients."""

    track: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    memory: TrackMemoryStats
    feedback: TrackFeedbackSummary
    saved_papers: TrackSavedPaperSummary
    eval_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchTrackContextService:
    """Compose a track-scoped read model from existing stores."""

    def __init__(
        self,
        *,
        track_reader: ResearchTrackReadPort,
        memory_store: MemoryPort,
    ) -> None:
        self._track_reader = track_reader
        self._memory_store = memory_store

    def get_track_context(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        query: Optional[TrackContextQuery] = None,
    ) -> Optional[TrackContextSnapshot]:
        limits = query or TrackContextQuery()
        track = self._resolve_track(user_id=user_id, track_id=track_id)
        if track is None:
            return None

        resolved_track_id = int(track.get("id") or 0)
        if resolved_track_id <= 0:
            return None

        tasks = self._track_reader.list_tasks(
            user_id=user_id,
            track_id=resolved_track_id,
            limit=limits.task_limit,
        )
        milestones = self._track_reader.list_milestones(
            user_id=user_id,
            track_id=resolved_track_id,
            limit=limits.milestone_limit,
        )
        memory = self._build_memory_stats(
            user_id=user_id,
            track_id=resolved_track_id,
            query=limits,
        )
        feedback = self._build_feedback_summary(
            user_id=user_id,
            track_id=resolved_track_id,
            query=limits,
        )
        saved_papers = self._build_saved_paper_summary(
            user_id=user_id,
            track_id=resolved_track_id,
            query=limits,
        )
        eval_summary = self._track_reader.summarize_eval(
            user_id=user_id,
            track_id=resolved_track_id,
            days=limits.eval_days,
        )

        return TrackContextSnapshot(
            track=track,
            tasks=tasks,
            milestones=milestones,
            memory=memory,
            feedback=feedback,
            saved_papers=saved_papers,
            eval_summary=eval_summary,
        )

    def _resolve_track(self, *, user_id: str, track_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if track_id is None:
            return self._track_reader.get_active_track(user_id=user_id)
        return self._track_reader.get_track(user_id=user_id, track_id=int(track_id))

    def _build_memory_stats(
        self,
        *,
        user_id: str,
        track_id: int,
        query: TrackContextQuery,
    ) -> TrackMemoryStats:
        scope_id = str(track_id)
        approved_items = self._memory_store.list_memories(
            user_id=user_id,
            limit=query.memory_scan_limit,
            scope_type="track",
            scope_id=scope_id,
            status="approved",
            include_pending=True,
            include_deleted=False,
        )
        pending_items = self._memory_store.list_memories(
            user_id=user_id,
            limit=query.memory_scan_limit,
            scope_type="track",
            scope_id=scope_id,
            status="pending",
            include_pending=True,
            include_deleted=False,
        )
        all_items = approved_items + pending_items
        tag_counts: Counter[str] = Counter()
        latest_memory_at: Optional[str] = None
        for item in all_items:
            for raw_tag in item.get("tags") or []:
                tag = str(raw_tag).strip()
                if tag:
                    tag_counts[tag] += 1
            latest_memory_at = self._pick_latest_timestamp(
                latest_memory_at,
                str(item.get("updated_at") or item.get("created_at") or "") or None,
            )
        return TrackMemoryStats(
            total_items=len(all_items),
            approved_items=len(approved_items),
            pending_items=len(pending_items),
            top_tags=[tag for tag, _ in tag_counts.most_common(query.top_tag_limit)],
            latest_memory_at=latest_memory_at,
        )

    def _build_feedback_summary(
        self,
        *,
        user_id: str,
        track_id: int,
        query: TrackContextQuery,
    ) -> TrackFeedbackSummary:
        rows = self._track_reader.list_effective_paper_feedback(
            user_id=user_id,
            track_id=track_id,
            limit=max(query.feedback_limit, query.feedback_scan_limit),
        )
        actions = Counter()
        latest_feedback_at: Optional[str] = None
        for row in rows:
            action = str(row.get("action") or "").strip()
            if action:
                actions[action] += 1
            latest_feedback_at = self._pick_latest_timestamp(
                latest_feedback_at,
                str(row.get("ts") or "") or None,
            )
        return TrackFeedbackSummary(
            total_items=len(rows),
            actions=dict(actions),
            latest_feedback_at=latest_feedback_at,
            recent_items=rows[: query.feedback_limit],
        )

    def _build_saved_paper_summary(
        self,
        *,
        user_id: str,
        track_id: int,
        query: TrackContextQuery,
    ) -> TrackSavedPaperSummary:
        rows = self._track_reader.list_saved_papers(
            user_id=user_id,
            track_id=track_id,
            limit=max(query.saved_preview_limit, query.saved_scan_limit),
            sort_by="saved_at",
        )
        latest_saved_at = None
        for row in rows:
            latest_saved_at = self._pick_latest_timestamp(
                latest_saved_at,
                str(row.get("saved_at") or "") or None,
            )
        return TrackSavedPaperSummary(
            total_items=len(rows),
            latest_saved_at=latest_saved_at,
            recent_items=rows[: query.saved_preview_limit],
        )

    @staticmethod
    def _pick_latest_timestamp(
        current_value: Optional[str],
        candidate_value: Optional[str],
    ) -> Optional[str]:
        if not candidate_value:
            return current_value
        if not current_value or candidate_value > current_value:
            return candidate_value
        return current_value
