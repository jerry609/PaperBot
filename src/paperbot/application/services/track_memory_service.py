"""Track-scoped memory facade used by research routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from paperbot.application.ports.research_track_read_port import ResearchTrackReadPort
from paperbot.application.ports.track_memory_store_port import TrackMemoryStorePort


class TrackMemoryScopeError(LookupError):
    """Raised when a required track scope cannot be resolved."""


class TrackMemoryValidationError(ValueError):
    """Raised when a track memory operation receives invalid input."""


@dataclass(frozen=True)
class TrackMemoryScope:
    track_id: int
    scope_id: str


@dataclass(frozen=True)
class TrackMemoryClearResult:
    track_id: int
    deleted_count: int
    retrieved_after_delete_count: int


@dataclass(frozen=True)
class TrackMemoryBulkResult:
    items_before: List[Dict[str, Any]]
    updated_items: List[Dict[str, Any]]
    affected_track_ids: List[int]


class TrackMemoryService:
    """Own track-scoped memory resolution and mutations."""

    def __init__(
        self,
        *,
        track_reader: ResearchTrackReadPort,
        memory_store: TrackMemoryStorePort,
    ) -> None:
        self._track_reader = track_reader
        self._memory_store = memory_store

    def resolve_scope_id(
        self,
        *,
        user_id: str,
        scope_type: str,
        scope_id: Optional[str],
    ) -> Optional[str]:
        normalized_scope_type = (scope_type or "global").strip() or "global"
        if normalized_scope_type != "track":
            return scope_id
        if scope_id:
            try:
                track_id = int(scope_id)
            except (TypeError, ValueError):
                return None
            track = self._track_reader.get_track(user_id=user_id, track_id=track_id)
            return str(track_id) if track is not None else None
        active = self._track_reader.get_active_track(user_id=user_id)
        if not active:
            return None
        track_id = int(active.get("id") or 0)
        return str(track_id) if track_id > 0 else None

    def list_inbox(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        scope = self.require_track_scope(user_id=user_id, track_id=track_id)
        return self._memory_store.list_memories(
            user_id=user_id,
            limit=limit,
            scope_type="track",
            scope_id=scope.scope_id,
            status="pending",
            include_deleted=False,
            include_pending=True,
        )

    def clear_track_memory(
        self,
        *,
        user_id: str,
        track_id: int,
        actor_id: str = "user",
        reason: str = "clear_track_memory",
        verification_limit: int = 100,
    ) -> TrackMemoryClearResult:
        scope = self.require_track_scope(user_id=user_id, track_id=track_id)
        deleted_count = self._memory_store.soft_delete_by_scope(
            user_id=user_id,
            scope_type="track",
            scope_id=scope.scope_id,
            actor_id=actor_id,
            reason=reason,
        )
        if deleted_count <= 0:
            return TrackMemoryClearResult(
                track_id=scope.track_id,
                deleted_count=0,
                retrieved_after_delete_count=0,
            )
        remaining_items = self._memory_store.list_memories(
            user_id=user_id,
            scope_type="track",
            scope_id=scope.scope_id,
            include_deleted=False,
            include_pending=True,
            limit=verification_limit,
        )
        search_results = self._memory_store.search_memories(
            user_id=user_id,
            query="*",
            scope_type="track",
            scope_id=scope.scope_id,
            limit=verification_limit,
        )
        return TrackMemoryClearResult(
            track_id=scope.track_id,
            deleted_count=deleted_count,
            retrieved_after_delete_count=len(remaining_items) + len(search_results),
        )

    def bulk_moderate(
        self,
        *,
        user_id: str,
        item_ids: List[int],
        status: str,
        actor_id: str = "user",
    ) -> TrackMemoryBulkResult:
        items_before = self._memory_store.get_items_by_ids(user_id=user_id, item_ids=item_ids)
        updated_items = self._memory_store.bulk_update_items(
            user_id=user_id,
            item_ids=item_ids,
            status=status,
            actor_id=actor_id,
        )
        return TrackMemoryBulkResult(
            items_before=items_before,
            updated_items=updated_items,
            affected_track_ids=self._extract_track_ids(updated_items),
        )

    def bulk_move(
        self,
        *,
        user_id: str,
        item_ids: List[int],
        scope_type: str,
        scope_id: Optional[str],
        actor_id: str = "user",
    ) -> TrackMemoryBulkResult:
        normalized_scope_type = (scope_type or "global").strip() or "global"
        resolved_scope_id = self.resolve_scope_id(
            user_id=user_id,
            scope_type=normalized_scope_type,
            scope_id=scope_id,
        )
        if normalized_scope_type == "track" and not resolved_scope_id:
            raise TrackMemoryValidationError(
                "track scope requires an existing track or an active track"
            )
        updated_items = self._memory_store.bulk_update_items(
            user_id=user_id,
            item_ids=item_ids,
            scope_type=normalized_scope_type,
            scope_id=resolved_scope_id,
            actor_id=actor_id,
        )
        return TrackMemoryBulkResult(
            items_before=[],
            updated_items=updated_items,
            affected_track_ids=self._extract_track_ids(updated_items),
        )

    def require_track_scope(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
    ) -> TrackMemoryScope:
        if track_id is None:
            track = self._track_reader.get_active_track(user_id=user_id)
            if not track:
                raise TrackMemoryScopeError("No active track for user")
        else:
            track = self._track_reader.get_track(user_id=user_id, track_id=int(track_id))
            if not track:
                raise TrackMemoryScopeError("Track not found")
        resolved_track_id = int(track.get("id") or 0)
        if resolved_track_id <= 0:
            raise TrackMemoryScopeError("Track not found")
        return TrackMemoryScope(track_id=resolved_track_id, scope_id=str(resolved_track_id))

    @staticmethod
    def _extract_track_ids(items: List[Dict[str, Any]]) -> List[int]:
        track_ids = {
            int(item.get("scope_id") or 0)
            for item in items
            if item.get("scope_type") == "track" and item.get("scope_id")
        }
        return sorted(track_id for track_id in track_ids if track_id > 0)
