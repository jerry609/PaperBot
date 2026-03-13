from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from paperbot.application.services.track_memory_service import (
    TrackMemoryScopeError,
    TrackMemoryService,
    TrackMemoryValidationError,
)


class _FakeTrackReader:
    def __init__(self) -> None:
        self.track = {"id": 7, "name": "Track 7", "is_active": True}

    def get_track(self, *, user_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        if user_id == "u1" and track_id == 7:
            return dict(self.track)
        return None

    def get_active_track(self, *, user_id: str) -> Optional[Dict[str, Any]]:
        if user_id == "u1":
            return dict(self.track)
        return None


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_memories(self, **kwargs) -> List[Dict[str, Any]]:
        self.calls.append({"method": "list_memories", **kwargs})
        return [{"id": 1, "scope_type": "track", "scope_id": "7", "status": "pending"}]

    def get_items_by_ids(self, **kwargs) -> List[Dict[str, Any]]:
        self.calls.append({"method": "get_items_by_ids", **kwargs})
        return [{"id": 1, "scope_type": "track", "scope_id": "7", "confidence": 0.9}]

    def search_memories(self, **kwargs) -> List[Dict[str, Any]]:
        self.calls.append({"method": "search_memories", **kwargs})
        return []

    def soft_delete_by_scope(self, **kwargs) -> int:
        self.calls.append({"method": "soft_delete_by_scope", **kwargs})
        return 1

    def bulk_update_items(self, **kwargs) -> List[Dict[str, Any]]:
        self.calls.append({"method": "bulk_update_items", **kwargs})
        return [{"id": 2, "scope_type": "track", "scope_id": "7", "status": kwargs.get("status")}]


def test_list_inbox_uses_active_track_scope() -> None:
    memory_store = _FakeMemoryStore()
    service = TrackMemoryService(
        track_reader=_FakeTrackReader(),
        memory_store=memory_store,
    )

    items = service.list_inbox(user_id="u1")

    assert items == [{"id": 1, "scope_type": "track", "scope_id": "7", "status": "pending"}]
    assert memory_store.calls[-1] == {
        "method": "list_memories",
        "user_id": "u1",
        "limit": 100,
        "scope_type": "track",
        "scope_id": "7",
        "status": "pending",
        "include_deleted": False,
        "include_pending": True,
    }


def test_clear_track_memory_reports_post_delete_retrieval_count() -> None:
    memory_store = _FakeMemoryStore()
    service = TrackMemoryService(
        track_reader=_FakeTrackReader(),
        memory_store=memory_store,
    )

    result = service.clear_track_memory(user_id="u1", track_id=7)

    assert result.track_id == 7
    assert result.deleted_count == 1
    assert result.retrieved_after_delete_count == 1


def test_bulk_move_requires_resolvable_track_scope() -> None:
    service = TrackMemoryService(
        track_reader=_FakeTrackReader(),
        memory_store=_FakeMemoryStore(),
    )

    with pytest.raises(TrackMemoryValidationError):
        service.bulk_move(
            user_id="missing",
            item_ids=[1, 2],
            scope_type="track",
            scope_id=None,
        )


def test_require_track_scope_rejects_missing_track() -> None:
    service = TrackMemoryService(
        track_reader=_FakeTrackReader(),
        memory_store=_FakeMemoryStore(),
    )

    with pytest.raises(TrackMemoryScopeError):
        service.require_track_scope(user_id="u1", track_id=99)
