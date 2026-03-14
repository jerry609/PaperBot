"""Unit tests for the track_papers MCP resource (MCP-07).

Tests _track_papers_impl with a fake SqlAlchemyResearchStore injected via
the module-level _store singleton pattern.
"""

import json

import pytest

TEST_USER_ID = "mcp-user"


class _FakeResearchStore:
    """ResearchStore stub returning canned feed data."""

    def __init__(self, items=None, track_exists=True, archived=False):
        self._items = items if items is not None else [{"title": "P1", "arxiv_id": "2401.0001"}]
        self._track_exists = track_exists
        self._archived = archived
        self.calls = []

    def get_track(self, *, user_id: str, track_id: int):
        self.calls.append({"fn": "get_track", "user_id": user_id, "track_id": track_id})
        if not self._track_exists:
            return None
        return {
            "id": track_id,
            "name": "Track",
            "archived_at": "2026-03-14T00:00:00+00:00" if self._archived else None,
        }

    def list_track_feed(self, *, user_id: str, track_id: int, limit: int = 50):
        self.calls.append(
            {"fn": "list_track_feed", "user_id": user_id, "track_id": track_id, "limit": limit}
        )
        return {"items": self._items, "total": len(self._items)}


class TestTrackPapersResource:
    @pytest.mark.asyncio
    async def test_returns_papers_for_valid_track(self):
        """_track_papers_impl('42') returns JSON with items list."""
        import paperbot.mcp.resources.track_papers as mod

        fake_store = _FakeResearchStore(items=[{"title": "P1"}])
        mod._store = fake_store
        try:
            result = await mod._track_papers_impl(user_id=TEST_USER_ID, track_id="42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "items" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "P1"
        assert fake_store.calls == [
            {"fn": "get_track", "user_id": TEST_USER_ID, "track_id": 42},
            {"fn": "list_track_feed", "user_id": TEST_USER_ID, "track_id": 42, "limit": 50},
        ]

    @pytest.mark.asyncio
    async def test_returns_empty_items_when_track_has_no_papers(self):
        """_track_papers_impl returns JSON with empty items list for empty track."""
        import paperbot.mcp.resources.track_papers as mod

        mod._store = _FakeResearchStore(items=[])
        try:
            result = await mod._track_papers_impl(user_id=TEST_USER_ID, track_id="42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "items" in data
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_track_id(self):
        """_track_papers_impl('xyz') returns JSON error for non-integer track_id."""
        import paperbot.mcp.resources.track_papers as mod

        mod._store = _FakeResearchStore()
        try:
            result = await mod._track_papers_impl(user_id=TEST_USER_ID, track_id="xyz")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_returns_error_when_track_not_found(self):
        """_track_papers_impl returns error JSON when the track does not exist."""
        import paperbot.mcp.resources.track_papers as mod

        fake_store = _FakeResearchStore(track_exists=False)
        mod._store = fake_store
        try:
            result = await mod._track_papers_impl(user_id=TEST_USER_ID, track_id="42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert data["error"] == "Track 42 not found."
        assert fake_store.calls == [{"fn": "get_track", "user_id": TEST_USER_ID, "track_id": 42}]

    @pytest.mark.asyncio
    async def test_returns_error_when_track_is_archived(self):
        """_track_papers_impl returns error JSON for archived tracks."""
        import paperbot.mcp.resources.track_papers as mod

        fake_store = _FakeResearchStore(archived=True)
        mod._store = fake_store
        try:
            result = await mod._track_papers_impl(user_id=TEST_USER_ID, track_id="42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert data["error"] == "Track 42 not found."
