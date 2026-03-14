"""Unit tests for the track_papers MCP resource (MCP-07).

Tests _track_papers_impl with a fake SqlAlchemyResearchStore injected via
the module-level _store singleton pattern.
"""

import json

import pytest


class _FakeResearchStore:
    """ResearchStore stub returning canned feed data."""

    def __init__(self, items=None):
        self._items = items if items is not None else [{"title": "P1", "arxiv_id": "2401.0001"}]

    def list_track_feed(self, user_id: str, track_id: int, limit: int = 50):
        return {"items": self._items, "total": len(self._items)}


class TestTrackPapersResource:
    @pytest.mark.asyncio
    async def test_returns_papers_for_valid_track(self):
        """_track_papers_impl('42') returns JSON with items list."""
        import paperbot.mcp.resources.track_papers as mod

        mod._store = _FakeResearchStore(items=[{"title": "P1"}])
        try:
            result = await mod._track_papers_impl("42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "items" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "P1"

    @pytest.mark.asyncio
    async def test_returns_empty_items_when_track_has_no_papers(self):
        """_track_papers_impl returns JSON with empty items list for empty track."""
        import paperbot.mcp.resources.track_papers as mod

        mod._store = _FakeResearchStore(items=[])
        try:
            result = await mod._track_papers_impl("42")
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
            result = await mod._track_papers_impl("xyz")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "error" in data
