"""Unit tests for the track_metadata MCP resource (MCP-06).

Tests _track_metadata_impl with a fake SqlAlchemyResearchStore injected via
the module-level _store singleton pattern.
"""

import json

import pytest


class _FakeResearchStore:
    """ResearchStore stub returning canned track data."""

    def __init__(self):
        self.calls = []

    def get_track(self, *, user_id: str, track_id: int):
        self.calls.append({"user_id": user_id, "track_id": track_id})
        if user_id == "default" and track_id == 42:
            return {
                "id": 42,
                "name": "ML",
                "description": "Machine learning research",
                "keywords": ["deep learning", "neural networks"],
                "venues": ["NeurIPS", "ICML"],
                "methods": ["transformers", "diffusion"],
            }
        return None


class TestTrackMetadataResource:
    @pytest.mark.asyncio
    async def test_returns_track_metadata_for_valid_id(self):
        """_track_metadata_impl('42') returns JSON with track fields."""
        import paperbot.mcp.resources.track_metadata as mod

        fake_store = _FakeResearchStore()
        mod._store = fake_store
        try:
            result = await mod._track_metadata_impl("42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert fake_store.calls == [{"user_id": "default", "track_id": 42}]
        assert data["id"] == 42
        assert data["name"] == "ML"
        assert "description" in data
        assert "keywords" in data
        assert "venues" in data
        assert "methods" in data

    @pytest.mark.asyncio
    async def test_returns_error_when_track_not_found(self):
        """_track_metadata_impl('99') returns JSON error when track not found."""
        import paperbot.mcp.resources.track_metadata as mod

        mod._store = _FakeResearchStore()
        try:
            result = await mod._track_metadata_impl("99")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_returns_error_for_non_integer_track_id(self):
        """_track_metadata_impl('abc') returns JSON error for invalid track_id."""
        import paperbot.mcp.resources.track_metadata as mod

        fake_store = _FakeResearchStore()
        mod._store = fake_store
        try:
            result = await mod._track_metadata_impl("abc")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "error" in data
        assert fake_store.calls == []
