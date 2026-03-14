"""Unit tests for the track_memory MCP resource (MCP-08).

Tests _track_memory_impl with a fake SqlAlchemyMemoryStore injected via
the module-level _store singleton pattern. Verifies scope_type="track"
filtering is applied correctly.
"""

import json
from typing import Optional

import pytest


class _FakeMemoryStore:
    """MemoryStore stub that captures args and returns canned memories."""

    def __init__(self, memories=None):
        self._memories = memories if memories is not None else []
        self.last_call_kwargs = {}

    def list_memories(
        self,
        *,
        user_id: str,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        limit: int = 100,
    ):
        self.last_call_kwargs = {
            "user_id": user_id,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "limit": limit,
        }
        return self._memories


class TestTrackMemoryResource:
    @pytest.mark.asyncio
    async def test_returns_memories_for_track(self):
        """_track_memory_impl('42') returns JSON list of memory dicts."""
        import paperbot.mcp.resources.track_memory as mod

        fake_store = _FakeMemoryStore(memories=[{"id": 1, "content": "note", "kind": "note"}])
        mod._store = fake_store
        try:
            result = await mod._track_memory_impl("42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "note"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_memories(self):
        """_track_memory_impl returns empty JSON list when no memories exist."""
        import paperbot.mcp.resources.track_memory as mod

        fake_store = _FakeMemoryStore(memories=[])
        mod._store = fake_store
        try:
            result = await mod._track_memory_impl("42")
        finally:
            mod._store = None

        data = json.loads(result)
        assert isinstance(data, list)
        assert data == []

    @pytest.mark.asyncio
    async def test_uses_track_scope_filtering(self):
        """_track_memory_impl calls list_memories with scope_type='track' and scope_id='42'."""
        import paperbot.mcp.resources.track_memory as mod

        fake_store = _FakeMemoryStore(memories=[])
        mod._store = fake_store
        try:
            await mod._track_memory_impl("42")
        finally:
            mod._store = None

        assert fake_store.last_call_kwargs["scope_type"] == "track"
        assert fake_store.last_call_kwargs["scope_id"] == "42"

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_track_id(self):
        """_track_memory_impl('bad') returns JSON error for non-integer track_id."""
        import paperbot.mcp.resources.track_memory as mod

        fake_store = _FakeMemoryStore()
        mod._store = fake_store
        try:
            result = await mod._track_memory_impl("bad")
        finally:
            mod._store = None

        data = json.loads(result)
        assert "error" in data
