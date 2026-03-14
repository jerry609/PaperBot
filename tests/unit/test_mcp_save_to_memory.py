"""Unit tests for the save_to_memory MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


class _FakeMemoryStore:
    """Fake memory store that records calls and returns canned results."""

    def __init__(self, created=1, skipped=0):
        self.calls = []
        self._created = created
        self._skipped = skipped

    def add_memories(self, user_id, memories):
        self.calls.append({"user_id": user_id, "memories": memories})
        return (self._created, self._skipped, [{"id": 1}])


class TestSaveToMemoryTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_saves_content_and_returns_counts(self):
        """_save_to_memory_impl with fake MemoryStore returns saved=True with created/skipped counts."""
        import paperbot.mcp.tools.save_to_memory as mod

        store = _FakeMemoryStore(created=1, skipped=0)
        mod._store = store
        try:
            result = await mod._save_to_memory_impl(
                content="Finding X: attention is all you need",
                kind="note",
                user_id="test_user",
            )
        finally:
            mod._store = None

        assert result["saved"] is True
        assert result["created"] == 1
        assert result["skipped"] == 0
        assert len(store.calls) == 1
        call = store.calls[0]
        assert call["user_id"] == "test_user"
        assert len(call["memories"]) == 1
        assert call["memories"][0].content == "Finding X: attention is all you need"
        assert call["memories"][0].kind == "note"

    @pytest.mark.asyncio
    async def test_handles_invalid_kind_gracefully(self):
        """_save_to_memory_impl with kind='invalid_kind' defaults to 'note' without raising."""
        import paperbot.mcp.tools.save_to_memory as mod

        store = _FakeMemoryStore(created=1, skipped=0)
        mod._store = store
        try:
            result = await mod._save_to_memory_impl(
                content="Some content",
                kind="invalid_kind",
                user_id="test_user",
            )
        finally:
            mod._store = None

        # Should not raise -- invalid kind defaults to "note"
        assert result["saved"] is True
        assert result["created"] == 1
        # The candidate should have kind="note" after defaulting
        assert store.calls[0]["memories"][0].kind == "note"

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """_save_to_memory_impl logs call via log_tool_call with correct tool name and workflow."""
        import paperbot.mcp.tools.save_to_memory as mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        store = _FakeMemoryStore()
        mod._store = store
        try:
            await mod._save_to_memory_impl(
                content="Hypothesis: scaling laws hold",
                kind="hypothesis",
            )
        finally:
            mod._store = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "save_to_memory"
        assert event["workflow"] == "mcp"

    @pytest.mark.asyncio
    async def test_returns_saved_false_when_store_skips_write(self):
        """_save_to_memory_impl returns saved=False when no new memory row is created."""
        import paperbot.mcp.tools.save_to_memory as mod

        store = _FakeMemoryStore(created=0, skipped=1)
        mod._store = store
        try:
            result = await mod._save_to_memory_impl(
                content="Duplicate finding",
                kind="note",
                user_id="test_user",
            )
        finally:
            mod._store = None

        assert result["saved"] is False
        assert result["created"] == 0
        assert result["skipped"] == 1
