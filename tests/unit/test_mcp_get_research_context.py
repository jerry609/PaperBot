"""Unit tests for the get_research_context MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


_CANNED_CONTEXT = {
    "papers": [{"title": "Test Paper", "abstract": "Test abstract"}],
    "memories": [],
    "track": None,
    "stage": "explore",
    "routing_suggestion": "default",
}


class _FakeContextEngine:
    """Fake ContextEngine returning a canned context pack dict."""

    async def build_context_pack(
        self, user_id: str, query: str, track_id=None
    ):
        return dict(_CANNED_CONTEXT)


class _SpyContextEngine:
    """Fake ContextEngine that records call args."""

    def __init__(self):
        self.calls = []

    async def build_context_pack(
        self, user_id: str, query: str, track_id=None
    ):
        self.calls.append({"user_id": user_id, "query": query, "track_id": track_id})
        return dict(_CANNED_CONTEXT)


class TestGetResearchContextTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_context_pack_dict(self):
        """_get_research_context_impl returns a context pack dict with expected keys."""
        import paperbot.mcp.tools.get_research_context as mod

        mod._engine = _FakeContextEngine()
        try:
            result = await mod._get_research_context_impl(query="transformers in NLP")
        finally:
            mod._engine = None

        assert isinstance(result, dict)
        assert "papers" in result
        assert "memories" in result
        assert "stage" in result
        assert result["stage"] == "explore"
        assert isinstance(result["papers"], list)
        assert result["papers"][0]["title"] == "Test Paper"

    @pytest.mark.asyncio
    async def test_accepts_user_id_and_track_id(self):
        """_get_research_context_impl passes user_id and track_id through to the engine."""
        import paperbot.mcp.tools.get_research_context as mod

        spy = _SpyContextEngine()
        mod._engine = spy
        try:
            await mod._get_research_context_impl(
                query="neural scaling laws",
                user_id="custom_user",
                track_id=42,
            )
        finally:
            mod._engine = None

        assert len(spy.calls) == 1
        call = spy.calls[0]
        assert call["user_id"] == "custom_user"
        assert call["track_id"] == 42
        assert call["query"] == "neural scaling laws"

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """_get_research_context_impl logs call via log_tool_call with correct tool name and workflow."""
        import paperbot.mcp.tools.get_research_context as mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        mod._engine = _FakeContextEngine()
        try:
            await mod._get_research_context_impl(query="attention mechanisms")
        finally:
            mod._engine = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "get_research_context"
        assert event["workflow"] == "mcp"
