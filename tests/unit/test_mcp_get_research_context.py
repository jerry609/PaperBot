"""Unit tests for the get_research_context MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


_CANNED_CONTEXT = {
    "paper_recommendations": [{"title": "Test Paper", "abstract": "Test abstract"}],
    "relevant_memories": [{"id": 1, "content": "Remember this"}],
    "active_track": {"id": 7, "name": "Transformers"},
    "routing": {"stage": "explore", "suggestion": {"track_id": 7, "score": 0.9}},
}


class _FakeContextEngine:
    """Fake ContextEngine returning a canned context pack dict."""

    async def build_context_pack(self, user_id: str, query: str, track_id=None):
        return dict(_CANNED_CONTEXT)


class _SpyContextEngine:
    """Fake ContextEngine that records call args."""

    def __init__(self):
        self.calls = []

    async def build_context_pack(self, user_id: str, query: str, track_id=None):
        self.calls.append({"user_id": user_id, "query": query, "track_id": track_id})
        return dict(_CANNED_CONTEXT)


class _CaptureContextEngine:
    """Fake ContextEngine that records constructor kwargs."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _CaptureContextEngineConfig:
    """Fake ContextEngineConfig that records constructor kwargs."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


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
        assert result["paper_recommendations"] == result["papers"]
        assert result["memories"] == result["relevant_memories"]
        assert result["track"] == result["active_track"]
        assert result["routing_suggestion"] == {"track_id": 7, "score": 0.9}

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

    def test_get_engine_wires_search_and_grounding_dependencies(self, monkeypatch):
        """_get_engine builds a ContextEngine with search, paper store, evidence, and grounding."""
        import paperbot.mcp.tools.get_research_context as mod

        mod._engine = None
        mod._paper_store = None
        mod._paper_search_service = None
        mod._document_index_store = None
        mod._query_grounder = None
        _CaptureContextEngine.last_kwargs = None
        _CaptureContextEngineConfig.last_kwargs = None

        monkeypatch.setattr(mod, "_get_paper_store", lambda: "paper-store")
        monkeypatch.setattr(mod, "_get_paper_search_service", lambda: "paper-search")
        monkeypatch.setattr(mod, "_get_document_index_store", lambda: "document-index")
        monkeypatch.setattr(mod, "_get_workflow_query_grounder", lambda: "grounder")
        monkeypatch.setattr(
            "paperbot.context_engine.engine.ContextEngine",
            _CaptureContextEngine,
        )
        monkeypatch.setattr(
            "paperbot.context_engine.engine.ContextEngineConfig",
            _CaptureContextEngineConfig,
        )

        try:
            engine = mod._get_engine()
        finally:
            mod._engine = None
            mod._paper_store = None
            mod._paper_search_service = None
            mod._document_index_store = None
            mod._query_grounder = None

        assert isinstance(engine, _CaptureContextEngine)
        assert _CaptureContextEngineConfig.last_kwargs == {}
        assert _CaptureContextEngine.last_kwargs["paper_store"] == "paper-store"
        assert _CaptureContextEngine.last_kwargs["search_service"] == "paper-search"
        assert _CaptureContextEngine.last_kwargs["evidence_retriever"] == "document-index"
        assert _CaptureContextEngine.last_kwargs["query_grounder"] == "grounder"
