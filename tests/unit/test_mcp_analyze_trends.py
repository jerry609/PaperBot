"""Unit tests for the analyze_trends MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


class _FakeTrendAnalyzer:
    """TrendAnalyzer stub returning a canned analysis string."""

    def analyze(self, *, topic: str, items) -> str:
        return "Trend: LLMs growing"


class _FakeEmptyTrendAnalyzer:
    """Simulates LLM unavailable -- returns empty string."""

    def analyze(self, *, topic: str, items) -> str:
        return ""


class _FailIfCalledTrendAnalyzer:
    """Fails fast if analyze() is invoked unexpectedly."""

    def analyze(self, *, topic: str, items) -> str:
        raise AssertionError("analyze() should not be called")


class TestAnalyzeTrendsTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_trend_analysis_dict(self):
        """_analyze_trends_impl with a fake TrendAnalyzer returns analysis dict."""
        import paperbot.mcp.tools.analyze_trends as mod

        mod._analyzer = _FakeTrendAnalyzer()
        try:
            result = await mod._analyze_trends_impl(
                topic="llms",
                papers=[{"title": "Paper A"}, {"title": "Paper B"}],
            )
        finally:
            mod._analyzer = None

        assert isinstance(result, dict)
        assert result["trend_analysis"] == "Trend: LLMs growing"
        assert result["topic"] == "llms"
        assert result["paper_count"] == 2
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_degraded_mode_when_llm_unavailable(self):
        """_analyze_trends_impl returns degraded=True when TrendAnalyzer returns empty string."""
        import paperbot.mcp.tools.analyze_trends as mod

        mod._analyzer = _FakeEmptyTrendAnalyzer()
        try:
            result = await mod._analyze_trends_impl(
                topic="llms",
                papers=[{"title": "Paper A"}, {"title": "Paper B"}],
            )
        finally:
            mod._analyzer = None

        assert isinstance(result, dict)
        assert result["degraded"] is True
        assert "error" in result
        assert "API_KEY" not in result["error"]
        assert "unavailable" in result["error"].lower() or "empty" in result["error"].lower()
        assert result["trend_analysis"] == ""
        assert result["topic"] == "llms"
        assert result["paper_count"] == 2

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """_analyze_trends_impl logs event with tool='analyze_trends', workflow='mcp'."""
        import paperbot.mcp.tools.analyze_trends as mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        mod._analyzer = _FakeTrendAnalyzer()
        try:
            await mod._analyze_trends_impl(
                topic="transformers",
                papers=[{"title": "Attention Is All You Need"}],
            )
        finally:
            mod._analyzer = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "analyze_trends"
        assert event["workflow"] == "mcp"

    @pytest.mark.asyncio
    async def test_treats_missing_papers_as_empty_list(self):
        """_analyze_trends_impl skips analyzer work when papers is missing."""
        import paperbot.mcp.tools.analyze_trends as mod

        mod._analyzer = _FailIfCalledTrendAnalyzer()
        try:
            result = await mod._analyze_trends_impl(
                topic="llms",
                papers=None,
            )
        finally:
            mod._analyzer = None

        assert result["paper_count"] == 0
        assert result["trend_analysis"] == ""
        assert result.get("degraded") is not True
