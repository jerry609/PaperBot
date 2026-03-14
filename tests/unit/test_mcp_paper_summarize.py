"""Unit tests for the paper_summarize MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.application.workflows.analysis.paper_summarizer import PaperSummarizer


class _FakeLLMService:
    """LLM service returning a canned summary."""

    def __init__(self, summary_text="This paper presents a novel approach."):
        self._summary = summary_text

    def summarize_paper(self, title: str, abstract: str) -> str:
        return self._summary

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "summary-model", "cost_tier": 1}


class _FakeEmptyLLMService:
    """Simulates missing API key -- returns empty output."""

    def summarize_paper(self, title: str, abstract: str) -> str:
        return ""

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "", "model_name": "", "cost_tier": 0}


class TestPaperSummarizeTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_summary_dict(self):
        """paper_summarize with fake LLM returns summary string in a result dict."""
        import paperbot.mcp.tools.paper_summarize as ps_mod

        summarizer = PaperSummarizer(llm_service=_FakeLLMService("Great paper summary."))
        ps_mod._summarizer = summarizer

        try:
            result = await ps_mod._paper_summarize_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            ps_mod._summarizer = None

        assert isinstance(result, dict)
        assert result["summary"] == "Great paper summary."
        assert "degraded" not in result

    @pytest.mark.asyncio
    async def test_degraded_mode_when_llm_returns_empty(self):
        """paper_summarize with empty LLM returns degraded=true and error message."""
        import paperbot.mcp.tools.paper_summarize as ps_mod

        summarizer = PaperSummarizer(llm_service=_FakeEmptyLLMService())
        ps_mod._summarizer = summarizer

        try:
            result = await ps_mod._paper_summarize_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            ps_mod._summarizer = None

        assert isinstance(result, dict)
        assert result["summary"] == ""
        assert result["degraded"] is True
        assert "error" in result
        assert "unavailable" in result["error"].lower() or "API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """paper_summarize logs call via log_tool_call with tool_name='paper_summarize'."""
        import paperbot.mcp.tools.paper_summarize as ps_mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        summarizer = PaperSummarizer(llm_service=_FakeLLMService("Summary."))
        ps_mod._summarizer = summarizer

        try:
            await ps_mod._paper_summarize_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            ps_mod._summarizer = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "paper_summarize"
        assert event["workflow"] == "mcp"
