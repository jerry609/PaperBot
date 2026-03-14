"""Unit tests for the relevance_assess MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.application.workflows.analysis.relevance_assessor import RelevanceAssessor


class _FakeLLMService:
    """LLM service returning a valid relevance assessment."""

    def assess_relevance(self, *, paper, query):
        return {"score": 85, "reason": "Highly relevant to the query topic."}

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "relevance-model", "cost_tier": 1}


class _FakeFallbackLLMService:
    """Simulates fallback scoring when LLM is unavailable."""

    def assess_relevance(self, *, paper, query):
        return {
            "score": 40,
            "reason": "Fallback score from token overlap (LLM output unavailable).",
        }

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "", "model_name": "", "cost_tier": 0}


class TestRelevanceAssessTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_score_and_reason_dict(self):
        """relevance_assess with fake LLM returns score and reason dict."""
        import paperbot.mcp.tools.relevance as rel_mod

        assessor = RelevanceAssessor(llm_service=_FakeLLMService())
        rel_mod._assessor = assessor

        try:
            result = await rel_mod._relevance_assess_impl(
                title="Test Paper",
                abstract="Test abstract",
                query="machine learning",
            )
        finally:
            rel_mod._assessor = None

        assert isinstance(result, dict)
        assert result["score"] == 85
        assert result["reason"] == "Highly relevant to the query topic."
        assert "degraded" not in result

    @pytest.mark.asyncio
    async def test_fallback_scoring_annotates_degraded(self):
        """relevance_assess with fallback scoring annotates result with degraded note."""
        import paperbot.mcp.tools.relevance as rel_mod

        assessor = RelevanceAssessor(llm_service=_FakeFallbackLLMService())
        rel_mod._assessor = assessor

        try:
            result = await rel_mod._relevance_assess_impl(
                title="Test Paper",
                abstract="Test abstract",
                query="machine learning",
            )
        finally:
            rel_mod._assessor = None

        assert isinstance(result, dict)
        assert result["degraded"] is True
        assert "note" in result
        assert "token" in result["note"].lower() or "fallback" in result["note"].lower()

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """relevance_assess logs call via log_tool_call with tool_name='relevance_assess'."""
        import paperbot.mcp.tools.relevance as rel_mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        assessor = RelevanceAssessor(llm_service=_FakeLLMService())
        rel_mod._assessor = assessor

        try:
            await rel_mod._relevance_assess_impl(
                title="Test Paper",
                abstract="Test abstract",
                query="machine learning",
            )
        finally:
            rel_mod._assessor = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "relevance_assess"
        assert event["workflow"] == "mcp"
