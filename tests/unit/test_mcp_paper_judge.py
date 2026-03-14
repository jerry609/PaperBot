"""Unit tests for the paper_judge MCP tool."""

import json

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.application.workflows.analysis.paper_judge import PaperJudge


class _FakeLLMService:
    """LLM service returning a valid JSON payload."""

    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        return json.dumps(self.payload)

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "judge-model", "cost_tier": 2}


class _FakeEmptyLLMService:
    """Simulates missing API key -- returns empty output."""

    def complete(self, **kwargs):
        return ""

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "", "model_name": "", "cost_tier": 0}


class _FakeConfiguredButEmptyLLMService:
    """Simulates a configured provider that returns empty output."""

    def complete(self, **kwargs):
        return ""

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "judge-model", "cost_tier": 2}


_VALID_PAYLOAD = {
    "relevance": {"score": 5, "rationale": "direct"},
    "novelty": {"score": 4, "rationale": "new"},
    "rigor": {"score": 4, "rationale": "solid"},
    "impact": {"score": 3, "rationale": "good"},
    "clarity": {"score": 5, "rationale": "clear"},
    "overall": 4.2,
    "one_line_summary": "strong paper",
    "recommendation": "must_read",
}


class TestPaperJudgeTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_judgment_dict_with_all_dimensions(self):
        """paper_judge with fake LLM returns judgment dict with all dimension scores."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        judge = PaperJudge(llm_service=_FakeLLMService(_VALID_PAYLOAD))
        pj_mod._judge = judge

        try:
            result = await pj_mod._paper_judge_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            pj_mod._judge = None

        assert isinstance(result, dict)
        assert result["relevance"]["score"] == 5
        assert result["novelty"]["score"] == 4
        assert result["rigor"]["score"] == 4
        assert result["impact"]["score"] == 3
        assert result["clarity"]["score"] == 5
        assert result["overall"] == 4.2
        assert result["recommendation"] == "must_read"
        assert result["judge_model"] == "judge-model"
        assert "degraded" not in result

    @pytest.mark.asyncio
    async def test_degraded_mode_when_llm_unavailable(self):
        """paper_judge with empty LLM (no API key) returns degraded=true."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        judge = PaperJudge(llm_service=_FakeEmptyLLMService())
        pj_mod._judge = judge

        try:
            result = await pj_mod._paper_judge_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            pj_mod._judge = None

        assert isinstance(result, dict)
        assert result["degraded"] is True
        assert "error" in result
        assert "API_KEY" in result["error"] or "unavailable" in result["error"].lower()
        assert result["judge_model"] == ""

    @pytest.mark.asyncio
    async def test_maps_abstract_to_snippet(self):
        """paper_judge maps 'abstract' parameter to 'snippet' key in paper dict."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        calls = []
        original_judge_single = PaperJudge.judge_single

        class _SpyJudge(PaperJudge):
            def judge_single(self, *, paper, query):
                calls.append(paper)
                return original_judge_single(self, paper=paper, query=query)

        spy = _SpyJudge(llm_service=_FakeLLMService(_VALID_PAYLOAD))
        pj_mod._judge = spy

        try:
            await pj_mod._paper_judge_impl(
                title="My Title",
                abstract="My abstract text",
            )
        finally:
            pj_mod._judge = None

        assert len(calls) == 1
        assert "snippet" in calls[0]
        assert calls[0]["snippet"] == "My abstract text"
        assert calls[0]["title"] == "My Title"

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """paper_judge logs call via log_tool_call with tool_name='paper_judge'."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        judge = PaperJudge(llm_service=_FakeLLMService(_VALID_PAYLOAD))
        pj_mod._judge = judge

        try:
            await pj_mod._paper_judge_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            pj_mod._judge = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "paper_judge"
        assert event["workflow"] == "mcp"

    @pytest.mark.asyncio
    async def test_degraded_mode_when_provider_is_configured_but_returns_empty_output(self):
        """paper_judge marks empty provider responses as degraded even when metadata exists."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        judge = PaperJudge(llm_service=_FakeConfiguredButEmptyLLMService())
        pj_mod._judge = judge

        try:
            result = await pj_mod._paper_judge_impl(
                title="Test Paper",
                abstract="Test abstract",
            )
        finally:
            pj_mod._judge = None

        assert result["degraded"] is True
        assert result["judge_model"] == ""
