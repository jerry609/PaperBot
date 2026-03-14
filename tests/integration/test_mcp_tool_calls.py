"""Integration tests verifying MCP tool listing and invocation via MCP protocol.

Tests verify:
1. All 4 tools are discoverable (register functions + _impl functions)
2. Each tool has correct parameter signatures (types, required vs optional)
3. Tools are callable through the implementation layer
4. Tool calls are logged to EventLogPort with correct workflow/stage

Note: The mcp package (FastMCP) requires Python 3.10+ and is not available
on Python 3.9.x. These tests exercise the tool implementations directly
via _impl functions, which is the same code path that FastMCP's @mcp.tool()
decorators call. When mcp is unavailable, server.py sets mcp=None and
tool registration is skipped, but the implementations remain fully functional.
"""

import inspect
import json
from typing import Any, Dict, List

import pytest

from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.application.workflows.analysis.paper_judge import PaperJudge
from paperbot.application.workflows.analysis.paper_summarizer import PaperSummarizer
from paperbot.application.workflows.analysis.relevance_assessor import RelevanceAssessor
from paperbot.core.di import Container
from paperbot.domain.paper import PaperCandidate
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSearchAdapter:
    """Implements SearchPort with canned results."""

    source_name = "fake"

    async def search(self, query, *, max_results=10, year_from=None, year_to=None):
        if query == "empty":
            return []
        return [
            PaperCandidate(
                title="Integration Test Paper",
                abstract="Abstract for integration test",
                authors=["Author A"],
            )
        ]

    async def close(self):
        pass


class _FakeJudgeLLM:
    """Returns a valid JSON payload for PaperJudge."""

    PAYLOAD = {
        "relevance": {"score": 4, "rationale": "relevant"},
        "novelty": {"score": 3, "rationale": "moderate"},
        "rigor": {"score": 4, "rationale": "solid"},
        "impact": {"score": 3, "rationale": "some"},
        "clarity": {"score": 5, "rationale": "clear"},
        "overall": 3.8,
        "one_line_summary": "integration test paper",
        "recommendation": "worth_reading",
    }

    def complete(self, **kwargs):
        return json.dumps(self.PAYLOAD)

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "test-model", "cost_tier": 1}


class _FakeSummarizerLLM:
    """Returns a canned summary for PaperSummarizer."""

    def summarize_paper(self, title: str, abstract: str) -> str:
        return "A concise summary of the paper."

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "sum-model", "cost_tier": 1}


class _FakeRelevanceLLM:
    """Returns a valid relevance assessment."""

    def assess_relevance(self, *, paper, query):
        return {"score": 90, "reason": "Highly relevant to the query."}

    def describe_task_provider(self, task_type="default"):
        return {"provider_name": "fake", "model_name": "rel-model", "cost_tier": 1}


# ---------------------------------------------------------------------------
# Tool listing (discovery) tests
# ---------------------------------------------------------------------------


EXPECTED_TOOLS = ["paper_search", "paper_judge", "paper_summarize", "relevance_assess"]


class TestMCPToolListing:
    """Verify all 4 tools are discoverable and have correct signatures."""

    def setup_method(self):
        Container._instance = None

    def test_all_four_tools_listed(self):
        """All 4 tool modules expose register() and _impl functions."""
        from paperbot.mcp.tools import paper_search, paper_judge, paper_summarize, relevance

        modules = {
            "paper_search": paper_search,
            "paper_judge": paper_judge,
            "paper_summarize": paper_summarize,
            "relevance_assess": relevance,
        }

        # Verify exactly 4 tools
        assert len(modules) == 4, f"Expected 4 tools, found {len(modules)}"

        for tool_name, mod in modules.items():
            # Each module has a register() function
            assert hasattr(mod, "register"), f"{tool_name} missing register()"
            assert callable(mod.register), f"{tool_name}.register is not callable"

            # Each module has a docstring (description)
            impl_name = f"_{tool_name}_impl"
            if tool_name == "relevance_assess":
                impl_name = "_relevance_assess_impl"
            impl_fn = getattr(mod, impl_name, None)
            assert impl_fn is not None, f"{tool_name} missing {impl_name}"
            assert impl_fn.__doc__, f"{tool_name} impl has no docstring (description)"

    def test_server_registers_all_four_tools(self):
        """server.py imports and calls register() for all 4 tool modules.

        Since mcp package is unavailable on Python 3.9, mcp=None. We verify
        by checking that the server module completed import without error and
        that all 4 tool register functions are referenced in the module source.
        """
        import importlib
        import paperbot.mcp.server as server_mod

        # server.py should have mcp attribute (None if package unavailable)
        assert hasattr(server_mod, "mcp")

        # Verify the 4 imports are present in the source
        source = inspect.getsource(server_mod)
        assert "paper_search.register" in source, "paper_search not registered in server.py"
        assert "paper_judge.register" in source, "paper_judge not registered in server.py"
        assert "paper_summarize.register" in source, "paper_summarize not registered in server.py"
        assert "relevance.register" in source, "relevance not registered in server.py"

    def test_each_tool_has_input_schema_via_signature(self):
        """Each tool impl has typed parameters serving as input schema."""
        from paperbot.mcp.tools.paper_search import _paper_search_impl
        from paperbot.mcp.tools.paper_judge import _paper_judge_impl
        from paperbot.mcp.tools.paper_summarize import _paper_summarize_impl
        from paperbot.mcp.tools.relevance import _relevance_assess_impl

        for name, fn in [
            ("paper_search", _paper_search_impl),
            ("paper_judge", _paper_judge_impl),
            ("paper_summarize", _paper_summarize_impl),
            ("relevance_assess", _relevance_assess_impl),
        ]:
            sig = inspect.signature(fn)
            params = sig.parameters

            # Each tool must have at least one required parameter
            required = [
                p
                for p in params.values()
                if p.default is inspect.Parameter.empty and p.name != "self"
            ]
            assert len(required) >= 1, f"{name} has no required parameters"

            # Each parameter must have a type annotation (may be a string
            # due to ``from __future__ import annotations``)
            for pname, param in params.items():
                assert param.annotation is not inspect.Parameter.empty, (
                    f"{name}.{pname} has no type annotation"
                )


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestMCPToolSchemas:
    """Verify each tool's parameter schema matches expectations.

    Note: Tool modules use ``from __future__ import annotations`` (PEP 563),
    so annotations are strings at runtime. We compare against the string
    representation (e.g. ``"str"``) rather than the type object itself.
    """

    def setup_method(self):
        Container._instance = None

    @staticmethod
    def _annotation_name(param: inspect.Parameter) -> str:
        """Return annotation as a plain string for comparison."""
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            return ""
        return ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))

    def test_paper_search_tool_has_correct_params(self):
        """paper_search has query (required, str), max_results (optional, int),
        sources (optional, list), _run_id (optional, str)."""
        from paperbot.mcp.tools.paper_search import _paper_search_impl

        sig = inspect.signature(_paper_search_impl)
        params = sig.parameters

        # query is required, str
        assert "query" in params
        assert params["query"].default is inspect.Parameter.empty
        assert self._annotation_name(params["query"]) == "str"

        # max_results is optional, int, default 10
        assert "max_results" in params
        assert params["max_results"].default == 10
        assert self._annotation_name(params["max_results"]) == "int"

        # sources is optional, list or None
        assert "sources" in params
        assert params["sources"].default is None

        # _run_id is present
        assert "_run_id" in params
        assert params["_run_id"].default == ""

    def test_paper_judge_tool_has_correct_params(self):
        """paper_judge has title (required, str), abstract (required, str),
        full_text (optional), rubric (optional)."""
        from paperbot.mcp.tools.paper_judge import _paper_judge_impl

        sig = inspect.signature(_paper_judge_impl)
        params = sig.parameters

        # title is required, str
        assert "title" in params
        assert params["title"].default is inspect.Parameter.empty
        assert self._annotation_name(params["title"]) == "str"

        # abstract is required, str
        assert "abstract" in params
        assert params["abstract"].default is inspect.Parameter.empty
        assert self._annotation_name(params["abstract"]) == "str"

        # full_text is optional
        assert "full_text" in params
        assert params["full_text"].default == ""

        # rubric is optional
        assert "rubric" in params
        assert params["rubric"].default == "default"

    def test_paper_summarize_tool_has_correct_params(self):
        """paper_summarize has title (required, str), abstract (required, str)."""
        from paperbot.mcp.tools.paper_summarize import _paper_summarize_impl

        sig = inspect.signature(_paper_summarize_impl)
        params = sig.parameters

        # title is required, str
        assert "title" in params
        assert params["title"].default is inspect.Parameter.empty
        assert self._annotation_name(params["title"]) == "str"

        # abstract is required, str
        assert "abstract" in params
        assert params["abstract"].default is inspect.Parameter.empty
        assert self._annotation_name(params["abstract"]) == "str"

        # _run_id is optional
        assert "_run_id" in params
        assert params["_run_id"].default == ""

    def test_relevance_assess_tool_has_correct_params(self):
        """relevance_assess has title (required), abstract (required),
        query (required), keywords (optional)."""
        from paperbot.mcp.tools.relevance import _relevance_assess_impl

        sig = inspect.signature(_relevance_assess_impl)
        params = sig.parameters

        # title is required, str
        assert "title" in params
        assert params["title"].default is inspect.Parameter.empty
        assert self._annotation_name(params["title"]) == "str"

        # abstract is required, str
        assert "abstract" in params
        assert params["abstract"].default is inspect.Parameter.empty

        # query is required, str
        assert "query" in params
        assert params["query"].default is inspect.Parameter.empty
        assert self._annotation_name(params["query"]) == "str"

        # keywords is optional
        assert "keywords" in params
        assert params["keywords"].default == ""


# ---------------------------------------------------------------------------
# Tool invocation tests (via _impl functions)
# ---------------------------------------------------------------------------


class TestMCPToolInvocation:
    """Verify tools are callable through the implementation layer.

    Note: FastMCP's in-process tool calling (e.g. mcp.call_tool) requires
    the mcp package which needs Python 3.10+. On Python 3.9, we test the
    _impl functions directly -- these are the exact same functions that
    the @mcp.tool() decorators wrap.
    """

    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_tool_call_paper_search_via_impl(self):
        """Calling paper_search through _impl returns paper results."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod

        service = PaperSearchService(adapters={"fake": _FakeSearchAdapter()})
        ps_mod._service = service

        try:
            result = await ps_mod._paper_search_impl(query="transformers", max_results=5)
        finally:
            ps_mod._service = None

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Integration Test Paper"
        assert "abstract" in result[0]
        assert "authors" in result[0]

    @pytest.mark.asyncio
    async def test_tool_call_paper_judge_via_impl(self):
        """Calling paper_judge through _impl returns judgment with scores."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        judge = PaperJudge(llm_service=_FakeJudgeLLM())
        pj_mod._judge = judge

        try:
            result = await pj_mod._paper_judge_impl(
                title="Integration Test",
                abstract="An abstract for integration testing.",
            )
        finally:
            pj_mod._judge = None

        assert isinstance(result, dict)
        assert "overall" in result
        assert "recommendation" in result
        assert result["recommendation"] == "worth_reading"

    @pytest.mark.asyncio
    async def test_tool_call_paper_summarize_via_impl(self):
        """Calling paper_summarize through _impl returns summary dict."""
        import paperbot.mcp.tools.paper_summarize as ps_mod

        summarizer = PaperSummarizer(llm_service=_FakeSummarizerLLM())
        ps_mod._summarizer = summarizer

        try:
            result = await ps_mod._paper_summarize_impl(
                title="Integration Test",
                abstract="An abstract for integration testing.",
            )
        finally:
            ps_mod._summarizer = None

        assert isinstance(result, dict)
        assert "summary" in result
        assert len(result["summary"]) > 0
        assert "degraded" not in result

    @pytest.mark.asyncio
    async def test_tool_call_relevance_assess_via_impl(self):
        """Calling relevance_assess through _impl returns score and reason."""
        import paperbot.mcp.tools.relevance as rel_mod

        assessor = RelevanceAssessor(llm_service=_FakeRelevanceLLM())
        rel_mod._assessor = assessor

        try:
            result = await rel_mod._relevance_assess_impl(
                title="Integration Test",
                abstract="An abstract for integration testing.",
                query="machine learning",
            )
        finally:
            rel_mod._assessor = None

        assert isinstance(result, dict)
        assert result["score"] == 90
        assert "reason" in result
        assert "degraded" not in result


# ---------------------------------------------------------------------------
# Event logging tests
# ---------------------------------------------------------------------------


class TestMCPToolEventLogging:
    """Verify tool calls are logged to EventLogPort during MCP protocol calls."""

    def setup_method(self):
        Container._instance = None

    def _register_event_log(self) -> InMemoryEventLog:
        """Register InMemoryEventLog in Container and return it."""
        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)
        return log

    @pytest.mark.asyncio
    async def test_tool_call_paper_search_logs_event(self):
        """paper_search logs event with workflow='mcp', stage='tool_call'."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod

        log = self._register_event_log()

        service = PaperSearchService(adapters={"fake": _FakeSearchAdapter()})
        ps_mod._service = service

        try:
            await ps_mod._paper_search_impl(query="test", max_results=5)
        finally:
            ps_mod._service = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "paper_search"

    @pytest.mark.asyncio
    async def test_tool_call_paper_judge_logs_event(self):
        """paper_judge logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.paper_judge as pj_mod

        log = self._register_event_log()

        judge = PaperJudge(llm_service=_FakeJudgeLLM())
        pj_mod._judge = judge

        try:
            await pj_mod._paper_judge_impl(title="Test", abstract="Abstract")
        finally:
            pj_mod._judge = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "paper_judge"

    @pytest.mark.asyncio
    async def test_tool_call_paper_summarize_logs_event(self):
        """paper_summarize logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.paper_summarize as ps_mod

        log = self._register_event_log()

        summarizer = PaperSummarizer(llm_service=_FakeSummarizerLLM())
        ps_mod._summarizer = summarizer

        try:
            await ps_mod._paper_summarize_impl(title="Test", abstract="Abstract")
        finally:
            ps_mod._summarizer = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "paper_summarize"

    @pytest.mark.asyncio
    async def test_tool_call_relevance_assess_logs_event(self):
        """relevance_assess logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.relevance as rel_mod

        log = self._register_event_log()

        assessor = RelevanceAssessor(llm_service=_FakeRelevanceLLM())
        rel_mod._assessor = assessor

        try:
            await rel_mod._relevance_assess_impl(
                title="Test", abstract="Abstract", query="ML"
            )
        finally:
            rel_mod._assessor = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "relevance_assess"

    @pytest.mark.asyncio
    async def test_all_tool_events_have_consistent_structure(self):
        """All 4 tool events share consistent structure: workflow, stage,
        agent_name, payload with tool name."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod
        import paperbot.mcp.tools.paper_judge as pj_mod
        import paperbot.mcp.tools.paper_summarize as psum_mod
        import paperbot.mcp.tools.relevance as rel_mod

        log = self._register_event_log()

        # Set up all fakes
        ps_mod._service = PaperSearchService(adapters={"fake": _FakeSearchAdapter()})
        pj_mod._judge = PaperJudge(llm_service=_FakeJudgeLLM())
        psum_mod._summarizer = PaperSummarizer(llm_service=_FakeSummarizerLLM())
        rel_mod._assessor = RelevanceAssessor(llm_service=_FakeRelevanceLLM())

        try:
            await ps_mod._paper_search_impl(query="test")
            await pj_mod._paper_judge_impl(title="T", abstract="A")
            await psum_mod._paper_summarize_impl(title="T", abstract="A")
            await rel_mod._relevance_assess_impl(title="T", abstract="A", query="Q")
        finally:
            ps_mod._service = None
            pj_mod._judge = None
            psum_mod._summarizer = None
            rel_mod._assessor = None

        assert len(log.events) == 4

        tool_names_logged = set()
        for event in log.events:
            assert event["workflow"] == "mcp"
            assert event["stage"] == "tool_call"
            assert event["agent_name"] == "paperbot-mcp"
            assert "tool" in event["payload"]
            assert "duration_ms" in event["metrics"]
            tool_names_logged.add(event["payload"]["tool"])

        assert tool_names_logged == {
            "paper_search",
            "paper_judge",
            "paper_summarize",
            "relevance_assess",
        }
