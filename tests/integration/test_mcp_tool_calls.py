"""Integration tests verifying MCP tool listing and invocation via MCP protocol.

Tests verify:
1. All 9 tools are discoverable (register functions + _impl functions)
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
from typing import Any, Dict, List, Optional

import pytest

from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.application.workflows.analysis.paper_judge import PaperJudge
from paperbot.application.workflows.analysis.paper_summarizer import PaperSummarizer
from paperbot.application.workflows.analysis.relevance_assessor import RelevanceAssessor
from paperbot.core.di import Container
from paperbot.domain.paper import PaperCandidate
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


# ---------------------------------------------------------------------------
# Fakes for existing 4 tools
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
# Fakes for 5 new tools (Plans 01 and 02)
# ---------------------------------------------------------------------------


class _FakeTrendAnalyzer:
    """TrendAnalyzer stub returning a canned analysis string."""

    def analyze(self, *, topic: str, items) -> str:
        return "Trend analysis result"


class _FakeS2Client:
    """Fake SemanticScholarClient returning a test author and papers."""

    async def search_authors(self, query, limit=10, fields=None):
        return [
            {
                "authorId": "123",
                "name": "Scholar",
                "hIndex": 10,
                "paperCount": 50,
                "citationCount": 1000,
            }
        ]

    async def get_author_papers(self, author_id, limit=10, fields=None):
        return [
            {
                "title": "Paper",
                "year": 2024,
                "citationCount": 5,
                "venue": "ICML",
            }
        ]


class _FakeContextEngine:
    """Fake ContextEngine returning a minimal context pack."""

    async def build_context_pack(self, user_id, query, track_id=None):
        return {"papers": [], "memories": [], "track": None, "stage": "explore"}


class _FakeMemoryStore:
    """Fake SqlAlchemyMemoryStore returning (created, skipped, rows) tuple."""

    def add_memories(self, user_id, memories):
        return (1, 0, [])


class _FakeExporter:
    """Fake ObsidianFilesystemExporter returning a minimal rendered note."""

    def _render_paper_note(self, **kwargs):
        return "# Title\n\nBody text"

    def _yaml_frontmatter(self, data):
        return "---\ntitle: Title\n---\n"


# ---------------------------------------------------------------------------
# Tool listing (discovery) tests
# ---------------------------------------------------------------------------


EXPECTED_TOOLS = [
    "paper_search",
    "paper_judge",
    "paper_summarize",
    "relevance_assess",
    "analyze_trends",
    "check_scholar",
    "get_research_context",
    "save_to_memory",
    "export_to_obsidian",
]


class TestMCPToolListing:
    """Verify all 9 tools are discoverable and have correct signatures."""

    def setup_method(self):
        Container._instance = None

    def test_all_nine_tools_listed(self):
        """All 9 tool modules expose register() and _impl functions."""
        from paperbot.mcp.tools import paper_search, paper_judge, paper_summarize, relevance
        from paperbot.mcp.tools import analyze_trends, check_scholar, get_research_context
        from paperbot.mcp.tools import save_to_memory, export_to_obsidian

        modules = {
            "paper_search": paper_search,
            "paper_judge": paper_judge,
            "paper_summarize": paper_summarize,
            "relevance_assess": relevance,
            "analyze_trends": analyze_trends,
            "check_scholar": check_scholar,
            "get_research_context": get_research_context,
            "save_to_memory": save_to_memory,
            "export_to_obsidian": export_to_obsidian,
        }

        # Verify exactly 9 tools
        assert len(modules) == 9, f"Expected 9 tools, found {len(modules)}"

        for tool_name, mod in modules.items():
            # Each module has a register() function
            assert hasattr(mod, "register"), f"{tool_name} missing register()"
            assert callable(mod.register), f"{tool_name}.register is not callable"

            # Each module has an _impl function with a docstring (description)
            impl_name = f"_{tool_name}_impl"
            if tool_name == "relevance_assess":
                impl_name = "_relevance_assess_impl"
            impl_fn = getattr(mod, impl_name, None)
            assert impl_fn is not None, f"{tool_name} missing {impl_name}"
            assert impl_fn.__doc__, f"{tool_name} impl has no docstring (description)"

    def test_server_registers_all_nine_tools(self):
        """server.py imports and calls register() for all 9 tool modules.

        Since mcp package is unavailable on Python 3.9, mcp=None. We verify
        by checking that the server module completed import without error and
        that all 9 tool register functions are referenced in the module source.
        """
        import paperbot.mcp.server as server_mod

        # server.py should have mcp attribute (None if package unavailable)
        assert hasattr(server_mod, "mcp")

        # Verify all 9 imports are present in the source
        source = inspect.getsource(server_mod)
        assert "paper_search.register" in source, "paper_search not registered in server.py"
        assert "paper_judge.register" in source, "paper_judge not registered in server.py"
        assert "paper_summarize.register" in source, "paper_summarize not registered in server.py"
        assert "relevance.register" in source, "relevance not registered in server.py"
        assert "analyze_trends.register" in source, "analyze_trends not registered in server.py"
        assert "check_scholar.register" in source, "check_scholar not registered in server.py"
        assert "get_research_context.register" in source, (
            "get_research_context not registered in server.py"
        )
        assert "save_to_memory.register" in source, "save_to_memory not registered in server.py"
        assert "export_to_obsidian.register" in source, (
            "export_to_obsidian not registered in server.py"
        )

    def test_each_tool_has_input_schema_via_signature(self):
        """Each tool impl has typed parameters serving as input schema."""
        from paperbot.mcp.tools.paper_search import _paper_search_impl
        from paperbot.mcp.tools.paper_judge import _paper_judge_impl
        from paperbot.mcp.tools.paper_summarize import _paper_summarize_impl
        from paperbot.mcp.tools.relevance import _relevance_assess_impl
        from paperbot.mcp.tools.analyze_trends import _analyze_trends_impl
        from paperbot.mcp.tools.check_scholar import _check_scholar_impl
        from paperbot.mcp.tools.get_research_context import _get_research_context_impl
        from paperbot.mcp.tools.save_to_memory import _save_to_memory_impl
        from paperbot.mcp.tools.export_to_obsidian import _export_to_obsidian_impl

        for name, fn in [
            ("paper_search", _paper_search_impl),
            ("paper_judge", _paper_judge_impl),
            ("paper_summarize", _paper_summarize_impl),
            ("relevance_assess", _relevance_assess_impl),
            ("analyze_trends", _analyze_trends_impl),
            ("check_scholar", _check_scholar_impl),
            ("get_research_context", _get_research_context_impl),
            ("save_to_memory", _save_to_memory_impl),
            ("export_to_obsidian", _export_to_obsidian_impl),
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
        sources (optional, list), _run_id (optional)."""
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

    def test_analyze_trends_tool_has_correct_params(self):
        """analyze_trends has topic (required, str), papers (required, list),
        _run_id (optional)."""
        from paperbot.mcp.tools.analyze_trends import _analyze_trends_impl

        sig = inspect.signature(_analyze_trends_impl)
        params = sig.parameters

        # topic is required, str
        assert "topic" in params
        assert params["topic"].default is inspect.Parameter.empty
        assert self._annotation_name(params["topic"]) == "str"

        # papers is required, list
        assert "papers" in params
        assert params["papers"].default is inspect.Parameter.empty

        # _run_id is optional
        assert "_run_id" in params
        assert params["_run_id"].default == ""

    def test_check_scholar_tool_has_correct_params(self):
        """check_scholar has scholar_name (required, str), max_papers (optional, int, default 10),
        _run_id (optional)."""
        from paperbot.mcp.tools.check_scholar import _check_scholar_impl

        sig = inspect.signature(_check_scholar_impl)
        params = sig.parameters

        # scholar_name is required, str
        assert "scholar_name" in params
        assert params["scholar_name"].default is inspect.Parameter.empty
        assert self._annotation_name(params["scholar_name"]) == "str"

        # max_papers is optional, int, default 10
        assert "max_papers" in params
        assert params["max_papers"].default == 10
        assert self._annotation_name(params["max_papers"]) == "int"

        # _run_id is optional
        assert "_run_id" in params
        assert params["_run_id"].default == ""

    def test_get_research_context_tool_has_correct_params(self):
        """get_research_context has query (required, str), user_id (optional, default 'default'),
        track_id (optional), _run_id (optional)."""
        from paperbot.mcp.tools.get_research_context import _get_research_context_impl

        sig = inspect.signature(_get_research_context_impl)
        params = sig.parameters

        # query is required, str
        assert "query" in params
        assert params["query"].default is inspect.Parameter.empty
        assert self._annotation_name(params["query"]) == "str"

        # user_id is optional with default "default"
        assert "user_id" in params
        assert params["user_id"].default == "default"

        # track_id is optional
        assert "track_id" in params
        assert params["track_id"].default is None

        # _run_id is optional
        assert "_run_id" in params
        assert params["_run_id"].default == ""

    def test_save_to_memory_tool_has_correct_params(self):
        """save_to_memory has content (required, str), kind (optional, default 'note'),
        user_id (optional), scope_type (optional), confidence (optional), _run_id (optional)."""
        from paperbot.mcp.tools.save_to_memory import _save_to_memory_impl

        sig = inspect.signature(_save_to_memory_impl)
        params = sig.parameters

        # content is required, str
        assert "content" in params
        assert params["content"].default is inspect.Parameter.empty
        assert self._annotation_name(params["content"]) == "str"

        # kind is optional with default "note"
        assert "kind" in params
        assert params["kind"].default == "note"

        # user_id is optional
        assert "user_id" in params

        # scope_type is optional
        assert "scope_type" in params

        # confidence is optional
        assert "confidence" in params

        # _run_id is optional
        assert "_run_id" in params
        assert params["_run_id"].default == ""

    def test_export_to_obsidian_tool_has_correct_params(self):
        """export_to_obsidian has title (required, str), abstract (required, str),
        authors (optional), year (optional), _run_id (optional)."""
        from paperbot.mcp.tools.export_to_obsidian import _export_to_obsidian_impl

        sig = inspect.signature(_export_to_obsidian_impl)
        params = sig.parameters

        # title is required, str
        assert "title" in params
        assert params["title"].default is inspect.Parameter.empty
        assert self._annotation_name(params["title"]) == "str"

        # abstract is required, str
        assert "abstract" in params
        assert params["abstract"].default is inspect.Parameter.empty
        assert self._annotation_name(params["abstract"]) == "str"

        # authors is optional
        assert "authors" in params

        # year is optional
        assert "year" in params
        assert params["year"].default is None

        # _run_id is optional
        assert "_run_id" in params
        assert params["_run_id"].default == ""


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

    @pytest.mark.asyncio
    async def test_tool_call_analyze_trends_via_impl(self):
        """Calling analyze_trends through _impl returns trend_analysis result."""
        import paperbot.mcp.tools.analyze_trends as at_mod

        at_mod._analyzer = _FakeTrendAnalyzer()

        try:
            result = await at_mod._analyze_trends_impl(
                topic="llms",
                papers=[{"title": "Paper A"}, {"title": "Paper B"}],
            )
        finally:
            at_mod._analyzer = None

        assert isinstance(result, dict)
        assert "trend_analysis" in result
        assert result["trend_analysis"] == "Trend analysis result"
        assert result["topic"] == "llms"
        assert result["paper_count"] == 2

    @pytest.mark.asyncio
    async def test_tool_call_check_scholar_via_impl(self):
        """Calling check_scholar through _impl returns scholar and recent_papers."""
        import paperbot.mcp.tools.check_scholar as cs_mod

        cs_mod._client = _FakeS2Client()

        try:
            result = await cs_mod._check_scholar_impl(scholar_name="Scholar")
        finally:
            cs_mod._client = None

        assert isinstance(result, dict)
        assert "scholar" in result
        assert "recent_papers" in result
        assert result["scholar"]["name"] == "Scholar"
        assert len(result["recent_papers"]) == 1

    @pytest.mark.asyncio
    async def test_tool_call_get_research_context_via_impl(self):
        """Calling get_research_context through _impl returns context pack with papers key."""
        import paperbot.mcp.tools.get_research_context as grc_mod

        grc_mod._engine = _FakeContextEngine()

        try:
            result = await grc_mod._get_research_context_impl(
                query="transformer architectures"
            )
        finally:
            grc_mod._engine = None

        assert isinstance(result, dict)
        assert "papers" in result
        assert result["stage"] == "explore"

    @pytest.mark.asyncio
    async def test_tool_call_save_to_memory_via_impl(self):
        """Calling save_to_memory through _impl returns dict with saved key."""
        import paperbot.mcp.tools.save_to_memory as stm_mod

        stm_mod._store = _FakeMemoryStore()

        try:
            result = await stm_mod._save_to_memory_impl(
                content="Important finding about attention mechanisms."
            )
        finally:
            stm_mod._store = None

        assert isinstance(result, dict)
        assert "saved" in result
        assert result["saved"] is True
        assert "created" in result
        assert result["created"] == 1

    @pytest.mark.asyncio
    async def test_tool_call_export_to_obsidian_via_impl(self):
        """Calling export_to_obsidian through _impl returns dict with markdown key."""
        import paperbot.mcp.tools.export_to_obsidian as eto_mod

        eto_mod._exporter = _FakeExporter()

        try:
            result = await eto_mod._export_to_obsidian_impl(
                title="Attention Is All You Need",
                abstract="We propose the Transformer, a model architecture...",
                authors=["Vaswani", "Shazeer"],
                year=2017,
            )
        finally:
            eto_mod._exporter = None

        assert isinstance(result, dict)
        assert "markdown" in result
        assert len(result["markdown"]) > 0


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
    async def test_tool_call_analyze_trends_logs_event(self):
        """analyze_trends logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.analyze_trends as at_mod

        log = self._register_event_log()

        at_mod._analyzer = _FakeTrendAnalyzer()

        try:
            await at_mod._analyze_trends_impl(
                topic="transformers",
                papers=[{"title": "Attention Is All You Need"}],
            )
        finally:
            at_mod._analyzer = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "analyze_trends"

    @pytest.mark.asyncio
    async def test_tool_call_check_scholar_logs_event(self):
        """check_scholar logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.check_scholar as cs_mod

        log = self._register_event_log()

        cs_mod._client = _FakeS2Client()

        try:
            await cs_mod._check_scholar_impl(scholar_name="Scholar")
        finally:
            cs_mod._client = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "check_scholar"

    @pytest.mark.asyncio
    async def test_tool_call_get_research_context_logs_event(self):
        """get_research_context logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.get_research_context as grc_mod

        log = self._register_event_log()

        grc_mod._engine = _FakeContextEngine()

        try:
            await grc_mod._get_research_context_impl(query="llms")
        finally:
            grc_mod._engine = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "get_research_context"

    @pytest.mark.asyncio
    async def test_tool_call_save_to_memory_logs_event(self):
        """save_to_memory logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.save_to_memory as stm_mod

        log = self._register_event_log()

        stm_mod._store = _FakeMemoryStore()

        try:
            await stm_mod._save_to_memory_impl(content="Important finding.")
        finally:
            stm_mod._store = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "save_to_memory"

    @pytest.mark.asyncio
    async def test_tool_call_export_to_obsidian_logs_event(self):
        """export_to_obsidian logs event with workflow='mcp', stage='tool_call'."""
        import paperbot.mcp.tools.export_to_obsidian as eto_mod

        log = self._register_event_log()

        eto_mod._exporter = _FakeExporter()

        try:
            await eto_mod._export_to_obsidian_impl(
                title="Test Paper",
                abstract="A test abstract.",
            )
        finally:
            eto_mod._exporter = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["payload"]["tool"] == "export_to_obsidian"

    @pytest.mark.asyncio
    async def test_all_tool_events_have_consistent_structure(self):
        """All 9 tool events share consistent structure: workflow, stage,
        agent_name, payload with tool name."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod
        import paperbot.mcp.tools.paper_judge as pj_mod
        import paperbot.mcp.tools.paper_summarize as psum_mod
        import paperbot.mcp.tools.relevance as rel_mod
        import paperbot.mcp.tools.analyze_trends as at_mod
        import paperbot.mcp.tools.check_scholar as cs_mod
        import paperbot.mcp.tools.get_research_context as grc_mod
        import paperbot.mcp.tools.save_to_memory as stm_mod
        import paperbot.mcp.tools.export_to_obsidian as eto_mod

        log = self._register_event_log()

        # Set up all fakes
        ps_mod._service = PaperSearchService(adapters={"fake": _FakeSearchAdapter()})
        pj_mod._judge = PaperJudge(llm_service=_FakeJudgeLLM())
        psum_mod._summarizer = PaperSummarizer(llm_service=_FakeSummarizerLLM())
        rel_mod._assessor = RelevanceAssessor(llm_service=_FakeRelevanceLLM())
        at_mod._analyzer = _FakeTrendAnalyzer()
        cs_mod._client = _FakeS2Client()
        grc_mod._engine = _FakeContextEngine()
        stm_mod._store = _FakeMemoryStore()
        eto_mod._exporter = _FakeExporter()

        try:
            await ps_mod._paper_search_impl(query="test")
            await pj_mod._paper_judge_impl(title="T", abstract="A")
            await psum_mod._paper_summarize_impl(title="T", abstract="A")
            await rel_mod._relevance_assess_impl(title="T", abstract="A", query="Q")
            await at_mod._analyze_trends_impl(topic="llms", papers=[{"title": "P"}])
            await cs_mod._check_scholar_impl(scholar_name="Scholar")
            await grc_mod._get_research_context_impl(query="transformers")
            await stm_mod._save_to_memory_impl(content="Finding.")
            await eto_mod._export_to_obsidian_impl(title="T", abstract="A")
        finally:
            ps_mod._service = None
            pj_mod._judge = None
            psum_mod._summarizer = None
            rel_mod._assessor = None
            at_mod._analyzer = None
            cs_mod._client = None
            grc_mod._engine = None
            stm_mod._store = None
            eto_mod._exporter = None

        assert len(log.events) == 9

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
            "analyze_trends",
            "check_scholar",
            "get_research_context",
            "save_to_memory",
            "export_to_obsidian",
        }


# ---------------------------------------------------------------------------
# Resource listing (discovery) tests
# ---------------------------------------------------------------------------


EXPECTED_RESOURCES = [
    "track_metadata",
    "track_papers",
    "track_memory",
    "scholars",
]


class TestMCPResourceListing:
    """Verify all 4 resources are discoverable and registered in server.py.

    URI template resources (track/{id}, track/{id}/papers, track/{id}/memory)
    appear in list_resource_templates; static resources (scholars) appear in
    list_resources. Both are verified via source inspection since FastMCP
    cannot be invoked directly on Python 3.9.
    """

    def setup_method(self):
        Container._instance = None

    def test_all_four_resources_listed(self):
        """All 4 resource modules expose register() and _impl functions."""
        from paperbot.mcp.resources import track_metadata, track_papers, track_memory, scholars

        modules = {
            "track_metadata": (track_metadata, "_track_metadata_impl"),
            "track_papers": (track_papers, "_track_papers_impl"),
            "track_memory": (track_memory, "_track_memory_impl"),
            "scholars": (scholars, "_scholars_impl"),
        }

        # Verify exactly 4 resources
        assert len(modules) == 4, f"Expected 4 resources, found {len(modules)}"

        for resource_name, (mod, impl_name) in modules.items():
            # Each module has a register() function
            assert hasattr(mod, "register"), f"{resource_name} missing register()"
            assert callable(mod.register), f"{resource_name}.register is not callable"

            # Each module has an _impl function
            impl_fn = getattr(mod, impl_name, None)
            assert impl_fn is not None, f"{resource_name} missing {impl_name}"
            assert impl_fn.__doc__, f"{resource_name} impl has no docstring (description)"

    def test_server_registers_all_four_resources(self):
        """server.py imports and calls register() for all 4 resource modules.

        Since mcp package is unavailable on Python 3.9, mcp=None. We verify
        by checking that the server module source contains all 4 resource
        register() calls (same approach as test_server_registers_all_nine_tools).
        """
        import paperbot.mcp.server as server_mod

        # server.py should have mcp attribute (None if package unavailable)
        assert hasattr(server_mod, "mcp")

        # Verify all 4 resource registrations are present in the source
        source = inspect.getsource(server_mod)
        assert "track_metadata.register" in source, (
            "track_metadata not registered in server.py"
        )
        assert "track_papers.register" in source, (
            "track_papers not registered in server.py"
        )
        assert "track_memory.register" in source, (
            "track_memory not registered in server.py"
        )
        assert "scholars.register" in source, (
            "scholars not registered in server.py"
        )

    def test_each_resource_impl_has_correct_signature(self):
        """Each resource _impl function has the expected parameter signature.

        URI template resources require track_id: str.
        The static scholars resource has no required parameters.
        """
        from paperbot.mcp.resources.track_metadata import _track_metadata_impl
        from paperbot.mcp.resources.track_papers import _track_papers_impl
        from paperbot.mcp.resources.track_memory import _track_memory_impl
        from paperbot.mcp.resources.scholars import _scholars_impl

        # track_metadata_impl: requires track_id: str
        sig = inspect.signature(_track_metadata_impl)
        params = sig.parameters
        assert "track_id" in params, "_track_metadata_impl missing track_id param"
        assert params["track_id"].default is inspect.Parameter.empty, (
            "track_id should be required"
        )

        # track_papers_impl: requires track_id: str
        sig = inspect.signature(_track_papers_impl)
        params = sig.parameters
        assert "track_id" in params, "_track_papers_impl missing track_id param"
        assert params["track_id"].default is inspect.Parameter.empty, (
            "track_id should be required"
        )

        # track_memory_impl: requires track_id: str
        sig = inspect.signature(_track_memory_impl)
        params = sig.parameters
        assert "track_id" in params, "_track_memory_impl missing track_id param"
        assert params["track_id"].default is inspect.Parameter.empty, (
            "track_id should be required"
        )

        # _scholars_impl: static resource — no required parameters
        sig = inspect.signature(_scholars_impl)
        required_params = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.name != "self"
        ]
        assert len(required_params) == 0, (
            f"_scholars_impl should have no required params, found: "
            f"{[p.name for p in required_params]}"
        )
