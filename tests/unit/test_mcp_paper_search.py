"""Unit tests for the paper_search MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.domain.paper import PaperCandidate


class _FakeSearchAdapter:
    """Implements SearchPort with canned results."""

    source_name = "fake"

    async def search(self, query, *, max_results=10, year_from=None, year_to=None):
        if query == "empty":
            return []
        return [
            PaperCandidate(
                title="Test Paper",
                abstract="Test abstract",
                authors=["Author A"],
            )
        ]

    async def close(self):
        pass


class TestPaperSearchTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_list_of_paper_dicts(self):
        """paper_search with a fake adapter returns list of paper dicts."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod

        service = PaperSearchService(
            adapters={"fake": _FakeSearchAdapter()},
        )
        ps_mod._service = service

        try:
            result = await ps_mod._paper_search_impl(
                query="transformers",
                max_results=5,
            )
        finally:
            ps_mod._service = None

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Paper"
        assert result[0]["abstract"] == "Test abstract"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_results(self):
        """paper_search with no adapters returns empty list."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod

        service = PaperSearchService(adapters={})
        ps_mod._service = service

        try:
            result = await ps_mod._paper_search_impl(query="empty", max_results=5)
        finally:
            ps_mod._service = None

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_calls_log_tool_call(self):
        """paper_search tool calls log_tool_call with tool_name='paper_search'."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        service = PaperSearchService(
            adapters={"fake": _FakeSearchAdapter()},
        )
        ps_mod._service = service

        try:
            await ps_mod._paper_search_impl(query="transformers", max_results=5)
        finally:
            ps_mod._service = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "paper_search"
        assert event["workflow"] == "mcp"

    @pytest.mark.asyncio
    async def test_rejects_out_of_range_max_results(self):
        """paper_search rejects max_results outside the supported range."""
        from paperbot.application.services.paper_search_service import PaperSearchService
        import paperbot.mcp.tools.paper_search as ps_mod

        ps_mod._service = PaperSearchService(adapters={"fake": _FakeSearchAdapter()})
        try:
            with pytest.raises(ValueError, match="max_results must be between 1 and 100"):
                await ps_mod._paper_search_impl(query="transformers", max_results=0)
        finally:
            ps_mod._service = None
