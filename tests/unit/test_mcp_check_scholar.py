"""Unit tests for the check_scholar MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


class _FakeS2Client:
    """Fake SemanticScholarClient returning a test author and papers."""

    async def search_authors(self, query, limit=10, fields=None):
        return [
            {
                "authorId": "123",
                "name": "Test Scholar",
                "hIndex": 42,
                "paperCount": 100,
                "citationCount": 5000,
            }
        ]

    async def get_author_papers(self, author_id, limit=10, fields=None):
        return [
            {
                "title": "Paper A",
                "year": 2024,
                "citationCount": 10,
                "venue": "NeurIPS",
            }
        ]


class _FakeEmptyS2Client:
    """Simulates scholar not found -- search_authors returns empty list."""

    async def search_authors(self, query, limit=10, fields=None):
        return []

    async def get_author_papers(self, author_id, limit=10, fields=None):
        return []


class TestCheckScholarTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_scholar_info_and_papers(self):
        """_check_scholar_impl with fake S2 client returns scholar info and recent papers."""
        import paperbot.mcp.tools.check_scholar as mod

        mod._client = _FakeS2Client()
        try:
            result = await mod._check_scholar_impl(scholar_name="Test Scholar")
        finally:
            mod._client = None

        assert isinstance(result, dict)
        scholar = result["scholar"]
        assert scholar["name"] == "Test Scholar"
        assert scholar["authorId"] == "123"
        assert scholar["hIndex"] == 42
        papers = result["recent_papers"]
        assert len(papers) == 1
        assert papers[0]["title"] == "Paper A"
        assert papers[0]["year"] == 2024
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_degraded_when_scholar_not_found(self):
        """_check_scholar_impl returns degraded=True when search_authors returns empty list."""
        import paperbot.mcp.tools.check_scholar as mod

        mod._client = _FakeEmptyS2Client()
        try:
            result = await mod._check_scholar_impl(scholar_name="Unknown Person")
        finally:
            mod._client = None

        assert isinstance(result, dict)
        assert result["degraded"] is True
        assert "Scholar not found" in result["error"]
        assert result["scholar"] is None
        assert result["recent_papers"] == []

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """_check_scholar_impl logs event with tool='check_scholar', workflow='mcp'."""
        import paperbot.mcp.tools.check_scholar as mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        mod._client = _FakeS2Client()
        try:
            await mod._check_scholar_impl(scholar_name="Test Scholar")
        finally:
            mod._client = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "check_scholar"
        assert event["workflow"] == "mcp"
