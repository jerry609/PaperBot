"""Unit tests for the export_to_obsidian MCP tool."""

import pytest

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


class _FakeExporter:
    """Fake ObsidianFilesystemExporter that returns a minimal markdown body."""

    def _render_paper_note(self, **kwargs):
        title = kwargs.get("title", "Untitled")
        abstract = kwargs.get("abstract", "")
        return f"# {title}\n\n{abstract}\n"


class TestExportToObsidianTool:
    def setup_method(self):
        Container._instance = None

    @pytest.mark.asyncio
    async def test_returns_dict_with_markdown_key(self):
        """_export_to_obsidian_impl returns a dict with a 'markdown' key containing a string."""
        import paperbot.mcp.tools.export_to_obsidian as mod

        mod._exporter = _FakeExporter()
        try:
            result = await mod._export_to_obsidian_impl(
                title="Attention Is All You Need",
                abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            )
        finally:
            mod._exporter = None

        assert isinstance(result, dict)
        assert "markdown" in result
        assert isinstance(result["markdown"], str)
        assert len(result["markdown"]) > 0

    @pytest.mark.asyncio
    async def test_markdown_contains_frontmatter_and_title(self):
        """Returned markdown contains YAML frontmatter delimiters '---' and the paper title."""
        import paperbot.mcp.tools.export_to_obsidian as mod

        mod._exporter = _FakeExporter()
        try:
            result = await mod._export_to_obsidian_impl(
                title="Attention Is All You Need",
                abstract="We propose the Transformer architecture.",
                authors=["Vaswani, A.", "Shazeer, N."],
                year=2017,
                venue="NeurIPS",
                arxiv_id="1706.03762",
            )
        finally:
            mod._exporter = None

        markdown = result["markdown"]
        # YAML frontmatter delimiters must be present
        assert "---" in markdown
        # Paper title must appear somewhere in the output
        assert "Attention Is All You Need" in markdown

    @pytest.mark.asyncio
    async def test_logs_call_via_log_tool_call(self):
        """_export_to_obsidian_impl logs call via log_tool_call with correct tool name and workflow."""
        import paperbot.mcp.tools.export_to_obsidian as mod

        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)

        mod._exporter = _FakeExporter()
        try:
            await mod._export_to_obsidian_impl(
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                abstract="We introduce BERT.",
            )
        finally:
            mod._exporter = None

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "export_to_obsidian"
        assert event["workflow"] == "mcp"
