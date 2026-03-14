"""PaperBot MCP server.

Provides the FastMCP instance with all tools registered.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("paperbot")

    # Register tools
    from paperbot.mcp.tools import paper_search
    from paperbot.mcp.tools import paper_judge
    from paperbot.mcp.tools import paper_summarize
    from paperbot.mcp.tools import relevance
    from paperbot.mcp.tools import analyze_trends
    from paperbot.mcp.tools import check_scholar
    from paperbot.mcp.tools import get_research_context
    from paperbot.mcp.tools import save_to_memory
    from paperbot.mcp.tools import export_to_obsidian

    paper_search.register(mcp)
    paper_judge.register(mcp)
    paper_summarize.register(mcp)
    relevance.register(mcp)
    analyze_trends.register(mcp)
    check_scholar.register(mcp)
    get_research_context.register(mcp)
    save_to_memory.register(mcp)
    export_to_obsidian.register(mcp)

except ImportError:
    # FastMCP not available -- create a minimal stub so tool modules
    # can still be imported and tested without the mcp package.
    logger.debug("mcp package not installed; MCP server unavailable")
    mcp = None  # type: ignore[assignment]
