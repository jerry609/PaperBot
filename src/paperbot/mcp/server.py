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

    paper_search.register(mcp)
    paper_judge.register(mcp)
    paper_summarize.register(mcp)
    relevance.register(mcp)

except ImportError:
    # FastMCP not available -- create a minimal stub so tool modules
    # can still be imported and tested without the mcp package.
    logger.debug("mcp package not installed; MCP server unavailable")
    mcp = None  # type: ignore[assignment]
