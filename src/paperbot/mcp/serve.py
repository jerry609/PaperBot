"""Transport dispatch for the PaperBot MCP server.

Provides two entry-points used by the CLI:

  run_stdio()  -- stdio transport (Claude Desktop / Claude Code)
  run_http()   -- Streamable HTTP transport (remote agents)

Example ``claude_desktop_config.json`` entry::

    {
      "mcpServers": {
        "paperbot": {
          "command": "paperbot",
          "args": ["mcp", "serve", "--stdio"]
        }
      }
    }
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def _get_mcp():
    """Return the mcp singleton (or None if mcp package not installed)."""
    from paperbot.mcp import server as _server_mod

    return _server_mod.mcp


def run_stdio() -> None:
    """Start the MCP server on stdio transport.

    All logging is redirected to stderr so that stdio remains clean for
    the MCP protocol framing (zero bytes on stdout from this process).

    Raises SystemExit(1) if the mcp package is not installed.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    mcp = _get_mcp()
    if mcp is None:
        print(
            "Error: mcp package is not installed. "
            "Install it with: pip install 'mcp[fastmcp]>=1.8.0,<2.0.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8001) -> None:
    """Start the MCP server on Streamable HTTP transport.

    Default port is 8001 to avoid conflicting with the FastAPI server (8000).

    Args:
        host: Bind address (default ``127.0.0.1``).
        port: Listen port (default ``8001``).

    Raises SystemExit(1) if the mcp package is not installed.
    """
    mcp = _get_mcp()
    if mcp is None:
        print(
            "Error: mcp package is not installed. "
            "Install it with: pip install 'mcp[fastmcp]>=1.8.0,<2.0.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp.run(transport="streamable-http", host=host, port=port)
