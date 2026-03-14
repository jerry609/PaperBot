"""check_scholar MCP tool wrapping SemanticScholarClient.

Checks a scholar's recent publications and activity by querying the
Semantic Scholar API. Uses the async SemanticScholarClient directly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the Semantic Scholar client
_client = None


def _get_client():
    """Construct SemanticScholarClient on first call (lazy singleton)."""
    global _client
    if _client is None:
        from paperbot.infrastructure.api_clients.semantic_scholar import SemanticScholarClient

        _client = SemanticScholarClient()
    return _client


async def _check_scholar_impl(
    scholar_name: str,
    max_papers: int = 10,
    _run_id: str = "",
) -> dict:
    """Core implementation of check_scholar, callable from both MCP registration and tests.

    Check a scholar's recent publications and activity.

    Args:
        scholar_name: Name or query string to search for the scholar.
        max_papers: Maximum number of recent papers to retrieve (default 10).
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with scholar info and recent_papers list.
        Includes degraded=True and error when scholar is not found.
    """
    start = time.monotonic()
    args = {"scholar_name": scholar_name, "max_papers": max_papers}

    try:
        client = _get_client()

        # Step 1: Search for the scholar
        authors = await client.search_authors(
            scholar_name,
            limit=3,
            fields=["name", "authorId", "hIndex", "paperCount", "citationCount"],
        )

        if not authors:
            output: Dict[str, Any] = {
                "degraded": True,
                "error": "Scholar not found",
                "scholar": None,
                "recent_papers": [],
                "candidates": [],
            }
            log_tool_call(
                tool_name="check_scholar",
                arguments=args,
                result_summary={"degraded": True, "error": "Scholar not found"},
                duration_ms=(time.monotonic() - start) * 1000,
                run_id=_run_id or None,
            )
            return output

        # Step 2: Pick top match (first result -- highest relevance from S2 API)
        top_author = authors[0]
        author_id = top_author.get("authorId", "")

        papers = await client.get_author_papers(
            author_id,
            limit=max_papers,
            fields=["title", "year", "citationCount", "venue", "abstract"],
        )

        output = {
            "scholar": top_author,
            "recent_papers": papers,
            "candidates": authors,
        }

        log_tool_call(
            tool_name="check_scholar",
            arguments=args,
            result_summary={
                "scholar": top_author.get("name"),
                "paper_count": len(papers),
                "degraded": False,
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="check_scholar",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the check_scholar tool on the given FastMCP instance."""

    @mcp.tool()
    async def check_scholar(
        scholar_name: str,
        max_papers: int = 10,
        _run_id: str = "",
    ) -> dict:
        """Check a scholar's recent publications and activity.

        Searches Semantic Scholar for the named scholar and returns their profile
        information (hIndex, citation count) along with their recent papers.
        Returns degraded=True if the scholar cannot be found.
        """
        return await _check_scholar_impl(scholar_name, max_papers, _run_id)
