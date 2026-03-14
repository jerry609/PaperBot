"""paper_search MCP tool wrapping PaperSearchService.

Provides paper search functionality over multiple academic data sources.
Uses the register(mcp) pattern to avoid circular imports.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the search service
_service = None


def _get_service():
    """Construct PaperSearchService on first call (lazy singleton)."""
    global _service
    if _service is None:
        from paperbot.infrastructure.adapters import build_adapter_registry
        from paperbot.application.services.paper_search_service import PaperSearchService

        adapters = build_adapter_registry()
        _service = PaperSearchService(adapters=adapters)
    return _service


async def _paper_search_impl(
    query: str,
    max_results: int = 10,
    sources: Optional[List[str]] = None,
    _run_id: str = "",
) -> List[Dict[str, Any]]:
    """Core implementation of paper_search, callable from both MCP registration and tests.

    Search for academic papers across multiple data sources.

    Args:
        query: Search query string describing the papers to find.
        max_results: Maximum number of papers to return (default 10).
        sources: Optional list of specific sources to search (e.g. ['arxiv', 'semantic_scholar']).
        _run_id: Optional run ID for event correlation.

    Returns:
        List of paper dictionaries with title, abstract, authors, and metadata.
    """
    service = _get_service()
    t0 = time.monotonic()

    try:
        result = await service.search(
            query,
            max_results=max_results,
            sources=sources,
            persist=False,
        )
        papers = [p.to_dict() for p in result.papers]

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        log_tool_call(
            tool_name="paper_search",
            arguments={
                "query": query,
                "max_results": max_results,
                "sources": sources,
            },
            result_summary=(
                f"returned {len(papers)} papers "
                f"(total_raw={result.total_raw}, "
                f"duplicates_removed={result.duplicates_removed})"
            ),
            duration_ms=elapsed_ms,
            run_id=_run_id or None,
        )

        return papers

    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        log_tool_call(
            tool_name="paper_search",
            arguments={
                "query": query,
                "max_results": max_results,
                "sources": sources,
            },
            result_summary="",
            duration_ms=elapsed_ms,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the paper_search tool on the given FastMCP instance."""

    @mcp.tool()
    async def paper_search(
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None,
        _run_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Search for academic papers across multiple data sources.

        Args:
            query: Search query string describing the papers to find.
            max_results: Maximum number of papers to return (default 10).
            sources: Optional list of specific sources to search.
            _run_id: Optional run ID for event correlation.

        Returns:
            List of paper dictionaries with title, abstract, authors, and metadata.
        """
        return await _paper_search_impl(
            query=query,
            max_results=max_results,
            sources=sources,
            _run_id=_run_id,
        )
