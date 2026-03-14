"""get_research_context MCP tool wrapping ContextEngine.

Retrieves research context for a query including relevant papers and memories.
Uses direct async await since ContextEngine.build_context_pack() is already async.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the context engine
_engine = None


def _get_engine():
    """Construct ContextEngine on first call (lazy singleton)."""
    global _engine
    if _engine is None:
        from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig

        _engine = ContextEngine(config=ContextEngineConfig(offline=True, paper_limit=0))
    return _engine


async def _get_research_context_impl(
    query: str,
    user_id: str = "default",
    track_id: Optional[int] = None,
    _run_id: str = "",
) -> Dict[str, Any]:
    """Core implementation of get_research_context, callable from both MCP registration and tests.

    Retrieve research context for a query, including relevant papers and memories.

    Args:
        query: The research query string.
        user_id: User identifier for personalized context.
        track_id: Optional track ID to scope the context retrieval.
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with keys: papers, memories, track, stage, routing_suggestion, and more.
    """
    start = time.monotonic()
    args = {"query": query, "user_id": user_id, "track_id": track_id}

    try:
        engine = _get_engine()
        result = await engine.build_context_pack(
            user_id=user_id,
            query=query,
            track_id=track_id,
        )

        log_tool_call(
            tool_name="get_research_context",
            arguments=args,
            result_summary={
                "paper_count": len(result.get("papers") or []),
                "memory_count": len(result.get("memories") or []),
                "stage": result.get("stage"),
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return result

    except Exception as exc:
        log_tool_call(
            tool_name="get_research_context",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the get_research_context tool on the given FastMCP instance."""

    @mcp.tool()
    async def get_research_context(
        query: str,
        user_id: str = "default",
        track_id: Optional[int] = None,
        _run_id: str = "",
    ) -> dict:
        """Retrieve research context for a query, including relevant papers and memories.

        Returns a context pack dict with papers, memories, track info, research stage,
        and routing suggestions based on the query and user profile.
        """
        return await _get_research_context_impl(query, user_id, track_id, _run_id)
