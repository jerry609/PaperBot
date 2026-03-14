"""analyze_trends MCP tool wrapping TrendAnalyzer.

Analyzes trends across a set of papers for a given topic using the
synchronous TrendAnalyzer service. Uses anyio.to_thread.run_sync() to
wrap the synchronous TrendAnalyzer.analyze() call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import anyio

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the trend analyzer service
_analyzer = None


def _get_analyzer():
    """Construct TrendAnalyzer on first call (lazy singleton)."""
    global _analyzer
    if _analyzer is None:
        from paperbot.application.workflows.analysis.trend_analyzer import TrendAnalyzer

        _analyzer = TrendAnalyzer()
    return _analyzer


async def _analyze_trends_impl(
    topic: str,
    papers: Optional[List[Dict[str, Any]]],
    _run_id: str = "",
) -> dict:
    """Core implementation of analyze_trends, callable from both MCP registration and tests.

    Analyze trends across a set of papers for a given topic.

    Args:
        topic: The research topic or theme to analyze trends for.
        papers: List of paper dicts (each may contain title, abstract, year, etc.).
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with trend_analysis string, topic, and paper_count.
        Includes degraded=True and error when LLM is unavailable.
    """
    start = time.monotonic()
    safe_papers = list(papers or [])
    args = {"topic": topic, "paper_count": len(safe_papers)}

    try:
        if not safe_papers:
            output: Dict[str, Any] = {
                "trend_analysis": "",
                "topic": topic,
                "paper_count": 0,
            }
            log_tool_call(
                tool_name="analyze_trends",
                arguments=args,
                result_summary={
                    "topic": topic,
                    "paper_count": 0,
                    "degraded": False,
                },
                duration_ms=(time.monotonic() - start) * 1000,
                run_id=_run_id or None,
            )
            return output

        analyzer = _get_analyzer()
        result = await anyio.to_thread.run_sync(
            lambda: analyzer.analyze(topic=topic, items=safe_papers)
        )

        output: Dict[str, Any] = {
            "trend_analysis": result,
            "topic": topic,
            "paper_count": len(safe_papers),
        }

        # Detect degraded LLM response (empty string when LLM unavailable)
        if not result or not result.strip():
            output["degraded"] = True
            output["error"] = "LLM response unavailable or empty. Check provider configuration."

        log_tool_call(
            tool_name="analyze_trends",
            arguments=args,
            result_summary={
                "topic": topic,
                "paper_count": len(safe_papers),
                "degraded": output.get("degraded", False),
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="analyze_trends",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the analyze_trends tool on the given FastMCP instance."""

    @mcp.tool()
    async def analyze_trends(
        topic: str,
        papers: Optional[List[Dict[str, Any]]],
        _run_id: str = "",
    ) -> dict:
        """Analyze trends across a set of papers for a given topic.

        Returns a natural language trend analysis string summarizing patterns,
        themes, and directions observed in the provided papers. Requires LLM API key.
        """
        return await _analyze_trends_impl(topic, papers, _run_id)
