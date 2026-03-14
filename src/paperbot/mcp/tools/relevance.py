"""relevance_assess MCP tool wrapping RelevanceAssessor.

Assesses a paper's relevance to a given research query. Uses
anyio.to_thread.run_sync() to wrap the synchronous
RelevanceAssessor.assess() call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import anyio

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the assessor service
_assessor = None


def _get_assessor():
    """Construct RelevanceAssessor on first call (lazy singleton)."""
    global _assessor
    if _assessor is None:
        from paperbot.application.workflows.analysis.relevance_assessor import RelevanceAssessor

        _assessor = RelevanceAssessor()
    return _assessor


async def _relevance_assess_impl(
    title: str,
    abstract: str,
    query: str,
    keywords: str = "",
    _run_id: str = "",
) -> dict:
    """Core implementation of relevance_assess, callable from both MCP registration and tests.

    Assess a paper's relevance to a research query.

    Args:
        title: Paper title.
        abstract: Paper abstract text.
        query: Research query to assess relevance against.
        keywords: Optional comma-separated keywords.
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with 'score' (0-100) and 'reason'. Includes degraded=True and
        note when fallback token-overlap scoring is used.
    """
    start = time.monotonic()
    args = {"title": title, "query": query, "abstract_len": len(abstract)}

    # Build paper dict with snippet key
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    paper: Dict[str, Any] = {
        "title": title,
        "snippet": abstract,
        "keywords": keyword_list,
    }

    try:
        assessor = _get_assessor()
        result = await anyio.to_thread.run_sync(lambda: assessor.assess(paper=paper, query=query))

        output: Dict[str, Any] = dict(result)

        # Prefer structured fallback metadata; keep reason matching as a compatibility fallback.
        reason = str(result.get("reason", ""))
        if bool(result.get("fallback") or result.get("degraded")) or "Fallback" in reason:
            output["degraded"] = True
            output["note"] = (
                "Score computed via token-overlap fallback. " "LLM-based assessment unavailable."
            )

        log_tool_call(
            tool_name="relevance_assess",
            arguments=args,
            result_summary={
                "score": output.get("score"),
                "degraded": output.get("degraded", False),
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="relevance_assess",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the relevance_assess tool on the given FastMCP instance."""

    @mcp.tool()
    async def relevance_assess(
        title: str,
        abstract: str,
        query: str,
        keywords: str = "",
        _run_id: str = "",
    ) -> dict:
        """Assess a paper's relevance to a research query.

        Returns a relevance score (0-100) and reasoning. Uses LLM when available,
        falls back to token-overlap scoring otherwise.
        """
        return await _relevance_assess_impl(title, abstract, query, keywords, _run_id)
