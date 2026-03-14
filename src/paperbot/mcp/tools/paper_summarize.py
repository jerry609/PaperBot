"""paper_summarize MCP tool wrapping PaperSummarizer.

Generates a concise summary of a paper using the LLM. Uses
anyio.to_thread.run_sync() to wrap the synchronous
PaperSummarizer.summarize_item() call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import anyio

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the summarizer service
_summarizer = None


def _get_summarizer():
    """Construct PaperSummarizer on first call (lazy singleton)."""
    global _summarizer
    if _summarizer is None:
        from paperbot.application.workflows.analysis.paper_summarizer import PaperSummarizer

        _summarizer = PaperSummarizer()
    return _summarizer


async def _paper_summarize_impl(
    title: str,
    abstract: str,
    _run_id: str = "",
) -> dict:
    """Core implementation of paper_summarize, callable from both MCP registration and tests.

    Summarize a paper given its title and abstract.

    Args:
        title: Paper title.
        abstract: Paper abstract text.
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with 'summary' key. Includes degraded=True and error when LLM
        returns empty output.
    """
    start = time.monotonic()
    args = {"title": title, "abstract_len": len(abstract)}

    # Build item dict with "snippet" key (PaperSummarizer reads snippet or abstract)
    item: Dict[str, Any] = {"title": title, "snippet": abstract}

    try:
        summarizer = _get_summarizer()
        summary = await anyio.to_thread.run_sync(
            lambda: summarizer.summarize_item(item)
        )

        output: Dict[str, Any] = {"summary": summary}

        # Detect degraded LLM response (empty summary when LLM unavailable)
        if not summary or not summary.strip():
            output["summary"] = ""
            output["degraded"] = True
            output["error"] = (
                "LLM service unavailable. "
                "Configure OPENAI_API_KEY or ANTHROPIC_API_KEY."
            )

        log_tool_call(
            tool_name="paper_summarize",
            arguments=args,
            result_summary={
                "summary_len": len(summary),
                "degraded": output.get("degraded", False),
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="paper_summarize",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the paper_summarize tool on the given FastMCP instance."""

    @mcp.tool()
    async def paper_summarize(
        title: str,
        abstract: str,
        _run_id: str = "",
    ) -> dict:
        """Summarize a paper given its title and abstract.

        Returns a concise summary of the paper's key contributions, methods,
        and findings. Requires LLM API key.
        """
        return await _paper_summarize_impl(title, abstract, _run_id)
