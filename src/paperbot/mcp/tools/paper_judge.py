"""paper_judge MCP tool wrapping PaperJudge.

Judges a paper's quality across multiple dimensions (relevance, novelty,
rigor, impact, clarity). Uses anyio.to_thread.run_sync() to wrap the
synchronous PaperJudge.judge_single() call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import anyio

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the judge service
_judge = None


def _get_judge():
    """Construct PaperJudge on first call (lazy singleton)."""
    global _judge
    if _judge is None:
        from paperbot.application.workflows.analysis.paper_judge import PaperJudge

        _judge = PaperJudge()
    return _judge


async def _paper_judge_impl(
    title: str,
    abstract: str,
    full_text: str = "",
    rubric: str = "default",
    _run_id: str = "",
) -> dict:
    """Core implementation of paper_judge, callable from both MCP registration and tests.

    Judge a paper's quality across multiple dimensions (relevance, novelty,
    rigor, impact, clarity).

    Args:
        title: Paper title.
        abstract: Paper abstract text.
        full_text: Optional full paper text for deeper analysis.
        rubric: Rubric name or query string for judging context.
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with dimension scores, overall score, recommendation, and judge_model.
        Includes degraded=True and error when LLM is unavailable.
    """
    start = time.monotonic()
    args = {"title": title, "abstract_len": len(abstract), "rubric": rubric}

    # CRITICAL: Map "abstract" to "snippet" -- PaperJudge expects "snippet" key
    paper: Dict[str, Any] = {"title": title, "snippet": abstract, "full_text": full_text}

    try:
        judge = _get_judge()
        result = await anyio.to_thread.run_sync(lambda: judge.judge_single(paper=paper, query=rubric))
        output = result.to_dict()

        # Treat parse/LLM fallbacks as degraded even if provider metadata is configured.
        if not result.judge_model:
            output["degraded"] = True
            output["error"] = (
                "LLM service unavailable. "
                "Configure OPENAI_API_KEY or ANTHROPIC_API_KEY."
            )

        log_tool_call(
            tool_name="paper_judge",
            arguments=args,
            result_summary={
                "overall": output.get("overall"),
                "recommendation": output.get("recommendation"),
                "degraded": output.get("degraded", False),
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="paper_judge",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the paper_judge tool on the given FastMCP instance."""

    @mcp.tool()
    async def paper_judge(
        title: str,
        abstract: str,
        full_text: str = "",
        rubric: str = "default",
        _run_id: str = "",
    ) -> dict:
        """Judge a paper's quality across multiple dimensions (relevance, novelty, rigor, impact, clarity).

        Returns dimension scores (1-5), overall score, one-line summary, and recommendation
        (must_read, worth_reading, skim, skip). Requires LLM API key.
        """
        return await _paper_judge_impl(title, abstract, full_text, rubric, _run_id)
