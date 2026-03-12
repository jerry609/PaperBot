"""
Paper Analysis API Route
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.core.abstractions import AgentRunContext, LegacyMethodRuntime

from ..streaming import StreamEvent, sse_response

router = APIRouter()


class AnalyzeRequest(BaseModel):
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None


async def analyze_paper_stream(request: AnalyzeRequest, *, run_id: str, trace_id: str):
    """Stream paper analysis progress via AgentRuntime contract."""
    try:
        yield StreamEvent(
            type="progress",
            data={
                "phase": "Fetching",
                "message": "Retrieving paper information...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        from ...agents.research import ResearchAgent

        agent = ResearchAgent({})
        runtime = LegacyMethodRuntime(agent=agent, method_name="analyze_paper")
        runtime_context = AgentRunContext(
            run_id=run_id,
            trace_id=trace_id,
            workflow="analyze",
            agent_name="ResearchAgent",
        )

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Analyzing",
                "message": "Extracting key contributions...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        runtime_result = await runtime.run(
            {
                "args": [],
                "kwargs": {
                    "title": request.title,
                    "abstract": request.abstract or "",
                },
            },
            context=runtime_context,
        )

        if not runtime_result.ok:
            message = runtime_result.error.message if runtime_result.error else "Analysis failed"
            raise RuntimeError(message)

        result = runtime_result.output

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Summarizing",
                "message": "Generating summary...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        yield StreamEvent(
            type="result",
            data={
                "title": request.title,
                "summary": result.summary if hasattr(result, "summary") else "Analysis complete",
                "keyContributions": (
                    result.key_contributions if hasattr(result, "key_contributions") else []
                ),
                "methodology": result.methodology if hasattr(result, "methodology") else "",
                "strengths": result.strengths if hasattr(result, "strengths") else [],
                "weaknesses": result.weaknesses if hasattr(result, "weaknesses") else [],
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

    except Exception as e:
        yield StreamEvent(
            type="error",
            message=str(e),
            data={"run_id": run_id, "trace_id": trace_id},
        )


@router.post("/analyze")
async def analyze_paper(request: AnalyzeRequest):
    """
    Analyze a paper and stream progress.

    Returns Server-Sent Events with analysis updates.
    """
    run_id = new_run_id()
    trace_id = new_trace_id()
    return sse_response(
        analyze_paper_stream(request, run_id=run_id, trace_id=trace_id),
        workflow="analyze",
        run_id=run_id,
        trace_id=trace_id,
    )
