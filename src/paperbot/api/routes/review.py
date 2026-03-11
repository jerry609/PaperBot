"""
Paper Review API Route
"""

from fastapi import APIRouter
from pydantic import BaseModel

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.core.abstractions import AgentRunContext, LegacyMethodRuntime

from ..streaming import StreamEvent, sse_response

router = APIRouter()


class ReviewRequest(BaseModel):
    title: str
    abstract: str


async def review_paper_stream(request: ReviewRequest, *, run_id: str, trace_id: str):
    """Stream deep review progress via AgentRuntime contract."""
    try:
        yield StreamEvent(
            type="progress",
            data={
                "phase": "Initializing",
                "message": "Starting deep review...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        from ...agents.review import ReviewerAgent

        agent = ReviewerAgent({})
        runtime = LegacyMethodRuntime(agent=agent, method_name="review")
        runtime_context = AgentRunContext(
            run_id=run_id,
            trace_id=trace_id,
            workflow="review",
            agent_name="ReviewerAgent",
        )

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Screening",
                "message": "Initial screening...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Critiquing",
                "message": "Deep critique analysis...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        runtime_result = await runtime.run(
            {
                "args": [],
                "kwargs": {
                    "title": request.title,
                    "abstract": request.abstract,
                },
            },
            context=runtime_context,
        )

        if not runtime_result.ok:
            message = runtime_result.error.message if runtime_result.error else "Review failed"
            raise RuntimeError(message)

        result = runtime_result.output

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Decision",
                "message": "Generating recommendation...",
                "run_id": run_id,
                "trace_id": trace_id,
            },
        )

        yield StreamEvent(
            type="result",
            data={
                "title": request.title,
                "summary": result.summary if hasattr(result, "summary") else "",
                "keyContributions": (
                    result.contributions if hasattr(result, "contributions") else []
                ),
                "methodology": "",
                "strengths": result.strengths if hasattr(result, "strengths") else [],
                "weaknesses": result.weaknesses if hasattr(result, "weaknesses") else [],
                "noveltyScore": result.novelty_score if hasattr(result, "novelty_score") else None,
                "recommendation": result.decision if hasattr(result, "decision") else None,
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


@router.post("/review")
async def review_paper(request: ReviewRequest):
    """
    Deep review a paper and stream progress.

    Returns Server-Sent Events with review updates.
    """
    run_id = new_run_id()
    trace_id = new_trace_id()
    return sse_response(
        review_paper_stream(request, run_id=run_id, trace_id=trace_id),
        workflow="review",
        run_id=run_id,
        trace_id=trace_id,
    )
