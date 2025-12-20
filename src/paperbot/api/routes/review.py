"""
Paper Review API Route
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


class ReviewRequest(BaseModel):
    title: str
    abstract: str


async def review_paper_stream(request: ReviewRequest):
    """Stream deep review progress"""
    try:
        yield StreamEvent(
            type="progress",
            data={"phase": "Initializing", "message": "Starting deep review..."},
        )

        # Import reviewer agent
        from ...agents.review import ReviewerAgent

        agent = ReviewerAgent({})

        yield StreamEvent(
            type="progress",
            data={"phase": "Screening", "message": "Initial screening..."},
        )

        yield StreamEvent(
            type="progress",
            data={"phase": "Critiquing", "message": "Deep critique analysis..."},
        )

        # Run review
        result = await agent.review(
            title=request.title,
            abstract=request.abstract,
        )

        yield StreamEvent(
            type="progress",
            data={"phase": "Decision", "message": "Generating recommendation..."},
        )

        yield StreamEvent(
            type="result",
            data={
                "title": request.title,
                "summary": result.summary if hasattr(result, 'summary') else "",
                "keyContributions": result.contributions if hasattr(result, 'contributions') else [],
                "methodology": "",
                "strengths": result.strengths if hasattr(result, 'strengths') else [],
                "weaknesses": result.weaknesses if hasattr(result, 'weaknesses') else [],
                "noveltyScore": result.novelty_score if hasattr(result, 'novelty_score') else None,
                "recommendation": result.decision if hasattr(result, 'decision') else None,
            },
        )

    except Exception as e:
        yield StreamEvent(type="error", message=str(e))


@router.post("/review")
async def review_paper(request: ReviewRequest):
    """
    Deep review a paper and stream progress.

    Returns Server-Sent Events with review updates.
    """
    return StreamingResponse(
        wrap_generator(review_paper_stream(request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
