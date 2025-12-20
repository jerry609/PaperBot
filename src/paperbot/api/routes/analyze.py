"""
Paper Analysis API Route
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


class AnalyzeRequest(BaseModel):
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None


async def analyze_paper_stream(request: AnalyzeRequest):
    """Stream paper analysis progress"""
    try:
        yield StreamEvent(
            type="progress",
            data={"phase": "Fetching", "message": "Retrieving paper information..."},
        )

        # Import agents
        from ...agents.research import ResearchAgent

        agent = ResearchAgent({})

        yield StreamEvent(
            type="progress",
            data={"phase": "Analyzing", "message": "Extracting key contributions..."},
        )

        # Analyze paper
        result = await agent.analyze_paper(
            title=request.title,
            abstract=request.abstract or "",
        )

        yield StreamEvent(
            type="progress",
            data={"phase": "Summarizing", "message": "Generating summary..."},
        )

        yield StreamEvent(
            type="result",
            data={
                "title": request.title,
                "summary": result.summary if hasattr(result, 'summary') else "Analysis complete",
                "keyContributions": result.key_contributions if hasattr(result, 'key_contributions') else [],
                "methodology": result.methodology if hasattr(result, 'methodology') else "",
                "strengths": result.strengths if hasattr(result, 'strengths') else [],
                "weaknesses": result.weaknesses if hasattr(result, 'weaknesses') else [],
            },
        )

    except Exception as e:
        yield StreamEvent(type="error", message=str(e))


@router.post("/analyze")
async def analyze_paper(request: AnalyzeRequest):
    """
    Analyze a paper and stream progress.

    Returns Server-Sent Events with analysis updates.
    """
    return StreamingResponse(
        wrap_generator(analyze_paper_stream(request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
