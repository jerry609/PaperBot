"""
Scholar Tracking API Route
"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


async def track_scholar_stream(
    scholar_id: Optional[str] = None,
    scholar_name: Optional[str] = None,
    force: bool = False,
):
    """Stream scholar tracking progress"""
    try:
        # Import here to avoid circular imports
        from ...workflows.scholar_tracking import ScholarTrackingWorkflow
        from ...core.di.container import Container

        yield StreamEvent(
            type="progress",
            data={"phase": "Initializing", "message": "Starting scholar tracking..."},
        )

        container = Container()
        workflow = ScholarTrackingWorkflow(container.config)

        yield StreamEvent(
            type="progress",
            data={"phase": "Fetching", "message": "Retrieving scholar information..."},
        )

        # Get scholar info
        scholar_info = await workflow.get_scholar_info(
            scholar_id=scholar_id,
            scholar_name=scholar_name,
        )

        if not scholar_info:
            yield StreamEvent(
                type="error",
                message="Scholar not found",
            )
            return

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Analyzing",
                "message": f"Found {len(scholar_info.papers)} papers",
                "percentage": 30,
            },
        )

        # Analyze papers
        for i, paper in enumerate(scholar_info.papers[:5]):
            yield StreamEvent(
                type="progress",
                data={
                    "phase": "Analyzing",
                    "message": f"Analyzing: {paper.title[:50]}...",
                    "percentage": 30 + int((i / 5) * 50),
                },
            )

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Calculating",
                "message": "Computing influence score...",
                "percentage": 80,
            },
        )

        # Calculate influence
        influence_score = await workflow.calculate_influence(scholar_info)

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Complete",
                "message": "Tracking complete",
                "percentage": 100,
            },
        )

        # Return result
        yield StreamEvent(
            type="result",
            data={
                "scholarName": scholar_info.name,
                "papers": [
                    {
                        "title": p.title,
                        "year": p.year,
                        "citations": p.citation_count,
                        "venue": p.venue,
                    }
                    for p in scholar_info.papers[:10]
                ],
                "influenceScore": influence_score,
            },
        )

    except Exception as e:
        yield StreamEvent(type="error", message=str(e))


@router.get("/track")
async def track_scholar(
    scholar_id: Optional[str] = Query(None, description="Semantic Scholar ID"),
    scholar_name: Optional[str] = Query(None, description="Scholar name"),
    force: bool = Query(False, description="Force refresh"),
):
    """
    Track a scholar and stream progress.

    Returns Server-Sent Events with progress updates.
    """
    if not scholar_id and not scholar_name:
        return {"error": "Either scholar_id or scholar_name is required"}

    return StreamingResponse(
        wrap_generator(track_scholar_stream(scholar_id, scholar_name, force)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
