"""
Scholar Tracking API Route
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


async def track_scholar_stream(
    http_request: Request,
    scholar_id: Optional[str] = None,
    scholar_name: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    max_new_papers: int = 5,
    persist_report: bool = False,
    offline: bool = False,
):
    """Stream scholar tracking progress"""
    # Imports here to avoid circular imports and reduce cold-start cost.
    from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
    from paperbot.application.workflows.scholar_pipeline import ScholarPipeline
    from paperbot.agents.scholar_tracking.paper_tracker_agent import PaperTrackerAgent
    from paperbot.domain.paper import PaperMeta

    run_id = new_run_id()
    request_trace_id = new_trace_id()
    event_log = getattr(http_request.app.state, "event_log", None)

    tracker_agent: Optional[PaperTrackerAgent] = None
    try:
        yield StreamEvent(
            type="progress",
            data={
                "phase": "Initializing",
                "message": "Starting scholar tracking...",
                "run_id": run_id,
                "trace_id": request_trace_id,
            },
        )

        tracker_agent = PaperTrackerAgent({"offline": offline})
        profile = tracker_agent.profile_agent

        # Resolve scholar by id/name from subscription list.
        scholar = None
        if scholar_id:
            scholar = profile.get_scholar_by_id(scholar_id)
        elif scholar_name:
            name_key = scholar_name.strip().lower()
            for s in profile.list_tracked_scholars():
                if (s.name or "").strip().lower() == name_key:
                    scholar = s
                    break

        if not scholar:
            yield StreamEvent(
                type="error",
                message="Scholar not found in subscriptions (provide scholar_id or scholar_name).",
            )
            return

        if force and scholar.semantic_scholar_id:
            profile.clear_scholar_cache(scholar.semantic_scholar_id)

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Tracking",
                "message": f"Checking new papers for {scholar.name}...",
                "run_id": run_id,
                "trace_id": request_trace_id,
            },
        )

        tracking_result = await tracker_agent.track_scholar(scholar, dry_run=dry_run)

        status = tracking_result.get("status", "unknown")
        new_papers: List[Dict[str, Any]] = tracking_result.get("new_papers", []) or []
        total_new = int(tracking_result.get("new_papers_count", len(new_papers)) or 0)

        yield StreamEvent(
            type="progress",
            data={
                "phase": "Tracking",
                "message": f"Tracking result: status={status}, new_papers={total_new}",
                "run_id": run_id,
                "trace_id": request_trace_id,
                "tracking": {
                    "status": status,
                    "new_papers_count": total_new,
                    "source": tracking_result.get("source"),
                },
            },
        )

        if status != "success":
            yield StreamEvent(
                type="result",
                data={
                    "run_id": run_id,
                    "trace_id": request_trace_id,
                    "scholar": scholar.to_dict(),
                    "tracking": tracking_result,
                    "analysis": [],
                },
            )
            return

        if not new_papers or total_new == 0:
            yield StreamEvent(
                type="result",
                data={
                    "run_id": run_id,
                    "trace_id": request_trace_id,
                    "scholar": scholar.to_dict(),
                    "tracking": tracking_result,
                    "analysis": [],
                },
            )
            return

        pipeline = ScholarPipeline({"enable_fail_fast": True})

        analyzed: List[Dict[str, Any]] = []
        limit = max(0, min(int(max_new_papers), len(new_papers)))
        for i, paper_dict in enumerate(new_papers[:limit], 1):
            paper = PaperMeta.from_dict(paper_dict)
            paper_trace_id = new_trace_id()

            yield StreamEvent(
                type="progress",
                data={
                    "phase": "Analyzing",
                    "message": f"Analyzing paper {i}/{limit}: {paper.title[:80]}",
                    "run_id": run_id,
                    "trace_id": paper_trace_id,
                    "paper_id": paper.paper_id,
                },
            )

            report_path, influence, pipeline_data = await pipeline.analyze_paper(
                paper=paper,
                scholar_name=scholar.name,
                persist_report=persist_report,
                event_log=event_log,
                run_id=run_id,
                trace_id=paper_trace_id,
            )

            analyzed.append(
                {
                    "paper": paper.to_dict(),
                    "trace_id": paper_trace_id,
                    "report_path": str(report_path) if report_path else None,
                    "influence": influence.to_dict() if hasattr(influence, "to_dict") else {},
                    "pipeline": pipeline_data,
                }
            )

        yield StreamEvent(
            type="result",
            data={
                "run_id": run_id,
                "trace_id": request_trace_id,
                "scholar": scholar.to_dict(),
                "tracking": tracking_result,
                "analysis": analyzed,
            },
        )

    except Exception as e:
        yield StreamEvent(
            type="error",
            data={"run_id": run_id, "trace_id": request_trace_id},
            message=str(e),
        )
    finally:
        if tracker_agent is not None:
            try:
                await tracker_agent.ss_agent.close()
            except Exception:
                pass


@router.get("/track")
async def track_scholar(
    http_request: Request,
    scholar_id: Optional[str] = Query(None, description="Semantic Scholar ID"),
    scholar_name: Optional[str] = Query(None, description="Scholar name"),
    force: bool = Query(False, description="Force refresh"),
    dry_run: bool = Query(False, description="Do not update caches"),
    max_new_papers: int = Query(5, description="Max new papers to analyze"),
    persist_report: bool = Query(False, description="Persist markdown report files"),
    offline: bool = Query(False, description="Offline mode (no remote API fallback)"),
):
    """
    Track a scholar and stream progress.

    Returns Server-Sent Events with progress updates.
    """
    if not scholar_id and not scholar_name:
        return {"error": "Either scholar_id or scholar_name is required"}

    return StreamingResponse(
        wrap_generator(
            track_scholar_stream(
                http_request,
                scholar_id,
                scholar_name,
                force,
                dry_run=dry_run,
                max_new_papers=max_new_papers,
                persist_report=persist_report,
                offline=offline,
            )
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
