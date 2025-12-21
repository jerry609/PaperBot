"""
Run/Event Replay API Route

Minimal endpoints for querying persisted runs/events.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/runs")
async def list_runs(
    http_request: Request,
    limit: int = Query(50, ge=1, le=500, description="Max runs to return"),
):
    event_log = getattr(http_request.app.state, "event_log", None)
    if event_log is None or not hasattr(event_log, "list_runs"):
        return {"error": "run persistence not enabled"}
    return {"runs": event_log.list_runs(limit=limit)}


@router.get("/runs/{run_id}/events")
async def list_run_events(
    run_id: str,
    http_request: Request,
    trace_id: Optional[str] = Query(None, description="Optional trace_id filter"),
    limit: int = Query(1000, ge=1, le=20000, description="Max events to return"),
):
    event_log = getattr(http_request.app.state, "event_log", None)
    if event_log is None or not hasattr(event_log, "list_events"):
        return {"error": "run persistence not enabled"}
    return {"run_id": run_id, "events": event_log.list_events(run_id, trace_id=trace_id, limit=limit)}


