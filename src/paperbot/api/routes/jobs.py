"""
Jobs API Route (ARQ)

Enqueue background jobs and query their status.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from arq.connections import create_pool
from arq.jobs import Job

from paperbot.infrastructure.queue.arq_worker import _redis_settings

router = APIRouter()


class TrackScholarJobRequest(BaseModel):
    scholar_id: str = Field(..., description="Semantic Scholar ID from subscriptions")
    dry_run: bool = True
    offline: bool = False


class AnalyzePaperJobRequest(BaseModel):
    paper: Dict[str, Any]
    scholar_name: str = ""


@router.post("/jobs/track-scholar")
async def enqueue_track_scholar(req: TrackScholarJobRequest):
    try:
        redis = await create_pool(_redis_settings())
        job = await redis.enqueue_job(
            "track_scholar_job",
            req.scholar_id,
            dry_run=req.dry_run,
            offline=req.offline,
        )
        return {"job_id": job.job_id}
    except Exception as e:
        return {"error": str(e)}


@router.post("/jobs/analyze-paper")
async def enqueue_analyze_paper(req: AnalyzePaperJobRequest):
    try:
        redis = await create_pool(_redis_settings())
        job = await redis.enqueue_job(
            "analyze_paper_job",
            req.paper,
            scholar_name=req.scholar_name,
        )
        return {"job_id": job.job_id}
    except Exception as e:
        return {"error": str(e)}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    try:
        redis = await create_pool(_redis_settings())
        job = Job(job_id, redis)
        info = await job.info()
        # When finished, result will be in job.result()
        result: Optional[Any] = None
        if info is not None and info.success is True:
            try:
                result = await job.result()
            except Exception:
                result = None
        return {"job_id": job_id, "info": info.model_dump() if info else None, "result": result}
    except Exception as e:
        return {"error": str(e)}


