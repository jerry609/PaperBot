from __future__ import annotations

import os
from typing import Any, Dict, Optional

from arq import cron
from arq.connections import RedisSettings

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id, make_event
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog


def _redis_settings() -> RedisSettings:
    return RedisSettings(
        host=os.getenv("PAPERBOT_REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("PAPERBOT_REDIS_PORT", "6379")),
        database=int(os.getenv("PAPERBOT_REDIS_DB", "0")),
        password=os.getenv("PAPERBOT_REDIS_PASSWORD") or None,
    )


def _event_log() -> SqlAlchemyEventLog:
    # Uses PAPERBOT_DB_URL if present.
    return SqlAlchemyEventLog()


async def track_scholar_job(ctx, scholar_id: str, *, dry_run: bool = True, offline: bool = False) -> Dict[str, Any]:
    """
    ARQ job: track a scholar and detect new papers.

    Notes:
    - This job is designed to be API-first and can be executed periodically by cron.
    - Results are written to Run/Event store for replay.
    """
    run_id = new_run_id()
    trace_id = new_trace_id()
    elog = _event_log()

    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="track_scholar",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_start",
            payload={"scholar_id": scholar_id, "dry_run": dry_run, "offline": offline},
        )
    )

    # Import inside job to keep worker import lightweight.
    from paperbot.agents.scholar_tracking.paper_tracker_agent import PaperTrackerAgent

    tracker = PaperTrackerAgent({"offline": offline})
    scholar = tracker.profile_agent.get_scholar_by_id(scholar_id)
    if not scholar:
        result = {"status": "not_found", "scholar_id": scholar_id}
        elog.append(
            make_event(
                run_id=run_id,
                trace_id=trace_id,
                workflow="jobs",
                stage="track_scholar",
                attempt=0,
                agent_name="ARQ",
                role="orchestrator",
                type="job_result",
                payload=result,
            )
        )
        await tracker.ss_agent.close()
        return {"run_id": run_id, "trace_id": trace_id, **result}

    try:
        tracking = await tracker.track_scholar(scholar, dry_run=dry_run)
        elog.append(
            make_event(
                run_id=run_id,
                trace_id=trace_id,
                workflow="jobs",
                stage="track_scholar",
                attempt=0,
                agent_name="ARQ",
                role="orchestrator",
                type="job_result",
                payload=tracking,
            )
        )
        return {"run_id": run_id, "trace_id": trace_id, "status": "ok", "tracking": tracking}
    finally:
        try:
            await tracker.ss_agent.close()
        except Exception:
            pass


async def analyze_paper_job(ctx, paper: Dict[str, Any], *, scholar_name: str = "") -> Dict[str, Any]:
    """
    ARQ job: analyze a single paper via ScholarPipeline.

    This is best-effort; in offline/fixture mode you can patch the coordinator.
    """
    run_id = new_run_id()
    trace_id = new_trace_id()
    elog = _event_log()

    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="analyze_paper",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_start",
            payload={"paper": paper, "scholar_name": scholar_name},
        )
    )

    from paperbot.domain.paper import PaperMeta
    from paperbot.application.workflows.scholar_pipeline import ScholarPipeline

    pipeline = ScholarPipeline({"enable_fail_fast": True})
    paper_meta = PaperMeta.from_dict(paper)
    report_path, influence, pipeline_data = await pipeline.analyze_paper(
        paper=paper_meta,
        scholar_name=scholar_name or None,
        persist_report=False,
        event_log=elog,
        run_id=run_id,
        trace_id=trace_id,
    )

    result = {
        "report_path": str(report_path) if report_path else None,
        "influence": influence.to_dict() if hasattr(influence, "to_dict") else {},
        "pipeline": pipeline_data,
    }
    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="analyze_paper",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_result",
            payload=result,
        )
    )
    return {"run_id": run_id, "trace_id": trace_id, "status": "ok", "result": result}


class WorkerSettings:
    functions = [track_scholar_job, analyze_paper_job]
    redis_settings = _redis_settings()

    # Example: daily tracking by scholar_id would require a configured list; keep as placeholder.
    cron_jobs = []


