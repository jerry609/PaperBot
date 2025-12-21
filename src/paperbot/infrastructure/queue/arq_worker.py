from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from arq import cron
from arq.connections import RedisSettings

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id, make_event
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog
from paperbot.infrastructure.services.subscription_service import SubscriptionService


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


def _subscription_service() -> SubscriptionService:
    """
    Load subscription list from a fixed YAML file.

    Override path with PAPERBOT_SUBSCRIPTIONS_PATH.
    """
    path = os.getenv("PAPERBOT_SUBSCRIPTIONS_PATH") or None
    return SubscriptionService(config_path=path)


def _load_subscription_scholar_ids() -> List[str]:
    svc = _subscription_service()
    scholars = svc.get_scholars()
    ids: List[str] = []
    for s in scholars:
        # Scholar dataclass (preferred) or raw dict fallback
        sid = getattr(s, "semantic_scholar_id", None) or (s.get("semantic_scholar_id") if isinstance(s, dict) else None)
        if sid:
            ids.append(str(sid))
    return ids


def _load_subscription_settings() -> Dict[str, Any]:
    try:
        svc = _subscription_service()
        return svc.get_settings()
    except Exception:
        # If config missing/invalid, keep worker running with no cron jobs.
        return {"check_interval": None}


async def cron_track_subscriptions(ctx) -> Dict[str, Any]:
    """
    Cron entrypoint: enqueue tracking jobs for all subscribed scholars.
    """
    run_id = new_run_id()
    trace_id = new_trace_id()
    elog = _event_log()

    dry_run = os.getenv("PAPERBOT_TRACK_DRY_RUN", "false").lower() in ("1", "true", "yes", "y")
    offline = os.getenv("PAPERBOT_TRACK_OFFLINE", "false").lower() in ("1", "true", "yes", "y")
    scholar_ids = _load_subscription_scholar_ids()

    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="cron_track_subscriptions",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_start",
            payload={"count": len(scholar_ids), "dry_run": dry_run, "offline": offline},
        )
    )

    redis = ctx.get("redis")
    enqueued: List[Dict[str, Any]] = []
    if redis is None:
        elog.append(
            make_event(
                run_id=run_id,
                trace_id=trace_id,
                workflow="jobs",
                stage="cron_track_subscriptions",
                attempt=0,
                agent_name="ARQ",
                role="orchestrator",
                type="job_result",
                payload={"status": "error", "error": "redis not available in ctx"},
            )
        )
        return {"run_id": run_id, "trace_id": trace_id, "status": "error", "error": "redis not available"}

    for sid in scholar_ids:
        job = await redis.enqueue_job(
            "track_scholar_job",
            sid,
            dry_run=dry_run,
            offline=offline,
        )
        info = {"scholar_id": sid, "job_id": job.job_id}
        enqueued.append(info)
        elog.append(
            make_event(
                run_id=run_id,
                trace_id=trace_id,
                workflow="jobs",
                stage="cron_track_subscriptions",
                attempt=0,
                agent_name="ARQ",
                role="orchestrator",
                type="job_enqueue",
                payload=info,
            )
        )

    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="cron_track_subscriptions",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_result",
            payload={"status": "ok", "enqueued": enqueued},
        )
    )
    return {"run_id": run_id, "trace_id": trace_id, "status": "ok", "enqueued": enqueued}


def _build_cron_jobs_from_subscriptions():
    """
    Build ARQ cron_jobs based on subscriptions.settings.check_interval.
    """
    settings = _load_subscription_settings()
    interval = (settings or {}).get("check_interval") or None

    minute = int(os.getenv("PAPERBOT_TRACK_CRON_MINUTE", "0"))
    hour = int(os.getenv("PAPERBOT_TRACK_CRON_HOUR", "2"))
    weekday_raw = os.getenv("PAPERBOT_TRACK_CRON_WEEKDAY", "mon")  # arq supports 0-6 or mon-sun
    day = int(os.getenv("PAPERBOT_TRACK_CRON_DAY", "1"))
    run_at_startup = os.getenv("PAPERBOT_TRACK_RUN_AT_STARTUP", "false").lower() in ("1", "true", "yes", "y")

    weekday: Any = weekday_raw
    try:
        weekday = int(weekday_raw)  # allow 0-6
    except Exception:
        weekday = weekday_raw

    if interval == "daily":
        return [cron(cron_track_subscriptions, minute=minute, hour=hour, run_at_startup=run_at_startup)]
    if interval == "weekly":
        return [cron(cron_track_subscriptions, minute=minute, hour=hour, weekday=weekday, run_at_startup=run_at_startup)]
    if interval == "monthly":
        return [cron(cron_track_subscriptions, minute=minute, hour=hour, day=day, run_at_startup=run_at_startup)]
    # Disabled / unknown interval -> no cron jobs
    return []


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
    functions = [track_scholar_job, analyze_paper_job, cron_track_subscriptions]
    redis_settings = _redis_settings()

    cron_jobs = _build_cron_jobs_from_subscriptions()


