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
        sid = getattr(s, "semantic_scholar_id", None) or (
            s.get("semantic_scholar_id") if isinstance(s, dict) else None
        )
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


def _parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        return {
            "run_id": run_id,
            "trace_id": trace_id,
            "status": "error",
            "error": "redis not available",
        }

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


async def cron_daily_papers(ctx) -> Dict[str, Any]:
    """Cron entrypoint: enqueue one DailyPaper topic search job."""
    run_id = new_run_id()
    trace_id = new_trace_id()
    elog = _event_log()

    queries = _parse_csv_env("PAPERBOT_DAILYPAPER_QUERIES", "ICL压缩,ICL隐式偏置,KV Cache加速")
    sources = _parse_csv_env("PAPERBOT_DAILYPAPER_SOURCES", "papers_cool")
    branches = _parse_csv_env("PAPERBOT_DAILYPAPER_BRANCHES", "arxiv,venue")

    redis = ctx.get("redis")
    if redis is None:
        elog.append(
            make_event(
                run_id=run_id,
                trace_id=trace_id,
                workflow="jobs",
                stage="cron_daily_papers",
                attempt=0,
                agent_name="ARQ",
                role="orchestrator",
                type="job_result",
                payload={"status": "error", "error": "redis not available in ctx"},
            )
        )
        return {
            "run_id": run_id,
            "trace_id": trace_id,
            "status": "error",
            "error": "redis not available",
        }

    enable_llm_analysis = os.getenv("PAPERBOT_DAILYPAPER_ENABLE_LLM", "false").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    llm_features = _parse_csv_env("PAPERBOT_DAILYPAPER_LLM_FEATURES", "summary")
    enable_judge = os.getenv("PAPERBOT_DAILYPAPER_ENABLE_JUDGE", "false").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    judge_runs = int(os.getenv("PAPERBOT_DAILYPAPER_JUDGE_RUNS", "1"))
    judge_max_items = int(os.getenv("PAPERBOT_DAILYPAPER_JUDGE_MAX_ITEMS", "5"))
    judge_token_budget = int(os.getenv("PAPERBOT_DAILYPAPER_JUDGE_TOKEN_BUDGET", "0"))

    job = await redis.enqueue_job(
        "daily_papers_job",
        queries=queries,
        sources=sources,
        branches=branches,
        top_k_per_query=int(os.getenv("PAPERBOT_DAILYPAPER_TOP_K", "5")),
        show_per_branch=int(os.getenv("PAPERBOT_DAILYPAPER_SHOW", "25")),
        top_n=int(os.getenv("PAPERBOT_DAILYPAPER_TOP_N", "10")),
        title=os.getenv("PAPERBOT_DAILYPAPER_TITLE", "DailyPaper Digest"),
        output_dir=os.getenv("PAPERBOT_DAILYPAPER_OUTPUT_DIR", "./reports/dailypaper"),
        enable_llm_analysis=enable_llm_analysis,
        llm_features=llm_features,
        enable_judge=enable_judge,
        judge_runs=judge_runs,
        judge_max_items_per_query=judge_max_items,
        judge_token_budget=judge_token_budget,
        save=True,
    )
    payload = {
        "status": "ok",
        "job_id": job.job_id,
        "queries": queries,
        "sources": sources,
        "branches": branches,
    }
    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="cron_daily_papers",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_enqueue",
            payload=payload,
        )
    )
    return {"run_id": run_id, "trace_id": trace_id, **payload}


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
    run_at_startup = os.getenv("PAPERBOT_TRACK_RUN_AT_STARTUP", "false").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    weekday: Any = weekday_raw
    try:
        weekday = int(weekday_raw)  # allow 0-6
    except Exception:
        weekday = weekday_raw

    if interval == "daily":
        return [
            cron(cron_track_subscriptions, minute=minute, hour=hour, run_at_startup=run_at_startup)
        ]
    if interval == "weekly":
        return [
            cron(
                cron_track_subscriptions,
                minute=minute,
                hour=hour,
                weekday=weekday,
                run_at_startup=run_at_startup,
            )
        ]
    if interval == "monthly":
        return [
            cron(
                cron_track_subscriptions,
                minute=minute,
                hour=hour,
                day=day,
                run_at_startup=run_at_startup,
            )
        ]
    # Disabled / unknown interval -> no cron jobs
    return []


def _build_daily_paper_cron_jobs():
    enabled = os.getenv("PAPERBOT_DAILYPAPER_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if not enabled:
        return []

    minute = int(os.getenv("PAPERBOT_DAILYPAPER_CRON_MINUTE", "30"))
    hour = int(os.getenv("PAPERBOT_DAILYPAPER_CRON_HOUR", "8"))
    run_at_startup = os.getenv("PAPERBOT_DAILYPAPER_RUN_AT_STARTUP", "false").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    return [cron(cron_daily_papers, minute=minute, hour=hour, run_at_startup=run_at_startup)]


async def daily_papers_job(
    ctx,
    *,
    queries: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    branches: Optional[List[str]] = None,
    top_k_per_query: int = 5,
    show_per_branch: int = 25,
    top_n: int = 10,
    title: str = "DailyPaper Digest",
    output_dir: str = "./reports/dailypaper",
    enable_llm_analysis: bool = False,
    llm_features: Optional[List[str]] = None,
    enable_judge: bool = False,
    judge_runs: int = 1,
    judge_max_items_per_query: int = 5,
    judge_token_budget: int = 0,
    save: bool = True,
) -> Dict[str, Any]:
    """ARQ job: generate DailyPaper report and bridge highlights into feed events."""
    run_id = new_run_id()
    trace_id = new_trace_id()
    elog = _event_log()

    job_queries = queries or ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]
    job_sources = sources or ["papers_cool"]
    job_branches = branches or ["arxiv", "venue"]

    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="daily_papers",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_start",
            payload={
                "queries": job_queries,
                "sources": job_sources,
                "branches": job_branches,
                "top_k_per_query": top_k_per_query,
                "show_per_branch": show_per_branch,
                "top_n": top_n,
                "enable_llm_analysis": enable_llm_analysis,
                "llm_features": llm_features or ["summary"],
                "enable_judge": enable_judge,
                "judge_runs": judge_runs,
                "judge_max_items_per_query": judge_max_items_per_query,
                "judge_token_budget": judge_token_budget,
            },
        )
    )

    from paperbot.application.workflows.dailypaper import (
        DailyPaperReporter,
        apply_judge_scores_to_report,
        build_daily_paper_report,
        enrich_daily_paper_report,
        normalize_llm_features,
        normalize_output_formats,
        render_daily_paper_markdown,
    )
    from paperbot.application.workflows.paperscool_topic_search import PapersCoolTopicSearchWorkflow
    from paperbot.workflows.feed import ScholarFeedService

    search_workflow = PapersCoolTopicSearchWorkflow()
    search_result = search_workflow.run(
        queries=job_queries,
        sources=job_sources,
        branches=job_branches,
        top_k_per_query=max(1, int(top_k_per_query)),
        show_per_branch=max(1, int(show_per_branch)),
    )
    report = build_daily_paper_report(
        search_result=search_result, title=title, top_n=max(1, int(top_n))
    )
    if enable_llm_analysis:
        report = enrich_daily_paper_report(
            report,
            llm_features=normalize_llm_features(llm_features or ["summary"]),
        )
    if enable_judge:
        report = apply_judge_scores_to_report(
            report,
            max_items_per_query=max(1, int(judge_max_items_per_query)),
            n_runs=max(1, int(judge_runs)),
            judge_token_budget=max(0, int(judge_token_budget)),
        )
    markdown = render_daily_paper_markdown(report)

    markdown_path = None
    json_path = None
    if save:
        reporter = DailyPaperReporter(output_dir=output_dir)
        artifacts = reporter.write(
            report=report,
            markdown=markdown,
            formats=normalize_output_formats(["both"]),
            slug=title,
        )
        markdown_path = artifacts.markdown_path
        json_path = artifacts.json_path

    feed_service = ScholarFeedService()
    feed_service.process_daily_paper_report(report)
    feed_events = [event.to_dict() for event in feed_service.get_feed(limit=30)]

    payload = {
        "report_date": report.get("date"),
        "unique_items": report.get("stats", {}).get("unique_items", 0),
        "markdown_path": markdown_path,
        "json_path": json_path,
        "feed_events": len(feed_events),
    }
    elog.append(
        make_event(
            run_id=run_id,
            trace_id=trace_id,
            workflow="jobs",
            stage="daily_papers",
            attempt=0,
            agent_name="ARQ",
            role="orchestrator",
            type="job_result",
            payload=payload,
        )
    )
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "status": "ok",
        "report": report,
        "markdown_path": markdown_path,
        "json_path": json_path,
        "feed_events": feed_events,
    }


async def track_scholar_job(
    ctx, scholar_id: str, *, dry_run: bool = True, offline: bool = False
) -> Dict[str, Any]:
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


async def analyze_paper_job(
    ctx, paper: Dict[str, Any], *, scholar_name: str = ""
) -> Dict[str, Any]:
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
    functions = [
        track_scholar_job,
        analyze_paper_job,
        cron_track_subscriptions,
        cron_daily_papers,
        daily_papers_job,
    ]
    redis_settings = _redis_settings()

    cron_jobs = _build_cron_jobs_from_subscriptions() + _build_daily_paper_cron_jobs()
