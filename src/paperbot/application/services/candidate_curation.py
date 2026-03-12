"""Explicit candidate curation and registry-ingest use cases."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from paperbot.application.services.enrichment_pipeline import (
    EnrichmentContext,
    EnrichmentPipeline,
    FilterStep,
    JudgeStep,
    LLMEnrichmentStep,
)
from paperbot.application.services.llm_service import get_llm_service
from paperbot.application.workflows.analysis.paper_judge import PaperJudge
from paperbot.application.workflows.dailypaper import (
    build_daily_paper_report,
    ingest_daily_report_to_registry,
    normalize_llm_features,
    persist_judge_scores_to_registry,
    select_judge_candidates,
)
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore

_KEEP_RECOMMENDATIONS = {"must_read", "worth_reading"}


@dataclass(frozen=True)
class CurationEvent:
    type: str
    data: Dict[str, Any]


@dataclass(frozen=True)
class CuratedReportResult:
    report: Dict[str, Any]
    events: List[CurationEvent]
    build_ms: float
    enrich_ms: float


@dataclass(frozen=True)
class IngestedReportResult:
    report: Dict[str, Any]


def _emit(events: List[CurationEvent], event_type: str, data: Dict[str, Any]) -> None:
    events.append(CurationEvent(type=event_type, data=data))


def _collect_query_items(report: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[int, str]]:
    query_items: List[Dict[str, Any]] = []
    paper_query_map: Dict[int, str] = {}

    for query in report.get("queries") or []:
        query_name = query.get("normalized_query") or query.get("raw_query") or ""
        for item in query.get("top_items") or []:
            query_items.append(item)
            paper_query_map[id(item)] = query_name

    return query_items, paper_query_map


async def curate_search_result(
    *,
    search_result: Dict[str, Any],
    title: str = "DailyPaper Digest",
    top_n: int = 10,
    enable_llm_analysis: bool = False,
    llm_features: Sequence[str] = ("summary",),
    enable_judge: bool = False,
    judge_runs: int = 1,
    judge_max_items_per_query: int = 5,
    judge_token_budget: int = 0,
    llm_service_factory: Callable[[], Any] = get_llm_service,
    judge_factory: Callable[[Any], Any] = PaperJudge,
) -> CuratedReportResult:
    """Turn a search result into a curated report without touching the registry."""

    events: List[CurationEvent] = []

    build_started = time.perf_counter()
    report = build_daily_paper_report(search_result=search_result, title=title, top_n=top_n)
    build_ms = round((time.perf_counter() - build_started) * 1000.0, 2)
    _emit(
        events,
        "report_built",
        {
            "queries_count": len(report.get("queries") or []),
            "global_top_count": len(report.get("global_top") or []),
            "report": report,
        },
    )

    enrich_started = time.perf_counter()
    query_items, paper_query_map = _collect_query_items(report)

    if enable_llm_analysis:
        features = normalize_llm_features(llm_features)
        if features:
            llm_service = llm_service_factory()
            llm_block: Dict[str, Any] = {
                "enabled": True,
                "features": features,
                "query_trends": [],
                "daily_insight": "",
            }

            llm_targets: set[int] = set()
            if "summary" in features or "relevance" in features:
                for query in report.get("queries") or []:
                    for item in (query.get("top_items") or [])[:3]:
                        llm_targets.add(id(item))

            _emit(
                events,
                "progress",
                {
                    "phase": "llm",
                    "message": "Starting LLM enrichment...",
                    "total": len(llm_targets),
                },
            )

            if llm_targets:
                pipeline = EnrichmentPipeline(
                    steps=[LLMEnrichmentStep(llm_service=llm_service, features=features)]
                )
                await pipeline.run(
                    query_items,
                    context=EnrichmentContext(
                        query="; ".join(
                            str(q.get("normalized_query") or q.get("raw_query") or "")
                            for q in (report.get("queries") or [])
                        ),
                        extra={
                            "llm_target_ids": llm_targets,
                            "query_for_relevance": "; ".join(
                                str(q.get("normalized_query") or q.get("raw_query") or "")
                                for q in (report.get("queries") or [])
                            ),
                        },
                    ),
                )

            if "trends" in features:
                for query in report.get("queries") or []:
                    query_name = query.get("normalized_query") or query.get("raw_query") or ""
                    top_items = (query.get("top_items") or [])[:3]
                    if not top_items:
                        continue
                    trend_text = llm_service.analyze_trends(topic=query_name, papers=top_items)
                    llm_block["query_trends"].append({"query": query_name, "analysis": trend_text})
                    _emit(
                        events,
                        "trend",
                        {
                            "query": query_name,
                            "analysis": trend_text,
                            "done": len(llm_block["query_trends"]),
                            "total": len(report.get("queries") or []),
                        },
                    )

            if "insight" in features:
                _emit(
                    events,
                    "progress",
                    {"phase": "insight", "message": "Generating daily insight..."},
                )
                llm_block["daily_insight"] = llm_service.generate_daily_insight(report)
                _emit(events, "insight", {"analysis": llm_block["daily_insight"]})

            report["llm_analysis"] = llm_block
            summary_done = sum(
                1
                for item in query_items
                if id(item) in llm_targets and (item.get("ai_summary") or item.get("relevance"))
            )
            _emit(
                events,
                "llm_done",
                {
                    "summaries_count": summary_done,
                    "trends_count": len(llm_block["query_trends"]),
                },
            )

    if enable_judge:
        llm_service_j = llm_service_factory()
        judge = judge_factory(llm_service=llm_service_j)
        selection = select_judge_candidates(
            report,
            max_items_per_query=judge_max_items_per_query,
            n_runs=judge_runs,
            token_budget=judge_token_budget,
        )
        selected = list(selection.get("selected") or [])
        judge_targets: set[int] = set()
        queries = list(report.get("queries") or [])
        for row in selected:
            query_index = int(row.get("query_index") or 0)
            item_index = int(row.get("item_index") or 0)
            if query_index >= len(queries):
                continue
            top_items = list(queries[query_index].get("top_items") or [])
            if item_index >= len(top_items):
                continue
            judge_targets.add(id(top_items[item_index]))

        _emit(
            events,
            "progress",
            {
                "phase": "judge",
                "message": "Starting judge scoring",
                "total": len(selected),
                "budget": selection.get("budget") or {},
            },
        )

        if judge_targets:
            judge_pipeline = EnrichmentPipeline(
                steps=[JudgeStep(judge=judge, n_runs=max(1, int(judge_runs)))]
            )
            await judge_pipeline.run(
                query_items,
                context=EnrichmentContext(
                    query="; ".join(
                        str(q.get("normalized_query") or q.get("raw_query") or "")
                        for q in (report.get("queries") or [])
                    ),
                    extra={"judge_target_ids": judge_targets, "paper_query_map": paper_query_map},
                ),
            )

        recommendation_count: Dict[str, int] = {
            "must_read": 0,
            "worth_reading": 0,
            "skim": 0,
            "skip": 0,
        }
        for item in query_items:
            if id(item) not in judge_targets:
                continue
            judge_payload = item.get("judge") if isinstance(item.get("judge"), dict) else {}
            recommendation = str(judge_payload.get("recommendation") or "")
            if recommendation in recommendation_count:
                recommendation_count[recommendation] += 1

        for query in report.get("queries") or []:
            top_items = list(query.get("top_items") or [])
            if not top_items:
                continue
            capped_count = min(len(top_items), max(1, int(judge_max_items_per_query)))
            capped = top_items[:capped_count]
            capped.sort(
                key=lambda item: float((item.get("judge") or {}).get("overall") or -1),
                reverse=True,
            )
            query["top_items"] = capped + top_items[capped_count:]

        report["judge"] = {
            "enabled": True,
            "max_items_per_query": int(judge_max_items_per_query),
            "n_runs": int(max(1, int(judge_runs))),
            "recommendation_count": recommendation_count,
            "budget": selection.get("budget") or {},
        }
        _emit(events, "judge_done", report["judge"])

        _emit(
            events,
            "progress",
            {
                "phase": "filter",
                "message": "Filtering papers by judge recommendation...",
            },
        )

        filter_pipeline = EnrichmentPipeline(steps=[FilterStep(keep=_KEEP_RECOMMENDATIONS)])
        await filter_pipeline.run(query_items, context=EnrichmentContext())

        filter_log: List[Dict[str, Any]] = []
        total_before = 0
        total_after = 0
        for query in report.get("queries") or []:
            query_name = query.get("normalized_query") or query.get("raw_query") or ""
            items_before = list(query.get("top_items") or [])
            total_before += len(items_before)
            kept: List[Dict[str, Any]] = []
            for item in items_before:
                if item.get("_filtered_out"):
                    judge_payload = item.get("judge") if isinstance(item.get("judge"), dict) else {}
                    filter_log.append(
                        {
                            "query": query_name,
                            "title": item.get("title") or "Untitled",
                            "recommendation": judge_payload.get("recommendation"),
                            "overall": judge_payload.get("overall"),
                            "action": "removed",
                        }
                    )
                    continue
                kept.append(item)
            total_after += len(kept)
            query["top_items"] = kept

        judge_by_key: Dict[str, Dict[str, Any]] = {}
        for item in query_items:
            if not isinstance(item.get("judge"), dict):
                continue
            key = f"{(item.get('url') or '').strip()}|{(item.get('title') or '').strip().lower()}"
            if key:
                judge_by_key[key] = item["judge"]

        global_before = list(report.get("global_top") or [])
        global_kept: List[Dict[str, Any]] = []
        for item in global_before:
            key = f"{(item.get('url') or '').strip()}|{(item.get('title') or '').strip().lower()}"
            if key in judge_by_key:
                item["judge"] = judge_by_key[key]
            judge_payload = item.get("judge")
            if isinstance(judge_payload, dict):
                recommendation = str(judge_payload.get("recommendation") or "")
                if recommendation in _KEEP_RECOMMENDATIONS:
                    global_kept.append(item)
            else:
                global_kept.append(item)
        report["global_top"] = global_kept

        report["filter"] = {
            "enabled": True,
            "keep_recommendations": list(_KEEP_RECOMMENDATIONS),
            "total_before": total_before,
            "total_after": total_after,
            "removed_count": total_before - total_after,
            "log": filter_log,
        }
        _emit(
            events,
            "filter_done",
            {
                "total_before": total_before,
                "total_after": total_after,
                "removed_count": total_before - total_after,
                "log": filter_log,
            },
        )

    enrich_ms = round((time.perf_counter() - enrich_started) * 1000.0, 2)
    return CuratedReportResult(
        report=report,
        events=events,
        build_ms=build_ms,
        enrich_ms=enrich_ms,
    )


def ingest_curated_report(
    *,
    report: Dict[str, Any],
    persist_judge_scores: bool = False,
    paper_store: Optional[SqlAlchemyPaperStore] = None,
) -> IngestedReportResult:
    """Explicitly ingest a curated report into the canonical registry."""

    ingested = copy.deepcopy(report)
    try:
        ingested["registry_ingest"] = ingest_daily_report_to_registry(
            ingested,
            paper_store=paper_store,
        )
    except Exception as exc:
        ingested["registry_ingest"] = {"error": str(exc)}

    if persist_judge_scores:
        try:
            ingested["judge_registry_ingest"] = persist_judge_scores_to_registry(
                ingested,
                paper_store=paper_store,
            )
        except Exception as exc:
            ingested["judge_registry_ingest"] = {"error": str(exc)}

    return IngestedReportResult(report=ingested)
