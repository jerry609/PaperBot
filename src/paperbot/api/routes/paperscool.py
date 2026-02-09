from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paperbot.api.streaming import StreamEvent, wrap_generator
from paperbot.application.services.llm_service import get_llm_service
from paperbot.application.workflows.analysis.paper_judge import PaperJudge
from paperbot.application.workflows.dailypaper import (
    DailyPaperReporter,
    apply_judge_scores_to_report,
    build_daily_paper_report,
    enrich_daily_paper_report,
    normalize_llm_features,
    normalize_output_formats,
    render_daily_paper_markdown,
    select_judge_candidates,
)
from paperbot.application.workflows.paperscool_topic_search import PapersCoolTopicSearchWorkflow

router = APIRouter()


class PapersCoolSearchRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=lambda: ["papers_cool"])
    branches: List[str] = Field(default_factory=lambda: ["arxiv", "venue"])
    top_k_per_query: int = Field(5, ge=1, le=50)
    show_per_branch: int = Field(25, ge=1, le=100)


class PapersCoolSearchResponse(BaseModel):
    source: str
    fetched_at: str
    sources: List[str]
    queries: List[Dict[str, Any]]
    items: List[Dict[str, Any]]
    summary: Dict[str, Any]


class DailyPaperRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=lambda: ["papers_cool"])
    branches: List[str] = Field(default_factory=lambda: ["arxiv", "venue"])
    top_k_per_query: int = Field(5, ge=1, le=50)
    show_per_branch: int = Field(25, ge=1, le=100)
    title: str = "DailyPaper Digest"
    top_n: int = Field(10, ge=1, le=50)
    formats: List[str] = Field(default_factory=lambda: ["both"])
    save: bool = False
    output_dir: str = "./reports/dailypaper"
    enable_llm_analysis: bool = False
    llm_features: List[str] = Field(default_factory=lambda: ["summary"])
    enable_judge: bool = False
    judge_runs: int = Field(1, ge=1, le=5)
    judge_max_items_per_query: int = Field(5, ge=1, le=20)
    judge_token_budget: int = Field(0, ge=0, le=2_000_000)


class DailyPaperResponse(BaseModel):
    report: Dict[str, Any]
    markdown: str
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None


class PapersCoolAnalyzeRequest(BaseModel):
    report: Dict[str, Any]
    run_judge: bool = False
    run_trends: bool = False
    judge_runs: int = Field(1, ge=1, le=5)
    judge_max_items_per_query: int = Field(5, ge=1, le=20)
    judge_token_budget: int = Field(0, ge=0, le=2_000_000)
    trend_max_items_per_query: int = Field(3, ge=1, le=20)


@router.post("/research/paperscool/search", response_model=PapersCoolSearchResponse)
def topic_search(req: PapersCoolSearchRequest):
    cleaned_queries = [q.strip() for q in req.queries if (q or "").strip()]
    if not cleaned_queries:
        raise HTTPException(status_code=400, detail="queries is required")

    workflow = PapersCoolTopicSearchWorkflow()
    result = workflow.run(
        queries=cleaned_queries,
        sources=req.sources,
        branches=req.branches,
        top_k_per_query=req.top_k_per_query,
        show_per_branch=req.show_per_branch,
    )
    return PapersCoolSearchResponse(**result)


@router.post("/research/paperscool/daily", response_model=DailyPaperResponse)
def generate_daily_report(req: DailyPaperRequest):
    cleaned_queries = [q.strip() for q in req.queries if (q or "").strip()]
    if not cleaned_queries:
        raise HTTPException(status_code=400, detail="queries is required")

    workflow = PapersCoolTopicSearchWorkflow()
    search_result = workflow.run(
        queries=cleaned_queries,
        sources=req.sources,
        branches=req.branches,
        top_k_per_query=req.top_k_per_query,
        show_per_branch=req.show_per_branch,
    )
    report = build_daily_paper_report(search_result=search_result, title=req.title, top_n=req.top_n)
    if req.enable_llm_analysis:
        report = enrich_daily_paper_report(
            report,
            llm_features=normalize_llm_features(req.llm_features),
        )
    if req.enable_judge:
        report = apply_judge_scores_to_report(
            report,
            max_items_per_query=req.judge_max_items_per_query,
            n_runs=req.judge_runs,
            judge_token_budget=req.judge_token_budget,
        )
    markdown = render_daily_paper_markdown(report)

    markdown_path = None
    json_path = None
    if req.save:
        reporter = DailyPaperReporter(output_dir=req.output_dir)
        artifacts = reporter.write(
            report=report,
            markdown=markdown,
            formats=normalize_output_formats(req.formats),
            slug=req.title,
        )
        markdown_path = artifacts.markdown_path
        json_path = artifacts.json_path

    return DailyPaperResponse(
        report=report,
        markdown=markdown,
        markdown_path=markdown_path,
        json_path=json_path,
    )


async def _paperscool_analyze_stream(req: PapersCoolAnalyzeRequest):
    report = copy.deepcopy(req.report)
    llm_service = get_llm_service()

    if req.run_trends:
        llm_block = report.get("llm_analysis")
        if not isinstance(llm_block, dict):
            llm_block = {}

        features = list(llm_block.get("features") or [])
        if "trends" not in features:
            features.append("trends")

        llm_block["enabled"] = True
        llm_block["features"] = features
        llm_block.setdefault("daily_insight", "")
        llm_block["query_trends"] = []

        queries = list(report.get("queries") or [])
        trend_total = sum(1 for query in queries if query.get("top_items"))
        trend_done = 0
        yield StreamEvent(
            type="progress",
            data={"phase": "trends", "message": "Starting trend analysis", "total": trend_total},
        )
        for query in queries:
            query_name = query.get("normalized_query") or query.get("raw_query") or ""
            top_items = list(query.get("top_items") or [])[
                : max(1, int(req.trend_max_items_per_query))
            ]
            if not top_items:
                continue

            trend_done += 1
            analysis = llm_service.analyze_trends(topic=query_name, papers=top_items)
            llm_block["query_trends"].append({"query": query_name, "analysis": analysis})

            yield StreamEvent(
                type="trend",
                data={
                    "query": query_name,
                    "analysis": analysis,
                    "done": trend_done,
                    "total": trend_total,
                },
            )

        report["llm_analysis"] = llm_block
        yield StreamEvent(type="trend_done", data={"count": trend_done})

    if req.run_judge:
        judge = PaperJudge(llm_service=llm_service)
        selection = select_judge_candidates(
            report,
            max_items_per_query=req.judge_max_items_per_query,
            n_runs=req.judge_runs,
            token_budget=req.judge_token_budget,
        )
        selected = list(selection.get("selected") or [])
        recommendation_count = {
            "must_read": 0,
            "worth_reading": 0,
            "skim": 0,
            "skip": 0,
        }

        yield StreamEvent(
            type="progress",
            data={
                "phase": "judge",
                "message": "Starting judge scoring",
                "total": len(selected),
                "budget": selection.get("budget") or {},
            },
        )

        for idx, row in enumerate(selected, start=1):
            query_index = int(row.get("query_index") or 0)
            item_index = int(row.get("item_index") or 0)

            queries = list(report.get("queries") or [])
            if query_index >= len(queries):
                continue

            query = queries[query_index]
            query_name = query.get("normalized_query") or query.get("raw_query") or ""
            top_items = list(query.get("top_items") or [])
            if item_index >= len(top_items):
                continue

            item = top_items[item_index]
            if req.judge_runs > 1:
                judgment = judge.judge_with_calibration(
                    paper=item,
                    query=query_name,
                    n_runs=max(1, int(req.judge_runs)),
                )
            else:
                judgment = judge.judge_single(paper=item, query=query_name)

            j_payload = judgment.to_dict()
            item["judge"] = j_payload
            rec = j_payload.get("recommendation")
            if rec in recommendation_count:
                recommendation_count[rec] += 1

            yield StreamEvent(
                type="judge",
                data={
                    "query": query_name,
                    "title": item.get("title") or "Untitled",
                    "judge": j_payload,
                    "done": idx,
                    "total": len(selected),
                },
            )

        for query in report.get("queries") or []:
            top_items = list(query.get("top_items") or [])
            if not top_items:
                continue
            capped_count = min(len(top_items), max(1, int(req.judge_max_items_per_query)))
            capped = top_items[:capped_count]
            capped.sort(
                key=lambda it: float((it.get("judge") or {}).get("overall") or -1), reverse=True
            )
            query["top_items"] = capped + top_items[capped_count:]

        report["judge"] = {
            "enabled": True,
            "max_items_per_query": int(req.judge_max_items_per_query),
            "n_runs": int(max(1, int(req.judge_runs))),
            "recommendation_count": recommendation_count,
            "budget": selection.get("budget") or {},
        }
        yield StreamEvent(type="judge_done", data=report["judge"])

    markdown = render_daily_paper_markdown(report)
    yield StreamEvent(type="result", data={"report": report, "markdown": markdown})


@router.post("/research/paperscool/analyze")
async def analyze_daily_report(req: PapersCoolAnalyzeRequest):
    if not req.run_judge and not req.run_trends:
        raise HTTPException(status_code=400, detail="run_judge or run_trends must be true")
    if not isinstance(req.report, dict) or not req.report.get("queries"):
        raise HTTPException(status_code=400, detail="report with queries is required")

    return StreamingResponse(
        wrap_generator(_paperscool_analyze_stream(req)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
