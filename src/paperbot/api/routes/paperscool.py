from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from paperbot.application.workflows.dailypaper import (
    DailyPaperReporter,
    build_daily_paper_report,
    enrich_daily_paper_report,
    normalize_llm_features,
    normalize_output_formats,
    render_daily_paper_markdown,
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


class DailyPaperResponse(BaseModel):
    report: Dict[str, Any]
    markdown: str
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None


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
