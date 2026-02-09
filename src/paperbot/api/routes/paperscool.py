from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
