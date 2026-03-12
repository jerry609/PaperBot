from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from paperbot.application.services.wiki_concept_service import WikiConceptService
from paperbot.infrastructure.stores.wiki_concept_store import WikiConceptStore

router = APIRouter()

_service: Optional[WikiConceptService] = None


def _get_service() -> WikiConceptService:
    global _service
    if _service is None:
        _service = WikiConceptService(WikiConceptStore())
    return _service


class WikiConceptResponse(BaseModel):
    id: str
    name: str
    description: str
    definition: str
    related_papers: List[str]
    related_concepts: List[str]
    examples: List[str]
    category: str
    icon: str
    paper_count: int = 0
    track_count: int = 0


class WikiConceptListResponse(BaseModel):
    items: List[WikiConceptResponse]
    categories: List[str]


@router.get("/wiki/concepts", response_model=WikiConceptListResponse)
def list_wiki_concepts(
    user_id: str = Query("default", description="User ID"),
    q: str = Query("", description="Keyword query"),
    category: Optional[str] = Query(None, description="Category filter"),
    limit: int = Query(100, ge=1, le=500),
):
    service = _get_service()
    items = service.list_concepts(user_id=user_id, query=q, category=category, limit=limit)
    return WikiConceptListResponse(
        items=[WikiConceptResponse(**item.to_dict()) for item in items],
        categories=service.categories(),
    )
