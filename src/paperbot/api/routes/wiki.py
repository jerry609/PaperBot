from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from paperbot.application.services.wiki_concept_service import WikiConceptService
from paperbot.infrastructure.stores.wiki_concept_store import WikiConceptStore

router = APIRouter()

_service: Optional[WikiConceptService] = None
_DEFAULT_USER_ID = "default"


def _get_service() -> WikiConceptService:
    global _service
    if _service is None:
        _service = WikiConceptService(WikiConceptStore())
    return _service


def _resolve_wiki_user_id(requested_user_id: Optional[str]) -> str:
    user_id = str(requested_user_id or _DEFAULT_USER_ID).strip() or _DEFAULT_USER_ID
    if user_id != _DEFAULT_USER_ID:
        raise HTTPException(
            status_code=403,
            detail="cross-user wiki grounding requires authenticated user context",
        )
    return _DEFAULT_USER_ID


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
    resolved_user_id = _resolve_wiki_user_id(user_id)
    service = _get_service()
    items = service.list_concepts(
        user_id=resolved_user_id,
        query=q,
        category=category,
        limit=limit,
    )
    return WikiConceptListResponse(
        items=[WikiConceptResponse(**item.to_dict()) for item in items],
        categories=service.categories(),
    )
