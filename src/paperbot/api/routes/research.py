from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from paperbot.context_engine import ContextEngine, ContextEngineConfig
from paperbot.context_engine.track_router import TrackRouter
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.memory.extractor import extract_memories
from paperbot.memory.schema import MemoryCandidate, NormalizedMessage

router = APIRouter()

_research_store = SqlAlchemyResearchStore()
_memory_store = SqlAlchemyMemoryStore()


class TrackCreateRequest(BaseModel):
    user_id: str = "default"
    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    keywords: List[str] = []
    venues: List[str] = []
    methods: List[str] = []
    activate: bool = True


class TrackResponse(BaseModel):
    track: Dict[str, Any]


@router.post("/research/tracks", response_model=TrackResponse)
def create_track(req: TrackCreateRequest):
    track = _research_store.create_track(
        user_id=req.user_id,
        name=req.name,
        description=req.description,
        keywords=req.keywords,
        venues=req.venues,
        methods=req.methods,
        activate=req.activate,
    )
    return TrackResponse(track=track)


class TrackListResponse(BaseModel):
    user_id: str
    tracks: List[Dict[str, Any]]


@router.get("/research/tracks", response_model=TrackListResponse)
def list_tracks(
    user_id: str = "default",
    include_archived: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
):
    tracks = _research_store.list_tracks(user_id=user_id, include_archived=include_archived, limit=limit)
    return TrackListResponse(user_id=user_id, tracks=tracks)


@router.get("/research/tracks/active", response_model=TrackResponse)
def get_active_track(user_id: str = "default"):
    track = _research_store.get_active_track(user_id=user_id)
    if not track:
        raise HTTPException(status_code=404, detail="No active track for user")
    return TrackResponse(track=track)


@router.post("/research/tracks/{track_id}/activate", response_model=TrackResponse)
def activate_track(track_id: int, user_id: str = "default"):
    track = _research_store.activate_track(user_id=user_id, track_id=track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return TrackResponse(track=track)


class TaskCreateRequest(BaseModel):
    user_id: str = "default"
    title: str = Field(..., min_length=1)
    status: str = "todo"
    priority: int = 0
    paper_id: Optional[str] = None
    paper_url: Optional[str] = None
    metadata: Dict[str, Any] = {}


class TaskResponse(BaseModel):
    task: Dict[str, Any]


@router.post("/research/tracks/{track_id}/tasks", response_model=TaskResponse)
def add_task(track_id: int, req: TaskCreateRequest):
    task = _research_store.add_task(
        user_id=req.user_id,
        track_id=track_id,
        title=req.title,
        status=req.status,
        priority=req.priority,
        paper_id=req.paper_id,
        paper_url=req.paper_url,
        metadata=req.metadata,
    )
    if not task:
        raise HTTPException(status_code=404, detail="Track not found")
    return TaskResponse(task=task)


class TaskListResponse(BaseModel):
    user_id: str
    track_id: int
    tasks: List[Dict[str, Any]]


@router.get("/research/tracks/{track_id}/tasks", response_model=TaskListResponse)
def list_tasks(
    track_id: int,
    user_id: str = "default",
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    tasks = _research_store.list_tasks(user_id=user_id, track_id=track_id, status=status, limit=limit)
    return TaskListResponse(user_id=user_id, track_id=track_id, tasks=tasks)


class MemoryItemCreateRequest(BaseModel):
    user_id: str = "default"
    scope_type: str = "track"  # global/track
    scope_id: Optional[str] = None
    kind: str = Field("note", min_length=1, max_length=32)
    content: str = Field(..., min_length=1)
    tags: List[str] = []
    status: str = "approved"
    confidence: float = 0.8
    evidence: Dict[str, Any] = {}


class MemoryItemResponse(BaseModel):
    item: Dict[str, Any]


def _resolve_track_scope_id(user_id: str, scope_type: str, scope_id: Optional[str]) -> Optional[str]:
    if scope_type != "track":
        return scope_id
    if scope_id:
        return scope_id
    active = _research_store.get_active_track(user_id=user_id)
    if not active:
        return None
    return str(active["id"])


@router.post("/research/memory/items", response_model=MemoryItemResponse)
def create_memory_item(req: MemoryItemCreateRequest):
    scope_type = (req.scope_type or "global").strip() or "global"
    scope_id = _resolve_track_scope_id(req.user_id, scope_type, req.scope_id)
    if scope_type == "track" and not scope_id:
        raise HTTPException(status_code=400, detail="scope_id missing and no active track")

    cand = MemoryCandidate(
        kind=req.kind,  # type: ignore[arg-type]
        content=req.content,
        confidence=float(req.confidence),
        tags=req.tags,
        evidence=req.evidence,
        scope_type=scope_type,
        scope_id=scope_id,
        status=req.status,
    )
    created, _, rows = _memory_store.add_memories(user_id=req.user_id, memories=[cand])
    if created <= 0 or not rows:
        raise HTTPException(status_code=409, detail="Duplicate memory item (same scope/kind/content)")
    return MemoryItemResponse(item=SqlAlchemyMemoryStore._row_to_dict(rows[0]))


class MemoryItemListResponse(BaseModel):
    user_id: str
    items: List[Dict[str, Any]]


@router.get("/research/memory/items", response_model=MemoryItemListResponse)
def list_memory_items(
    user_id: str = "default",
    scope_type: Optional[str] = None,
    scope_id: Optional[str] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    include_pending: bool = False,
    limit: int = Query(100, ge=1, le=500),
):
    items = _memory_store.list_memories(
        user_id=user_id,
        limit=limit,
        kind=kind,
        scope_type=scope_type,
        scope_id=scope_id,
        include_pending=include_pending,
        status=status,
    )
    return MemoryItemListResponse(user_id=user_id, items=items)


@router.get("/research/memory/inbox", response_model=MemoryItemListResponse)
def list_memory_inbox(
    user_id: str = "default",
    track_id: Optional[int] = None,
    limit: int = Query(100, ge=1, le=500),
):
    if track_id is None:
        active = _research_store.get_active_track(user_id=user_id)
        if not active:
            raise HTTPException(status_code=404, detail="No active track for user")
        track_id = int(active["id"])

    items = _memory_store.list_memories(
        user_id=user_id,
        limit=limit,
        scope_type="track",
        scope_id=str(track_id),
        status="pending",
        include_deleted=False,
        include_pending=True,
    )
    return MemoryItemListResponse(user_id=user_id, items=items)


class MemorySuggestRequest(BaseModel):
    user_id: str = "default"
    text: str = Field(..., min_length=1)
    scope_type: str = "track"
    scope_id: Optional[str] = None
    use_llm: bool = False
    redact: bool = True
    language_hint: Optional[str] = None


class MemorySuggestResponse(BaseModel):
    user_id: str
    created: int
    skipped: int
    candidates: List[Dict[str, Any]]


@router.post("/research/memory/suggest", response_model=MemorySuggestResponse)
def suggest_memories(req: MemorySuggestRequest):
    scope_type = (req.scope_type or "global").strip() or "global"
    scope_id = _resolve_track_scope_id(req.user_id, scope_type, req.scope_id)
    if scope_type == "track" and not scope_id:
        raise HTTPException(status_code=400, detail="scope_id missing and no active track")

    msgs = [NormalizedMessage(role="user", content=req.text)]
    extracted = extract_memories(msgs, use_llm=req.use_llm, redact=req.redact, language_hint=req.language_hint)
    pending = [
        MemoryCandidate(
            kind=m.kind,
            content=m.content,
            confidence=m.confidence,
            tags=m.tags,
            evidence=m.evidence,
            scope_type=scope_type,
            scope_id=scope_id,
            status="pending",
        )
        for m in extracted
    ]
    created, skipped, rows = _memory_store.add_memories(user_id=req.user_id, memories=pending)
    return MemorySuggestResponse(
        user_id=req.user_id,
        created=created,
        skipped=skipped,
        candidates=[SqlAlchemyMemoryStore._row_to_dict(r) for r in rows],
    )


class MemoryModerateRequest(BaseModel):
    user_id: str = "default"
    status: str = Field(..., min_length=1)  # approved/rejected/pending/superseded
    content: Optional[str] = None
    kind: Optional[str] = None
    tags: Optional[List[str]] = None
    scope_type: Optional[str] = None
    scope_id: Optional[str] = None


@router.post("/research/memory/items/{item_id}/moderate", response_model=MemoryItemResponse)
def moderate_memory_item(item_id: int, req: MemoryModerateRequest):
    updated = _memory_store.update_item(
        user_id=req.user_id,
        item_id=item_id,
        status=req.status,
        content=req.content,
        kind=req.kind,
        tags=req.tags,
        scope_type=req.scope_type,
        scope_id=req.scope_id,
        actor_id="user",
    )
    if not updated:
        raise HTTPException(status_code=409, detail="Memory item not found or update conflict")
    return MemoryItemResponse(item=updated)


class BulkModerateRequest(BaseModel):
    user_id: str = "default"
    item_ids: List[int] = Field(default_factory=list)
    status: str = Field(..., min_length=1)


class BulkModerateResponse(BaseModel):
    user_id: str
    updated: List[Dict[str, Any]]


@router.post("/research/memory/bulk_moderate", response_model=BulkModerateResponse)
def bulk_moderate(req: BulkModerateRequest):
    updated = _memory_store.bulk_update_items(
        user_id=req.user_id,
        item_ids=req.item_ids,
        status=req.status,
        actor_id="user",
    )
    return BulkModerateResponse(user_id=req.user_id, updated=updated)


class BulkMoveRequest(BaseModel):
    user_id: str = "default"
    item_ids: List[int] = Field(default_factory=list)
    scope_type: str = Field(..., min_length=1)
    scope_id: Optional[str] = None


class BulkMoveResponse(BaseModel):
    user_id: str
    updated: List[Dict[str, Any]]


@router.post("/research/memory/bulk_move", response_model=BulkMoveResponse)
def bulk_move(req: BulkMoveRequest):
    scope_type = (req.scope_type or "global").strip() or "global"
    scope_id = _resolve_track_scope_id(req.user_id, scope_type, req.scope_id)
    if scope_type == "track" and not scope_id:
        raise HTTPException(status_code=400, detail="scope_id missing and no active track")
    updated = _memory_store.bulk_update_items(
        user_id=req.user_id,
        item_ids=req.item_ids,
        scope_type=scope_type,
        scope_id=scope_id,
        actor_id="user",
    )
    return BulkMoveResponse(user_id=req.user_id, updated=updated)


class ClearTrackMemoryResponse(BaseModel):
    user_id: str
    track_id: int
    deleted_count: int


@router.post("/research/tracks/{track_id}/memory/clear", response_model=ClearTrackMemoryResponse)
def clear_track_memory(track_id: int, user_id: str = "default", confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true required")
    deleted = _memory_store.soft_delete_by_scope(
        user_id=user_id,
        scope_type="track",
        scope_id=str(track_id),
        actor_id="user",
        reason="clear_track_memory",
    )
    return ClearTrackMemoryResponse(user_id=user_id, track_id=track_id, deleted_count=deleted)


class PaperFeedbackRequest(BaseModel):
    user_id: str = "default"
    track_id: Optional[int] = None
    paper_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)  # like/dislike/skip/save/cite
    weight: float = 0.0
    metadata: Dict[str, Any] = {}


class PaperFeedbackResponse(BaseModel):
    feedback: Dict[str, Any]


@router.post("/research/papers/feedback", response_model=PaperFeedbackResponse)
def add_paper_feedback(req: PaperFeedbackRequest):
    track_id = req.track_id
    if track_id is None:
        active = _research_store.get_active_track(user_id=req.user_id)
        if not active:
            raise HTTPException(status_code=400, detail="track_id missing and no active track")
        track_id = int(active["id"])

    fb = _research_store.add_paper_feedback(
        user_id=req.user_id,
        track_id=track_id,
        paper_id=req.paper_id,
        action=req.action,
        weight=req.weight,
        metadata=req.metadata,
    )
    if not fb:
        raise HTTPException(status_code=404, detail="Track not found")
    return PaperFeedbackResponse(feedback=fb)


class PaperFeedbackListResponse(BaseModel):
    user_id: str
    track_id: int
    items: List[Dict[str, Any]]


@router.get("/research/tracks/{track_id}/papers/feedback", response_model=PaperFeedbackListResponse)
def list_paper_feedback(
    track_id: int,
    user_id: str = "default",
    action: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
):
    items = _research_store.list_paper_feedback(user_id=user_id, track_id=track_id, action=action, limit=limit)
    return PaperFeedbackListResponse(user_id=user_id, track_id=track_id, items=items)


class RouterSuggestRequest(BaseModel):
    user_id: str = "default"
    query: str = Field(..., min_length=1)


class RouterSuggestResponse(BaseModel):
    suggestion: Optional[Dict[str, Any]]


@router.post("/research/router/suggest", response_model=RouterSuggestResponse)
def suggest_track(req: RouterSuggestRequest):
    active = _research_store.get_active_track(user_id=req.user_id)
    if not active:
        return RouterSuggestResponse(suggestion=None)
    router = TrackRouter(research_store=_research_store, memory_store=_memory_store)
    suggestion = router.suggest_track(user_id=req.user_id, query=req.query, active_track_id=int(active["id"]))
    return RouterSuggestResponse(suggestion=suggestion)


class ContextRequest(BaseModel):
    user_id: str = "default"
    query: str = Field(..., min_length=1)
    track_id: Optional[int] = None
    activate_track_id: Optional[int] = None  # confirm switch: activates then uses it
    memory_limit: int = Field(8, ge=1, le=50)
    paper_limit: int = Field(8, ge=0, le=50)
    offline: bool = False
    include_cross_track: bool = False


class ContextResponse(BaseModel):
    context_pack: Dict[str, Any]


@router.post("/research/context", response_model=ContextResponse)
async def build_context(req: ContextRequest):
    if req.activate_track_id is not None:
        activated = _research_store.activate_track(user_id=req.user_id, track_id=req.activate_track_id)
        if not activated:
            raise HTTPException(status_code=404, detail="Track not found")

    engine = ContextEngine(
        research_store=_research_store,
        memory_store=_memory_store,
        config=ContextEngineConfig(
            memory_limit=req.memory_limit,
            paper_limit=req.paper_limit,
            offline=req.offline,
        ),
    )
    try:
        pack = await engine.build_context_pack(
            user_id=req.user_id,
            query=req.query,
            track_id=req.track_id,
            include_cross_track=req.include_cross_track,
        )
        return ContextResponse(context_pack=pack)
    finally:
        await engine.close()
