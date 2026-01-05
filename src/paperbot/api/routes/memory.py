from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from pydantic import BaseModel, Field

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory import build_memory_context, extract_memories, parse_chat_log
from paperbot.memory.schema import MemoryCandidate

from paperbot.memory.eval.collector import MemoryMetricCollector
router = APIRouter()

_store = SqlAlchemyMemoryStore()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MemoryItemOut(BaseModel):
    id: Optional[int] = None
    workspace_id: Optional[str] = None
    kind: str
    content: str
    confidence: float = 0.6
    status: Optional[str] = None
    pii_risk: Optional[int] = None
    use_count: Optional[int] = None
    last_used_at: Optional[str] = None
    tags: List[str] = []
    evidence: Dict[str, Any] = {}


class IngestResponse(BaseModel):
    user_id: str
    platform: str
    filename: str
    source_sha256: str
    messages_parsed: int
    memory_items_created: int
    memory_items_skipped: int
    extracted: List[MemoryItemOut] = []
    metadata: Dict[str, Any] = {}


@router.post("/memory/ingest", response_model=IngestResponse)
async def ingest_memory(
    file: UploadFile = File(...),
    user_id: str = Query("default", description="Memory namespace; use one id per person/team."),
    workspace_id: Optional[str] = Query(None, description="Optional workspace/project namespace."),
    platform: Optional[str] = Query(None, description="Hint: chatgpt/gemini/claude/..."),
    use_llm: bool = Query(False, description="Use configured LLM to extract memories (falls back on heuristics)."),
    redact: bool = Query(True, description="Redact basic PII (email/phone) before extraction."),
    language_hint: Optional[str] = Query(None, description="Optional hint: zh/en/..."),
    actor_id: str = Query("system", description="Audit actor id (user/admin/service)."),
):
    raw = await file.read()
    filename = file.filename or ""
    parsed = parse_chat_log(raw, filename=filename, platform_hint=platform)

    effective_platform = platform or parsed.platform or "unknown"
    candidates = extract_memories(parsed.messages, use_llm=use_llm, redact=redact, language_hint=language_hint)

    src = _store.upsert_source(
        user_id=user_id,
        platform=effective_platform,
        filename=filename,
        raw_bytes=raw,
        message_count=len(parsed.messages),
        conversation_count=int(parsed.metadata.get("conversation_count") or 0),
        metadata={**parsed.metadata, "parsed_platform": parsed.platform},
    )

    created, skipped, _ = _store.add_memories(
        user_id=user_id,
        workspace_id=workspace_id,
        memories=candidates,
        source_id=src.id,
        actor_id=actor_id,
    )

    extracted_out: List[MemoryItemOut] = []
    for cand in candidates[:50]:
        extracted_out.append(
            MemoryItemOut(
                id=None,
                kind=cand.kind,
                content=cand.content,
                confidence=cand.confidence,
                tags=cand.tags,
                evidence=cand.evidence,
            )
        )

    return IngestResponse(
        user_id=user_id,
        platform=effective_platform,
        filename=filename,
        source_sha256=_sha256_bytes(raw),
        messages_parsed=len(parsed.messages),
        memory_items_created=created,
        memory_items_skipped=skipped,
        extracted=extracted_out,
        metadata={"source_id": src.id, **parsed.metadata},
    )


class MemoryListResponse(BaseModel):
    user_id: str
    items: List[MemoryItemOut]


@router.get("/memory/list", response_model=MemoryListResponse)
def list_memories(
    user_id: str = "default",
    limit: int = 100,
    kind: Optional[str] = None,
    workspace_id: Optional[str] = None,
    include_pending: bool = False,
    include_deleted: bool = False,
):
    items = _store.list_memories(
        user_id=user_id,
        limit=limit,
        kind=kind,
        workspace_id=workspace_id,
        include_pending=include_pending,
        include_deleted=include_deleted,
    )
    return MemoryListResponse(
        user_id=user_id,
        items=[
            MemoryItemOut(
                id=i.get("id"),
                workspace_id=i.get("workspace_id"),
                kind=i.get("kind") or "fact",
                content=i.get("content") or "",
                confidence=float(i.get("confidence") or 0.6),
                status=i.get("status"),
                pii_risk=i.get("pii_risk"),
                use_count=i.get("use_count"),
                last_used_at=i.get("last_used_at"),
                tags=i.get("tags") or [],
                evidence=i.get("evidence") or {},
            )
            for i in items
        ],
    )


class ContextRequest(BaseModel):
    user_id: str = "default"
    workspace_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    limit: int = 8
    actor_id: str = "system"


class ContextResponse(BaseModel):
    user_id: str
    query: str
    context: str
    items: List[MemoryItemOut]


@router.post("/memory/context", response_model=ContextResponse)
def memory_context(req: ContextRequest):
    items = _store.search_memories(
        user_id=req.user_id,
        workspace_id=req.workspace_id,
        query=req.query,
        limit=req.limit,
    )
    _store.touch_usage(item_ids=[int(i["id"]) for i in items if i.get("id")], actor_id=req.actor_id)
    cands = [
        MemoryCandidate(
            kind=i.get("kind") or "fact",  # type: ignore[arg-type]
            content=i.get("content") or "",
            confidence=float(i.get("confidence") or 0.6),
            tags=i.get("tags") or [],
            evidence=i.get("evidence") or {},
        )
        for i in items
        if (i.get("content") or "").strip()
    ]
    ctx = build_memory_context(cands, max_items=req.limit)
    return ContextResponse(
        user_id=req.user_id,
        query=req.query,
        context=ctx,
        items=[
            MemoryItemOut(
                id=i.get("id"),
                workspace_id=i.get("workspace_id"),
                kind=i.get("kind") or "fact",
                content=i.get("content") or "",
                confidence=float(i.get("confidence") or 0.6),
                status=i.get("status"),
                pii_risk=i.get("pii_risk"),
                use_count=i.get("use_count"),
                last_used_at=i.get("last_used_at"),
                tags=i.get("tags") or [],
                evidence=i.get("evidence") or {},
            )
            for i in items
        ],
    )


class MemoryItemUpdateRequest(BaseModel):
    kind: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    workspace_id: Optional[str] = None
    actor_id: str = "system"


@router.patch("/memory/items/{item_id}", response_model=MemoryItemOut)
def update_memory_item(user_id: str, item_id: int, body: MemoryItemUpdateRequest):
    updated = _store.update_item(
        user_id=user_id,
        item_id=item_id,
        actor_id=body.actor_id,
        content=body.content,
        kind=body.kind,
        tags=body.tags,
        status=body.status,
        workspace_id=body.workspace_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="memory item not found or update failed")
    return MemoryItemOut(
        id=updated.get("id"),
        workspace_id=updated.get("workspace_id"),
        kind=updated.get("kind") or "fact",
        content=updated.get("content") or "",
        confidence=float(updated.get("confidence") or 0.6),
        status=updated.get("status"),
        pii_risk=updated.get("pii_risk"),
        use_count=updated.get("use_count"),
        last_used_at=updated.get("last_used_at"),
        tags=updated.get("tags") or [],
        evidence=updated.get("evidence") or {},
    )


@router.delete("/memory/items/{item_id}")
def delete_memory_item(
    user_id: str,
    item_id: int,
    actor_id: str = "system",
    reason: str = "",
    hard: bool = False,
):
    if hard:
        ok = _store.hard_delete_item(user_id=user_id, item_id=item_id, actor_id=actor_id)
    else:
        ok = _store.soft_delete_item(user_id=user_id, item_id=item_id, actor_id=actor_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail="memory item not found")
    return {"status": "ok"}


# --- Metrics Endpoints (Scope and Acceptance Criteria) ---

_metric_collector = MemoryMetricCollector()


class MetricsSummaryResponse(BaseModel):
    status: str  # "pass" or "fail"
    metrics: Dict[str, Any]
    targets: Dict[str, float]


@router.get("/memory/metrics", response_model=MetricsSummaryResponse)
def get_memory_metrics():
    """
    Get summary of memory system evaluation metrics.

    Returns the latest value for each P0 acceptance metric:
    - extraction_precision (target: >= 85%)
    - false_positive_rate (target: <= 5%)
    - retrieval_hit_rate (target: >= 80%)
    - injection_pollution_rate (target: <= 2%)
    - deletion_compliance (target: 100%)
    """
    return _metric_collector.get_metrics_summary()


class MetricHistoryResponse(BaseModel):
    metric_name: str
    history: List[Dict[str, Any]]


@router.get("/memory/metrics/{metric_name}", response_model=MetricHistoryResponse)
def get_metric_history(metric_name: str, limit: int = Query(30, ge=1, le=100)):
    """Get historical values for a specific metric."""
    if metric_name not in MemoryMetricCollector.TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric: {metric_name}. Valid: {list(MemoryMetricCollector.TARGETS.keys())}",
        )
    history = _metric_collector.get_metric_history(metric_name, limit=limit)
    return MetricHistoryResponse(metric_name=metric_name, history=history)
