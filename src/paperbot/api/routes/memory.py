from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile, Query
from pydantic import BaseModel, Field

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory import build_memory_context, extract_memories, parse_chat_log
from paperbot.memory.schema import MemoryCandidate

router = APIRouter()

_store = SqlAlchemyMemoryStore()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MemoryItemOut(BaseModel):
    id: Optional[int] = None
    kind: str
    content: str
    confidence: float = 0.6
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
    platform: Optional[str] = Query(None, description="Hint: chatgpt/gemini/claude/..."),
    use_llm: bool = Query(False, description="Use configured LLM to extract memories (falls back on heuristics)."),
    redact: bool = Query(True, description="Redact basic PII (email/phone) before extraction."),
    language_hint: Optional[str] = Query(None, description="Optional hint: zh/en/..."),
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

    created, skipped, _ = _store.add_memories(user_id=user_id, memories=candidates, source_id=src.id)

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
def list_memories(user_id: str = "default", limit: int = 100, kind: Optional[str] = None):
    items = _store.list_memories(user_id=user_id, limit=limit, kind=kind)
    return MemoryListResponse(
        user_id=user_id,
        items=[
            MemoryItemOut(
                id=i.get("id"),
                kind=i.get("kind") or "fact",
                content=i.get("content") or "",
                confidence=float(i.get("confidence") or 0.6),
                tags=i.get("tags") or [],
                evidence=i.get("evidence") or {},
            )
            for i in items
        ],
    )


class ContextRequest(BaseModel):
    user_id: str = "default"
    query: str = Field(..., min_length=1)
    limit: int = 8


class ContextResponse(BaseModel):
    user_id: str
    query: str
    context: str
    items: List[MemoryItemOut]


@router.post("/memory/context", response_model=ContextResponse)
def memory_context(req: ContextRequest):
    items = _store.search_memories(user_id=req.user_id, query=req.query, limit=req.limit)
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
                kind=i.get("kind") or "fact",
                content=i.get("content") or "",
                confidence=float(i.get("confidence") or 0.6),
                tags=i.get("tags") or [],
                evidence=i.get("evidence") or {},
            )
            for i in items
        ],
    )
