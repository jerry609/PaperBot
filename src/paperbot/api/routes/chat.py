"""
Chat API Route - Interactive conversation with AI about papers
"""

from typing import List, Optional

from fastapi import APIRouter, Depends

from paperbot.api.auth.dependencies import get_required_user_id
from pydantic import BaseModel, Field, field_validator

from ...application.collaboration.message_schema import new_trace_id
from ..error_handling import GENERIC_STREAM_ERROR_MESSAGE, resolve_chat_max_history
from ..streaming import StreamEvent, sse_response

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    use_memory: bool = False

    @field_validator("history")
    @classmethod
    def _trim_history(cls, history: List[ChatMessage]) -> List[ChatMessage]:
        return list(history or [])[-resolve_chat_max_history() :]


async def chat_stream(request: ChatRequest, *, trace_id: str):
    """Stream chat response"""
    try:
        yield StreamEvent(
            type="progress",
            data={"phase": "Processing", "message": "Thinking..."},
        )

        # Import LLM client
        from ...infrastructure.llm import ModelRouter, TaskType

        router = ModelRouter.from_env()
        provider = router.get_provider(TaskType.CHAT)

        # Build conversation
        messages = [
            {
                "role": "system",
                "content": """You are PaperBot, an AI assistant specialized in academic research.
You help users:
- Find and analyze research papers
- Track scholars and their publications
- Understand research methodologies
- Generate code implementations from papers
- Review papers for quality and novelty

Be concise and helpful. When discussing papers, cite specific details when available.""",
            },
        ]

        # Optional long-term memory augmentation (cross-platform).
        if request.use_memory and request.user_id:
            try:
                from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
                from paperbot.memory.extractor import build_memory_context
                from paperbot.memory.schema import MemoryCandidate

                store = SqlAlchemyMemoryStore()
                items = store.search_memories(
                    user_id=request.user_id, query=request.message, limit=8
                )
                store.touch_usage(
                    item_ids=[int(i["id"]) for i in items if i.get("id")], actor_id=request.user_id
                )
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
                mem_ctx = build_memory_context(cands, max_items=8)
                if mem_ctx:
                    messages.append({"role": "system", "content": mem_ctx})
            except Exception:
                pass

        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": request.message})

        # Stream response
        full_content = ""
        async for chunk in provider.stream(messages):
            if chunk.delta:
                full_content += chunk.delta
                yield StreamEvent(
                    type="progress",
                    data={"delta": chunk.delta, "content": full_content},
                )

        yield StreamEvent(
            type="result",
            data={"content": full_content},
        )

    except Exception:
        # Fallback to simple response if LLM fails
        yield StreamEvent(
            type="error",
            message=GENERIC_STREAM_ERROR_MESSAGE,
            data={
                "content": (
                    "I apologize, but I encountered an internal error. "
                    "Please retry and provide the trace_id if this keeps failing."
                ),
                "trace_id": trace_id,
            },
        )


@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_required_user_id)):
    """
    Chat with PaperBot AI and stream response.

    Returns Server-Sent Events with streaming text.
    """
    trace_id = new_trace_id()
    # Override any client-provided user_id with authenticated user
    request.user_id = user_id
    return sse_response(chat_stream(request, trace_id=trace_id), workflow="chat", trace_id=trace_id)
