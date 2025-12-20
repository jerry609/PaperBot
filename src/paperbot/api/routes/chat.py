"""
Chat API Route - Interactive conversation with AI about papers
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from ..streaming import StreamEvent, wrap_generator

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


async def chat_stream(request: ChatRequest):
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
            {"role": "system", "content": """You are PaperBot, an AI assistant specialized in academic research.
You help users:
- Find and analyze research papers
- Track scholars and their publications
- Understand research methodologies
- Generate code implementations from papers
- Review papers for quality and novelty

Be concise and helpful. When discussing papers, cite specific details when available."""},
        ]

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

    except Exception as e:
        # Fallback to simple response if LLM fails
        yield StreamEvent(
            type="result",
            data={
                "content": f"I apologize, but I encountered an error: {str(e)}. "
                "Please make sure the API server is properly configured with LLM credentials."
            },
        )


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with PaperBot AI and stream response.

    Returns Server-Sent Events with streaming text.
    """
    return StreamingResponse(
        wrap_generator(chat_stream(request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
