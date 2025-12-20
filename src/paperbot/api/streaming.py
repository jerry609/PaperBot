"""
Streaming utilities for Server-Sent Events (SSE)
"""

import json
from typing import Any, AsyncGenerator
from dataclasses import dataclass


@dataclass
class StreamEvent:
    """SSE event structure"""
    type: str  # progress, result, error, done
    data: Any = None
    message: str | None = None

    def to_sse(self) -> str:
        """Convert to SSE format"""
        payload = {
            "type": self.type,
            "data": self.data,
            "message": self.message,
        }
        return f"data: {json.dumps(payload)}\n\n"


def sse_done() -> str:
    """Return SSE done signal"""
    return "data: [DONE]\n\n"


async def wrap_generator(
    generator: AsyncGenerator[StreamEvent, None]
) -> AsyncGenerator[str, None]:
    """Wrap a StreamEvent generator to SSE strings"""
    try:
        async for event in generator:
            yield event.to_sse()
        yield sse_done()
    except Exception as e:
        yield StreamEvent(type="error", message=str(e)).to_sse()
        yield sse_done()
