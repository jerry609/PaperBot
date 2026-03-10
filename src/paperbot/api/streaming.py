"""
Streaming utilities for Server-Sent Events (SSE).

Provides a normalized envelope for stream observability:
- workflow
- run_id
- trace_id
- seq
- phase
- ts
"""

from __future__ import annotations

import json
import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional
from uuid import uuid4

from fastapi.responses import StreamingResponse


DEFAULT_SSE_HEARTBEAT_SECONDS = 15.0
DEFAULT_SSE_TIMEOUT_SECONDS = 30 * 60.0
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class StandardEvent(str, Enum):
    STATUS = "status"
    PROGRESS = "progress"
    TOOL = "tool"
    RESULT = "result"
    ERROR = "error"
    DONE = "done"


def _new_stream_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class StreamEvent:
    """SSE event structure."""

    type: str  # progress, result, error, done
    event: Optional[str] = None  # canonical event kind for unified frontend handling
    data: Any = None
    message: Optional[str] = None
    envelope: Optional[Dict[str, Any]] = None

    def to_sse(self) -> str:
        """Convert to SSE format."""
        payload = {
            "type": self.type,
            "event": self.event,
            "data": self.data,
            "message": self.message,
            "envelope": self.envelope,
        }
        return f"data: {json.dumps(payload)}\n\n"


def sse_done() -> str:
    """Return SSE done signal."""
    return "data: [DONE]\n\n"


def sse_comment(comment: str = "keepalive") -> str:
    """Return an SSE comment frame for idle heartbeats."""
    return f": {comment}\n\n"


def _canonical_event_kind(
    *,
    event_type: str,
    data: Any,
    explicit_event: Optional[str],
) -> str:
    if explicit_event:
        return str(explicit_event)

    t = str(event_type or "").strip().lower()
    if t in {"error", "failed", "failure"}:
        return StandardEvent.ERROR.value
    if t in {"result", "final", "final_result"}:
        return StandardEvent.RESULT.value
    if t in {"done", "completed", "complete"}:
        return StandardEvent.DONE.value
    if t == "status":
        return StandardEvent.STATUS.value
    if t.startswith("tool"):
        return StandardEvent.TOOL.value
    if t in {
        "progress",
        "search_done",
        "report_built",
        "llm_summary",
        "llm_done",
        "trend",
        "insight",
        "judge",
        "judge_done",
        "filter_done",
    }:
        return StandardEvent.PROGRESS.value

    if isinstance(data, dict):
        if any(k in data for k in ("phase", "delta", "done", "total")):
            return StandardEvent.PROGRESS.value
    return StandardEvent.STATUS.value


def _with_envelope(
    event: StreamEvent,
    *,
    workflow: str,
    run_id: str,
    trace_id: str,
    seq: int,
) -> StreamEvent:
    canonical_event = _canonical_event_kind(
        event_type=event.type,
        data=event.data,
        explicit_event=event.event,
    )
    event.event = canonical_event

    if event.envelope:
        if "event" not in event.envelope:
            event.envelope["event"] = canonical_event
        return event

    phase = None
    if isinstance(event.data, dict):
        phase = event.data.get("phase")

    event.envelope = {
        "workflow": workflow or "unknown",
        "run_id": run_id,
        "trace_id": trace_id,
        "seq": seq,
        "phase": phase,
        "event": canonical_event,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    return event


async def wrap_generator(
    generator: AsyncGenerator[StreamEvent, None],
    *,
    workflow: str = "",
    run_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    heartbeat_seconds: Optional[float] = DEFAULT_SSE_HEARTBEAT_SECONDS,
    timeout_seconds: Optional[float] = DEFAULT_SSE_TIMEOUT_SECONDS,
) -> AsyncGenerator[str, None]:
    """Wrap a StreamEvent generator to SSE strings with a normalized envelope."""
    resolved_run_id = run_id or _new_stream_id("run")
    resolved_trace_id = trace_id or _new_stream_id("trace")
    seq = 0
    started_at = time.monotonic()
    pending_next: Optional[asyncio.Task[StreamEvent]] = asyncio.create_task(generator.__anext__())

    try:
        while True:
            wait_timeout = heartbeat_seconds
            if timeout_seconds is not None:
                remaining = timeout_seconds - (time.monotonic() - started_at)
                if remaining <= 0:
                    raise asyncio.TimeoutError
                if wait_timeout is None or remaining < wait_timeout:
                    wait_timeout = remaining

            try:
                if wait_timeout is None:
                    event = await pending_next
                else:
                    event = await asyncio.wait_for(asyncio.shield(pending_next), timeout=wait_timeout)
            except StopAsyncIteration:
                yield sse_done()
                return
            except asyncio.TimeoutError:
                if timeout_seconds is not None and (time.monotonic() - started_at) >= timeout_seconds:
                    if pending_next is not None:
                        pending_next.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await pending_next
                    raise
                yield sse_comment()
                continue

            seq += 1
            yield _with_envelope(
                event,
                workflow=workflow,
                run_id=resolved_run_id,
                trace_id=resolved_trace_id,
                seq=seq,
            ).to_sse()
            pending_next = asyncio.create_task(generator.__anext__())
    except asyncio.CancelledError:
        return
    except asyncio.TimeoutError:
        seq += 1
        yield _with_envelope(
            StreamEvent(type="error", message="stream timed out"),
            workflow=workflow,
            run_id=resolved_run_id,
            trace_id=resolved_trace_id,
            seq=seq,
        ).to_sse()
        yield sse_done()
    except Exception as e:
        seq += 1
        yield _with_envelope(
            StreamEvent(type="error", message=str(e)),
            workflow=workflow,
            run_id=resolved_run_id,
            trace_id=resolved_trace_id,
            seq=seq,
        ).to_sse()
        yield sse_done()
    finally:
        if pending_next is not None and not pending_next.done():
            pending_next.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending_next
        with contextlib.suppress(Exception):
            await generator.aclose()


def sse_response(
    generator: AsyncGenerator[StreamEvent, None],
    *,
    workflow: str = "",
    run_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    heartbeat_seconds: Optional[float] = DEFAULT_SSE_HEARTBEAT_SECONDS,
    timeout_seconds: Optional[float] = DEFAULT_SSE_TIMEOUT_SECONDS,
    headers: Optional[Dict[str, str]] = None,
) -> StreamingResponse:
    merged_headers = dict(SSE_HEADERS)
    if headers:
        merged_headers.update(headers)
    return StreamingResponse(
        wrap_generator(
            generator,
            workflow=workflow,
            run_id=run_id,
            trace_id=trace_id,
            heartbeat_seconds=heartbeat_seconds,
            timeout_seconds=timeout_seconds,
        ),
        media_type="text/event-stream",
        headers=merged_headers,
    )
