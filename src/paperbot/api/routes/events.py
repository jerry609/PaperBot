"""
SSE fan-out endpoint for the EventBus.

GET /api/events/stream streams every event_log.append() call to all connected
SSE clients without any additional envelope wrapping — events already carry their
own AgentEventEnvelope fields (run_id, trace_id, workflow, etc.).

Design:
- _get_bus()         — locates the EventBusEventLog inside app.state.event_log
- _event_generator() — subscribe → drain queue with heartbeat → unsubscribe (finally)
- events_stream()    — returns StreamingResponse using the generator

Anti-patterns avoided (per 07-RESEARCH.md):
- No wrap_generator(): would add a second envelope layer around events that already
  carry AgentEventEnvelope fields (run_id, trace_id, seq, etc.).
- No sse_response(): same concern.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from paperbot.api.streaming import SSE_HEADERS, sse_comment

router = APIRouter(prefix="/events")

_HEARTBEAT_SECONDS = 15.0


def _get_bus(request: Request):
    """
    Locate the EventBusEventLog backend inside app.state.event_log.

    The import is done inside the function to avoid a circular import at
    module load time (events.py loaded during app creation, before event_log
    infrastructure is wired).

    Raises RuntimeError if the bus is not registered (misconfigured startup).
    """
    from paperbot.infrastructure.event_log.event_bus_event_log import EventBusEventLog  # noqa: PLC0415

    event_log = request.app.state.event_log
    backends = getattr(event_log, "_backends", None)
    if backends is not None:
        for backend in backends:
            if isinstance(backend, EventBusEventLog):
                return backend

    # event_log itself might be the bus (useful in tests)
    if isinstance(event_log, EventBusEventLog):
        return event_log

    raise RuntimeError("EventBusEventLog not registered in CompositeEventLog")


async def _event_generator(request: Request, bus):
    """
    Async generator: subscribe → yield events / heartbeats → unsubscribe.

    The try/finally guarantees bus.unsubscribe(q) is called even if the
    client disconnects mid-stream (ASGI sends CancelledError) or if the
    generator is garbage-collected.
    """
    q = bus.subscribe()
    try:
        while True:
            # Fast disconnect check — avoids waiting a full heartbeat cycle after drop
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_SECONDS)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                # No event arrived within the heartbeat window — send a keepalive comment
                yield sse_comment()
            except asyncio.CancelledError:
                # ASGI framework signals client disconnect via CancelledError
                break
    finally:
        bus.unsubscribe(q)


@router.get("/stream")
async def events_stream(request: Request) -> StreamingResponse:
    """
    Stream all global events to the SSE client.

    Each event is sent as a plain ``data: {...}\\n\\n`` frame.
    When the queue is idle for 15 seconds a ``: keepalive\\n\\n`` comment is sent.
    """
    bus = _get_bus(request)
    return StreamingResponse(
        _event_generator(request, bus),
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
    )
