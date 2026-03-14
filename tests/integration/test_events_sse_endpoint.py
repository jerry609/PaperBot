"""
Integration tests for the /api/events/stream SSE endpoint.

These tests exercise:
  1. Event delivery latency — event appended to EventBusEventLog arrives in queue < 1s.
  2. Heartbeat on idle — generator yields a keepalive comment when no events arrive
     within the heartbeat window.

Strategy (per plan 07-02 guidance):
  - test_event_delivered_within_1s: calls _startup_eventlog() manually, then exercises
    the EventBusEventLog subscribe/append/get path directly.  This validates the full
    round-trip from event_log.append() to the subscriber queue without needing an HTTP
    server.  The HTTP layer (StreamingResponse) is tested in e2e / manual UAT.
  - test_heartbeat_on_idle: instantiates _event_generator() directly with a mock
    request, drives it with a very short heartbeat timeout, and confirms the keepalive
    comment is yielded before any data event.

asyncio_mode = "strict" (pyproject.toml) — every async test needs @pytest.mark.asyncio.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from paperbot.infrastructure.event_log.event_bus_event_log import EventBusEventLog
from paperbot.infrastructure.event_log.composite_event_log import CompositeEventLog
from paperbot.infrastructure.event_log.logging_event_log import LoggingEventLog
from paperbot.api.routes.events import _event_generator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mock_request(disconnected: bool = False):
    """
    Build a minimal async mock of starlette.requests.Request.

    Only `is_disconnected()` is needed by _event_generator.
    """
    req = types.SimpleNamespace()

    async def is_disconnected():
        return disconnected

    req.is_disconnected = is_disconnected
    return req


# ---------------------------------------------------------------------------
# test_event_delivered_within_1s
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
async def test_event_delivered_within_1s():
    """
    Event appended to the bus reaches a subscriber queue within 1 second.

    Validates the core delivery guarantee for EVNT-04 without a live HTTP server.
    The test:
      1. Creates an EventBusEventLog and wraps it in CompositeEventLog (as main.py does).
      2. Subscribes a queue directly.
      3. Appends a test event via composite.append().
      4. Awaits the queue item with a 1-second budget.
      5. Asserts the item contains the original run_id.
    """
    bus = EventBusEventLog()
    composite = CompositeEventLog([LoggingEventLog(), bus])

    q = bus.subscribe()

    test_event = {"run_id": "test-run-123", "type": "test_event", "payload": "hello"}
    composite.append(test_event)

    item = await asyncio.wait_for(q.get(), timeout=1.0)
    assert item["run_id"] == "test-run-123"
    assert item["type"] == "test_event"

    # Cleanup
    bus.unsubscribe(q)
    assert q not in bus._queues, "unsubscribe must remove queue from fan-out set"


# ---------------------------------------------------------------------------
# test_heartbeat_on_idle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
async def test_heartbeat_on_idle():
    """
    _event_generator emits a keepalive SSE comment when no events arrive
    within the heartbeat window.

    We patch _HEARTBEAT_SECONDS to 0.05 s (50 ms) for speed then drive the
    generator one iteration — it should yield the ``: keepalive\\n\\n`` comment.
    """
    import paperbot.api.routes.events as events_module

    original_heartbeat = events_module._HEARTBEAT_SECONDS
    events_module._HEARTBEAT_SECONDS = 0.05  # speed up the test

    try:
        bus = EventBusEventLog()
        mock_req = _make_mock_request(disconnected=False)

        gen = _event_generator(mock_req, bus)

        # The generator should yield a heartbeat comment (no events were appended)
        frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)

        assert frame == ": keepalive\n\n", (
            f"Expected keepalive comment, got: {frame!r}"
        )
    finally:
        events_module._HEARTBEAT_SECONDS = original_heartbeat
        # Close the generator so unsubscribe() runs
        await gen.aclose()
