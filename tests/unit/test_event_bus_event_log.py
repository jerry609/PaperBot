"""
Unit tests for EventBusEventLog — asyncio fan-out ring buffer backend.

RED phase: These tests are written before the implementation file exists.
Expected to fail with ImportError / ModuleNotFoundError until Task 2 creates
src/paperbot/infrastructure/event_log/event_bus_event_log.py.

asyncio_mode = "strict" in pyproject.toml — every async test must carry
@pytest.mark.asyncio explicitly.
"""
from __future__ import annotations

import asyncio

import pytest

from paperbot.infrastructure.event_log.event_bus_event_log import EventBusEventLog
from paperbot.infrastructure.event_log.composite_event_log import CompositeEventLog
from paperbot.application.collaboration.message_schema import (
    AgentEventEnvelope,
    new_run_id,
    new_trace_id,
)


def _make_envelope(payload: dict | None = None) -> AgentEventEnvelope:
    return AgentEventEnvelope(
        run_id=new_run_id(),
        trace_id=new_trace_id(),
        workflow="test_workflow",
        stage="test_stage",
        attempt=0,
        agent_name="test_agent",
        role="worker",
        type="test_event",
        payload=payload or {"key": "value"},
    )


@pytest.mark.asyncio
async def test_fan_out_to_multiple_subscribers():
    """append(event) fans out to all registered subscriber queues."""
    bus = EventBusEventLog()

    q1: asyncio.Queue = bus.subscribe()
    q2: asyncio.Queue = bus.subscribe()

    event = _make_envelope()
    bus.append(event)

    # Both queues must have received the event
    assert not q1.empty(), "q1 did not receive the fanned-out event"
    assert not q2.empty(), "q2 did not receive the fanned-out event"

    item1 = q1.get_nowait()
    item2 = q2.get_nowait()

    # Both should be the serialized dict form
    assert isinstance(item1, dict)
    assert isinstance(item2, dict)
    assert item1["type"] == "test_event"
    assert item2["type"] == "test_event"


@pytest.mark.asyncio
async def test_ring_buffer_catch_up():
    """New subscriber receives ring buffer contents as catch-up burst on subscribe()."""
    bus = EventBusEventLog()

    # Append 3 events BEFORE any subscriber
    for i in range(3):
        bus.append(_make_envelope({"index": i}))

    # Now subscribe — should receive catch-up burst of those 3 events
    q: asyncio.Queue = bus.subscribe()

    assert q.qsize() == 3, f"Expected 3 catch-up events, got {q.qsize()}"

    items = [q.get_nowait() for _ in range(3)]
    indexes = [item["payload"]["index"] for item in items]
    assert indexes == [0, 1, 2], f"Expected indexes [0,1,2], got {indexes}"


@pytest.mark.asyncio
async def test_backpressure_drops_oldest():
    """Full client queue drops oldest event on overflow — producer never blocks."""
    bus = EventBusEventLog(client_queue_size=2)
    q: asyncio.Queue = bus.subscribe()

    # Fill queue to maxsize=2
    bus.append(_make_envelope({"seq": 0}))
    bus.append(_make_envelope({"seq": 1}))
    assert q.qsize() == 2

    # Append one more — should drop oldest (seq=0), keep seq=1 and seq=2
    bus.append(_make_envelope({"seq": 2}))

    assert q.qsize() == 2, f"Queue should still have 2 items, got {q.qsize()}"

    items = [q.get_nowait() for _ in range(2)]
    seqs = [item["payload"]["seq"] for item in items]

    # Oldest (seq=0) must be gone; newest (seq=2) must be present
    assert 0 not in seqs, f"Oldest item (seq=0) should have been dropped, got seqs={seqs}"
    assert 2 in seqs, f"Newest item (seq=2) should be present, got seqs={seqs}"


@pytest.mark.asyncio
async def test_unsubscribe_cleans_up():
    """unsubscribe(q) removes queue from fan-out set; append after unsubscribe delivers nothing."""
    bus = EventBusEventLog()
    q: asyncio.Queue = bus.subscribe()

    bus.unsubscribe(q)

    # After unsubscribe, internal set should be empty
    assert len(bus._queues) == 0, f"Expected 0 queues after unsubscribe, got {len(bus._queues)}"

    # Appending should not put anything in q
    bus.append(_make_envelope())
    assert q.empty(), "Queue should be empty after unsubscribe — no fan-out expected"


@pytest.mark.asyncio
async def test_composite_includes_bus():
    """CompositeEventLog delegates append() to EventBusEventLog backend."""
    bus = EventBusEventLog()
    composite = CompositeEventLog([bus])

    event = _make_envelope()
    composite.append(event)

    # The bus ring buffer should contain the event
    assert len(bus._ring) == 1, f"Expected 1 item in ring buffer, got {len(bus._ring)}"

    ring_item = list(bus._ring)[0]
    assert isinstance(ring_item, dict)
    assert ring_item["type"] == "test_event"
