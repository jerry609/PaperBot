"""
EventBusEventLog — asyncio fan-out ring buffer backend for SSE delivery.

Purpose (EVNT-04):
    Intercepts every event_log.append() call and delivers the event to all
    connected SSE client queues via a non-blocking put_nowait() fan-out.
    Plan 07-02 wires this into the FastAPI /api/events SSE endpoint.

Design decisions (locked in 07-CONTEXT.md):
    - Ring buffer:   collections.deque(maxlen=ring_buffer_size) — default 200
    - Client queue:  asyncio.Queue(maxsize=client_queue_size) — default 256
    - Backpressure:  drop-oldest (get_nowait then put_nowait) — NEVER block producer
    - No filtering:  all events go to all subscriber queues
    - Serialisation: AgentEventEnvelope serialized via .to_dict() once in append();
                     fan-out distributes the already-serialised dict

Thread-safety note:
    append() is called from the async event loop only (uvicorn single-process).
    put_nowait() is safe; no thread bridging needed for current architecture.
    _fan_out() uses list(self._queues) snapshot to guard against concurrent
    unsubscribe() calls inside the same event-loop tick.
"""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from typing import Iterable, Set, Union

from paperbot.application.collaboration.message_schema import AgentEventEnvelope


class EventBusEventLog:
    """
    In-process SSE fan-out backend implementing EventLogPort.

    Usage (Plan 07-02 will wire this into the DI container)::

        bus = EventBusEventLog()
        composite = CompositeEventLog([existing_backend, bus])

        # SSE handler
        q = bus.subscribe()
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\\n\\n"
        finally:
            bus.unsubscribe(q)
    """

    def __init__(
        self,
        *,
        ring_buffer_size: int = 200,
        client_queue_size: int = 256,
    ) -> None:
        self._ring: deque[dict] = deque(maxlen=ring_buffer_size)
        self._client_queue_size = client_queue_size
        self._queues: Set[asyncio.Queue] = set()

    # ------------------------------------------------------------------
    # EventLogPort interface
    # ------------------------------------------------------------------

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        """
        Serialize the event once, store in ring buffer, and fan out to all queues.

        This is a synchronous def — no await, no blocking.
        """
        if isinstance(event, AgentEventEnvelope):
            data = event.to_dict()
        else:
            # Already a dict; snapshot it so caller-side mutations cannot leak into the bus.
            data = deepcopy(event)

        # Store in ring buffer (oldest auto-evicted when deque is full)
        self._ring.append(deepcopy(data))

        # Fan out to all subscriber queues (non-blocking)
        self._fan_out(data)

    def stream(self, run_id: str) -> Iterable[dict]:
        """
        Bus does not support run_id-based historical replay.

        Returns an empty iterator to satisfy the EventLogPort protocol.
        Use subscribe() / unsubscribe() for real-time delivery.
        """
        return iter(())

    def close(self) -> None:
        """Disconnect all subscribers (drop references, let queues GC)."""
        self._queues.clear()

    # ------------------------------------------------------------------
    # Fan-out helpers (public so tests can inspect _queues)
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        """
        Register a new subscriber queue and pre-load it with ring buffer contents.

        The catch-up burst lets SSE clients see recent events immediately on connect.
        Returns the asyncio.Queue; the caller is responsible for calling unsubscribe()
        when the connection closes.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._client_queue_size)
        self._queues.add(q)

        # Register first, then replay a snapshot. This favors duplicate delivery
        # over silent gaps if an append lands during subscribe().
        ring_snapshot = list(self._ring)
        for event in ring_snapshot:
            self._put_nowait_drop_oldest(q, deepcopy(event))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove queue from fan-out set (idempotent)."""
        self._queues.discard(q)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fan_out(self, data: dict) -> None:
        """
        Deliver data to all subscriber queues using a snapshot iteration.

        Snapshot (list(...)) protects against concurrent unsubscribe() calls
        that modify _queues while we iterate.
        """
        for q in list(self._queues):
            self._put_nowait_drop_oldest(q, deepcopy(data))

    @staticmethod
    def _put_nowait_drop_oldest(q: asyncio.Queue, data: dict) -> None:
        """
        Non-blocking put with drop-oldest backpressure.

        If the queue is full, evict the oldest item then insert the new one.
        This guarantees the producer never blocks and the client always sees
        the most recent events.
        """
        if q.full():
            try:
                q.get_nowait()  # Discard oldest
            except asyncio.QueueEmpty:
                pass  # Race is harmless — queue emptied between full() and get_nowait()
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            # Extremely unlikely race; silently drop to protect producer
            pass
