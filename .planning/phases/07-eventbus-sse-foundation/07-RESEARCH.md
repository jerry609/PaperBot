# Phase 7: EventBus + SSE Foundation - Research

**Researched:** 2026-03-14
**Domain:** Python asyncio in-process event fan-out + FastAPI SSE endpoint
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Event filtering:**
- No server-side filtering — all events go to all connected clients
- Dashboard filters client-side (Zustand store in Phase 9)
- Rationale: single-user tool, low event volume (dozens/min), allows cross-workflow views without reconnecting

**Reconnection behavior:**
- Small in-memory ring buffer (last ~200 events) kept by the event bus
- On connect, client receives buffer contents as a catch-up burst, then switches to live streaming
- No Last-Event-ID support — unnecessary complexity for a dashboard
- If client was away longer than the buffer, old events are simply missed (acceptable)

**Backpressure handling:**
- Each SSE client gets an `asyncio.Queue` with a fixed max size (~256)
- When queue is full, drop the oldest event and enqueue the new one
- Never block the producer — `event_log.append()` must stay fast (called from agent hot paths)
- Never disconnect slow clients — just drop their oldest queued events

**SSE endpoint design:**
- Single new endpoint: `GET /api/events/stream`
- No query params (all events to all clients)
- Existing per-feature SSE endpoints (agent_board, gen_code, track) stay untouched
- This endpoint is specifically for the event bus fan-out to the dashboard

**Integration architecture:**
- EventBus plugs in as a new backend in CompositeEventLog
- Zero changes to existing code that calls `event_log.append()`
- Zero new dependencies — uses asyncio.Queue, collections.deque, existing streaming.py

### Claude's Discretion

- Ring buffer implementation details (collections.deque vs list slice)
- Exact queue size tuning (200 buffer, 256 per-client are guidelines)
- Internal event serialization format within the bus
- SSE event `id` field format (sequential int, UUID, etc.)
- Heartbeat interval for the new endpoint

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVNT-04 | Agent events are pushed to connected dashboard clients in real-time via SSE (no polling) | EventBus as CompositeEventLog backend; asyncio.Queue fan-out per client; `GET /api/events/stream` SSE endpoint |
</phase_requirements>

---

## Summary

Phase 7 builds an in-process event bus that intercepts every `event_log.append()` call and fan-outs the event to all connected SSE clients. The mechanism is a new `EventBusEventLog` class that implements `EventLogPort` and is inserted into `CompositeEventLog`'s backend list at startup. No caller of `append()` ever changes. The event bus holds a `collections.deque` ring buffer for catch-up on connect, and a set of per-client `asyncio.Queue` instances for live delivery.

The single new API endpoint `GET /api/events/stream` is a long-lived SSE stream that yields the catch-up burst then blocks on the client's queue. FastAPI's `StreamingResponse` with the existing `SSE_HEADERS` and `sse_comment()` heartbeat pattern is reused verbatim from `api/streaming.py`. The endpoint uses `asyncio.Queue.get()` with a timeout loop to interleave heartbeats and events.

The central correctness concern is **thread/coroutine safety**: `append()` is called synchronously from agent hot-paths (potentially from non-async code), while the SSE generator awaits queue items from the async event loop. The standard Python idiom is `loop.call_soon_threadsafe(queue.put_nowait, item)` if `append()` is called from a thread, or simply `queue.put_nowait()` if always called from the same async context. Because `append()` is defined as `def` (not `async def`) in the port, and because FastAPI runs uvicorn in a single-process async loop, all handlers calling `append()` do so from within the event loop — making `put_nowait` safe without thread bridging.

**Primary recommendation:** Implement `EventBusEventLog` with a `collections.deque(maxlen=200)` ring buffer, a `set` of `asyncio.Queue(maxsize=256)` subscriber queues, drop-oldest backpressure on full queues, and register it as the third backend in `CompositeEventLog` at startup. The SSE generator yields buffer contents on connect, then loops on `asyncio.wait_for(queue.get(), timeout=heartbeat_interval)` with `sse_comment()` on timeout.

---

## Standard Stack

### Core (all already in pyproject.toml — zero new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `asyncio.Queue` | stdlib | Per-client event delivery queue | Async-native, bounded, put_nowait safe from same event loop |
| `collections.deque` | stdlib | Ring buffer for catch-up events | O(1) append/popleft, maxlen enforces ring size |
| `fastapi.StreamingResponse` | 0.115.0 (pinned) | SSE HTTP response | Already used by all 6+ existing SSE endpoints |
| `starlette.responses.StreamingResponse` | >=0.37.2 | Underlying SSE transport | Handles chunked transfer encoding, no buffering |
| `asyncio.wait_for` | stdlib | Heartbeat timeout during queue wait | Standard pattern for interleaving heartbeats |

### Reused Project Utilities (from `api/streaming.py`)

| Asset | What It Provides |
|-------|-----------------|
| `SSE_HEADERS` | `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no` |
| `sse_comment()` | Returns `: keepalive\n\n` — heartbeat frame |
| `StreamEvent.to_sse()` | Serializes event to `data: {...}\n\n` |
| `sse_done()` | Returns `data: [DONE]\n\n` |
| `wrap_generator()` | Adds envelope, heartbeat, timeout — usable for the new endpoint |

**Note:** `wrap_generator()` adds an outer envelope (workflow/run_id/trace_id/seq) designed for workflow-scoped streams. The events endpoint is global fan-out; use a simpler inline generator to avoid injecting misleading envelope fields. Raw `StreamEvent.to_sse()` or a custom serializer that directly emits `AgentEventEnvelope.to_json()` is cleaner for this endpoint.

### Installation

No new packages needed. All primitives are stdlib or already in `pyproject.toml`.

---

## Architecture Patterns

### Recommended File Structure

```
src/paperbot/
├── infrastructure/
│   └── event_log/
│       ├── event_bus_event_log.py   # NEW: EventBusEventLog class
│       └── composite_event_log.py   # unchanged
├── api/
│   ├── routes/
│   │   └── events.py                # NEW: GET /api/events/stream
│   ├── main.py                      # MODIFIED: add bus to CompositeEventLog, register events router
│   └── streaming.py                 # unchanged (reused)
└── ...

tests/
├── unit/
│   └── test_event_bus_event_log.py  # NEW: unit tests for bus
└── integration/
    └── test_events_sse_endpoint.py  # NEW: httpx/TestClient SSE test
```

### Pattern 1: EventBusEventLog — Composite Backend

`EventBusEventLog` implements `EventLogPort` by:
1. Storing each appended event in a `deque(maxlen=200)` ring buffer
2. Fanning out to all registered subscriber queues via `put_nowait()`
3. Dropping the oldest item from full queues before enqueueing the new item

```python
# src/paperbot/infrastructure/event_log/event_bus_event_log.py
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Iterable, Set, Union

from paperbot.application.collaboration.message_schema import AgentEventEnvelope
from paperbot.application.ports.event_log_port import EventLogPort

logger = logging.getLogger(__name__)

_RING_BUFFER_SIZE = 200
_CLIENT_QUEUE_SIZE = 256


class EventBusEventLog(EventLogPort):
    """
    In-process fan-out bus. Plugs into CompositeEventLog as a backend.

    append() is always called from the async event loop (FastAPI/uvicorn
    single-process model), so put_nowait() is safe without thread bridging.
    """

    def __init__(
        self,
        ring_buffer_size: int = _RING_BUFFER_SIZE,
        client_queue_size: int = _CLIENT_QUEUE_SIZE,
    ) -> None:
        self._ring: deque[dict] = deque(maxlen=ring_buffer_size)
        self._queues: Set[asyncio.Queue] = set()
        self._client_queue_size = client_queue_size

    # --- EventLogPort interface ---

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        serialized: dict
        if isinstance(event, AgentEventEnvelope):
            serialized = event.to_dict()
        else:
            serialized = dict(event)
        self._ring.append(serialized)
        self._fan_out(serialized)

    def stream(self, run_id: str) -> Iterable[dict]:
        # Bus does not support historical replay by run_id; that is SQLAlchemy's job.
        return iter(())

    def close(self) -> None:
        self._queues.clear()

    # --- Subscription management ---

    def subscribe(self) -> asyncio.Queue:
        """Return a new per-client queue pre-loaded with ring buffer contents."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._client_queue_size)
        # Drain ring buffer into the new queue (catch-up burst)
        for event in list(self._ring):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                break  # client queue is already saturated by the buffer itself
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._queues.discard(q)

    # --- Internal ---

    def _fan_out(self, event: dict) -> None:
        dead: list = []
        for q in self._queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest, enqueue newest — never block producer
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.debug("EventBus: client queue still full after drop, skipping")
        for q in dead:
            self._queues.discard(q)
```

### Pattern 2: SSE Generator — Queue Drain with Heartbeat

The SSE generator subscribes, yields catch-up events (already in the queue from `subscribe()`), then loops reading live events with a heartbeat timeout.

```python
# src/paperbot/api/routes/events.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from paperbot.api.streaming import SSE_HEADERS, sse_comment, sse_done

router = APIRouter(prefix="/api/events")
log = logging.getLogger(__name__)

_HEARTBEAT_SECONDS = 15.0


@router.get("/stream")
async def events_stream(request: Request) -> StreamingResponse:
    """Fan-out SSE endpoint. All agent events go to all connected clients."""
    bus = _get_bus(request)
    return StreamingResponse(
        _event_generator(request, bus),
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
    )


async def _event_generator(request: Request, bus) -> AsyncGenerator[str, None]:
    q = bus.subscribe()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event: dict = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_SECONDS)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield sse_comment()  # keepalive heartbeat
    except asyncio.CancelledError:
        pass
    finally:
        bus.unsubscribe(q)
        log.debug("EventBus SSE client disconnected, queue cleaned up")


def _get_bus(request: Request):
    # Bus is stored on app.state by main.py startup handler
    event_log = request.app.state.event_log
    # CompositeEventLog exposes backends
    for backend in event_log._backends:
        from paperbot.infrastructure.event_log.event_bus_event_log import EventBusEventLog
        if isinstance(backend, EventBusEventLog):
            return backend
    raise RuntimeError("EventBusEventLog not registered in CompositeEventLog")
```

### Pattern 3: Startup Wiring in main.py

```python
# api/main.py — modify _startup_eventlog()
from paperbot.infrastructure.event_log.event_bus_event_log import EventBusEventLog

@app.on_event("startup")
async def _startup_eventlog():
    try:
        bus = EventBusEventLog()
        app.state.event_log = CompositeEventLog([
            LoggingEventLog(),
            SqlAlchemyEventLog(),
            bus,                  # NEW: fan-out backend
        ])
        app.state.event_bus = bus  # also stored directly for convenience
    except Exception:
        app.state.event_log = LoggingEventLog()
```

And register the new router in `main.py`:

```python
from .routes import events as events_route
app.include_router(events_route.router, prefix="", tags=["Events"])
```

### Anti-Patterns to Avoid

- **Calling `queue.put()` (awaitable) inside `append()`:** `append()` is a sync method; using `await` inside it is impossible. Use `put_nowait()` only.
- **Storing the event loop reference at import time:** `asyncio.get_event_loop()` at module level can return a closed loop after hot reload. Retrieve via `asyncio.get_running_loop()` inside a running coroutine instead, or just use `put_nowait()` which needs no loop reference.
- **Using `wrap_generator()` for this endpoint:** It injects per-workflow envelope fields (run_id, trace_id, seq, workflow). The events endpoint serves global fan-out — events already carry their own envelope via `AgentEventEnvelope.to_dict()`. Adding a second envelope layer confuses consumers.
- **Iterating `self._queues` while modifying it:** Dead/unsubscribed queues should be collected into a `dead` list and discarded after iteration to avoid `RuntimeError: Set changed size during iteration`.
- **Leaking the subscriber queue on disconnect:** FastAPI's `StreamingResponse` generator is cancelled by ASGI middleware when the client drops. The `finally: bus.unsubscribe(q)` block in the generator is the guaranteed cleanup path.
- **Re-serializing inside `_fan_out`:** Convert once in `append()` (to `dict`) and store/fan-out that dict. Do not call `to_json()` inside `_fan_out` — unnecessary allocation per subscriber.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Catch-up replay on connect | Custom DB query in the SSE handler | `deque(maxlen=N)` in EventBusEventLog | DB query adds latency; ring buffer is O(N) copy and already in memory |
| Heartbeat mechanism | Custom sleep loop | `asyncio.wait_for(queue.get(), timeout=N)` + `sse_comment()` | Already proven pattern in streaming.py; handles `TimeoutError` correctly |
| SSE framing | Custom `data:` string builder | `json.dumps(event) + "\n\n"` or `StreamEvent.to_sse()` | One-liner; no edge cases for this simple case |
| Disconnect detection | Polling `request.is_disconnected()` on a timer | ASGI CancelledError on generator + `finally` cleanup | FastAPI cancels the generator; `finally` block is guaranteed |
| Queue bounded drop | Manual size check + conditional | `queue.get_nowait()` then `put_nowait()` in QueueFull handler | Built-in QueueFull exception is the correct hook |

**Key insight:** asyncio.Queue with maxsize is the correct bounded-buffer primitive. Anything hand-rolled around a list + asyncio.Event has subtle race conditions in an async context. Use the queue.

---

## Common Pitfalls

### Pitfall 1: Sync `append()` calling async queue methods

**What goes wrong:** Developer writes `await q.put(event)` inside `append()`, causing a `RuntimeError: no running event loop` or `SyntaxError` because `append()` is a plain `def`.

**Why it happens:** The `EventLogPort.append()` protocol is synchronous (returns `None`, not a coroutine). Changing it to `async def` would break all existing callers.

**How to avoid:** Always use `put_nowait()` in `append()`. If the queue is full, handle `QueueFull` synchronously by dropping oldest.

**Warning signs:** Any `await` inside `append()` or `_fan_out()`.

### Pitfall 2: Generator not cleaned up on client disconnect

**What goes wrong:** Client disconnects, but the generator coroutine is left suspended at `await q.get()`, and the queue stays registered in `self._queues` forever. Memory and set size grow with every connection.

**Why it happens:** Missing `finally` block in the generator, or `unsubscribe()` only called on `StopAsyncIteration` (which never fires for a long-lived SSE stream).

**How to avoid:** Always wrap the `await q.get()` loop in `try/finally: bus.unsubscribe(q)`. FastAPI sends `CancelledError` to the generator on disconnect; the `finally` runs.

**Warning signs:** `len(bus._queues)` grows monotonically. Unit test: connect + disconnect + assert `len(bus._queues) == 0`.

### Pitfall 3: Ring buffer not thread-safe with non-async callers

**What goes wrong:** `deque.append()` is thread-safe in CPython (GIL), but if `append()` is ever called from a thread (e.g., ARQ worker thread), `put_nowait()` on an asyncio.Queue that belongs to a different event loop will raise `RuntimeError`.

**Why it happens:** asyncio.Queue is bound to the event loop it was created in. If called from a thread, `put_nowait()` is unsafe.

**How to avoid:** In the current architecture, `append()` is called exclusively from FastAPI request handlers that run in uvicorn's single async event loop. Document this assumption. If ARQ workers ever call `append()` directly in the future, use `loop.call_soon_threadsafe(q.put_nowait, event)`.

**Warning signs:** `RuntimeError: Event loop is closed` or `got Future attached to a different loop`.

### Pitfall 4: Bus referenced before startup completes

**What goes wrong:** A test or early request calls `_get_bus(request)` before `_startup_eventlog()` has run, causing `AttributeError: 'State' object has no attribute 'event_log'`.

**Why it happens:** FastAPI `app.state` is empty until startup hooks fire. Tests using `TestClient` may not trigger startup unless using the context manager form (`with TestClient(app) as client:`).

**How to avoid:** Always use `with TestClient(app) as client:` in tests (triggers lifespan). In `_get_bus`, add a fallback or raise a clear error if `event_log` is missing.

**Warning signs:** `AttributeError` in tests that don't use `with TestClient(app) as client:`.

### Pitfall 5: `_queues` set mutation during `_fan_out` iteration

**What goes wrong:** `_fan_out` iterates `self._queues` while a concurrent coroutine calls `unsubscribe()`, raising `RuntimeError: Set changed size during iteration`.

**Why it happens:** Both operations occur in the async event loop, but `asyncio` tasks can interleave at `await` points. However, since `_fan_out` has no `await` calls it is effectively atomic — this is safe. The concern applies only if `unsubscribe()` is called from a different thread.

**How to avoid:** Since everything runs in the same event loop thread, the iteration is safe. Document the assumption. Alternatively, iterate a snapshot: `for q in list(self._queues):`.

**Warning signs:** `RuntimeError: Set changed size during iteration` in logs.

---

## Code Examples

### Full EventBusEventLog — correct backpressure

```python
# Source: architecture pattern derived from Python asyncio documentation
# https://docs.python.org/3/library/asyncio-queue.html

def _fan_out(self, event: dict) -> None:
    for q in list(self._queues):   # snapshot to avoid mutation-during-iteration
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Drop-oldest strategy: discard head, insert new tail
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("EventBus: client queue still full after drop, skipping")
```

### SSE generator — correct disconnect cleanup

```python
# Source: FastAPI StreamingResponse CancelledError pattern
# https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse

async def _event_generator(request: Request, bus: EventBusEventLog):
    q = bus.subscribe()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_SECONDS)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield sse_comment()
    except asyncio.CancelledError:
        pass   # ASGI cancelled us on disconnect
    finally:
        bus.unsubscribe(q)  # always runs — no queue leak
```

### Wiring CompositeEventLog with EventBus at startup

```python
# Source: existing pattern in api/main.py _startup_eventlog()

@app.on_event("startup")
async def _startup_eventlog():
    try:
        bus = EventBusEventLog()
        app.state.event_log = CompositeEventLog([
            LoggingEventLog(),
            SqlAlchemyEventLog(),
            bus,
        ])
        app.state.event_bus = bus
    except Exception:
        app.state.event_log = LoggingEventLog()
```

### Test pattern — multiple clients receive independent events

```python
# Pattern: use asyncio directly (not TestClient) to test async SSE generator
import asyncio
from paperbot.infrastructure.event_log.event_bus_event_log import EventBusEventLog
from paperbot.application.collaboration.message_schema import make_event, new_run_id, new_trace_id

@pytest.mark.asyncio
async def test_multiple_clients_receive_independent_events():
    bus = EventBusEventLog()
    q1 = bus.subscribe()
    q2 = bus.subscribe()

    run_id = new_run_id()
    evt = make_event(
        run_id=run_id, trace_id=new_trace_id(),
        workflow="test", stage="s1", attempt=0,
        agent_name="A", role="worker", type="score_update",
    )
    bus.append(evt)

    item1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    item2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert item1["run_id"] == run_id
    assert item2["run_id"] == run_id
    assert item1 is not item2  # independent queue items (same dict value, different refs OK)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Polling `/api/runs` for new events | SSE push via `GET /api/events/stream` | Phase 7 | Eliminates polling latency; sub-1-second delivery |
| LoggingEventLog + SqlAlchemyEventLog only | CompositeEventLog with added EventBusEventLog | Phase 7 | Zero changes to callers; bus is transparent |
| Per-workflow SSE endpoints (track, analyze, etc.) | Global event bus fan-out as additional channel | Phase 7 | Existing endpoints unchanged; bus is additive |

**Deprecated/outdated:**
- Polling-based dashboard updates: replaced by the new SSE endpoint.

---

## Open Questions

1. **`append()` called from ARQ worker thread?**
   - What we know: ARQ workers run in a separate process/thread; they currently use their own event log instance (not the in-process bus)
   - What's unclear: If an ARQ worker ever calls the shared `app.state.event_log`, the bus would be invoked from a thread
   - Recommendation: Verify ARQ workers use their own isolated event log (not app.state). If they ever share it in the future, use `loop.call_soon_threadsafe()` in `_fan_out`. For Phase 7, document the single-loop assumption.

2. **`request.is_disconnected()` polling overhead?**
   - What we know: FastAPI's `is_disconnected()` sends a receive call to the ASGI scope; it is O(1) but has some overhead
   - What's unclear: Whether polling it on every heartbeat cycle (every 15 seconds) is sufficient or too slow
   - Recommendation: The 15-second heartbeat loop means `is_disconnected()` is polled at most once per 15 seconds. This is fine. The `CancelledError` path handles immediate disconnects during `await q.get()`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio 0.21+ |
| Config file | `pyproject.toml` — `asyncio_mode = "strict"` |
| Quick run command | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py -q` |
| Full suite command | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py tests/integration/test_events_sse_endpoint.py -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVNT-04 | append() fans out to all subscriber queues | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_fan_out_to_multiple_subscribers -x` | Wave 0 |
| EVNT-04 | Ring buffer replayed to new subscriber | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_ring_buffer_catch_up -x` | Wave 0 |
| EVNT-04 | Full queue drops oldest, never blocks | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_backpressure_drops_oldest -x` | Wave 0 |
| EVNT-04 | Unsubscribe removes queue (no leak) | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_unsubscribe_cleans_up -x` | Wave 0 |
| EVNT-04 | SSE endpoint delivers events within 1 second | integration | `PYTHONPATH=src pytest tests/integration/test_events_sse_endpoint.py::test_event_delivered_within_1s -x` | Wave 0 |
| EVNT-04 | SSE endpoint sends heartbeat comment | integration | `PYTHONPATH=src pytest tests/integration/test_events_sse_endpoint.py::test_heartbeat_on_idle -x` | Wave 0 |
| EVNT-04 | CompositeEventLog wires bus as backend | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_composite_includes_bus -x` | Wave 0 |

**Note:** EVNT-04 success criterion 1 ("within 1 second") is tested via asyncio timing in integration tests using `asyncio.wait_for()` with a 1-second budget. Success criterion 4 ("clean up on disconnect") is tested via queue size assertion after generator `finally` block.

### Sampling Rate

- **Per task commit:** `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py -q`
- **Per wave merge:** `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py tests/integration/test_events_sse_endpoint.py -q`
- **Phase gate:** Full CI suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/test_event_bus_event_log.py` — covers all unit cases above (fan-out, ring buffer, backpressure, unsubscribe, composite wiring)
- [ ] `tests/integration/test_events_sse_endpoint.py` — covers SSE delivery latency and heartbeat; uses `TestClient` with `with` form to trigger startup

*(No framework install needed — pytest-asyncio already present in `[dev]` extras)*

---

## Sources

### Primary (HIGH confidence)

- Codebase direct read: `src/paperbot/infrastructure/event_log/` — all 5 event log files read in full
- Codebase direct read: `src/paperbot/api/streaming.py` — SSE utilities, heartbeat pattern, headers
- Codebase direct read: `src/paperbot/application/ports/event_log_port.py` — Protocol definition
- Codebase direct read: `src/paperbot/application/collaboration/message_schema.py` — AgentEventEnvelope
- Codebase direct read: `src/paperbot/infrastructure/event_log/composite_event_log.py` — backend fan-out pattern
- Codebase direct read: `src/paperbot/api/main.py` — startup hook, CompositeEventLog wiring
- Codebase direct read: `src/paperbot/core/di/container.py` — DI pattern
- Codebase direct read: `pyproject.toml` — asyncio_mode=strict, dependency versions
- Codebase direct read: `tests/e2e/test_api_track_fullstack_offline.py` — existing SSE test pattern
- Python docs: `asyncio.Queue` — `put_nowait`, `QueueFull`, `QueueEmpty` behavior (stdlib, no staleness risk)
- Python docs: `collections.deque(maxlen=N)` — ring buffer behavior (stdlib)

### Secondary (MEDIUM confidence)

- FastAPI docs pattern: `StreamingResponse` + `CancelledError` for SSE disconnect — consistent with `streaming.py` implementation observed in codebase
- uvicorn single-process async model — confirms `put_nowait()` is safe without thread bridging for FastAPI handlers

### Tertiary (LOW confidence)

- None — all research based on direct codebase inspection + stdlib documentation

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all libraries are stdlib or pinned in pyproject.toml; read directly
- Architecture: HIGH — derived from direct inspection of 10+ existing source files in the repo; patterns match exactly
- Pitfalls: HIGH for asyncio pitfalls (well-known); MEDIUM for thread-safety edge case (ARQ interaction — theoretical, not observed)

**Research date:** 2026-03-14
**Valid until:** 2026-09-14 (stable stdlib primitives; FastAPI 0.115.0 is pinned; re-verify if FastAPI is upgraded)
