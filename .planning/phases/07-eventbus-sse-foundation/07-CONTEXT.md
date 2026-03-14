# Phase 7: EventBus + SSE Foundation - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

In-process event bus with SSE subscription endpoint for real-time push. Agent events are pushed to connected clients in real-time without polling. Existing `event_log.append()` calls automatically push to SSE subscribers with zero changes to calling code.

Success criteria:
1. Dashboard client connected via SSE receives agent events within 1 second of emission
2. Multiple simultaneous SSE clients each receive all events independently
3. Existing event_log.append() calls automatically push to SSE subscribers with zero changes to calling code
4. SSE connections clean up gracefully on client disconnect (no leaked queues or background tasks)

</domain>

<decisions>
## Implementation Decisions

### Event filtering
- No server-side filtering — all events go to all connected clients
- Dashboard filters client-side (Zustand store in Phase 9)
- Rationale: single-user tool, low event volume (dozens/min), allows cross-workflow views without reconnecting

### Reconnection behavior
- Small in-memory ring buffer (last ~200 events) kept by the event bus
- On connect, client receives buffer contents as a catch-up burst, then switches to live streaming
- No Last-Event-ID support — unnecessary complexity for a dashboard
- If client was away longer than the buffer, old events are simply missed (acceptable for real-time dashboard, not a message queue)

### Backpressure handling
- Each SSE client gets an `asyncio.Queue` with a fixed max size (~256)
- When queue is full, drop the oldest event and enqueue the new one
- Never block the producer — `event_log.append()` must stay fast (called from agent hot paths)
- Never disconnect slow clients — just drop their oldest queued events

### SSE endpoint design
- Single new endpoint: `GET /api/events/stream`
- No query params needed (all events to all clients)
- Existing per-feature SSE endpoints (agent_board, gen_code, track) stay untouched — they serve different purposes (streaming workflow results)
- This endpoint is specifically for the event bus fan-out to the dashboard

### Claude's Discretion
- Ring buffer implementation details (collections.deque vs list slice)
- Exact queue size tuning (200 buffer, 256 per-client are guidelines)
- Internal event serialization format within the bus
- SSE event `id` field format (sequential int, UUID, etc.)
- Heartbeat interval for the new endpoint

</decisions>

<specifics>
## Specific Ideas

- EventBus plugs in as a new backend in CompositeEventLog — when `append()` is called, the event fans out to all registered SSE subscriber queues automatically
- Zero changes to existing code that calls `event_log.append()` — the bus is transparent
- Zero new dependencies — uses asyncio.Queue, collections.deque, existing streaming.py infrastructure

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EventLogPort` (application/ports/event_log_port.py): Protocol with `append()`, `stream()`, `close()` — new bus implements this
- `CompositeEventLog` (infrastructure/event_log/composite_event_log.py): Tees to multiple backends — bus plugs in as another backend
- `AgentEventEnvelope` (application/collaboration/message_schema.py): Rich event envelope with run_id, trace_id, span_id, workflow, stage, agent_name, type, payload — already JSON-serializable via `to_dict()`/`to_json()`
- `api/streaming.py`: `sse_response()`, `wrap_generator()`, `StreamEvent`, heartbeat, timeout, SSE_HEADERS — reuse for the new endpoint

### Established Patterns
- SSE endpoints return `sse_response(async_generator)` — follow same pattern
- Event log backends are registered in DI container (`Container.instance()`)
- CompositeEventLog is assembled in DI container with list of backends

### Integration Points
- DI container: Add event bus as backend in CompositeEventLog's backend list
- New route file: `api/routes/events.py` with `GET /api/events/stream`
- Router registration in `api/main.py`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-eventbus-sse-foundation*
*Context gathered: 2026-03-14*
