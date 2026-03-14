---
phase: 7
slug: eventbus-sse-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio 0.21+ |
| **Config file** | `pyproject.toml` — `asyncio_mode = "strict"` |
| **Quick run command** | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py -q` |
| **Full suite command** | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py tests/integration/test_events_sse_endpoint.py -q` |
| **Estimated runtime** | ~3 seconds |

---

## Sampling Rate

- **After every task commit:** Run `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py -q`
- **After every plan wave:** Run `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py tests/integration/test_events_sse_endpoint.py -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 3 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | EVNT-04 | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_fan_out_to_multiple_subscribers -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | EVNT-04 | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_ring_buffer_catch_up -x` | ❌ W0 | ⬜ pending |
| 07-01-03 | 01 | 1 | EVNT-04 | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_backpressure_drops_oldest -x` | ❌ W0 | ⬜ pending |
| 07-01-04 | 01 | 1 | EVNT-04 | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_unsubscribe_cleans_up -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | EVNT-04 | integration | `PYTHONPATH=src pytest tests/integration/test_events_sse_endpoint.py::test_event_delivered_within_1s -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 1 | EVNT-04 | integration | `PYTHONPATH=src pytest tests/integration/test_events_sse_endpoint.py::test_heartbeat_on_idle -x` | ❌ W0 | ⬜ pending |
| 07-02-03 | 02 | 1 | EVNT-04 | unit | `PYTHONPATH=src pytest tests/unit/test_event_bus_event_log.py::test_composite_includes_bus -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_event_bus_event_log.py` — stubs for fan-out, ring buffer, backpressure, unsubscribe, composite wiring
- [ ] `tests/integration/test_events_sse_endpoint.py` — stubs for SSE delivery latency and heartbeat

*No framework install needed — pytest-asyncio already in `[dev]` extras*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 3s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
