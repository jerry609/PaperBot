---
phase: 8
slug: agent-event-vocabulary
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + pytest-asyncio 0.21+ (backend); vitest 2.1.4 (frontend) |
| **Config file** | `pyproject.toml` — `asyncio_mode = "strict"` |
| **Quick run command** | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -q` |
| **Full suite command** | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py tests/integration/test_events_sse_endpoint.py -q && cd web && npm test` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -q`
- **After every plan wave:** Run `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py tests/integration/test_events_sse_endpoint.py -q && cd web && npm test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | EVNT-01/02/03 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_event_type_constants -x` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | EVNT-02 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_lifecycle_event_types -x` | ❌ W0 | ⬜ pending |
| 08-01-03 | 01 | 1 | EVNT-03 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_tool_call_event_error_type -x` | ❌ W0 | ⬜ pending |
| 08-01-04 | 01 | 1 | EVNT-03 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py::test_audit_uses_constants -x` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 1 | EVNT-01 | unit (vitest) | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 1 | EVNT-02 | unit (vitest) | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 08-02-03 | 02 | 1 | EVNT-03 | unit (vitest) | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_agent_events_vocab.py` — EventType constants, make_lifecycle_event, make_tool_call_event, _audit.py constant usage
- [ ] `web/src/lib/agent-events/parsers.test.ts` — parseActivityItem, parseAgentStatus, parseToolCall with fixture envelopes
- [ ] `web/src/lib/agent-events/store.test.ts` — feed cap, addFeedItem, updateAgentStatus, addToolCall

*Existing infrastructure covers framework installation — pytest-asyncio and vitest already present.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Activity feed scrolls and updates visually | EVNT-01 | Visual rendering behavior | Open `/agent-events` test page, trigger events, observe scrolling |
| Agent status indicators change color/icon | EVNT-02 | Visual state transitions | Trigger lifecycle events, observe status panel |
| Tool call timeline renders correctly | EVNT-03 | Visual layout and interaction | Trigger tool call events, observe timeline component |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
