---
phase: 10
slug: agent-board-codex-bridge
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest 2.1.4 (frontend); pytest + pytest-asyncio (backend) |
| **Config file** | `web/vitest.config.ts` — environment: "node", alias: "@" → "./src" |
| **Quick run command** | `cd web && npm test -- KanbanBoard agent-events` |
| **Full suite command** | `cd web && npm test -- KanbanBoard agent-events && PYTHONPATH=src pytest tests/unit/test_agent_board_route.py tests/unit/test_codex_overflow.py -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd web && npm test -- KanbanBoard agent-events 2>&1 | tail -10`
- **After every plan wave:** Run `cd web && npm test -- KanbanBoard agent-events && PYTHONPATH=src pytest tests/unit/test_agent_board_route.py tests/unit/test_codex_overflow.py -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | DASH-02 | unit (vitest) | `cd web && npm test -- KanbanBoard` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | DASH-02 | unit (vitest) | `cd web && npm test -- KanbanBoard` | ❌ W0 | ⬜ pending |
| 10-01-03 | 01 | 1 | DASH-03 | unit (vitest) | `cd web && npm test -- KanbanBoard` | ❌ W0 | ⬜ pending |
| 10-02-01 | 02 | 1 | CDX-01 | file check (bash) | `test -f .claude/agents/codex-worker.md` | ❌ W0 | ⬜ pending |
| 10-02-02 | 02 | 1 | CDX-02 | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_codex_overflow.py -x` | ❌ W0 | ⬜ pending |
| 10-02-03 | 02 | 1 | CDX-03 | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_agent_board_route.py -k codex_event -x` | ❌ W0 | ⬜ pending |
| 10-03-01 | 03 | 2 | CDX-03 | unit (vitest) | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 10-03-02 | 03 | 2 | CDX-03 | unit (vitest) | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `web/src/components/agent-dashboard/KanbanBoard.test.tsx` — stubs for DASH-02, DASH-03
- [ ] `web/src/lib/agent-events/parsers.test.ts` — extended stubs for CDX-03
- [ ] `tests/unit/test_codex_overflow.py` — stubs for CDX-02
- [ ] `tests/unit/test_agent_board_route.py` — extended stubs for CDX-03 emission

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Kanban board column layout renders correctly | DASH-02 | Visual layout | Open /agent-dashboard, click "Kanban" view, verify 5 columns visible |
| Codex error badge is red and prominent | DASH-03 | Visual styling | Trigger a failed Codex task, verify red Error badge on card |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
