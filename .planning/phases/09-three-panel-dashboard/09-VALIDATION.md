---
phase: 9
slug: three-panel-dashboard
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest 2.1.4 (frontend); pytest + pytest-asyncio 0.21+ (backend) |
| **Config file** | `web/vitest.config.ts` — environment: "node", alias: "@" → "./src" |
| **Quick run command** | `cd web && npm test -- agent-dashboard` |
| **Full suite command** | `cd web && npm test -- agent-dashboard agent-events` |
| **Estimated runtime** | ~8 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd web && npm test -- agent-dashboard agent-events 2>&1 | tail -10`
- **After every plan wave:** Run `cd web && npm test -- agent-dashboard agent-events`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | DASH-01 | unit | `cd web && npm test -- agent-dashboard` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 1 | DASH-04 | unit | `cd web && npm test -- SplitPanels` | ✅ | ⬜ pending |
| 09-01-03 | 01 | 1 | FILE-01 | unit | `cd web && npm test -- agent-dashboard` | ❌ W0 | ⬜ pending |
| 09-01-04 | 01 | 1 | FILE-02 | unit | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 09-01-05 | 01 | 1 | FILE-02 | unit | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 09-01-06 | 01 | 1 | FILE-02 | unit | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 09-01-07 | 01 | 1 | FILE-02 | unit | `cd web && npm test -- agent-events` | ❌ W0 | ⬜ pending |
| 09-01-08 | 01 | 1 | DASH-01 | unit | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `web/src/components/agent-dashboard/TasksPanel.test.tsx` — renders "No runs yet" when feed is empty
- [ ] `web/src/components/agent-dashboard/FileListPanel.test.tsx` — renders file list, handles empty state, navigates to diff on click
- [ ] `web/src/components/agent-dashboard/InlineDiffPanel.test.tsx` — renders DiffViewer, renders fallback when no content
- [ ] `web/src/lib/agent-events/parsers.test.ts` — EXTENDED: add parseFileTouched test cases
- [ ] `web/src/lib/agent-events/store.test.ts` — EXTENDED: addFileTouched dedup + eviction tests

*No new framework install needed — vitest already configured*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Drag panel dividers and verify sizes persist across page navigation | DASH-04 | Requires browser interaction with drag handles + page reload | 1. Open `/agent-dashboard` 2. Drag rail/list divider 3. Navigate away 4. Return — sizes should restore |
| Three-panel layout renders correctly on mobile breakpoint | DASH-01 | Requires viewport resize below 768px | 1. Open DevTools responsive mode 2. Set viewport <768px 3. Verify tab strip fallback |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
