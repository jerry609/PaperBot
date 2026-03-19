---
phase: 11
slug: dag-visualization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest 2.1.4 |
| **Config file** | `web/vitest.config.ts` |
| **Quick run command** | `cd web && npm test -- --reporter=verbose src/lib/agent-events/` |
| **Full suite command** | `cd web && npm test` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd web && npm test -- src/lib/agent-events/`
- **After every plan wave:** Run `cd web && npm test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | VIZ-01 | unit (vitest) | `cd web && npm test -- src/lib/agent-events/dag.test.ts` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | VIZ-02 | unit (vitest) | `cd web && npm test -- src/lib/agent-events/parsers.test.ts` | ✅ extend | ⬜ pending |
| 11-01-03 | 01 | 1 | VIZ-02 | unit (vitest) | `cd web && npm test -- src/lib/agent-events/store.test.ts` | ✅ extend | ⬜ pending |
| 11-02-01 | 02 | 1 | VIZ-01+02 | smoke (jsdom) | `cd web && npm test -- src/components/agent-dashboard/AgentDagPanel.test.tsx` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `web/src/lib/agent-events/dag.test.ts` — stubs for VIZ-01 (buildDagNodes status mapping, column placement)
- [ ] `web/src/components/agent-dashboard/AgentDagPanel.test.tsx` — smoke render test (jsdom, mocks ReactFlow)
- [ ] Extend `web/src/lib/agent-events/parsers.test.ts` — add `parseScoreEdge` test cases
- [ ] Extend `web/src/lib/agent-events/store.test.ts` — add `addScoreEdge` dedup test cases

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| DAG layout is visually readable with nodes arranged logically | VIZ-01 | Visual layout judgment | Open /agent-dashboard, click "DAG" view, verify nodes are arranged in a readable graph |
| ScoreShareBus edges animate between connected nodes | VIZ-02 | Visual animation | Trigger a score_update event, verify animated dashed edge appears between relevant nodes |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
