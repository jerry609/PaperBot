---
phase: 4
slug: mcp-resources
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest with pytest-asyncio (asyncio_mode = "strict") |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` |
| **Quick run command** | `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py tests/unit/test_mcp_track_papers.py tests/unit/test_mcp_track_memory.py tests/unit/test_mcp_scholars.py -q` |
| **Full suite command** | `PYTHONPATH=src pytest -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py tests/unit/test_mcp_track_papers.py tests/unit/test_mcp_track_memory.py tests/unit/test_mcp_scholars.py -q`
- **After every plan wave:** Run `PYTHONPATH=src pytest -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | MCP-06 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | MCP-06 | unit | same | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | MCP-06 | unit | same | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | MCP-07 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_track_papers.py -x` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 1 | MCP-07 | unit | same | ❌ W0 | ⬜ pending |
| 04-01-06 | 01 | 1 | MCP-08 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_track_memory.py -x` | ❌ W0 | ⬜ pending |
| 04-01-07 | 01 | 1 | MCP-08 | unit | same | ❌ W0 | ⬜ pending |
| 04-01-08 | 01 | 1 | MCP-09 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_scholars.py -x` | ❌ W0 | ⬜ pending |
| 04-01-09 | 01 | 1 | MCP-09 | unit | same | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | All 4 | integration | `PYTHONPATH=src pytest tests/integration/test_mcp_tool_calls.py -x -k resource` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 2 | All 4 | integration | same | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_mcp_track_metadata.py` — 3+ tests for MCP-06 (valid, not found, invalid ID)
- [ ] `tests/unit/test_mcp_track_papers.py` — 2+ tests for MCP-07 (with papers, empty)
- [ ] `tests/unit/test_mcp_track_memory.py` — 2+ tests for MCP-08 (with memories, empty)
- [ ] `tests/unit/test_mcp_scholars.py` — 2+ tests for MCP-09 (with scholars, missing config)
- [ ] `tests/integration/test_mcp_tool_calls.py` — add `TestMCPResourceListing` class

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
