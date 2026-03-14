---
phase: 3
slug: remaining-mcp-tools
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x with pytest-asyncio (asyncio_mode = "strict") |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` |
| **Quick run command** | `PYTHONPATH=src pytest tests/unit/test_mcp_analyze_trends.py tests/unit/test_mcp_check_scholar.py tests/unit/test_mcp_get_research_context.py tests/unit/test_mcp_save_to_memory.py tests/unit/test_mcp_export_to_obsidian.py -q` |
| **Full suite command** | `PYTHONPATH=src pytest tests/unit/test_mcp_*.py tests/integration/test_mcp_tool_calls.py -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `PYTHONPATH=src pytest tests/unit/test_mcp_<tool>.py -q`
- **After every plan wave:** Run `PYTHONPATH=src pytest tests/unit/test_mcp_*.py tests/integration/test_mcp_tool_calls.py -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | MCP-01 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_analyze_trends.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 0 | MCP-02 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_check_scholar.py -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 0 | MCP-03 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_get_research_context.py -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 0 | MCP-04 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_save_to_memory.py -x` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 0 | MCP-05 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_export_to_obsidian.py -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | MCP-01 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_analyze_trends.py -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | MCP-02 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_check_scholar.py -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | MCP-03 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_get_research_context.py -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | MCP-04 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_save_to_memory.py -x` | ❌ W0 | ⬜ pending |
| 03-02-05 | 02 | 1 | MCP-05 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_export_to_obsidian.py -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | All | integration | `PYTHONPATH=src pytest tests/integration/test_mcp_tool_calls.py -x` | ✅ (update) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_mcp_analyze_trends.py` — stubs for MCP-01 (3 tests: success, degraded, audit)
- [ ] `tests/unit/test_mcp_check_scholar.py` — stubs for MCP-02 (3 tests: success, not-found, audit)
- [ ] `tests/unit/test_mcp_get_research_context.py` — stubs for MCP-03 (3 tests: success, params, audit)
- [ ] `tests/unit/test_mcp_save_to_memory.py` — stubs for MCP-04 (3 tests: success, invalid-kind, audit)
- [ ] `tests/unit/test_mcp_export_to_obsidian.py` — stubs for MCP-05 (3 tests: success, frontmatter, audit)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| All 9 tools appear in MCP tools/list | All | Server startup required | Start MCP server, call `tools/list`, verify 9 entries |

*Note: Integration test `test_mcp_tool_calls.py` covers this via FastMCP test client, so effectively automated.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
