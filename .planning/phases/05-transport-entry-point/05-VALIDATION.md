---
phase: 5
slug: transport-entry-point
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x with pytest-asyncio (asyncio_mode = "strict") |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` |
| **Quick run command** | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py -q` |
| **Full suite command** | `PYTHONPATH=src pytest -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py -q`
- **After every plan wave:** Run `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py tests/unit/test_mcp_bootstrap.py -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | MCP-12 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestCLIServeCommand -x` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | MCP-10 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestRunStdio -x` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | MCP-11 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestRunHttp -x` | ❌ W0 | ⬜ pending |
| 05-01-04 | 01 | 1 | MCP-10 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestRunStdio::test_logging_stderr -x` | ❌ W0 | ⬜ pending |
| 05-01-05 | 01 | 1 | MCP-10/11 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestServeModuleImport -x` | ❌ W0 | ⬜ pending |
| 05-01-06 | 01 | 1 | MCP-10/11 | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestMcpNoneGuard -x` | ❌ W0 | ⬜ pending |
| 05-01-07 | 01 | 1 | MCP-12 | static | `grep -q 'project.scripts' pyproject.toml` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_mcp_serve_cli.py` — stubs for MCP-10, MCP-11, MCP-12
- [ ] `src/paperbot/mcp/serve.py` — `run_stdio()` and `run_http()` entry functions

*Existing infrastructure covers test framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Claude Code connects via stdio in `claude_desktop_config.json` | MCP-10 | Requires Claude Desktop/Code runtime | 1. Add config snippet to `~/.claude.json` 2. Start Claude Code 3. Verify tools appear |
| Remote agent connects via HTTP and calls tools | MCP-11 | Requires external agent client | 1. Start `paperbot mcp serve --http` 2. Use MCP client to connect to `http://127.0.0.1:8001/mcp` 3. Call a tool |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
