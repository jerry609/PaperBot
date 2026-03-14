---
phase: 05-transport-entry-point
verified: 2026-03-14T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 5: Transport & Entry Point Verification Report

**Phase Goal:** MCP server is runnable via stdio (local) and Streamable HTTP (remote) with a CLI command
**Verified:** 2026-03-14
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `paperbot mcp serve --stdio` starts MCP server on stdio transport | VERIFIED | `_run_mcp_serve` in `main.py:693-700` dispatches to `run_stdio()` which calls `mcp.run(transport="stdio")`; test `test_run_cli_mcp_serve_stdio_dispatches` PASSES |
| 2 | `paperbot mcp serve --http` starts MCP server on Streamable HTTP transport | VERIFIED | `_run_mcp_serve` dispatches to `run_http()` which calls `mcp.run(transport="streamable-http", host=host, port=port)`; test `test_run_cli_mcp_serve_http_dispatches` PASSES |
| 3 | `paperbot mcp serve --http --host 0.0.0.0 --port 9000` allows host/port override | VERIFIED | `run_http(host, port)` signature and argparse wiring confirmed; `test_parse_mcp_serve_http_custom_host_port` and `test_passes_custom_host_and_port` both PASS |
| 4 | `paperbot mcp serve` (no flag) prints help without crashing | VERIFIED | mutually exclusive group with `required=True` causes argparse to exit non-zero; `test_run_cli_mcp_serve_no_transport_exits_nonzero` PASSES |
| 5 | `paperbot mcp` (no subcommand) prints help without crashing | VERIFIED | `run_cli(["mcp"])` branch at `main.py:332-338` prints inline help and returns 0; `test_run_cli_mcp_no_subcommand_returns_zero` PASSES |
| 6 | stdio mode sends zero bytes to stdout (logging goes to stderr only) | VERIFIED | `run_stdio()` calls `logging.basicConfig(stream=sys.stderr, ...)` before `mcp.run()`; `test_configures_logging_to_stderr` PASSES |
| 7 | Missing mcp package produces clear error message to stderr and exits non-zero | VERIFIED | Both `run_stdio()` and `run_http()` check `if mcp is None` -> print to `sys.stderr` + `sys.exit(1)`; `TestMcpNoneGuard` tests (2) PASS |

**Score:** 7/7 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/paperbot/mcp/serve.py` | Transport dispatch functions | VERIFIED | 81 lines; exports `run_stdio`, `run_http`, `_get_mcp`; substantive implementation with None guard, logging config, `mcp.run()` calls |
| `src/paperbot/presentation/cli/main.py` | mcp serve subcommand in CLI parser | VERIFIED | `mcp_parser` added at lines 214-231; `_run_mcp_serve` handler at lines 693-701; `mcp` dispatch branch at lines 331-338 |
| `pyproject.toml` | `[project.scripts]` entry + mcp dependency | VERIFIED | `[project.scripts]` at line 101; `paperbot = "paperbot.presentation.cli.main:run_cli"` at line 102; `mcp[fastmcp]>=1.8.0,<2.0.0` at line 62 |
| `requirements.txt` | mcp[fastmcp] dependency line | VERIFIED | `mcp[fastmcp]>=1.8.0,<2.0.0` found at line 112 |
| `tests/unit/test_mcp_serve_cli.py` | Unit tests for all transport and CLI behaviors | VERIFIED | 18 tests across 6 test classes; all 18 PASS |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/paperbot/presentation/cli/main.py` | `src/paperbot/mcp/serve.py` | lazy import in `_run_mcp_serve` | WIRED | `from paperbot.mcp.serve import run_http, run_stdio` at line 695; import is exercised by `run_cli(["mcp", "serve", ...])` |
| `src/paperbot/mcp/serve.py` | `src/paperbot/mcp/server.py` | `_get_mcp()` helper | WIRED | `from paperbot.mcp import server as _server_mod; return _server_mod.mcp` at lines 29-31; called at runtime in both `run_stdio()` and `run_http()` |
| `pyproject.toml` | `src/paperbot/presentation/cli/main.py` | `[project.scripts]` entry point | WIRED | `paperbot = "paperbot.presentation.cli.main:run_cli"` confirmed at pyproject.toml line 102 |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MCP-10 | 05-01-PLAN.md | MCP server runs via stdio transport for local agent integration | SATISFIED | `run_stdio()` calls `mcp.run(transport="stdio")`; CLI wiring confirmed; 18/18 tests pass |
| MCP-11 | 05-01-PLAN.md | MCP server runs via Streamable HTTP transport for remote agent integration | SATISFIED | `run_http()` calls `mcp.run(transport="streamable-http", host, port)`; default port 8001 avoids FastAPI collision |
| MCP-12 | 05-01-PLAN.md | User can start MCP server via `paperbot mcp serve` CLI command | SATISFIED | `mcp` subparser + `serve` sub-subparser wired in `create_parser()`; `_run_mcp_serve` dispatches correctly; `[project.scripts]` makes `paperbot` installable |

No orphaned requirements for Phase 5: all three (MCP-10, MCP-11, MCP-12) are declared in 05-01-PLAN.md and verified in the codebase.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/paperbot/presentation/cli/main.py` | 357 | `# TODO: 实现 Semantic Scholar 客户端调用` | Info | Pre-existing in `_quick_score` function (unrelated to Phase 5 scope); zero impact on transport/MCP goal |

No blockers. No warnings in Phase 5 scope. The one TODO is in pre-existing `_quick_score` code, not in any Phase 5 deliverable.

---

## Human Verification Required

### 1. Claude Desktop integration round-trip

**Test:** Add the following to `claude_desktop_config.json` and launch Claude Desktop:
```json
{
  "mcpServers": {
    "paperbot": {
      "command": "paperbot",
      "args": ["mcp", "serve", "--stdio"]
    }
  }
}
```
**Expected:** PaperBot tools appear in the Claude Desktop tool picker; calling `paper_search` returns results.
**Why human:** Requires an installed package, a running MCP package, and a Claude Desktop instance — cannot verify end-to-end connectivity programmatically.

### 2. Remote agent HTTP connection

**Test:** Run `paperbot mcp serve --http` and connect a remote MCP client to `http://127.0.0.1:8001`.
**Expected:** Client receives tool list (9 tools) and can call any tool successfully.
**Why human:** Requires `mcp[fastmcp]` installed in the active environment, a running server process, and a live MCP client.

---

## Gaps Summary

None. All 7 observable truths verified. All 5 artifacts exist, are substantive, and are wired. All 3 key links confirmed. All 3 requirement IDs (MCP-10, MCP-11, MCP-12) satisfied. No blocker anti-patterns.

---

_Verified: 2026-03-14_
_Verifier: Claude (gsd-verifier)_
