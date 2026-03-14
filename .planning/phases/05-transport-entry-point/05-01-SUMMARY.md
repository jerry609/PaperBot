---
phase: 05-transport-entry-point
plan: "01"
subsystem: mcp
tags: [mcp, transport, stdio, http, cli, packaging]
dependency_graph:
  requires: [04-mcp-resources]
  provides: [mcp-serve-module, paperbot-cli-entry, transport-wiring]
  affects: [cli, mcp-server, packaging]
tech_stack:
  added: [mcp[fastmcp]>=1.8.0,<2.0.0]
  patterns: [TDD-fake-stub, lazy-import, mutually-exclusive-argparse]
key_files:
  created:
    - src/paperbot/mcp/serve.py
    - tests/unit/test_mcp_serve_cli.py
  modified:
    - src/paperbot/presentation/cli/main.py
    - pyproject.toml
    - requirements.txt
decisions:
  - "_get_mcp() helper function used for testable lazy import of mcp singleton"
  - "Default HTTP port is 8001 (not 8000) to avoid conflict with FastAPI server"
  - "serve.py uses logging.basicConfig(stream=sys.stderr) before mcp.run() for stdio purity"
  - "mcp_parser help printed via inline string (not re-creating parser) for simplicity"
  - "_run_mcp_serve uses lazy import of run_stdio/run_http to avoid circular imports"
metrics:
  duration: "3 min"
  completed: "2026-03-14"
  tasks_completed: 2
  files_created: 2
  files_modified: 3
---

# Phase 05 Plan 01: Transport Entry Point Summary

**One-liner:** stdio and Streamable HTTP transport dispatch wired to CLI `paperbot mcp serve` with `mcp[fastmcp]` packaging.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Create serve.py module and update packaging | b2580d3 | serve.py, pyproject.toml, requirements.txt, test_mcp_serve_cli.py |
| 2 | Add mcp serve subcommand to CLI | b3d4224 | main.py (test_mcp_serve_cli.py updated in place) |

## What Was Built

### src/paperbot/mcp/serve.py

Transport dispatch module with two public functions:

- `run_stdio()`: redirects all logging to stderr, lazy-imports the `mcp` singleton from `paperbot.mcp.server`, checks None guard with clear error + `sys.exit(1)`, calls `mcp.run(transport="stdio")`
- `run_http(host="127.0.0.1", port=8001)`: same None guard, calls `mcp.run(transport="streamable-http", host=host, port=port)`

Internal `_get_mcp()` helper is the testable injection point (monkeypatched in tests).

### CLI: paperbot mcp serve

Added to `src/paperbot/presentation/cli/main.py`:

- `mcp` subparser with `serve` sub-subparser
- `--stdio` / `--http` as a `required=True` mutually exclusive group
- `--host` (default `127.0.0.1`) and `--port` (default `8001`) for HTTP mode
- `_run_mcp_serve(parsed)` handler with lazy import of `run_stdio`/`run_http`
- `paperbot mcp` (no subcommand) prints inline help and returns 0

### Packaging

- `pyproject.toml`: added `[project.scripts]` entry `paperbot = "paperbot.presentation.cli.main:run_cli"` and `mcp[fastmcp]>=1.8.0,<2.0.0` to `dependencies`
- `requirements.txt`: added `mcp[fastmcp]>=1.8.0,<2.0.0` line

## Tests

18 tests in `tests/unit/test_mcp_serve_cli.py` covering all behaviors:
- `TestServeModuleImport` (1 test)
- `TestRunStdio` (2 tests)
- `TestRunHttp` (2 tests)
- `TestMcpNoneGuard` (2 tests)
- `TestPyprojectScripts` (3 tests)
- `TestCLIServeCommand` (8 tests)

All 18 pass. No regressions in `test_mcp_bootstrap.py` (3 tests).

## Deviations from Plan

### Minor Implementation Deviation

**Deviation:** `_get_mcp()` helper function introduced instead of direct `from paperbot.mcp.server import mcp` import at module level.

**Reason:** Python 3.9 module-level import caching makes it impossible to monkeypatch `mcp` after `serve` is imported once — the module-level name is already bound. A `_get_mcp()` function is called at runtime and can be monkeypatched in tests cleanly.

**Impact:** Functionally equivalent. Tests are cleaner. Pattern is consistent with project's fake-stub convention.

## Self-Check: PASSED

Files created:
- FOUND: src/paperbot/mcp/serve.py
- FOUND: tests/unit/test_mcp_serve_cli.py

Files modified:
- FOUND: src/paperbot/presentation/cli/main.py
- FOUND: pyproject.toml
- FOUND: requirements.txt

Commits:
- FOUND: b2580d3 (Task 1)
- FOUND: b3d4224 (Task 2)

Verification:
- 18/18 tests pass in test_mcp_serve_cli.py
- 3/3 tests pass in test_mcp_bootstrap.py
- pyproject.toml has [project.scripts] entry
- pyproject.toml has mcp[fastmcp] dependency
- requirements.txt has mcp[fastmcp] line
- `from paperbot.mcp.serve import run_stdio, run_http` imports OK
