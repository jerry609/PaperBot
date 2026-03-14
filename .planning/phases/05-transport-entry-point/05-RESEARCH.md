# Phase 5: Transport & Entry Point - Research

**Researched:** 2026-03-14
**Domain:** FastMCP transport configuration (stdio / Streamable HTTP), Python CLI entry points (argparse + pyproject.toml `[project.scripts]`), claude_desktop_config.json integration
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCP-10 | MCP server runs via stdio transport for local agent integration | `mcp.run(transport="stdio")` — FastMCP built-in; no extra deps; default transport when no arg passed |
| MCP-11 | MCP server runs via Streamable HTTP transport for remote agent integration | `mcp.run(transport="streamable-http")` — FastMCP built-in; server binds to configurable host:port, endpoint at `/mcp` |
| MCP-12 | User can start MCP server via `paperbot mcp serve` CLI command | Add `mcp` subcommand + `serve` sub-subcommand to existing `argparse` CLI in `src/paperbot/presentation/cli/main.py`; wire `--stdio` / `--http` flags to `mcp.run()` |
</phase_requirements>

---

## Summary

Phase 5 makes the already-complete MCP server (9 tools + 4 resources, built in Phases 2–4) runnable by agents. Two transport modes are required: **stdio** for local Claude Code/Claude Desktop integration and **Streamable HTTP** for remote agents. Both modes are provided natively by the `mcp[fastmcp]` package already established in earlier phases — no new dependencies are needed.

The CLI entry point extends the existing `argparse`-based CLI in `src/paperbot/presentation/cli/main.py` with a `mcp` subcommand and a `serve` sub-subcommand. The `--stdio` flag calls `mcp.run(transport="stdio")` and `--http` calls `mcp.run(transport="streamable-http", host=..., port=...)`. The package's `[project.scripts]` entry in `pyproject.toml` (currently absent) must be added so that `pip install -e .` makes `paperbot` available on `$PATH`.

The `claude_desktop_config.json` pattern for stdio is a standard JSON block that tells Claude Desktop to spawn the server process and communicate via stdin/stdout. The exact command depends on whether the package is installed (`paperbot mcp serve --stdio`) or run from the project root (`python -m paperbot.mcp.serve_stdio` or similar).

**Primary recommendation:** Add `[project.scripts]` entry to `pyproject.toml`, add `mcp serve` subcommand to the CLI, and document the `claude_desktop_config.json` snippet in a comment block in the serve module.

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mcp[fastmcp]` | `>=1.8.0,<2.0.0` | `FastMCP.run(transport=...)` for both stdio and Streamable HTTP | Established in Phase 1; already used for all tools and resources |
| `argparse` | stdlib | CLI subcommand parsing (`paperbot mcp serve --stdio/--http`) | Already used in `src/paperbot/presentation/cli/main.py` |
| `uvicorn` | existing dep | Optional ASGI runner for HTTP transport in production | Already a project dependency |

### No New Dependencies
Both stdio and Streamable HTTP transports are built into `mcp[fastmcp]`. Zero new packages required.

**Installation (for new environments):**
```bash
pip install "mcp[fastmcp]>=1.8.0,<2.0.0"
```
This must be added to `pyproject.toml` dependencies (currently missing).

---

## Architecture Patterns

### Recommended Project Structure (Phase 5 additions)
```
src/paperbot/
├── mcp/
│   ├── server.py              # existing: FastMCP instance + all registrations
│   └── serve.py               # NEW: run_stdio() / run_http() entry functions
├── presentation/
│   └── cli/
│       └── main.py            # MODIFY: add 'mcp' subcommand + serve sub-subcommand

pyproject.toml                 # MODIFY: add [project.scripts] + mcp dependency

tests/
└── unit/
    └── test_mcp_serve_cli.py  # NEW: covers MCP-10, MCP-11, MCP-12

docs/ (optional)
└── claude_desktop_config_example.json  # NEW: example config for users
```

### Pattern 1: Transport dispatch via `serve.py` module
**What:** Create `src/paperbot/mcp/serve.py` with two public functions — `run_stdio()` and `run_http(host, port)` — that import the `mcp` singleton from `server.py` and call `mcp.run()` with the appropriate transport.
**When to use:** These functions are called from the CLI handler and can also be called directly in scripts or tests.
**Why:** Keeps transport dispatch logic in the `mcp/` package, separate from the CLI argument parsing layer. Makes it testable without argparse.

```python
# src/paperbot/mcp/serve.py
from __future__ import annotations

import sys


def run_stdio() -> None:
    """Start MCP server on stdio transport (local / Claude Desktop mode)."""
    from paperbot.mcp.server import mcp

    if mcp is None:
        print("Error: mcp package not installed. Run: pip install 'mcp[fastmcp]>=1.8.0,<2.0.0'",
              file=sys.stderr)
        sys.exit(1)

    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start MCP server on Streamable HTTP transport (remote agent mode)."""
    from paperbot.mcp.server import mcp

    if mcp is None:
        print("Error: mcp package not installed.", file=sys.stderr)
        sys.exit(1)

    mcp.run(transport="streamable-http", host=host, port=port)
```

### Pattern 2: `mcp` subcommand added to existing argparse CLI
**What:** Add a `mcp` subparser to `create_parser()` in `main.py`, and a `serve` sub-subparser under it. Wire `--stdio` / `--http` flags and optional `--host` / `--port` parameters.
**When to use:** Follows the pattern already established by `track`, `analyze`, `topic-search`, etc. in the CLI.

```python
# In create_parser() — add after existing subparser definitions:
mcp_parser = subparsers.add_parser("mcp", help="MCP server commands")
mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP sub-commands")

serve_parser = mcp_subparsers.add_parser("serve", help="Start MCP server")
serve_transport = serve_parser.add_mutually_exclusive_group(required=True)
serve_transport.add_argument("--stdio", action="store_true",
                             help="Run on stdio transport (for Claude Desktop / Claude Code)")
serve_transport.add_argument("--http", action="store_true",
                             help="Run on Streamable HTTP transport (for remote agents)")
serve_parser.add_argument("--host", default="127.0.0.1",
                          help="HTTP host (default: 127.0.0.1)")
serve_parser.add_argument("--port", type=int, default=8000,
                          help="HTTP port (default: 8000)")
```

```python
# In run_cli() — add to command dispatch:
elif parsed.command == "mcp":
    if parsed.mcp_command == "serve":
        return _run_mcp_serve(parsed)
    mcp_parser.print_help()
    return 0
```

```python
# New handler function:
def _run_mcp_serve(parsed: argparse.Namespace) -> int:
    from paperbot.mcp.serve import run_stdio, run_http
    if parsed.stdio:
        run_stdio()
    else:
        run_http(host=parsed.host, port=parsed.port)
    return 0
```

### Pattern 3: `[project.scripts]` entry in `pyproject.toml`
**What:** Add a `[project.scripts]` section so that `pip install -e .` installs a `paperbot` command on `$PATH`.
**When to use:** Required for `claude_desktop_config.json` to reference `paperbot` as the command name.

```toml
# pyproject.toml — add this section:
[project.scripts]
paperbot = "paperbot.presentation.cli.main:run_cli"
```

This is currently absent from `pyproject.toml` despite the CLI being fully implemented. The planner MUST add it.

### Pattern 4: `claude_desktop_config.json` stdio integration
**What:** Document the JSON block users add to Claude Desktop config to connect to PaperBot MCP via stdio.
**When to use:** MCP-10 success criterion — "Claude Code can connect to PaperBot MCP server via stdio in `claude_desktop_config.json`."

```json
{
  "mcpServers": {
    "paperbot": {
      "command": "paperbot",
      "args": ["mcp", "serve", "--stdio"],
      "env": {
        "PAPERBOT_DB_URL": "sqlite:////absolute/path/to/data/paperbot.db",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

If the package is not installed but the project is cloned:
```json
{
  "mcpServers": {
    "paperbot": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "paperbot.mcp.serve_stdio"],
      "env": {
        "PAPERBOT_DB_URL": "sqlite:////absolute/path/to/data/paperbot.db"
      }
    }
  }
}
```

### Pattern 5: Stdio logging safety
**What:** Ensure NO output goes to stdout when running in stdio transport. All logging must go to stderr or a file.
**When to use:** Critical for stdio transport — stdout is the MCP protocol channel. Any non-JSON-RPC bytes on stdout corrupt the protocol.

```python
# In serve.py — configure logging before mcp.run() for stdio mode:
import logging
import sys

def run_stdio() -> None:
    # Redirect all logging to stderr so stdout stays clean for MCP protocol
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    ...
    mcp.run(transport="stdio")
```

### Anti-Patterns to Avoid
- **Calling `print()` without `file=sys.stderr` in stdio mode**: stdout is the MCP JSON-RPC channel. Any stray bytes corrupt the protocol.
- **Using `transport="sse"`**: SSE is legacy and deprecated. Always use `"streamable-http"` for HTTP transport.
- **Missing `[project.scripts]` in `pyproject.toml`**: Without this, `paperbot` is not on PATH and `claude_desktop_config.json` will not work with the simple `"command": "paperbot"` form.
- **Calling `mcp.run()` from inside an async function**: `FastMCP.run()` creates its own event loop. Call it only from synchronous code (the CLI handler is already synchronous). Use `mcp.run_async()` only if an event loop is already running.
- **Hardcoding `host="0.0.0.0"` as default**: The default should be `127.0.0.1` for security. Users can override via `--host` flag.
- **Blocking the test suite with `mcp.run()`**: Tests MUST NOT call `run_stdio()` or `run_http()` directly — those block. Test via `run_stdio` / `run_http` being callable, or mock `mcp.run` in unit tests.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| stdio MCP protocol framing | Custom stdin/stdout JSON-RPC loop | `mcp.run(transport="stdio")` | Protocol framing, lifecycle, error recovery all handled |
| HTTP MCP endpoint | Custom FastAPI route | `mcp.run(transport="streamable-http")` | Streamable HTTP spec implementation, session mgmt, SSE upgrade all handled |
| ASGI server for HTTP MCP | Custom uvicorn setup | `mcp.run(transport="streamable-http")` | FastMCP starts its own ASGI server internally (uses uvicorn) |
| Transport negotiation | Custom detection logic | `mcp.run(transport=...)` with explicit string | Explicit transport string is clear and unambiguous |

**Key insight:** The entire Phase 5 implementation is plumbing — connecting the existing `mcp` singleton (built in Phases 1–4) to a transport. All protocol complexity lives in `mcp[fastmcp]`.

---

## Common Pitfalls

### Pitfall 1: stdout pollution in stdio mode
**What goes wrong:** Any `print()` call, startup banner, or logging line that goes to stdout corrupts the MCP JSON-RPC stream. Claude Desktop will fail to parse the server's responses.
**Why it happens:** Python `print()` defaults to stdout. Many frameworks (loguru, standard `logging` with default `StreamHandler`) also write to stdout by default.
**How to avoid:** In `run_stdio()`, configure logging to go to stderr before calling `mcp.run()`. Audit all startup code paths for `print()` calls.
**Warning signs:** Claude Desktop shows "Failed to read from MCP server" or JSON parse errors.

### Pitfall 2: Missing `[project.scripts]` entry
**What goes wrong:** `paperbot mcp serve --stdio` only works if the `paperbot` binary is on PATH. Without `[project.scripts]`, users must run `python -m paperbot.presentation.cli.main mcp serve --stdio` — a path that `claude_desktop_config.json` makes cumbersome.
**Why it happens:** `pyproject.toml` currently has no `[project.scripts]` section.
**How to avoid:** Add `[project.scripts]` as Task 1 of this phase (prerequisite for success criterion 3).
**Warning signs:** `command not found: paperbot` when testing claude_desktop_config.json.

### Pitfall 3: `mcp` singleton is `None` when mcp package not installed
**What goes wrong:** `server.py` sets `mcp = None` when `import mcp` fails. Calling `mcp.run()` on `None` raises `AttributeError`.
**Why it happens:** Defensive design from Phase 1 — the stub allows imports to succeed in test environments without mcp installed.
**How to avoid:** `serve.py` must check `if mcp is None` and exit with a clear error message before calling `mcp.run()`.
**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'run'` at startup.

### Pitfall 4: `mcp` package not in `pyproject.toml` dependencies
**What goes wrong:** The `mcp[fastmcp]` package is used in `server.py` but is not listed in `pyproject.toml` dependencies or `requirements.txt`. Fresh installs will fail at runtime.
**Why it happens:** Previous phases deferred the packaging work (server.py uses a try/except import guard). Phase 5 is when this gets fixed.
**How to avoid:** Add `"mcp[fastmcp]>=1.8.0,<2.0.0"` to `pyproject.toml` `dependencies` list AND to `requirements.txt`.
**Warning signs:** `ModuleNotFoundError: No module named 'mcp'` in production or after a clean install.

### Pitfall 5: Port conflict when testing HTTP transport
**What goes wrong:** If the HTTP server is already running (or the port is in use), `mcp.run(transport="streamable-http", port=8000)` will fail to bind.
**Why it happens:** `127.0.0.1:8000` is the default for uvicorn-based services (including PaperBot's FastAPI server).
**How to avoid:** Use a different default port for MCP HTTP (e.g., `8001`) or make it configurable via `--port`. Document the conflict risk.
**Warning signs:** `OSError: [Errno 98] Address already in use`.

### Pitfall 6: Async loop conflict with `mcp.run()` inside FastAPI
**What goes wrong:** If a user attempts to call `mcp.run()` from within an async context (e.g., a FastAPI startup event), it will fail because `run()` creates its own event loop.
**Why it happens:** `FastMCP.run()` is a blocking synchronous method that calls `asyncio.run()` internally.
**How to avoid:** The CLI handler (`_run_mcp_serve`) is synchronous. Keep it that way. Document that HTTP transport does NOT require the FastAPI server — they are separate processes.

---

## Code Examples

Verified patterns from official sources and established codebase:

### run() with transport string (mcp SDK v1.x)
```python
# Source: MCP Python SDK official docs (modelcontextprotocol.io)
# transport="stdio" is the default; explicit for clarity
mcp.run(transport="stdio")

# Streamable HTTP — production-recommended for network access
mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
# MCP endpoint: http://127.0.0.1:8000/mcp
```

### serve.py — complete module
```python
# src/paperbot/mcp/serve.py
from __future__ import annotations

import logging
import sys


def run_stdio() -> None:
    """Start MCP server on stdio transport.

    Configures logging to stderr (stdout must remain clean for JSON-RPC).
    Blocks until the client closes the connection.

    claude_desktop_config.json usage:
        {
          "mcpServers": {
            "paperbot": {
              "command": "paperbot",
              "args": ["mcp", "serve", "--stdio"]
            }
          }
        }
    """
    # Redirect logging to stderr — stdout is the MCP JSON-RPC channel
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from paperbot.mcp.server import mcp  # late import to avoid circular
    if mcp is None:
        print(
            "Error: mcp package not installed. "
            "Run: pip install 'mcp[fastmcp]>=1.8.0,<2.0.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8001) -> None:
    """Start MCP server on Streamable HTTP transport.

    Blocks until the server is stopped (Ctrl+C).
    MCP endpoint: http://{host}:{port}/mcp
    """
    from paperbot.mcp.server import mcp
    if mcp is None:
        print("Error: mcp package not installed.", file=sys.stderr)
        sys.exit(1)

    mcp.run(transport="streamable-http", host=host, port=port)
```

### pyproject.toml additions
```toml
# Add to [project] dependencies list:
"mcp[fastmcp]>=1.8.0,<2.0.0",

# Add as new top-level section:
[project.scripts]
paperbot = "paperbot.presentation.cli.main:run_cli"
```

### CLI addition (main.py excerpt)
```python
# Source: src/paperbot/presentation/cli/main.py — extend create_parser()

# After existing subparsers:
mcp_parser = subparsers.add_parser("mcp", help="MCP server commands")
mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="Available commands")

serve_parser = mcp_subparsers.add_parser("serve", help="Start MCP server")
serve_transport = serve_parser.add_mutually_exclusive_group(required=True)
serve_transport.add_argument(
    "--stdio", action="store_true",
    help="stdio transport — for Claude Desktop and Claude Code local integration",
)
serve_transport.add_argument(
    "--http", action="store_true",
    help="Streamable HTTP transport — for remote agents",
)
serve_parser.add_argument(
    "--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)"
)
serve_parser.add_argument(
    "--port", type=int, default=8001, help="HTTP port (default: 8001)"
)
```

```python
# In run_cli() dispatch block:
elif parsed.command == "mcp":
    if not getattr(parsed, "mcp_command", None):
        mcp_parser.print_help()
        return 0
    if parsed.mcp_command == "serve":
        return _run_mcp_serve(parsed)
    mcp_parser.print_help()
    return 0

# Handler function:
def _run_mcp_serve(parsed: argparse.Namespace) -> int:
    from paperbot.mcp.serve import run_stdio, run_http
    if parsed.stdio:
        run_stdio()  # blocks
    else:
        run_http(host=parsed.host, port=parsed.port)  # blocks
    return 0
```

### claude_desktop_config.json (for users)
```json
{
  "mcpServers": {
    "paperbot": {
      "command": "paperbot",
      "args": ["mcp", "serve", "--stdio"],
      "env": {
        "PAPERBOT_DB_URL": "sqlite:////Users/yourname/PaperBot/data/paperbot.db",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Note for Claude Code users: Claude Code reads MCP config from `~/.claude.json` under `"mcpServers"` key, not `claude_desktop_config.json`. Same format applies.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SSE transport (`transport="sse"`) | Streamable HTTP (`transport="streamable-http"`) | 2024–2025 MCP spec evolution | SSE deprecated; new projects MUST use streamable-http |
| Manual stdio framing | `mcp.run(transport="stdio")` built-in | MCP SDK v1.x | No hand-rolling needed |
| Separate `fastmcp` PyPI package | `mcp[fastmcp]` from official SDK | 2024 (FastMCP 1.0 merged into SDK) | Import path is `mcp.server.fastmcp.FastMCP` |

**Deprecated:**
- `transport="sse"`: Do not use in new code. SSE exists for backward compatibility only.
- Standalone `fastmcp` PyPI package (v3.x): This is a different, independently maintained project by PrefectHQ (`from fastmcp import FastMCP`). PaperBot uses the official Anthropic SDK (`from mcp.server.fastmcp import FastMCP`). Do not mix the two.

---

## Open Questions

1. **Default HTTP port: 8000 or 8001?**
   - What we know: PaperBot FastAPI server uses `--port 8000` (from CLAUDE.md). FastMCP HTTP defaults to `8000` in official examples.
   - What's unclear: Will users run both the FastAPI server and the MCP HTTP server simultaneously?
   - Recommendation: Default MCP HTTP port to `8001` to avoid collision with the existing FastAPI server at `8000`.

2. **`mcp.run()` parameters for Streamable HTTP — path configuration?**
   - What we know: The MCP endpoint appears at `/mcp` with Streamable HTTP transport. There is no documented parameter to change the path.
   - What's unclear: Whether `FastMCP` in the `mcp` SDK v1.x exposes a `path` parameter for the HTTP endpoint (standalone fastmcp does, but the APIs diverge after v1.0).
   - Recommendation: Accept `/mcp` as the endpoint path — it's the standard. If path configuration is needed, document it as a Phase 5 open item.

3. **`mcp` package version pin — v1.x vs v2.x**
   - What we know: Current PyPI version of `mcp` is 1.26.0 (Jan 2026). Previous phases established `>=1.8.0,<2.0.0`.
   - What's unclear: Whether v1.26.0 has any breaking changes in the `FastMCP.run()` API compared to what was used in Phases 2–4.
   - Recommendation: Maintain the established pin `>=1.8.0,<2.0.0` for consistency with all previous phases. Verify `transport="streamable-http"` string works in v1.26.0 (verified via official MCP Python SDK docs).

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest with pytest-asyncio (asyncio_mode = "strict") |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` |
| Quick run command | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py -q` |
| Full suite command | `PYTHONPATH=src pytest -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCP-10 | `run_stdio()` calls `mcp.run(transport="stdio")` | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestRunStdio -x` | ❌ Wave 0 |
| MCP-10 | `run_stdio()` exits with error when `mcp is None` | unit | same | ❌ Wave 0 |
| MCP-10 | `run_stdio()` configures logging to stderr (not stdout) | unit | same | ❌ Wave 0 |
| MCP-11 | `run_http(host, port)` calls `mcp.run(transport="streamable-http", host=..., port=...)` | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestRunHttp -x` | ❌ Wave 0 |
| MCP-11 | `run_http()` exits with error when `mcp is None` | unit | same | ❌ Wave 0 |
| MCP-12 | `paperbot mcp serve --stdio` parses to `run_stdio()` call | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestCLIServeCommand -x` | ❌ Wave 0 |
| MCP-12 | `paperbot mcp serve --http --port 9000` parses to `run_http(port=9000)` call | unit | same | ❌ Wave 0 |
| MCP-12 | `paperbot mcp serve` (no flag) shows help without error | unit | same | ❌ Wave 0 |
| MCP-12 | `paperbot mcp` (no subcommand) shows help without error | unit | same | ❌ Wave 0 |
| MCP-10/11 | `serve.py` module exists and exports `run_stdio` and `run_http` | unit (import) | `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py::TestServeModuleImport -x` | ❌ Wave 0 |
| MCP-10/11/12 | `pyproject.toml` contains `[project.scripts]` with `paperbot` entry | static check | `grep -q 'paperbot' pyproject.toml` | ❌ Wave 0 |

### Test Implementation Notes
Tests MUST NOT call `run_stdio()` or `run_http()` directly — those block indefinitely. Instead:
- Inject a mock for `mcp.run` via monkeypatching: `monkeypatch.setattr("paperbot.mcp.server.mcp", FakeMCP())`
- Or test that the correct arguments are passed to `mcp.run` by capturing the call

```python
# Example test pattern (does NOT block):
class _FakeMCP:
    def __init__(self):
        self.run_calls = []

    def run(self, transport, **kwargs):
        self.run_calls.append({"transport": transport, **kwargs})

def test_run_stdio_calls_correct_transport(monkeypatch):
    fake = _FakeMCP()
    monkeypatch.setattr("paperbot.mcp.server.mcp", fake)
    from paperbot.mcp import serve
    import importlib
    importlib.reload(serve)  # reload to pick up monkeypatched mcp
    serve.run_stdio()
    assert fake.run_calls == [{"transport": "stdio"}]
```

### Sampling Rate
- **Per task commit:** `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py -q`
- **Per wave merge:** `PYTHONPATH=src pytest tests/unit/test_mcp_serve_cli.py tests/unit/test_mcp_bootstrap.py -q`
- **Phase gate:** Full CI offline suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/paperbot/mcp/serve.py` — `run_stdio()` and `run_http()` functions
- [ ] `tests/unit/test_mcp_serve_cli.py` — covers MCP-10, MCP-11, MCP-12 (all tests)
- [ ] `pyproject.toml` — add `[project.scripts]` section and `mcp[fastmcp]` dependency
- [ ] `requirements.txt` — add `mcp[fastmcp]>=1.8.0,<2.0.0`

---

## Sources

### Primary (HIGH confidence)
- `src/paperbot/mcp/server.py` — FastMCP singleton, `mcp = FastMCP("paperbot")`, confirmed `mcp=None` stub when package absent
- `src/paperbot/presentation/cli/main.py` — existing argparse structure, `create_parser()` + `run_cli()` pattern
- `pyproject.toml` — confirmed absence of `[project.scripts]` and `mcp` package in deps
- [MCP Python SDK quickstart](https://modelcontextprotocol.io/quickstart/server) — `mcp.run(transport="stdio")` pattern, `claude_desktop_config.json` format (verified from official docs)
- [FastMCP running-server docs](https://gofastmcp.com/deployment/running-server) — `mcp.run(transport="http", host=..., port=...)`, endpoint at `/mcp`, stdio as default
- `.planning/phases/03-remaining-mcp-tools/03-RESEARCH.md` — established `mcp[fastmcp]>=1.8.0,<2.0.0` pin
- `.planning/phases/04-mcp-resources/04-RESEARCH.md` — confirmed same mcp package version

### Secondary (MEDIUM confidence)
- [MCP Python SDK PyPI page](https://pypi.org/project/mcp/) — v1.26.0 is current (Jan 2026); package name `mcp`, extras: `cli`, `rich`, `ws`
- [WebSearch results](https://github.com/modelcontextprotocol/python-sdk) — `transport="streamable-http"` confirmed as the correct string for Streamable HTTP (not "http" or "http-streamable")
- [setuptools entry_points docs](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) — `[project.scripts]` format for pyproject.toml

### Tertiary (LOW confidence)
- FastMCP HTTP default port (`8000`) — from official docs examples; marked LOW because it may be illustrative rather than a hard default; use explicit `port=8001` to avoid conflict

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — `mcp[fastmcp]` already in use; `argparse` already in use; no new packages needed
- `mcp.run()` API (stdio): HIGH — verified from official MCP Python SDK quickstart docs
- `mcp.run()` API (streamable-http): HIGH — verified from FastMCP docs and official SDK README
- CLI pattern: HIGH — directly extending existing argparse structure in codebase
- `[project.scripts]`: HIGH — standard setuptools pyproject.toml spec
- `claude_desktop_config.json` format: HIGH — verified from official MCP quickstart
- HTTP endpoint path (`/mcp`): MEDIUM — stated in official docs, but no source code verification
- Default port selection: MEDIUM — `8001` is a recommendation to avoid collision, not from official spec

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable MCP SDK API; argparse is stdlib)
