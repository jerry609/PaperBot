"""Unit tests for MCP serve module and CLI subcommand.

Covers:
  - serve.py module imports and exports
  - run_stdio() / run_http() dispatch to mcp.run() with correct args
  - stdio mode configures logging to stderr
  - None guard exits with error message when mcp is unavailable
  - pyproject.toml has [project.scripts] entry and mcp[fastmcp] dependency
  - requirements.txt includes mcp[fastmcp]
  - CLI parses `paperbot mcp serve --stdio` / `--http` correctly
  - CLI dispatch calls run_stdio() / run_http() with correct args
  - Mutual exclusion of --stdio and --http
  - `paperbot mcp` (no subcommand) prints help and returns 0
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent


class _FakeMCP:
    """Minimal stub for mcp singleton — records .run() calls without blocking."""

    def __init__(self) -> None:
        self.run_calls: List[Tuple[tuple, dict]] = []

    def run(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self.run_calls.append((args, kwargs))


# ---------------------------------------------------------------------------
# Task 1 tests: serve.py module
# ---------------------------------------------------------------------------


class TestServeModuleImport:
    def test_import_run_stdio_and_run_http(self):
        """from paperbot.mcp.serve import run_stdio, run_http succeeds."""
        from paperbot.mcp.serve import run_http, run_stdio

        assert callable(run_stdio)
        assert callable(run_http)


class TestRunStdio:
    def test_calls_mcp_run_with_stdio_transport(self, monkeypatch):
        """run_stdio() calls mcp.run(transport='stdio')."""
        fake = _FakeMCP()
        monkeypatch.setattr("paperbot.mcp.server.mcp", fake)

        import paperbot.mcp.serve as serve_mod

        importlib.reload(serve_mod)

        # Patch the module-level reference that serve_mod uses
        monkeypatch.setattr(serve_mod, "_get_mcp", lambda: fake)

        serve_mod.run_stdio()

        assert len(fake.run_calls) == 1
        _, kwargs = fake.run_calls[0]
        assert kwargs.get("transport") == "stdio"

    def test_configures_logging_to_stderr(self, monkeypatch):
        """run_stdio() calls logging.basicConfig(stream=sys.stderr, ...)."""
        fake = _FakeMCP()
        import paperbot.mcp.serve as serve_mod

        importlib.reload(serve_mod)
        monkeypatch.setattr(serve_mod, "_get_mcp", lambda: fake)

        captured_calls: List[dict] = []
        original_basicConfig = logging.basicConfig

        def fake_basicConfig(**kwargs):
            captured_calls.append(kwargs)

        monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)

        serve_mod.run_stdio()

        assert any(c.get("stream") is sys.stderr for c in captured_calls)


class TestRunHttp:
    def test_calls_mcp_run_with_default_host_port(self, monkeypatch):
        """run_http() calls mcp.run(transport='streamable-http', host='127.0.0.1', port=8001)."""
        fake = _FakeMCP()
        import paperbot.mcp.serve as serve_mod

        importlib.reload(serve_mod)
        monkeypatch.setattr(serve_mod, "_get_mcp", lambda: fake)

        serve_mod.run_http()

        assert len(fake.run_calls) == 1
        _, kwargs = fake.run_calls[0]
        assert kwargs.get("transport") == "streamable-http"
        assert kwargs.get("host") == "127.0.0.1"
        assert kwargs.get("port") == 8001

    def test_passes_custom_host_and_port(self, monkeypatch):
        """run_http(host='0.0.0.0', port=9000) passes those values to mcp.run()."""
        fake = _FakeMCP()
        import paperbot.mcp.serve as serve_mod

        importlib.reload(serve_mod)
        monkeypatch.setattr(serve_mod, "_get_mcp", lambda: fake)

        serve_mod.run_http(host="0.0.0.0", port=9000)

        assert len(fake.run_calls) == 1
        _, kwargs = fake.run_calls[0]
        assert kwargs.get("host") == "0.0.0.0"
        assert kwargs.get("port") == 9000


class TestMcpNoneGuard:
    def test_run_stdio_exits_when_mcp_none(self, monkeypatch, capsys):
        """run_stdio() prints error to stderr and exits 1 when mcp is None."""
        import paperbot.mcp.serve as serve_mod

        importlib.reload(serve_mod)
        monkeypatch.setattr(serve_mod, "_get_mcp", lambda: None)

        with pytest.raises(SystemExit) as exc_info:
            serve_mod.run_stdio()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.err  # some message written to stderr

    def test_run_http_exits_when_mcp_none(self, monkeypatch, capsys):
        """run_http() prints error to stderr and exits 1 when mcp is None."""
        import paperbot.mcp.serve as serve_mod

        importlib.reload(serve_mod)
        monkeypatch.setattr(serve_mod, "_get_mcp", lambda: None)

        with pytest.raises(SystemExit) as exc_info:
            serve_mod.run_http()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.err


class TestPyprojectScripts:
    def test_project_scripts_entry_exists(self):
        """pyproject.toml has [project.scripts] with paperbot = '...:run_cli'."""
        content = (REPO_ROOT / "pyproject.toml").read_text()
        assert "[project.scripts]" in content
        assert "paperbot" in content
        assert "run_cli" in content

    def test_mcp_fastmcp_in_dependencies(self):
        """pyproject.toml dependencies include mcp[fastmcp]."""
        content = (REPO_ROOT / "pyproject.toml").read_text()
        assert "mcp[fastmcp]" in content

    def test_requirements_txt_has_mcp_fastmcp(self):
        """requirements.txt includes mcp[fastmcp]."""
        content = (REPO_ROOT / "requirements.txt").read_text()
        assert "mcp[fastmcp]" in content


# ---------------------------------------------------------------------------
# Task 2 tests: CLI subcommand
# ---------------------------------------------------------------------------


class TestCLIServeCommand:
    def _parser(self):
        from paperbot.presentation.cli.main import create_parser

        return create_parser()

    def test_parse_mcp_serve_stdio(self):
        """parse_args(['mcp', 'serve', '--stdio']) sets expected attributes."""
        parsed = self._parser().parse_args(["mcp", "serve", "--stdio"])
        assert parsed.command == "mcp"
        assert parsed.mcp_command == "serve"
        assert parsed.stdio is True
        assert parsed.http is False

    def test_parse_mcp_serve_http_defaults(self):
        """parse_args(['mcp', 'serve', '--http']) sets http=True, default host/port."""
        parsed = self._parser().parse_args(["mcp", "serve", "--http"])
        assert parsed.http is True
        assert parsed.stdio is False
        assert parsed.host == "127.0.0.1"
        assert parsed.port == 8001

    def test_parse_mcp_serve_http_custom_host_port(self):
        """parse_args with --host/--port passes custom values."""
        parsed = self._parser().parse_args(
            ["mcp", "serve", "--http", "--host", "0.0.0.0", "--port", "9000"]
        )
        assert parsed.host == "0.0.0.0"
        assert parsed.port == 9000

    def test_parse_mcp_serve_mutually_exclusive(self):
        """--stdio and --http are mutually exclusive."""
        with pytest.raises(SystemExit):
            self._parser().parse_args(["mcp", "serve", "--stdio", "--http"])

    def test_run_cli_mcp_serve_stdio_dispatches(self, monkeypatch):
        """run_cli(['mcp', 'serve', '--stdio']) calls run_stdio()."""
        called_with: List[dict] = []

        def fake_run_stdio():
            called_with.append({"fn": "run_stdio"})

        monkeypatch.setattr("paperbot.mcp.serve.run_stdio", fake_run_stdio)

        from paperbot.presentation.cli.main import run_cli

        run_cli(["mcp", "serve", "--stdio"])

        assert len(called_with) == 1
        assert called_with[0]["fn"] == "run_stdio"

    def test_run_cli_mcp_serve_http_dispatches(self, monkeypatch):
        """run_cli(['mcp', 'serve', '--http', '--port', '9000']) calls run_http with correct args."""
        called_with: List[dict] = []

        def fake_run_http(host: str = "127.0.0.1", port: int = 8001):
            called_with.append({"fn": "run_http", "host": host, "port": port})

        monkeypatch.setattr("paperbot.mcp.serve.run_http", fake_run_http)

        from paperbot.presentation.cli.main import run_cli

        run_cli(["mcp", "serve", "--http", "--port", "9000"])

        assert len(called_with) == 1
        assert called_with[0]["fn"] == "run_http"
        assert called_with[0]["host"] == "127.0.0.1"
        assert called_with[0]["port"] == 9000

    def test_run_cli_mcp_no_subcommand_returns_zero(self):
        """run_cli(['mcp']) prints help and returns 0."""
        from paperbot.presentation.cli.main import run_cli

        result = run_cli(["mcp"])
        assert result == 0

    def test_run_cli_mcp_serve_no_transport_exits_nonzero(self):
        """run_cli(['mcp', 'serve']) with no --stdio/--http exits or returns non-zero."""
        from paperbot.presentation.cli.main import run_cli

        try:
            result = run_cli(["mcp", "serve"])
            assert result != 0
        except SystemExit as e:
            assert e.code != 0
