"""Unit tests for MCP server bootstrap and tool registration."""

import importlib


class TestMCPServerBootstrap:
    def test_server_module_imports_without_error(self):
        """server.py imports cleanly regardless of mcp package availability."""
        mod = importlib.import_module("paperbot.mcp.server")
        # mcp may be None (stub) or a FastMCP instance
        assert hasattr(mod, "mcp")

    def test_paper_search_register_function_exists(self):
        """paper_search module exposes a register() function."""
        from paperbot.mcp.tools.paper_search import register

        assert callable(register)

    def test_audit_log_tool_call_importable(self):
        """_audit module exposes log_tool_call function."""
        from paperbot.mcp.tools._audit import log_tool_call

        assert callable(log_tool_call)
