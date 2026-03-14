"""Unit tests for the scholars MCP resource (MCP-09).

Tests _scholars_impl with a fake SubscriptionService injected via
the module-level _service singleton pattern.
"""

import json

import pytest


class _FakeSubscriptionService:
    """SubscriptionService stub returning a canned scholar list."""

    def get_scholar_configs(self):
        return [
            {"name": "Dawn Song", "semantic_scholar_id": "123", "keywords": ["security", "ML"]},
            {"name": "Yoshua Bengio", "semantic_scholar_id": "456", "keywords": ["deep learning"]},
        ]


class _FakeMissingConfigService:
    """SubscriptionService stub that raises FileNotFoundError (config file missing)."""

    def get_scholar_configs(self):
        raise FileNotFoundError("config/scholar_subscriptions.yaml not found")


class _FakeInvalidConfigService:
    """SubscriptionService stub that raises ValueError for malformed config."""

    def get_scholar_configs(self):
        raise ValueError("Invalid YAML in config file: unexpected token")


class TestScholarsResource:
    @pytest.mark.asyncio
    async def test_returns_scholar_list(self):
        """_scholars_impl() returns JSON list with name and semantic_scholar_id."""
        import paperbot.mcp.resources.scholars as mod

        mod._service = _FakeSubscriptionService()
        try:
            result = await mod._scholars_impl()
        finally:
            mod._service = None

        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "Dawn Song"
        assert data[0]["semantic_scholar_id"] == "123"

    @pytest.mark.asyncio
    async def test_returns_error_json_when_config_not_found(self):
        """_scholars_impl() returns JSON error when config file not found."""
        import paperbot.mcp.resources.scholars as mod

        mod._service = _FakeMissingConfigService()
        try:
            result = await mod._scholars_impl()
        finally:
            mod._service = None

        data = json.loads(result)
        assert "error" in data
        assert "scholars" in data
        assert data["scholars"] == []

    @pytest.mark.asyncio
    async def test_returns_error_json_when_config_is_invalid(self):
        """_scholars_impl() returns JSON error when config validation fails."""
        import paperbot.mcp.resources.scholars as mod

        mod._service = _FakeInvalidConfigService()
        try:
            result = await mod._scholars_impl()
        finally:
            mod._service = None

        data = json.loads(result)
        assert data["error"].startswith("Invalid YAML in config file:")
        assert data["scholars"] == []
