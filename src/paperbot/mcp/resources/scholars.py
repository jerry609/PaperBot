"""scholars MCP resource wrapping SubscriptionService.

Exposes paperbot://scholars as a read-only JSON resource.
Returns the list of tracked scholars from config/scholar_subscriptions.yaml.
"""

from __future__ import annotations

import json
import logging

import anyio

logger = logging.getLogger(__name__)

# Module-level service reference for test injection only.
# Set to None by default; tests set it to a fake service.
# Production code instantiates a fresh SubscriptionService() each call for fresh config reads.
_service = None


async def _scholars_impl() -> str:
    """Return tracked scholars in a stable JSON envelope.

    Returns:
        JSON string with a stable ``{"scholars": [...], "error": ...}`` shape.
    """
    # Use injected service for tests; otherwise instantiate fresh for each call
    # to ensure we always read the latest config file.
    if _service is not None:
        service = _service
    else:
        from paperbot.infrastructure.services.subscription_service import SubscriptionService

        service = SubscriptionService()

    try:
        scholars = await anyio.to_thread.run_sync(service.get_scholar_configs)
        return json.dumps({"scholars": scholars, "error": None})
    except FileNotFoundError:
        return json.dumps({"error": "Scholar config not found", "scholars": []})


def register(mcp) -> None:
    """Register the scholars resource on the given FastMCP instance."""

    @mcp.resource("paperbot://scholars", mime_type="application/json")
    async def scholars() -> str:
        """Return the list of PaperBot tracked scholars.

        Returns scholar configurations including name, semantic_scholar_id, and
        keyword interests in a stable envelope with optional error information.
        """
        return await _scholars_impl()
