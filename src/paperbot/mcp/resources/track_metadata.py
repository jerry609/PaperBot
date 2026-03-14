"""track_metadata MCP resource wrapping SqlAlchemyResearchStore.

Exposes paperbot://track/{track_id} as a read-only JSON resource.
Returns track metadata including name, description, keywords, venues, methods.
"""

from __future__ import annotations

import json
import logging

import anyio

from paperbot.utils.user_identity import require_user_identity

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the research store (can be overridden in tests)
_store = None


def _get_store():
    """Construct SqlAlchemyResearchStore on first call (lazy singleton)."""
    global _store
    if _store is None:
        from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore

        _store = SqlAlchemyResearchStore()
    return _store


async def _track_metadata_impl(user_id: str, track_id: str) -> str:
    """Return JSON metadata for the given track.

    Args:
        track_id: Track identifier as a string (will be cast to int).

    Returns:
        JSON string with track fields, or JSON error object.
    """
    try:
        tid = int(track_id)
    except (ValueError, TypeError):
        return json.dumps({"error": f"Invalid track_id: {track_id!r}. Must be an integer."})

    resolved_user_id = require_user_identity(user_id)
    store = _get_store()
    track = await anyio.to_thread.run_sync(
        lambda: store.get_track(user_id=resolved_user_id, track_id=tid)
    )

    if track is None or track.get("archived_at") is not None:
        return json.dumps({"error": f"Track {tid} not found."})

    return json.dumps(track)


def register(mcp) -> None:
    """Register the track_metadata resource on the given FastMCP instance."""

    @mcp.resource("paperbot://users/{user_id}/tracks/{track_id}", mime_type="application/json")
    async def track_metadata(user_id: str, track_id: str) -> str:
        """Return metadata for a PaperBot research track.

        Provides track name, description, keywords, venues, methods, and status.
        Use this to understand what a track monitors before fetching its papers.
        """
        return await _track_metadata_impl(user_id, track_id)
