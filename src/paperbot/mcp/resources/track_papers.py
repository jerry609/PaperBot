"""track_papers MCP resource wrapping SqlAlchemyResearchStore.

Exposes paperbot://track/{track_id}/papers as a read-only JSON resource.
Returns the list of papers in a track's feed.
"""

from __future__ import annotations

import json
import logging

import anyio

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


async def _track_papers_impl(track_id: str) -> str:
    """Return JSON list of papers in the given track's feed.

    Args:
        track_id: Track identifier as a string (will be cast to int).

    Returns:
        JSON string with items list and total count, or JSON error object.
    """
    try:
        tid = int(track_id)
    except (ValueError, TypeError):
        return json.dumps({"error": f"Invalid track_id: {track_id!r}. Must be an integer."})

    store = _get_store()
    track = await anyio.to_thread.run_sync(lambda: store.get_track(user_id="default", track_id=tid))
    if track is None or track.get("archived_at") is not None:
        return json.dumps({"error": f"Track {tid} not found."})

    feed = await anyio.to_thread.run_sync(
        lambda: store.list_track_feed(user_id="default", track_id=tid, limit=50)
    )

    return json.dumps(feed)


def register(mcp) -> None:
    """Register the track_papers resource on the given FastMCP instance."""

    @mcp.resource("paperbot://track/{track_id}/papers", mime_type="application/json")
    async def track_papers(track_id: str) -> str:
        """Return papers in a PaperBot research track's feed.

        Returns up to 50 most recent papers with metadata including title, authors,
        abstract, arxiv_id, and relevance scores. Use track_metadata first to verify
        the track exists.
        """
        return await _track_papers_impl(track_id)
