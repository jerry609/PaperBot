"""track_memory MCP resource wrapping SqlAlchemyMemoryStore.

Exposes paperbot://track/{track_id}/memory as a read-only JSON resource.
Returns memories scoped to a specific track.
"""

from __future__ import annotations

import json
import logging

import anyio

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the memory store (can be overridden in tests)
_store = None


def _get_store():
    """Construct SqlAlchemyMemoryStore on first call (lazy singleton)."""
    global _store
    if _store is None:
        from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore

        _store = SqlAlchemyMemoryStore()
    return _store


async def _track_memory_impl(track_id: str) -> str:
    """Return JSON list of memories scoped to the given track.

    Args:
        track_id: Track identifier as a string (will be cast to int).

    Returns:
        JSON string with list of memory dicts, or JSON error object.
    """
    try:
        tid = int(track_id)
    except (ValueError, TypeError):
        return json.dumps({"error": f"Invalid track_id: {track_id!r}. Must be an integer."})

    store = _get_store()
    memories = await anyio.to_thread.run_sync(
        lambda: store.list_memories(
            user_id="default",
            scope_type="track",
            scope_id=str(tid),
            limit=100,
        )
    )

    return json.dumps(memories)


def register(mcp) -> None:
    """Register the track_memory resource on the given FastMCP instance."""

    @mcp.resource("paperbot://track/{track_id}/memory", mime_type="application/json")
    async def track_memory(track_id: str) -> str:
        """Return memories scoped to a PaperBot research track.

        Returns approved, non-expired memory entries (notes, hypotheses, decisions, etc.)
        that were recorded in the context of this track. Useful for retrieving persistent
        agent observations and research findings.
        """
        return await _track_memory_impl(track_id)
