"""save_to_memory MCP tool wrapping SqlAlchemyMemoryStore.

Persists research findings, notes, and structured knowledge to the memory store.
Uses anyio.to_thread.run_sync() to wrap the synchronous store.add_memories() call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import anyio

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Allowed MemoryKind values (must match paperbot.memory.schema.MemoryKind)
_ALLOWED_KINDS = frozenset(
    [
        "profile",
        "preference",
        "goal",
        "project",
        "constraint",
        "todo",
        "fact",
        "note",
        "decision",
        "hypothesis",
        "keyword_set",
    ]
)

# Module-level lazy singleton for the memory store
_store = None


def _get_store():
    """Construct SqlAlchemyMemoryStore on first call (lazy singleton)."""
    global _store
    if _store is None:
        from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore

        _store = SqlAlchemyMemoryStore()
    return _store


async def _save_to_memory_impl(
    content: str,
    kind: str = "note",
    user_id: str = "default",
    scope_type: str = "global",
    scope_id: str = "",
    confidence: float = 0.8,
    _run_id: str = "",
) -> Dict[str, Any]:
    """Core implementation of save_to_memory, callable from both MCP registration and tests.

    Save research findings to memory for later retrieval.

    Args:
        content: The content to store in memory.
        kind: Memory kind. One of: profile, preference, goal, project, constraint,
              todo, fact, note, decision, hypothesis, keyword_set. Defaults to 'note'.
        user_id: User identifier to scope the memory.
        scope_type: Scope type: global, track, project, or paper. Defaults to 'global'.
        scope_id: Scope identifier (e.g., track ID or project ID).
        confidence: Confidence score for the memory candidate (0.0-1.0).
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with keys: saved (bool), created (int), skipped (int).
    """
    start = time.monotonic()

    # Validate kind; default to "note" if invalid
    effective_kind = kind if kind in _ALLOWED_KINDS else "note"
    if kind not in _ALLOWED_KINDS:
        logger.warning(
            "save_to_memory: invalid kind=%r; defaulting to 'note'. " "Allowed values: %s",
            kind,
            sorted(_ALLOWED_KINDS),
        )

    args = {
        "content_len": len(content),
        "kind": effective_kind,
        "user_id": user_id,
        "scope_type": scope_type,
        "confidence": confidence,
    }

    try:
        from paperbot.memory.schema import MemoryCandidate

        normalized_confidence = float(confidence)
        if not (0.0 <= normalized_confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        candidate = MemoryCandidate(
            kind=effective_kind,
            content=content,
            confidence=normalized_confidence,
            scope_type=scope_type or None,
            scope_id=scope_id or None,
        )

        store = _get_store()
        created_count, skipped_count, _rows = await anyio.to_thread.run_sync(
            lambda: store.add_memories(user_id=user_id, memories=[candidate])
        )

        output = {"saved": True, "created": created_count, "skipped": skipped_count}

        log_tool_call(
            tool_name="save_to_memory",
            arguments=args,
            result_summary=output,
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="save_to_memory",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the save_to_memory tool on the given FastMCP instance."""

    @mcp.tool()
    async def save_to_memory(
        content: str,
        kind: str = "note",
        user_id: str = "default",
        scope_type: str = "global",
        scope_id: str = "",
        confidence: float = 0.8,
        _run_id: str = "",
    ) -> dict:
        """Save research findings to memory for later retrieval.

        Persists content as a typed memory candidate. Use kind to categorize
        (e.g., 'note', 'hypothesis', 'decision', 'keyword_set'). Invalid kinds
        default to 'note'. Returns created and skipped counts.
        """
        return await _save_to_memory_impl(
            content, kind, user_id, scope_type, scope_id, confidence, _run_id
        )
