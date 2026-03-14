"""get_research_context MCP tool wrapping ContextEngine.

Retrieves research context for a query including relevant papers and memories.
Uses direct async await since ContextEngine.build_context_pack() is already async.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the context engine
_engine = None
_paper_store = None
_paper_search_service = None
_document_index_store = None
_query_grounder = None


def _get_paper_store():
    """Construct PaperStore on first call (lazy singleton)."""
    global _paper_store
    if _paper_store is None:
        from paperbot.infrastructure.stores.paper_store import PaperStore

        _paper_store = PaperStore()
    return _paper_store


def _get_paper_search_service():
    """Construct PaperSearchService on first call (lazy singleton)."""
    global _paper_search_service
    if _paper_search_service is None:
        from paperbot.application.services.paper_search_service import PaperSearchService
        from paperbot.infrastructure.adapters import build_adapter_registry

        _paper_search_service = PaperSearchService(
            adapters=build_adapter_registry(),
            registry=_get_paper_store(),
        )
    return _paper_search_service


def _get_document_index_store():
    """Construct DocumentIndexStore on first call (lazy singleton)."""
    global _document_index_store
    if _document_index_store is None:
        from paperbot.infrastructure.stores.document_index_store import DocumentIndexStore

        _document_index_store = DocumentIndexStore()
    return _document_index_store


def _get_workflow_query_grounder():
    """Construct WorkflowQueryGrounder on first call (lazy singleton)."""
    global _query_grounder
    if _query_grounder is None:
        from paperbot.application.services.workflow_query_grounder import WorkflowQueryGrounder
        from paperbot.application.services.wiki_concept_service import WikiConceptService
        from paperbot.infrastructure.stores.wiki_concept_store import WikiConceptStore

        _query_grounder = WorkflowQueryGrounder(
            concept_service=WikiConceptService(WikiConceptStore())
        )
    return _query_grounder


def _get_engine():
    """Construct ContextEngine on first call (lazy singleton)."""
    global _engine
    if _engine is None:
        from paperbot.context_engine.engine import ContextEngine, ContextEngineConfig

        _engine = ContextEngine(
            paper_store=_get_paper_store(),
            search_service=_get_paper_search_service(),
            evidence_retriever=_get_document_index_store(),
            query_grounder=_get_workflow_query_grounder(),
            config=ContextEngineConfig(),
        )
    return _engine


def _normalize_context_pack(result: Dict[str, Any]) -> Dict[str, Any]:
    """Expose stable MCP aliases while preserving ContextEngine-native keys."""
    normalized = dict(result)
    routing = normalized.get("routing") if isinstance(normalized.get("routing"), dict) else {}

    papers = normalized.get("papers")
    if papers is None:
        papers = normalized.get("paper_recommendations")
    if papers is None:
        papers = []

    memories = normalized.get("memories")
    if memories is None:
        memories = normalized.get("relevant_memories")
    if memories is None:
        memories = []

    track = normalized.get("track", normalized.get("active_track"))
    stage = normalized.get("stage", routing.get("stage"))
    routing_suggestion = normalized.get("routing_suggestion", routing.get("suggestion"))

    normalized.setdefault("paper_recommendations", papers)
    normalized.setdefault("relevant_memories", memories)
    normalized.setdefault("active_track", track)
    normalized["papers"] = papers
    normalized["memories"] = memories
    normalized["track"] = track
    normalized["stage"] = stage
    normalized["routing_suggestion"] = routing_suggestion
    return normalized


async def _get_research_context_impl(
    query: str,
    user_id: str = "default",
    track_id: Optional[int] = None,
    _run_id: str = "",
) -> Dict[str, Any]:
    """Core implementation of get_research_context, callable from both MCP registration and tests.

    Retrieve research context for a query, including relevant papers and memories.

    Args:
        query: The research query string.
        user_id: User identifier for personalized context.
        track_id: Optional track ID to scope the context retrieval.
        _run_id: Optional run ID for event correlation.

    Returns:
        Context pack containing ContextEngine-native keys plus stable MCP aliases
        for papers, memories, track, stage, and routing_suggestion.
    """
    start = time.monotonic()
    args = {"query": query, "user_id": user_id, "track_id": track_id}

    try:
        engine = _get_engine()
        result = await engine.build_context_pack(
            user_id=user_id,
            query=query,
            track_id=track_id,
        )
        normalized = _normalize_context_pack(result)

        log_tool_call(
            tool_name="get_research_context",
            arguments=args,
            result_summary={
                "paper_count": len(normalized.get("papers") or []),
                "memory_count": len(normalized.get("memories") or []),
                "stage": normalized.get("stage"),
            },
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return normalized

    except Exception as exc:
        log_tool_call(
            tool_name="get_research_context",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the get_research_context tool on the given FastMCP instance."""

    @mcp.tool()
    async def get_research_context(
        query: str,
        user_id: str = "default",
        track_id: Optional[int] = None,
        _run_id: str = "",
    ) -> dict:
        """Retrieve research context for a query, including relevant papers and memories.

        Returns a context pack dict with ContextEngine-native fields plus MCP
        compatibility aliases for papers, memories, track info, research stage,
        and routing suggestions.
        """
        return await _get_research_context_impl(query, user_id, track_id, _run_id)
