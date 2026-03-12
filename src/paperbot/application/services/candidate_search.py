"""Shared candidate-search and explicit-ingest helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence

from paperbot.application.ports.paper_registry_port import RegistryPort
from paperbot.application.services.paper_search_service import PaperSearchService, SearchResult

_PREFERRED_IDENTITY_SOURCES: tuple[str, ...] = (
    "semantic_scholar",
    "arxiv",
    "openalex",
    "papers_cool",
    "hf_daily",
    "doi",
)


async def search_candidate_papers(
    search_service: PaperSearchService,
    *,
    query: str,
    sources: Sequence[str] | None = None,
    max_results: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    source_weights: Optional[Dict[str, float]] = None,
    persist: bool = False,
) -> SearchResult:
    """Fetch candidate papers without implicitly ingesting them by default."""

    selected_sources = [str(source).strip() for source in (sources or []) if str(source).strip()]
    return await search_service.search(
        query,
        sources=selected_sources or None,
        max_results=max_results,
        year_from=year_from,
        year_to=year_to,
        persist=bool(persist),
        source_weights=source_weights,
    )


def resolve_candidate_paper_id(paper: Dict[str, Any]) -> str:
    """Resolve the best available identifier for a candidate paper."""

    canonical_id = paper.get("canonical_paper_id") or paper.get("canonical_id")
    if canonical_id not in (None, ""):
        return str(canonical_id)

    current_id = str(paper.get("paper_id") or "").strip()
    if current_id:
        return current_id

    identities = paper.get("identities") or []
    by_source: Dict[str, str] = {}
    if isinstance(identities, list):
        for identity in identities:
            if not isinstance(identity, dict):
                continue
            source = str(identity.get("source") or "").strip().lower()
            external_id = str(identity.get("external_id") or "").strip()
            if source and external_id and source not in by_source:
                by_source[source] = external_id

    for source in _PREFERRED_IDENTITY_SOURCES:
        value = by_source.get(source)
        if value:
            return value

    title_hash = str(paper.get("title_hash") or "").strip()
    if title_hash:
        return f"title:{title_hash}"

    title = str(paper.get("title") or "").strip()
    if title:
        return f"title:{title.lower()}"

    return ""


def resolve_existing_canonical_paper_id(
    paper: Dict[str, Any],
    *,
    registry: Optional[RegistryPort] = None,
) -> Optional[int]:
    """Best-effort canonical id lookup without creating new rows."""

    canonical_id = paper.get("canonical_paper_id") or paper.get("canonical_id")
    if canonical_id not in (None, ""):
        try:
            return int(canonical_id)
        except (TypeError, ValueError):
            return None

    if registry is None:
        return None

    lookup_by_source_id = getattr(registry, "get_paper_by_source_id_any", None)
    if not callable(lookup_by_source_id):
        return None

    candidate_ids = [resolve_candidate_paper_id(paper)]
    identities = paper.get("identities") or []
    if isinstance(identities, list):
        for identity in identities:
            if not isinstance(identity, dict):
                continue
            external_id = str(identity.get("external_id") or "").strip()
            if external_id:
                candidate_ids.append(external_id)

    for candidate_id in candidate_ids:
        if not candidate_id:
            continue
        row = lookup_by_source_id(candidate_id)
        if row is None:
            continue
        try:
            return int(getattr(row, "id"))
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def search_result_to_candidate_dicts(
    search_result: SearchResult,
    *,
    registry: Optional[RegistryPort] = None,
) -> list[Dict[str, Any]]:
    """Convert SearchResult into candidate dictionaries with stable ids."""

    rows: list[Dict[str, Any]] = []
    for paper in search_result.papers:
        payload = paper.to_dict()
        canonical_id = resolve_existing_canonical_paper_id(payload, registry=registry)
        if canonical_id is not None:
            payload["canonical_id"] = canonical_id
            payload["canonical_paper_id"] = canonical_id
            payload["paper_id"] = str(canonical_id)
        else:
            payload["paper_id"] = resolve_candidate_paper_id(payload)
        rows.append(payload)
    return rows


def ingest_candidate_papers(
    *,
    papers: Iterable[Dict[str, Any]],
    registry: RegistryPort,
    source_hint: Optional[str] = None,
    seen_at: Optional[datetime] = None,
) -> Dict[str, int]:
    """Explicitly ingest candidate papers into the canonical registry."""

    return registry.upsert_many(
        papers=papers,
        source_hint=source_hint,
        seen_at=seen_at,
    )
