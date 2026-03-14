from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from paperbot.application.services.candidate_search import (
    resolve_candidate_paper_id,
    search_candidate_papers,
)
from paperbot.application.services.paper_search_service import PaperSearchService, SearchResult
from paperbot.utils.user_identity import has_user_identity, optional_user_identity

if TYPE_CHECKING:
    from paperbot.application.services.workflow_query_grounder import (
        GroundedQuery,
        WorkflowQueryGrounderPort,
    )


_QUERY_ALIASES = {
    "icl压缩": "icl compression",
    "icl 压缩": "icl compression",
    "icl 隐式偏置": "icl implicit bias",
    "icl隐式偏置": "icl implicit bias",
    "kv cache加速": "kv cache acceleration",
    "kv cache 加速": "kv cache acceleration",
}

_SOURCE_ALIASES = {
    "papers_cool": "papers_cool",
    "paperscool": "papers_cool",
    "arxiv": "arxiv",
    "arxiv_api": "arxiv",
    "hf_daily": "hf_daily",
    "openalex": "openalex",
    "semantic_scholar": "semantic_scholar",
    "s2": "semantic_scholar",
}


def make_default_search_service(*, registry=None) -> PaperSearchService:
    from paperbot.infrastructure.adapters import build_adapter_registry

    return PaperSearchService(adapters=build_adapter_registry(), registry=registry)


def normalize_topic_sources(sources: Sequence[str] | None) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for raw in sources or ["papers_cool"]:
        normalized = _SOURCE_ALIASES.get((raw or "").strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        names.append(normalized)
    if not names:
        names = ["papers_cool"]
    return names


def _normalize_query(query: str) -> str:
    base = re.sub(r"\s+", " ", (query or "").strip()).lower()
    return _QUERY_ALIASES.get(base, base)


def _tokenize_query(query: str) -> List[str]:
    seen: set[str] = set()
    tokens: List[str] = []
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _unique_preserve(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        v = (value or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _default_grounded_query(query: str) -> "GroundedQuery | Dict[str, Any]":
    cleaned_query = (query or "").strip()
    return {
        "original_query": cleaned_query,
        "canonical_query": cleaned_query,
        "search_queries": [cleaned_query] if cleaned_query else [],
        "concepts": [],
    }


def _matched_keywords(paper: Dict[str, Any], tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    blob = " ".join(
        [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or paper.get("snippet") or ""),
            " ".join(str(v) for v in (paper.get("keywords") or [])),
            " ".join(str(v) for v in (paper.get("fields_of_study") or [])),
        ]
    ).lower()
    return [tok for tok in tokens if tok in blob]


def _score_paper(paper: Dict[str, Any], query_tokens: List[str], matched: List[str]) -> float:
    token_score = (len(matched) / max(1, len(query_tokens))) * 3.0
    citation_count = 0
    try:
        citation_count = int(paper.get("citation_count") or 0)
    except Exception:
        citation_count = 0
    citation_score = min(1.25, math.log10(max(1, citation_count + 1)) / 3.0)

    year = None
    try:
        year = int(paper.get("year")) if paper.get("year") is not None else None
    except Exception:
        year = None
    if year is not None:
        age = max(0, datetime.now(timezone.utc).year - year)
        recency_score = max(0.0, 1.0 - age / 10.0)
    else:
        recency_score = 0.25

    return round(token_score + citation_score + recency_score, 4)


def _paper_id_from_item(item: Dict[str, Any]) -> str:
    return resolve_candidate_paper_id(item)


def _search_result_to_items(
    *,
    search_result: SearchResult,
    normalized_query: str,
    query_tokens: List[str],
    branches: Sequence[str],
    fallback_sources: List[str],
    min_score: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for paper in search_result.papers:
        p = paper.to_dict()
        key = p.get("title_hash") or p.get("title")
        provenances = search_result.provenance.get(str(key), fallback_sources)
        matched = _matched_keywords(p, query_tokens)
        score = _score_paper(p, query_tokens, matched)
        if score < float(min_score or 0.0):
            continue

        url = str(p.get("url") or "").strip()
        venue = str(p.get("venue") or "").strip()
        rows.append(
            {
                "paper_id": _paper_id_from_item(p),
                "title": str(p.get("title") or "").strip(),
                "url": url,
                "external_url": url,
                "pdf_url": str(p.get("pdf_url") or "").strip(),
                "authors": list(p.get("authors") or []),
                "subject_or_venue": venue,
                "published_at": p.get("publication_date")
                or (str(p.get("year")) if p.get("year") else ""),
                "snippet": str(p.get("abstract") or "").strip(),
                "keywords": _unique_preserve(
                    [
                        *[str(v) for v in (p.get("keywords") or [])],
                        *[str(v) for v in (p.get("fields_of_study") or [])],
                    ]
                ),
                "branches": list(branches or ["arxiv", "venue"]),
                "sources": _unique_preserve([str(v) for v in provenances]) or fallback_sources,
                "matched_keywords": matched,
                "matched_queries": [normalized_query],
                "score": score,
                "pdf_stars": 0,
                "kimi_stars": 0,
                "alternative_urls": [],
            }
        )

    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return rows


def _merge_item(target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    target["matched_queries"] = _unique_preserve(
        [*target.get("matched_queries", []), *incoming.get("matched_queries", [])]
    )
    target["matched_keywords"] = _unique_preserve(
        [*target.get("matched_keywords", []), *incoming.get("matched_keywords", [])]
    )
    target["branches"] = _unique_preserve(
        [*target.get("branches", []), *incoming.get("branches", [])]
    )
    target["sources"] = _unique_preserve([*target.get("sources", []), *incoming.get("sources", [])])
    target["keywords"] = _unique_preserve(
        [*target.get("keywords", []), *incoming.get("keywords", [])]
    )
    target["authors"] = _unique_preserve([*target.get("authors", []), *incoming.get("authors", [])])

    incoming_url = str(incoming.get("url") or "").strip()
    target_url = str(target.get("url") or "").strip()
    if incoming_url and incoming_url != target_url:
        target["alternative_urls"] = _unique_preserve(
            [*target.get("alternative_urls", []), incoming_url]
        )

    if float(incoming.get("score") or 0.0) > float(target.get("score") or 0.0):
        target["score"] = float(incoming.get("score") or 0.0)


async def run_unified_topic_search(
    *,
    queries: Sequence[str],
    user_id: Optional[str] = None,
    branches: Sequence[str] = ("arxiv", "venue"),
    sources: Sequence[str] = ("papers_cool",),
    top_k_per_query: int = 5,
    show_per_branch: int = 25,
    min_score: float = 0.0,
    search_service: Optional[PaperSearchService] = None,
    query_grounder: Optional["WorkflowQueryGrounderPort"] = None,
    persist: bool = False,
) -> Dict[str, Any]:
    normalized_sources = normalize_topic_sources(sources)
    resolved_user_id = optional_user_identity(user_id)

    query_specs: List[Dict[str, Any]] = []
    seen_queries: set[str] = set()
    for raw in queries:
        raw_query = (raw or "").strip()
        if not raw_query:
            continue
        grounded = (
            query_grounder.ground_query(user_id=resolved_user_id, query=raw_query)
            if query_grounder is not None and has_user_identity(resolved_user_id)
            else _default_grounded_query(raw_query)
        )
        canonical_query = str(
            grounded["canonical_query"] if isinstance(grounded, dict) else grounded.canonical_query
        )
        search_queries = list(
            grounded["search_queries"] if isinstance(grounded, dict) else grounded.search_queries
        )
        concepts = list(grounded["concepts"] if isinstance(grounded, dict) else grounded.concepts)
        normalized_query = _normalize_query(canonical_query or raw_query)
        if normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        query_specs.append(
            {
                "raw_query": raw_query,
                "canonical_query": canonical_query or raw_query,
                "normalized_query": normalized_query,
                "search_queries": _unique_preserve(
                    _normalize_query(value) for value in search_queries
                ),
                "grounded_concepts": [
                    concept.to_dict() if hasattr(concept, "to_dict") else dict(concept)
                    for concept in concepts
                ],
                "tokens": _tokenize_query(normalized_query),
            }
        )

    if not query_specs:
        return {
            "source": "papers.cool",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "sources": normalized_sources,
            "queries": [],
            "items": [],
            "summary": {
                "unique_items": 0,
                "total_query_hits": 0,
                "top_titles": [],
                "query_highlights": [],
                "source_breakdown": {},
                "source_errors": [],
            },
        }

    service = search_service or make_default_search_service()
    max_results = max(1, int(show_per_branch))

    tasks = [
        asyncio.gather(
            *[
                search_candidate_papers(
                    service,
                    query=search_query,
                    sources=normalized_sources,
                    max_results=max_results,
                    persist=bool(persist),
                )
                for search_query in (spec["search_queries"] or [spec["normalized_query"]])
            ]
        )
        for spec in query_specs
    ]
    search_results = await asyncio.gather(*tasks)

    query_views: List[Dict[str, Any]] = []
    aggregated: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}

    for spec, spec_search_results in zip(query_specs, search_results):
        query_items: List[Dict[str, Any]] = []
        query_items_by_key: Dict[str, Dict[str, Any]] = {}
        for search_query, search_result in zip(
            spec["search_queries"] or [spec["normalized_query"]],
            spec_search_results,
        ):
            current_items = _search_result_to_items(
                search_result=search_result,
                normalized_query=search_query,
                query_tokens=spec["tokens"],
                branches=branches,
                fallback_sources=normalized_sources,
                min_score=min_score,
            )
            for item in current_items:
                url = str(item.get("url") or "").strip().lower()
                title = str(item.get("title") or "").strip().lower()
                key = url or title
                if not key:
                    continue
                existing_query_item = query_items_by_key.get(key)
                if existing_query_item is None:
                    cloned_query_item = dict(item)
                    query_items_by_key[key] = cloned_query_item
                    query_items.append(cloned_query_item)
                else:
                    _merge_item(existing_query_item, item)

        query_items.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)

        query_views.append(
            {
                "raw_query": spec["raw_query"],
                "canonical_query": spec["canonical_query"],
                "normalized_query": spec["normalized_query"],
                "search_queries": spec["search_queries"] or [spec["normalized_query"]],
                "grounded_concepts": spec["grounded_concepts"],
                "tokens": spec["tokens"],
                "total_hits": len(query_items),
                "items": query_items[: max(0, int(top_k_per_query))],
            }
        )

        for item in query_items:
            url = str(item.get("url") or "").strip().lower()
            title = str(item.get("title") or "").strip().lower()
            key = url or title
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                cloned = dict(item)
                by_key[key] = cloned
                aggregated.append(cloned)
            else:
                _merge_item(existing, item)

    aggregated.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)

    query_highlights: List[Dict[str, Any]] = []
    total_query_hits = 0
    for row in query_views:
        total_hits = int(row.get("total_hits") or 0)
        total_query_hits += total_hits
        top_item = (row.get("items") or [None])[0]
        query_highlights.append(
            {
                "raw_query": row.get("raw_query") or "",
                "canonical_query": row.get("canonical_query") or "",
                "normalized_query": row.get("normalized_query") or "",
                "hit_count": total_hits,
                "top_title": (top_item or {}).get("title") or "",
                "top_keywords": ((top_item or {}).get("matched_keywords") or [])[:5],
            }
        )

    source_breakdown: Dict[str, int] = {}
    for item in aggregated:
        for source in item.get("sources") or []:
            source_breakdown[source] = source_breakdown.get(source, 0) + 1

    return {
        "source": "papers.cool",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": normalized_sources,
        "queries": query_views,
        "items": aggregated,
        "summary": {
            "unique_items": len(aggregated),
            "total_query_hits": total_query_hits,
            "top_titles": [item.get("title") or "" for item in aggregated[:5]],
            "query_highlights": query_highlights,
            "source_breakdown": source_breakdown,
            "source_errors": [],
        },
    }
