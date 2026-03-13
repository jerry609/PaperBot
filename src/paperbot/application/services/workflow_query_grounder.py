from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Dict, List, Protocol, runtime_checkable

from paperbot.application.services.wiki_concept_service import (
    ResolvedWikiConcept,
    WikiConceptService,
)


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip())


def _normalize_query(query: str) -> str:
    return _clean_query(query).lower()


def _unique_preserve(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        cleaned = _clean_query(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _replace_id_term(query: str, *, term: str, replacement: str) -> str:
    cleaned_query = _clean_query(query)
    cleaned_term = _clean_query(term)
    cleaned_replacement = _clean_query(replacement)
    if not cleaned_query or not cleaned_term or not cleaned_replacement:
        return cleaned_query
    pattern = re.compile(rf"\b{re.escape(cleaned_term)}\b", re.IGNORECASE)
    return pattern.sub(cleaned_replacement, cleaned_query, count=1)


@dataclass(frozen=True)
class GroundedWorkflowConcept:
    id: str
    name: str
    category: str
    canonical_query: str
    matched_terms: List[str]
    paper_count: int
    track_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedQuery:
    original_query: str
    canonical_query: str
    search_queries: List[str]
    concepts: List[GroundedWorkflowConcept]

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_query": self.original_query,
            "canonical_query": self.canonical_query,
            "search_queries": list(self.search_queries),
            "concepts": [concept.to_dict() for concept in self.concepts],
        }


@runtime_checkable
class WorkflowQueryGrounderPort(Protocol):
    def ground_query(self, *, user_id: str, query: str, limit: int = 3) -> GroundedQuery: ...


class WorkflowQueryGrounder:
    """Resolve workflow queries to grounded concepts without over-expanding intent."""

    def __init__(self, concept_service: WikiConceptService):
        self._concept_service = concept_service

    def ground_query(self, *, user_id: str, query: str, limit: int = 3) -> GroundedQuery:
        original_query = _clean_query(query)
        if not original_query:
            return GroundedQuery(
                original_query="",
                canonical_query="",
                search_queries=[],
                concepts=[],
            )

        resolved = self._concept_service.resolve_concepts(
            user_id=user_id, query=original_query, limit=limit
        )
        concepts = [self._to_grounded_concept(item) for item in resolved]

        canonical_query = original_query
        for concept in resolved:
            canonical_query = self._apply_concept_resolution(canonical_query, concept)

        search_queries = _unique_preserve([original_query, canonical_query])
        return GroundedQuery(
            original_query=original_query,
            canonical_query=_clean_query(canonical_query),
            search_queries=search_queries,
            concepts=concepts,
        )

    @staticmethod
    def _to_grounded_concept(item: ResolvedWikiConcept) -> GroundedWorkflowConcept:
        return GroundedWorkflowConcept(
            id=item.id,
            name=item.name,
            category=item.category,
            canonical_query=item.canonical_query,
            matched_terms=list(item.matched_terms),
            paper_count=item.paper_count,
            track_count=item.track_count,
        )

    @staticmethod
    def _apply_concept_resolution(query: str, concept: ResolvedWikiConcept) -> str:
        normalized_id = _normalize_query(concept.id)
        if normalized_id not in {_normalize_query(term) for term in concept.matched_terms}:
            return query
        if _normalize_query(concept.canonical_query) == normalized_id:
            return query
        return _replace_id_term(query, term=concept.id, replacement=concept.canonical_query)


def build_grounded_routing_query(
    *,
    original_query: str,
    grounded_query: GroundedQuery | None,
) -> str:
    cleaned_original = _clean_query(original_query)
    if grounded_query is None:
        return cleaned_original
    search_queries = getattr(grounded_query, "search_queries", None)
    canonical_query = getattr(grounded_query, "canonical_query", "")
    if search_queries is None and hasattr(grounded_query, "to_dict"):
        payload = grounded_query.to_dict()
        search_queries = payload.get("search_queries") if isinstance(payload, dict) else None
        canonical_query = str(payload.get("canonical_query") or canonical_query)
    values = _unique_preserve([cleaned_original, *list(search_queries or []), canonical_query])
    return " ".join(values)
