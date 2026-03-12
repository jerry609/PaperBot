"""Multi-hop citation graph traversal built on top of Semantic Scholar."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Literal, Optional

from paperbot.infrastructure.api_clients.semantic_scholar import SemanticScholarClient

TraversalDirection = Literal["references", "citations", "both"]
RelevanceFilter = Callable[[Dict[str, Any]], float]

_DEFAULT_FIELDS = [
    "title",
    "year",
    "citationCount",
    "authors",
    "references",
    "citations",
]


@dataclass
class CitationGraphNode:
    paper_id: str
    title: str
    year: Optional[int] = None
    citation_count: int = 0
    authors: List[str] = field(default_factory=list)
    hop: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "citation_count": self.citation_count,
            "authors": list(self.authors),
            "hop": self.hop,
        }


@dataclass
class CitationGraphEdge:
    source_id: str
    target_id: str
    relation: Literal["references", "citations"]
    hop: int
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "hop": self.hop,
            "weight": self.weight,
        }


@dataclass
class CitationGraph:
    seed_paper_id: str
    direction: TraversalDirection
    nodes: Dict[str, CitationGraphNode] = field(default_factory=dict)
    edges: List[CitationGraphEdge] = field(default_factory=list)

    def add_node(self, node: CitationGraphNode) -> None:
        existing = self.nodes.get(node.paper_id)
        if existing is None:
            self.nodes[node.paper_id] = node
            return
        if node.hop < existing.hop:
            existing.hop = node.hop
        if not existing.title and node.title:
            existing.title = node.title
        if existing.year is None and node.year is not None:
            existing.year = node.year
        if not existing.authors and node.authors:
            existing.authors = list(node.authors)
        if node.citation_count > existing.citation_count:
            existing.citation_count = node.citation_count

    def add_edge(self, edge: CitationGraphEdge) -> None:
        for existing in self.edges:
            if (
                existing.source_id == edge.source_id
                and existing.target_id == edge.target_id
                and existing.relation == edge.relation
            ):
                if edge.hop < existing.hop:
                    existing.hop = edge.hop
                if edge.weight > existing.weight:
                    existing.weight = edge.weight
                return
        self.edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_paper_id": self.seed_paper_id,
            "direction": self.direction,
            "nodes": [node.to_dict() for node in sorted(self.nodes.values(), key=lambda item: (item.hop, item.paper_id))],
            "edges": [edge.to_dict() for edge in self.edges],
        }


class CitationGraphClient:
    """Traverse references and citations up to a configurable depth."""

    def __init__(
        self,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        *,
        api_key: Optional[str] = None,
        timeout: int = 30,
        request_interval: float = 1.0,
        max_concurrency: int = 4,
    ) -> None:
        self._owns_client = semantic_scholar_client is None
        self._client = semantic_scholar_client or SemanticScholarClient(
            api_key=api_key,
            timeout=timeout,
            request_interval=request_interval,
        )
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            await self._client.close()

    async def __aenter__(self) -> "CitationGraphClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def traverse(
        self,
        seed_paper_id: str,
        *,
        direction: TraversalDirection,
        max_hops: int = 2,
        max_papers_per_hop: int = 10,
        relevance_filter: Optional[RelevanceFilter] = None,
        fields: Optional[List[str]] = None,
    ) -> CitationGraph:
        normalized_seed = str(seed_paper_id or "").strip()
        if not normalized_seed:
            raise ValueError("seed_paper_id is required")
        if direction not in {"references", "citations", "both"}:
            raise ValueError("direction must be one of: references, citations, both")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if max_papers_per_hop < 1:
            raise ValueError("max_papers_per_hop must be >= 1")

        graph = CitationGraph(seed_paper_id=normalized_seed, direction=direction)
        visited: set[str] = set()
        frontier: Dict[str, int] = {normalized_seed: 0}
        request_fields = fields or list(_DEFAULT_FIELDS)

        for hop in range(max_hops):
            batch = [(paper_id, frontier_hop) for paper_id, frontier_hop in frontier.items() if paper_id and paper_id not in visited]
            if not batch:
                break

            tasks = [
                self._fetch_paper(paper_id=paper_id, fields=request_fields)
                for paper_id, _ in batch
            ]
            responses = await asyncio.gather(*tasks)
            next_frontier: Dict[str, int] = {}

            for (requested_id, current_hop), payload in zip(batch, responses):
                visited.add(requested_id)
                paper = payload or {}
                if not paper:
                    continue

                canonical_id = self._paper_id(paper) or requested_id
                graph.add_node(self._node_from_paper(paper=paper, paper_id=canonical_id, hop=current_hop))

                for relation, related in self._related_candidates(paper=paper, direction=direction):
                    scored = self._score_related(related, relevance_filter=relevance_filter)
                    for related_paper, weight in scored[:max_papers_per_hop]:
                        related_id = self._paper_id(related_paper)
                        if not related_id:
                            continue
                        graph.add_node(self._node_from_paper(paper=related_paper, paper_id=related_id, hop=current_hop + 1))
                        if relation == "references":
                            source_id, target_id = canonical_id, related_id
                        else:
                            source_id, target_id = related_id, canonical_id
                        graph.add_edge(
                            CitationGraphEdge(
                                source_id=source_id,
                                target_id=target_id,
                                relation=relation,
                                hop=current_hop + 1,
                                weight=weight,
                            )
                        )
                        if related_id not in visited and related_id not in next_frontier:
                            next_frontier[related_id] = current_hop + 1

            frontier = next_frontier

        return graph

    async def _fetch_paper(self, *, paper_id: str, fields: List[str]) -> Dict[str, Any]:
        async with self._semaphore:
            payload = await self._client.get_paper(paper_id, fields=fields)
            return payload or {}

    @staticmethod
    def _paper_id(paper: Dict[str, Any]) -> str:
        for key in ("paperId", "paper_id", "id", "externalId"):
            value = str(paper.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _paper_authors(paper: Dict[str, Any]) -> List[str]:
        rows: List[str] = []
        for author in paper.get("authors") or []:
            if not isinstance(author, dict):
                continue
            name = str(author.get("name") or author.get("display_name") or "").strip()
            if name:
                rows.append(name)
        return rows

    @classmethod
    def _node_from_paper(cls, *, paper: Dict[str, Any], paper_id: str, hop: int) -> CitationGraphNode:
        year = paper.get("year")
        try:
            parsed_year = int(year) if year is not None else None
        except (TypeError, ValueError):
            parsed_year = None
        citation_count = paper.get("citationCount", paper.get("citation_count", 0))
        try:
            parsed_citations = int(citation_count or 0)
        except (TypeError, ValueError):
            parsed_citations = 0
        return CitationGraphNode(
            paper_id=paper_id,
            title=str(paper.get("title") or paper_id).strip(),
            year=parsed_year,
            citation_count=parsed_citations,
            authors=cls._paper_authors(paper),
            hop=hop,
        )

    @classmethod
    def _related_candidates(
        cls,
        *,
        paper: Dict[str, Any],
        direction: TraversalDirection,
    ) -> List[tuple[Literal["references", "citations"], List[Dict[str, Any]]]]:
        groups: List[tuple[Literal["references", "citations"], List[Dict[str, Any]]]] = []
        if direction in {"references", "both"}:
            groups.append(("references", cls._extract_related(paper.get("references") or [], nested_key="citedPaper")))
        if direction in {"citations", "both"}:
            groups.append(("citations", cls._extract_related(paper.get("citations") or [], nested_key="citingPaper")))
        return groups

    @classmethod
    def _extract_related(
        cls,
        entries: Iterable[Any],
        *,
        nested_key: str,
    ) -> List[Dict[str, Any]]:
        related: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            nested = entry.get(nested_key)
            if isinstance(nested, dict):
                payload = dict(nested)
            else:
                payload = dict(entry)
            if cls._paper_id(payload):
                related.append(payload)
        return related

    @staticmethod
    def _score_related(
        papers: List[Dict[str, Any]],
        *,
        relevance_filter: Optional[RelevanceFilter],
    ) -> List[tuple[Dict[str, Any], float]]:
        deduped: Dict[str, tuple[Dict[str, Any], float]] = {}
        for paper in papers:
            paper_id = CitationGraphClient._paper_id(paper)
            if not paper_id:
                continue
            try:
                score = float(relevance_filter(paper)) if relevance_filter is not None else 1.0
            except Exception:
                score = 0.0
            current = deduped.get(paper_id)
            if current is None or score > current[1]:
                deduped[paper_id] = (paper, score)

        def _sort_key(item: tuple[Dict[str, Any], float]) -> tuple[float, int, str]:
            paper, score = item
            citation_count = paper.get("citationCount", paper.get("citation_count", 0))
            try:
                parsed_citations = int(citation_count or 0)
            except (TypeError, ValueError):
                parsed_citations = 0
            return (score, parsed_citations, str(paper.get("title") or ""))

        return sorted(deduped.values(), key=_sort_key, reverse=True)
