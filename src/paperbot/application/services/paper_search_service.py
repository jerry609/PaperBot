"""PaperSearchService — unified search facade across all data sources."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from paperbot.application.ports.paper_search_port import SearchPort
from paperbot.domain.paper import PaperCandidate

logger = logging.getLogger(__name__)


_TOKEN_SEP_RX = re.compile(r"\s+")


@dataclass
class SearchResult:
    """Aggregated search result from PaperSearchService."""

    papers: List[PaperCandidate] = field(default_factory=list)
    provenance: Dict[str, List[str]] = field(default_factory=dict)
    total_raw: int = 0
    duplicates_removed: int = 0

    def to_legacy_format(self) -> List[Dict[str, Any]]:
        """Convert to the dict-list format used by existing code."""
        return [p.to_dict() for p in self.papers]


class PaperSearchService:
    """Facade that fans out queries to multiple SearchPort adapters,
    deduplicates, and optionally persists results."""

    DEFAULT_RRF_K = 60.0
    DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
        "semantic_scholar": 1.0,
        "openalex": 0.9,
        "arxiv": 0.8,
        "papers_cool": 0.7,
        "hf_daily": 0.6,
    }

    def __init__(
        self,
        adapters: Dict[str, SearchPort],
        deduplicator=None,
        registry=None,
        identity_store=None,
    ):
        self._adapters = adapters
        self._deduplicator = deduplicator
        self._registry = registry
        self._identity_store = identity_store

    async def search(
        self,
        query: str,
        *,
        sources: Optional[List[str]] = None,
        max_results: int = 30,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        persist: bool = True,
        source_weights: Optional[Dict[str, float]] = None,
        rrf_k: float = DEFAULT_RRF_K,
    ) -> SearchResult:
        """Fan-out search across selected adapters, deduplicate, optionally persist."""
        selected = self._select_adapters(sources)
        if not selected:
            return SearchResult()

        # 1. Concurrent search across adapters with per-adapter timeout
        PER_ADAPTER_TIMEOUT = 25.0  # seconds — don't let one slow source block everything

        async def _guarded_search(adapter):
            try:
                return await asyncio.wait_for(
                    adapter.search(
                        query,
                        max_results=max_results,
                        year_from=year_from,
                        year_to=year_to,
                    ),
                    timeout=PER_ADAPTER_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Adapter %s timed out after %.0fs", adapter.source_name, PER_ADAPTER_TIMEOUT
                )
                return TimeoutError(f"{adapter.source_name} timed out")
            except Exception as exc:
                return exc

        tasks = [_guarded_search(adapter) for adapter in selected]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        results_by_source: Dict[str, List[PaperCandidate]] = {}
        failed_sources: List[str] = []
        total_raw = 0
        for adapter, result in zip(selected, results):
            if isinstance(result, BaseException):
                logger.warning("Adapter %s failed: %s", adapter.source_name, result)
                failed_sources.append(adapter.source_name)
                continue
            papers = list(result)
            results_by_source[adapter.source_name] = papers
            total_raw += len(papers)

        if failed_sources:
            logger.info(
                "Search degraded: %d/%d sources failed (%s), continuing with %s",
                len(failed_sources),
                len(selected),
                ", ".join(failed_sources),
                ", ".join(results_by_source.keys()) or "none",
            )

        if not results_by_source:
            return SearchResult()

        # 2. RRF fusion + dedup merge
        effective_weights = dict(self.DEFAULT_SOURCE_WEIGHTS)
        if source_weights:
            for source, weight in source_weights.items():
                if not source:
                    continue
                try:
                    effective_weights[str(source)] = float(weight)
                except (TypeError, ValueError):
                    continue

        fused, provenance = self._fuse_with_rrf(
            results_by_source=results_by_source,
            source_weights=effective_weights,
            rrf_k=rrf_k,
        )

        unique = [paper for _, paper in fused]
        duplicates_removed = total_raw - len(unique)

        # 3. Persist if requested
        if persist and self._registry:
            for paper in unique:
                try:
                    source_hint = (paper.retrieval_sources or ["unknown"])[0]
                    upsert_kwargs = {
                        "paper": paper.to_dict(),
                        "source_hint": source_hint,
                        # Interactive search path: avoid blocking user requests on best-effort
                        # author-link syncing when SQLite is busy.
                        "sync_authors": False,
                    }
                    try:
                        result_dict = self._registry.upsert_paper(**upsert_kwargs)
                    except TypeError as exc:
                        if "sync_authors" not in str(exc):
                            raise
                        upsert_kwargs.pop("sync_authors", None)
                        result_dict = self._registry.upsert_paper(**upsert_kwargs)
                    paper.canonical_id = result_dict.get("id")
                except Exception as e:
                    logger.warning("Failed to persist paper %s: %s", paper.title[:50], e)

        return SearchResult(
            papers=unique[:max_results],
            provenance=provenance,
            total_raw=total_raw,
            duplicates_removed=duplicates_removed,
        )

    def _select_adapters(self, sources: Optional[List[str]]) -> List[SearchPort]:
        if sources is None:
            return list(self._adapters.values())
        return [self._adapters[s] for s in sources if s in self._adapters]

    async def close(self) -> None:
        for adapter in self._adapters.values():
            try:
                await adapter.close()
            except Exception:
                pass

    @staticmethod
    def _paper_key(paper: PaperCandidate) -> str:
        if paper.title_hash:
            return paper.title_hash
        normalized = _TOKEN_SEP_RX.sub(" ", (paper.title or "").strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _paper_quality(paper: PaperCandidate) -> Tuple[int, int, int, int]:
        """Heuristic for picking best canonical copy when duplicate appears in many sources."""
        has_abstract = 1 if (paper.abstract or "").strip() else 0
        citation_count = int(paper.citation_count or 0)
        has_year = 1 if paper.year else 0
        identity_count = len(paper.identities or [])
        return (has_abstract, citation_count, has_year, identity_count)

    def _fuse_with_rrf(
        self,
        *,
        results_by_source: Dict[str, List[PaperCandidate]],
        source_weights: Dict[str, float],
        rrf_k: float,
    ) -> Tuple[List[Tuple[float, PaperCandidate]], Dict[str, List[str]]]:
        rrf_k = max(1.0, float(rrf_k or self.DEFAULT_RRF_K))

        scores: Dict[str, float] = defaultdict(float)
        provenance: Dict[str, List[str]] = defaultdict(list)
        source_contrib: Dict[str, Dict[str, float]] = defaultdict(dict)
        best_by_key: Dict[str, PaperCandidate] = {}

        for source, papers in results_by_source.items():
            weight = float(source_weights.get(source, 0.5))
            for rank, paper in enumerate(papers, start=1):
                key = self._paper_key(paper)
                contrib = weight / (rrf_k + rank)
                scores[key] += contrib
                source_contrib[key][source] = source_contrib[key].get(source, 0.0) + contrib
                if source not in provenance[key]:
                    provenance[key].append(source)

                best = best_by_key.get(key)
                if best is None or self._paper_quality(paper) > self._paper_quality(best):
                    best_by_key[key] = paper

        fused: List[Tuple[float, PaperCandidate]] = []
        for key, paper in best_by_key.items():
            score = float(scores.get(key, 0.0))
            ranked_sources = sorted(
                source_contrib.get(key, {}).items(), key=lambda item: (-item[1], item[0])
            )
            paper.title_hash = key
            paper.retrieval_score = score
            paper.retrieval_sources = [name for name, _ in ranked_sources]
            fused.append((score, paper))

        fused.sort(
            key=lambda item: (
                -float(item[0]),
                -(int(item[1].citation_count or 0)),
                -(int(item[1].year or 0)),
                (item[1].title or "").lower(),
            )
        )

        normalized_provenance = {k: list(v) for k, v in provenance.items()}
        return fused, normalized_provenance
