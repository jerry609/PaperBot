from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from paperbot.application.ports.paper_search_port import SearchPort
from paperbot.application.services.paper_search_service import PaperSearchService
from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper import PaperCandidate


@dataclass(frozen=True)
class RetrievalJudgment:
    doc_id: str
    relevance: int
    title: str = ""


@dataclass
class RetrievalBenchmarkCase:
    query_id: str
    query: str
    query_type: str = "generic"
    source: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    max_results: int = 50
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    judgments: List[RetrievalJudgment] = field(default_factory=list)
    results_by_source: Dict[str, List[PaperCandidate]] = field(default_factory=dict)


class _FixtureSearchAdapter(SearchPort):
    def __init__(self, source_name: str, papers: Sequence[PaperCandidate]):
        self._source_name = str(source_name)
        self._papers = [self._clone_paper(paper) for paper in papers]

    @property
    def source_name(self) -> str:
        return self._source_name

    async def search(
        self,
        query: str,
        *,
        max_results: int = 30,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[PaperCandidate]:
        del query
        filtered: List[PaperCandidate] = []
        for paper in self._papers:
            if year_from is not None and paper.year is not None and paper.year < year_from:
                continue
            if year_to is not None and paper.year is not None and paper.year > year_to:
                continue
            filtered.append(self._clone_paper(paper))
        return filtered[: max(0, int(max_results))]

    async def close(self) -> None:
        return None

    @staticmethod
    def _clone_paper(paper: PaperCandidate) -> PaperCandidate:
        return PaperCandidate(
            title=paper.title,
            abstract=paper.abstract,
            authors=list(paper.authors or []),
            year=paper.year,
            venue=paper.venue,
            citation_count=int(paper.citation_count or 0),
            url=paper.url,
            pdf_url=paper.pdf_url,
            keywords=list(paper.keywords or []),
            fields_of_study=list(paper.fields_of_study or []),
            publication_date=paper.publication_date,
            identities=[
                PaperIdentity(source=identity.source, external_id=identity.external_id)
                for identity in (paper.identities or [])
            ],
            title_hash=paper.title_hash,
            canonical_id=paper.canonical_id,
            retrieval_score=float(paper.retrieval_score or 0.0),
            retrieval_sources=list(paper.retrieval_sources or []),
        )


def _metric_key(name: str, k: int) -> str:
    return f"{name}_at_{int(k)}"


def _case_source_label(case: RetrievalBenchmarkCase) -> str:
    if case.source and str(case.source).strip():
        return str(case.source).strip()
    if case.sources:
        return "+".join(str(source).strip() for source in case.sources if str(source).strip())
    return "all"


def _paper_from_payload(payload: Dict[str, Any]) -> PaperCandidate:
    identities = [
        PaperIdentity(
            source=str(item.get("source") or "").strip(),
            external_id=str(item.get("external_id") or "").strip(),
        )
        for item in (payload.get("identities") or [])
        if item.get("source") and item.get("external_id")
    ]
    year = payload.get("year")
    citation_count = payload.get("citation_count")
    return PaperCandidate(
        title=str(payload.get("title") or ""),
        abstract=str(payload.get("abstract") or ""),
        authors=[str(item) for item in (payload.get("authors") or []) if str(item)],
        year=int(year) if year not in (None, "") else None,
        venue=(str(payload.get("venue")) if payload.get("venue") else None),
        citation_count=int(citation_count or 0),
        url=(str(payload.get("url")) if payload.get("url") else None),
        pdf_url=(str(payload.get("pdf_url")) if payload.get("pdf_url") else None),
        keywords=[str(item) for item in (payload.get("keywords") or []) if str(item)],
        fields_of_study=[str(item) for item in (payload.get("fields_of_study") or []) if str(item)],
        publication_date=(
            str(payload.get("publication_date")) if payload.get("publication_date") else None
        ),
        identities=identities,
        title_hash=str(payload.get("doc_id") or payload.get("title_hash") or ""),
    )


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    payload = json.loads(text)
    if isinstance(payload, dict):
        rows = payload.get("cases") or []
    else:
        rows = payload
    return list(rows)


def load_retrieval_benchmark_cases(path: str | Path) -> List[RetrievalBenchmarkCase]:
    fixture_path = Path(path)
    rows = _read_rows(fixture_path)
    cases: List[RetrievalBenchmarkCase] = []
    for row in rows:
        source_payload = row.get("results_by_source") or {}
        results_by_source = {
            str(source): [_paper_from_payload(item) for item in (papers or [])]
            for source, papers in source_payload.items()
        }
        sources = [
            str(source).strip() for source in (row.get("sources") or []) if str(source).strip()
        ]
        if not sources and results_by_source:
            sources = list(results_by_source.keys())

        source_label = row.get("source")
        if not source_label:
            source_label = "+".join(sources) if sources else "all"

        judgments = [
            RetrievalJudgment(
                doc_id=str(item.get("doc_id") or "").strip(),
                relevance=int(item.get("relevance") or 0),
                title=str(item.get("title") or ""),
            )
            for item in (row.get("judgments") or [])
            if str(item.get("doc_id") or "").strip()
        ]
        cases.append(
            RetrievalBenchmarkCase(
                query_id=str(row.get("query_id") or "").strip(),
                query=str(row.get("query") or "").strip(),
                query_type=str(row.get("query_type") or "generic").strip(),
                source=str(source_label).strip() if str(source_label).strip() else None,
                sources=sources,
                max_results=int(row.get("max_results") or 50),
                year_from=(
                    int(row["year_from"]) if row.get("year_from") not in (None, "") else None
                ),
                year_to=(int(row["year_to"]) if row.get("year_to") not in (None, "") else None),
                judgments=judgments,
                results_by_source=results_by_source,
            )
        )
    return cases


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (math.pow(2.0, float(relevance)) - 1.0) / math.log2(index + 2.0)
        for index, relevance in enumerate(relevances)
    )


def ndcg_at_k(
    judgments: Dict[str, int],
    ranked_doc_ids: Sequence[str],
    *,
    k: int = 10,
) -> float:
    actual = [int(judgments.get(doc_id, 0)) for doc_id in list(ranked_doc_ids)[:k]]
    ideal = sorted((int(score) for score in judgments.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 1.0
    return _dcg(actual) / ideal_dcg


def mrr_at_k(
    judgments: Dict[str, int],
    ranked_doc_ids: Sequence[str],
    *,
    k: int = 10,
    relevant_threshold: int = 1,
) -> float:
    relevant_ids = {
        doc_id for doc_id, score in judgments.items() if int(score) >= relevant_threshold
    }
    if not relevant_ids:
        return 1.0
    for index, doc_id in enumerate(list(ranked_doc_ids)[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / float(index)
    return 0.0


def recall_at_k(
    judgments: Dict[str, int],
    ranked_doc_ids: Sequence[str],
    *,
    k: int = 50,
    relevant_threshold: int = 1,
) -> float:
    relevant_ids = {
        doc_id for doc_id, score in judgments.items() if int(score) >= relevant_threshold
    }
    if not relevant_ids:
        return 1.0
    hits = sum(1 for doc_id in list(ranked_doc_ids)[:k] if doc_id in relevant_ids)
    return hits / float(len(relevant_ids))


def evaluate_retrieval_case(
    case: RetrievalBenchmarkCase,
    ranked_doc_ids: Sequence[str],
    latency_ms: float,
    *,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 50,
    relevant_threshold: int = 1,
    total_raw: int = 0,
    duplicates_removed: int = 0,
) -> Dict[str, Any]:
    judgments = {item.doc_id: int(item.relevance) for item in case.judgments}
    ndcg_key = _metric_key("ndcg", ndcg_k)
    mrr_key = _metric_key("mrr", mrr_k)
    recall_key = _metric_key("recall", recall_k)

    relevant_ids = {
        doc_id for doc_id, relevance in judgments.items() if int(relevance) >= relevant_threshold
    }
    top_hits = [
        {
            "doc_id": doc_id,
            "rank": rank,
            "relevance": int(judgments[doc_id]),
        }
        for rank, doc_id in enumerate(list(ranked_doc_ids)[:recall_k], start=1)
        if doc_id in relevant_ids
    ]

    return {
        "query_id": case.query_id,
        "query": case.query,
        "query_type": case.query_type,
        "source": _case_source_label(case),
        "selected_sources": list(case.sources),
        "judged_docs": len(judgments),
        "judged_relevant": len(relevant_ids),
        "retrieved_count": len(list(ranked_doc_ids)),
        "total_raw": int(total_raw),
        "duplicates_removed": int(duplicates_removed),
        "latency_ms": float(latency_ms),
        ndcg_key: float(ndcg_at_k(judgments, ranked_doc_ids, k=ndcg_k)),
        mrr_key: float(
            mrr_at_k(
                judgments,
                ranked_doc_ids,
                k=mrr_k,
                relevant_threshold=relevant_threshold,
            )
        ),
        recall_key: float(
            recall_at_k(
                judgments,
                ranked_doc_ids,
                k=recall_k,
                relevant_threshold=relevant_threshold,
            )
        ),
        "top_hits": top_hits,
        "ranked_doc_ids": list(ranked_doc_ids)[: max(ndcg_k, mrr_k, recall_k)],
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(0, math.ceil((float(percentile) / 100.0) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _aggregate_rows(
    case_results: Sequence[Dict[str, Any]],
    *,
    ndcg_k: int,
    mrr_k: int,
    recall_k: int,
) -> Dict[str, float]:
    ndcg_key = _metric_key("ndcg", ndcg_k)
    mrr_key = _metric_key("mrr", mrr_k)
    recall_key = _metric_key("recall", recall_k)

    if not case_results:
        return {
            "case_count": 0.0,
            ndcg_key: 0.0,
            mrr_key: 0.0,
            recall_key: 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    n = float(len(case_results))
    latencies = [float(row.get("latency_ms", 0.0)) for row in case_results]
    return {
        "case_count": n,
        ndcg_key: sum(float(row.get(ndcg_key, 0.0)) for row in case_results) / n,
        mrr_key: sum(float(row.get(mrr_key, 0.0)) for row in case_results) / n,
        recall_key: sum(float(row.get(recall_key, 0.0)) for row in case_results) / n,
        "avg_latency_ms": sum(latencies) / n,
        "p95_latency_ms": _percentile(latencies, 95.0),
    }


def aggregate_retrieval_results(
    case_results: Sequence[Dict[str, Any]],
    *,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 50,
) -> Dict[str, Any]:
    by_query_type: Dict[str, List[Dict[str, Any]]] = {}
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in case_results:
        by_query_type.setdefault(str(row.get("query_type") or "generic"), []).append(row)
        by_source.setdefault(str(row.get("source") or "all"), []).append(row)

    return {
        "overall": _aggregate_rows(
            case_results,
            ndcg_k=ndcg_k,
            mrr_k=mrr_k,
            recall_k=recall_k,
        ),
        "by_query_type": {
            name: _aggregate_rows(rows, ndcg_k=ndcg_k, mrr_k=mrr_k, recall_k=recall_k)
            for name, rows in sorted(by_query_type.items())
        },
        "by_source": {
            name: _aggregate_rows(rows, ndcg_k=ndcg_k, mrr_k=mrr_k, recall_k=recall_k)
            for name, rows in sorted(by_source.items())
        },
    }


def _build_fixture_service(case: RetrievalBenchmarkCase) -> PaperSearchService:
    if not case.results_by_source:
        raise ValueError(
            f"Benchmark case '{case.query_id}' is missing results_by_source and no search service was provided"
        )
    adapters = {
        source: _FixtureSearchAdapter(source, papers)
        for source, papers in case.results_by_source.items()
    }
    return PaperSearchService(adapters=adapters)


async def run_retrieval_benchmark(
    cases: Sequence[RetrievalBenchmarkCase],
    *,
    search_service: Optional[PaperSearchService] = None,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 50,
    relevant_threshold: int = 1,
    source_weights: Optional[Dict[str, float]] = None,
    rrf_k: float = PaperSearchService.DEFAULT_RRF_K,
) -> Dict[str, Any]:
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        current_service = search_service or _build_fixture_service(case)
        started = time.perf_counter()
        search_result = await current_service.search(
            case.query,
            sources=case.sources or None,
            max_results=max(int(case.max_results), int(recall_k)),
            year_from=case.year_from,
            year_to=case.year_to,
            persist=False,
            source_weights=source_weights,
            rrf_k=rrf_k,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_doc_ids = [PaperSearchService._paper_key(paper) for paper in search_result.papers]
        case_results.append(
            evaluate_retrieval_case(
                case,
                ranked_doc_ids,
                latency_ms,
                ndcg_k=ndcg_k,
                mrr_k=mrr_k,
                recall_k=recall_k,
                relevant_threshold=relevant_threshold,
                total_raw=search_result.total_raw,
                duplicates_removed=search_result.duplicates_removed,
            )
        )
        if search_service is None:
            await current_service.close()

    return {
        "cases": case_results,
        "summary": aggregate_retrieval_results(
            case_results,
            ndcg_k=ndcg_k,
            mrr_k=mrr_k,
            recall_k=recall_k,
        ),
        "config": {
            "case_count": len(cases),
            "ndcg_k": int(ndcg_k),
            "mrr_k": int(mrr_k),
            "recall_k": int(recall_k),
            "relevant_threshold": int(relevant_threshold),
            "rrf_k": float(rrf_k),
            "source_weights": dict(source_weights or {}),
        },
    }


def format_benchmark_report(result: Dict[str, Any]) -> str:
    config = result.get("config") or {}
    summary = result.get("summary") or {}
    overall = summary.get("overall") or {}
    ndcg_key = _metric_key("ndcg", int(config.get("ndcg_k", 10)))
    mrr_key = _metric_key("mrr", int(config.get("mrr_k", 10)))
    recall_key = _metric_key("recall", int(config.get("recall_k", 50)))

    lines = [
        "Retrieval Benchmark",
        f"Cases: {int(config.get('case_count', 0))}",
        (
            f"Overall: {ndcg_key}={float(overall.get(ndcg_key, 0.0)):.3f} | "
            f"{mrr_key}={float(overall.get(mrr_key, 0.0)):.3f} | "
            f"{recall_key}={float(overall.get(recall_key, 0.0)):.3f} | "
            f"p95_latency_ms={float(overall.get('p95_latency_ms', 0.0)):.2f}"
        ),
        "By query_type:",
    ]

    for name, metrics in (summary.get("by_query_type") or {}).items():
        lines.append(
            (
                f"  - {name}: {ndcg_key}={float(metrics.get(ndcg_key, 0.0)):.3f}, "
                f"{mrr_key}={float(metrics.get(mrr_key, 0.0)):.3f}, "
                f"{recall_key}={float(metrics.get(recall_key, 0.0)):.3f}, "
                f"p95_latency_ms={float(metrics.get('p95_latency_ms', 0.0)):.2f}"
            )
        )

    lines.append("By source:")
    for name, metrics in (summary.get("by_source") or {}).items():
        lines.append(
            (
                f"  - {name}: {ndcg_key}={float(metrics.get(ndcg_key, 0.0)):.3f}, "
                f"{mrr_key}={float(metrics.get(mrr_key, 0.0)):.3f}, "
                f"{recall_key}={float(metrics.get(recall_key, 0.0)):.3f}, "
                f"p95_latency_ms={float(metrics.get('p95_latency_ms', 0.0)):.2f}"
            )
        )
    return "\n".join(lines)


__all__ = [
    "RetrievalBenchmarkCase",
    "RetrievalJudgment",
    "aggregate_retrieval_results",
    "evaluate_retrieval_case",
    "format_benchmark_report",
    "load_retrieval_benchmark_cases",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "run_retrieval_benchmark",
]
