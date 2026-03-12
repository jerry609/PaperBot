from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from paperbot.application.services.retrieval_benchmark import mrr_at_k, ndcg_at_k, recall_at_k
from paperbot.context_engine.embeddings import EmbeddingProvider, HashEmbeddingProvider
from paperbot.infrastructure.services.document_indexing_service import DocumentIndexingService
from paperbot.infrastructure.stores.document_index_store import DocumentIndexStore
from paperbot.infrastructure.stores.paper_store import PaperStore


@dataclass(frozen=True)
class DocumentEvidenceJudgment:
    paper_id: int
    chunk_ref: str
    relevance: int


@dataclass(frozen=True)
class DocumentEvidencePaper:
    paper_id: int
    title: str
    abstract: str = ""
    structured_card: Dict[str, Any] = field(default_factory=dict)
    year: Optional[int] = None
    venue: Optional[str] = None


@dataclass
class DocumentEvidenceBenchmarkCase:
    case_id: str
    query: str
    query_type: str = "generic"
    top_k: int = 5
    expected_paper_ids: List[int] = field(default_factory=list)
    expected_chunk_refs: List[str] = field(default_factory=list)
    judgments: List[DocumentEvidenceJudgment] = field(default_factory=list)


@dataclass
class DocumentEvidenceBenchmarkFixture:
    version: str
    description: str
    papers: List[DocumentEvidencePaper] = field(default_factory=list)
    cases: List[DocumentEvidenceBenchmarkCase] = field(default_factory=list)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_document_evidence_benchmark_fixture(
    path: str | Path,
) -> DocumentEvidenceBenchmarkFixture:
    payload = _read_json(Path(path))
    papers = [
        DocumentEvidencePaper(
            paper_id=int(row["paper_id"]),
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            structured_card=dict(row.get("structured_card") or {}),
            year=int(row["year"]) if row.get("year") not in (None, "") else None,
            venue=str(row.get("venue") or "") or None,
        )
        for row in (payload.get("papers") or [])
    ]
    cases = [
        DocumentEvidenceBenchmarkCase(
            case_id=str(row.get("case_id") or ""),
            query=str(row.get("query") or ""),
            query_type=str(row.get("query_type") or "generic"),
            top_k=max(1, int(row.get("top_k") or 5)),
            expected_paper_ids=[int(value) for value in (row.get("expected_paper_ids") or [])],
            expected_chunk_refs=[str(value) for value in (row.get("expected_chunk_refs") or [])],
            judgments=[
                DocumentEvidenceJudgment(
                    paper_id=int(item["paper_id"]),
                    chunk_ref=str(item.get("chunk_ref") or ""),
                    relevance=int(item.get("relevance") or 0),
                )
                for item in (row.get("judgments") or [])
                if str(item.get("chunk_ref") or "")
            ],
        )
        for row in (payload.get("cases") or [])
    ]
    return DocumentEvidenceBenchmarkFixture(
        version=str(payload.get("version") or "unknown"),
        description=str(payload.get("description") or ""),
        papers=papers,
        cases=cases,
    )


def _metric_key(name: str, k: int) -> str:
    return f"{name}_at_{int(k)}"


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(0, int((float(percentile) / 100.0) * len(ordered) + 0.999999) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _aggregate_rows(case_results: Sequence[Dict[str, Any]], *, top_k: int) -> Dict[str, float]:
    recall_key = _metric_key("recall", top_k)
    mrr_key = _metric_key("mrr", top_k)
    ndcg_key = _metric_key("ndcg", top_k)
    if not case_results:
        return {
            "case_count": 0.0,
            recall_key: 0.0,
            mrr_key: 0.0,
            ndcg_key: 0.0,
            "evidence_hit_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    count = float(len(case_results))
    latencies = [float(row.get("latency_ms", 0.0)) for row in case_results]
    return {
        "case_count": count,
        recall_key: sum(float(row.get(recall_key, 0.0)) for row in case_results) / count,
        mrr_key: sum(float(row.get(mrr_key, 0.0)) for row in case_results) / count,
        ndcg_key: sum(float(row.get(ndcg_key, 0.0)) for row in case_results) / count,
        "evidence_hit_rate": sum(float(row.get("evidence_hit", 0.0)) for row in case_results)
        / count,
        "avg_latency_ms": sum(latencies) / count,
        "p95_latency_ms": _percentile(latencies, 95.0),
    }


def _group_case_results(case_results: Sequence[Dict[str, Any]], *, top_k: int) -> Dict[str, Any]:
    by_query_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in case_results:
        by_query_type.setdefault(str(row.get("query_type") or "generic"), []).append(row)
    return {
        "overall": _aggregate_rows(case_results, top_k=top_k),
        "by_query_type": {
            query_type: _aggregate_rows(rows, top_k=top_k)
            for query_type, rows in sorted(by_query_type.items())
        },
    }


def _seed_fixture_into_store(
    fixture: DocumentEvidenceBenchmarkFixture,
    *,
    db_url: str,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> Dict[int, int]:
    provider = embedding_provider or HashEmbeddingProvider(dim=128)
    paper_store = PaperStore(db_url=db_url, auto_create_schema=True)
    service = DocumentIndexingService(
        paper_store=paper_store,
        index_store=DocumentIndexStore(
            db_url=db_url,
            auto_create_schema=True,
            embedding_provider=provider,
        ),
        embedding_provider=provider,
    )
    id_map: Dict[int, int] = {}
    try:
        paper_ids: List[int] = []
        for paper in fixture.papers:
            row = paper_store.upsert_paper(
                paper={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                },
                source_hint="document_evidence_bench",
            )
            actual_paper_id = int(row["id"])
            id_map[int(paper.paper_id)] = actual_paper_id
            paper_ids.append(actual_paper_id)
            if paper.structured_card:
                paper_store.update_structured_card(
                    actual_paper_id,
                    json.dumps(paper.structured_card, ensure_ascii=False),
                )

        service.enqueue_papers(paper_ids=paper_ids, trigger_source="document_evidence_bench")
        service.process_pending_jobs(limit=max(1, len(paper_ids)))
    finally:
        service.close()
    return id_map


def _remap_chunk_ref(chunk_ref: str, id_map: Dict[int, int]) -> str:
    parts = str(chunk_ref or "").split(":")
    if len(parts) != 3:
        return str(chunk_ref or "")
    try:
        fixture_paper_id = int(parts[0])
    except (TypeError, ValueError):
        return str(chunk_ref or "")
    actual_paper_id = id_map.get(fixture_paper_id, fixture_paper_id)
    return f"{actual_paper_id}:{parts[1]}:{parts[2]}"


def _remap_case(
    case: DocumentEvidenceBenchmarkCase,
    *,
    id_map: Dict[int, int],
) -> DocumentEvidenceBenchmarkCase:
    return DocumentEvidenceBenchmarkCase(
        case_id=case.case_id,
        query=case.query,
        query_type=case.query_type,
        top_k=case.top_k,
        expected_paper_ids=[
            id_map.get(int(paper_id), int(paper_id)) for paper_id in case.expected_paper_ids
        ],
        expected_chunk_refs=[
            _remap_chunk_ref(chunk_ref, id_map) for chunk_ref in case.expected_chunk_refs
        ],
        judgments=[
            DocumentEvidenceJudgment(
                paper_id=id_map.get(int(judgment.paper_id), int(judgment.paper_id)),
                chunk_ref=_remap_chunk_ref(judgment.chunk_ref, id_map),
                relevance=int(judgment.relevance),
            )
            for judgment in case.judgments
        ],
    )


def _rank_embedding_only(
    chunks: Sequence[Dict[str, Any]],
    *,
    query: str,
    top_k: int,
    embedding_provider: Optional[EmbeddingProvider],
) -> List[Dict[str, Any]]:
    if embedding_provider is None:
        return []

    query_embedding = embedding_provider.embed(query)
    if not query_embedding:
        return []

    ranked: List[Dict[str, Any]] = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        score = 0.0
        for left, right in zip(query_embedding, embedding):
            score += float(left) * float(right)
        ranked.append({**chunk, "score": float(score)})
    ranked.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return ranked[:top_k]


def _rank_mode(
    mode: str,
    *,
    store: DocumentIndexStore,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "embedding_only":
        return _rank_embedding_only(
            store.list_chunks(),
            query=query,
            top_k=top_k,
            embedding_provider=store.embedding_provider,
        )

    if normalized_mode == "fts_only":
        fts_store = DocumentIndexStore(db_url=store.db_url, auto_create_schema=False)
        try:
            hits = fts_store.retrieve_evidence(query=query, limit=top_k)
        finally:
            fts_store.close()
    else:
        hits = store.retrieve_evidence(query=query, limit=top_k)

    ranked_rows: List[Dict[str, Any]] = []
    for hit in hits:
        if isinstance(hit, dict):
            payload = dict(hit)
        else:
            payload = {
                "paper_id": int(hit.paper_id),
                "chunk_id": int(hit.chunk_id),
                "chunk_index": int(hit.chunk_index),
                "paper_title": hit.paper_title,
                "section": hit.section,
                "heading": hit.heading,
                "content": hit.snippet,
                "score": float(hit.score),
                "metadata": dict(hit.metadata or {}),
            }
        section_chunk_index = int((payload.get("metadata") or {}).get("section_chunk_index") or 0)
        payload["chunk_ref"] = (
            f"{int(payload['paper_id'])}:{payload.get('section') or ''}:{section_chunk_index}"
        )
        ranked_rows.append(payload)
    return ranked_rows[:top_k]


def _evaluate_case(
    case: DocumentEvidenceBenchmarkCase,
    *,
    mode: str,
    ranked_rows: Sequence[Dict[str, Any]],
    latency_ms: float,
) -> Dict[str, Any]:
    top_k = max(1, int(case.top_k))
    recall_key = _metric_key("recall", top_k)
    mrr_key = _metric_key("mrr", top_k)
    ndcg_key = _metric_key("ndcg", top_k)
    judgments = {judgment.chunk_ref: int(judgment.relevance) for judgment in case.judgments}
    ranked_chunk_refs = [
        str(row.get("chunk_ref") or "") for row in ranked_rows if str(row.get("chunk_ref") or "")
    ]
    top_hits = [
        chunk_ref
        for chunk_ref in ranked_chunk_refs[:top_k]
        if int(judgments.get(chunk_ref, 0)) >= 1
    ]
    return {
        "case_id": case.case_id,
        "query": case.query,
        "query_type": case.query_type,
        "mode": mode,
        "top_k": top_k,
        "latency_ms": float(latency_ms),
        "retrieved_count": len(ranked_chunk_refs),
        recall_key: float(recall_at_k(judgments, ranked_chunk_refs, k=top_k)),
        mrr_key: float(mrr_at_k(judgments, ranked_chunk_refs, k=top_k)),
        ndcg_key: float(ndcg_at_k(judgments, ranked_chunk_refs, k=top_k)),
        "evidence_hit": 1.0 if top_hits else 0.0,
        "ranked_chunk_refs": ranked_chunk_refs[:top_k],
        "expected_chunk_refs": list(case.expected_chunk_refs),
        "expected_paper_ids": list(case.expected_paper_ids),
    }


def run_document_evidence_benchmark(
    fixture: DocumentEvidenceBenchmarkFixture,
    *,
    modes: Sequence[str] = ("fts_only", "embedding_only", "hybrid"),
    embedding_provider: Optional[EmbeddingProvider] = None,
    provider_label: str = "hash",
) -> Dict[str, Any]:
    provider = embedding_provider or HashEmbeddingProvider(dim=128)
    with tempfile.TemporaryDirectory(prefix="paperbot-document-evidence-bench-") as temp_dir:
        db_url = f"sqlite:///{Path(temp_dir) / 'document_evidence_bench.db'}"
        id_map = _seed_fixture_into_store(
            fixture,
            db_url=db_url,
            embedding_provider=provider,
        )
        remapped_cases = [_remap_case(case, id_map=id_map) for case in fixture.cases]
        hybrid_store = DocumentIndexStore(
            db_url=db_url,
            auto_create_schema=False,
            embedding_provider=provider,
        )
        try:
            case_results: List[Dict[str, Any]] = []
            for case in remapped_cases:
                for mode in modes:
                    started = time.perf_counter()
                    ranked_rows = _rank_mode(
                        mode,
                        store=hybrid_store,
                        query=case.query,
                        top_k=max(1, int(case.top_k)),
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    case_results.append(
                        _evaluate_case(
                            case,
                            mode=mode,
                            ranked_rows=ranked_rows,
                            latency_ms=latency_ms,
                        )
                    )
        finally:
            hybrid_store.close()

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in case_results:
        by_mode.setdefault(str(row.get("mode") or "unknown"), []).append(row)

    return {
        "config": {
            "fixture_version": fixture.version,
            "case_count": len(fixture.cases),
            "modes": [str(mode) for mode in modes],
            "embedding_provider": str(provider_label or "hash"),
        },
        "cases": case_results,
        "summary": {
            mode: _group_case_results(rows, top_k=max(int(row.get("top_k") or 5) for row in rows))
            for mode, rows in sorted(by_mode.items())
        },
    }


def format_document_evidence_benchmark_report(result: Dict[str, Any]) -> str:
    config = result.get("config") or {}
    summary = result.get("summary") or {}
    lines = [
        "Document Evidence Benchmark",
        f"Fixture: {config.get('fixture_version')}",
        f"Embedding Provider: {config.get('embedding_provider') or 'unknown'}",
        f"Cases: {int(config.get('case_count', 0))}",
    ]
    for mode, metrics_by_group in summary.items():
        overall = metrics_by_group.get("overall") or {}
        top_k = 5
        for key in overall.keys():
            if key.startswith("recall_at_"):
                top_k = int(key.rsplit("_", 1)[-1])
                break
        lines.append(
            (
                f"- {mode}: "
                f"recall_at_{top_k}={float(overall.get(f'recall_at_{top_k}', 0.0)):.3f}, "
                f"mrr_at_{top_k}={float(overall.get(f'mrr_at_{top_k}', 0.0)):.3f}, "
                f"ndcg_at_{top_k}={float(overall.get(f'ndcg_at_{top_k}', 0.0)):.3f}, "
                f"evidence_hit_rate={float(overall.get('evidence_hit_rate', 0.0)):.3f}, "
                f"p95_latency_ms={float(overall.get('p95_latency_ms', 0.0)):.2f}"
            )
        )
    return "\n".join(lines)


__all__ = [
    "DocumentEvidenceBenchmarkCase",
    "DocumentEvidenceBenchmarkFixture",
    "DocumentEvidenceJudgment",
    "DocumentEvidencePaper",
    "format_document_evidence_benchmark_report",
    "load_document_evidence_benchmark_fixture",
    "run_document_evidence_benchmark",
]
