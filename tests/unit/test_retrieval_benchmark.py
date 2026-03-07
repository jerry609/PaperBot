from __future__ import annotations

from pathlib import Path

import pytest

from paperbot.application.services.retrieval_benchmark import (
    RetrievalBenchmarkCase,
    RetrievalJudgment,
    aggregate_retrieval_results,
    evaluate_retrieval_case,
    load_retrieval_benchmark_cases,
    run_retrieval_benchmark,
)


def test_load_retrieval_benchmark_cases_parses_jsonl_fixture(tmp_path):
    fixture = tmp_path / "retrieval_fixture.jsonl"
    fixture.write_text(
        "\n".join(
            [
                (
                    '{"query_id":"q1","query":"rag","query_type":"short","'
                    'source":"semantic_scholar","sources":["semantic_scholar"],'
                    '"judgments":[{"doc_id":"id:doi:10.1/demo","relevance":3}],'
                    '"results_by_source":{"semantic_scholar":[{"title":"Demo","'
                    'abstract":"Benchmark demo paper.","identities":[{"source":"doi",'
                    '"external_id":"10.1/demo"}]}]}}'
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_retrieval_benchmark_cases(fixture)

    assert len(cases) == 1
    assert cases[0].query_id == "q1"
    assert cases[0].source == "semantic_scholar"
    assert cases[0].sources == ["semantic_scholar"]
    assert cases[0].results_by_source["semantic_scholar"][0].get_identity("doi") == "10.1/demo"


def test_evaluate_retrieval_case_scores_metrics_and_hits():
    case = RetrievalBenchmarkCase(
        query_id="q1",
        query="rag",
        query_type="short",
        source="semantic_scholar",
        judgments=[
            RetrievalJudgment(doc_id="d1", relevance=3),
            RetrievalJudgment(doc_id="d2", relevance=2),
        ],
    )

    result = evaluate_retrieval_case(
        case,
        ranked_doc_ids=["d3", "d1", "d2"],
        latency_ms=12.5,
        ndcg_k=10,
        mrr_k=10,
        recall_k=2,
    )

    assert result["mrr_at_10"] == pytest.approx(0.5)
    assert result["recall_at_2"] == pytest.approx(0.5)
    assert 0.0 < result["ndcg_at_10"] < 1.0
    assert result["top_hits"] == [{"doc_id": "d1", "rank": 2, "relevance": 3}]


def test_aggregate_retrieval_results_groups_by_query_type_and_source():
    summary = aggregate_retrieval_results(
        [
            {
                "query_type": "short",
                "source": "semantic_scholar",
                "ndcg_at_10": 1.0,
                "mrr_at_10": 1.0,
                "recall_at_50": 1.0,
                "latency_ms": 10.0,
            },
            {
                "query_type": "long",
                "source": "arxiv",
                "ndcg_at_10": 0.5,
                "mrr_at_10": 0.5,
                "recall_at_50": 1.0,
                "latency_ms": 30.0,
            },
            {
                "query_type": "short",
                "source": "semantic_scholar",
                "ndcg_at_10": 0.5,
                "mrr_at_10": 1.0,
                "recall_at_50": 0.5,
                "latency_ms": 20.0,
            },
        ],
        ndcg_k=10,
        mrr_k=10,
        recall_k=50,
    )

    assert summary["overall"]["case_count"] == 3.0
    assert summary["by_query_type"]["short"]["case_count"] == 2.0
    assert summary["by_source"]["semantic_scholar"]["ndcg_at_10"] == pytest.approx(0.75)
    assert summary["by_source"]["semantic_scholar"]["p95_latency_ms"] == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_run_retrieval_benchmark_uses_repo_fixture_and_tracks_dedup():
    fixture = Path("evals/fixtures/retrieval/bench_v2.jsonl")
    cases = load_retrieval_benchmark_cases(fixture)

    result = await run_retrieval_benchmark(cases, ndcg_k=10, mrr_k=10, recall_k=50)

    assert result["summary"]["overall"]["case_count"] >= 6.0
    assert result["summary"]["overall"]["recall_at_50"] == pytest.approx(1.0)
    assert "semantic_scholar+arxiv" in result["summary"]["by_source"]
    assert any(case["duplicates_removed"] > 0 for case in result["cases"])
