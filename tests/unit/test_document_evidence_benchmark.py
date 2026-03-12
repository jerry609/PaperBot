from __future__ import annotations

from pathlib import Path

from paperbot.application.services.document_evidence_benchmark import (
    format_document_evidence_benchmark_report,
    load_document_evidence_benchmark_fixture,
    run_document_evidence_benchmark,
)


def test_document_evidence_benchmark_runs_all_modes():
    fixture = load_document_evidence_benchmark_fixture(
        Path("evals/fixtures/document_evidence/bench_v1.json")
    )
    result = run_document_evidence_benchmark(fixture)

    assert set(result["summary"].keys()) == {"embedding_only", "fts_only", "hybrid"}
    assert result["summary"]["hybrid"]["overall"]["case_count"] == 2.0
    assert "evidence_hit_rate" in result["summary"]["hybrid"]["overall"]

    report = format_document_evidence_benchmark_report(result)
    assert "Document Evidence Benchmark" in report
    assert "hybrid" in report
