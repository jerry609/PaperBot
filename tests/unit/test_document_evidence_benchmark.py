from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from paperbot.application.services.document_evidence_benchmark import (
    format_document_evidence_benchmark_report,
    load_document_evidence_benchmark_fixture,
    run_document_evidence_benchmark,
)
from paperbot.context_engine.embeddings import EmbeddingProvider


class _FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> Optional[List[float]]:
        self.calls += 1
        size = max(1, len((text or "").split()))
        return [float(size), 1.0, 0.5]


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


def test_document_evidence_benchmark_uses_injected_embedding_provider():
    fixture = load_document_evidence_benchmark_fixture(
        Path("evals/fixtures/document_evidence/bench_v1.json")
    )
    provider = _FakeEmbeddingProvider()

    result = run_document_evidence_benchmark(
        fixture,
        embedding_provider=provider,
        provider_label="fake",
    )

    assert provider.calls > 0
    assert result["config"]["embedding_provider"] == "fake"
