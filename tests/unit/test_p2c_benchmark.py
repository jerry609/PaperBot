from __future__ import annotations

import json

import pytest

from paperbot.application.services.p2c import (
    BenchmarkCase,
    ReproContextPack,
    aggregate_results,
    evaluate_case,
    load_benchmark_cases,
    run_module1_benchmark,
)
from paperbot.application.services.p2c.models import (
    EvidenceLink,
    ExtractionObservation,
    PaperIdentity,
)


def test_load_benchmark_cases_parses_fixture(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "case_id": "c1",
                    "title": "Case 1",
                    "abstract": "A",
                    "full_text": "B",
                    "year": 2026,
                    "expected": {
                        "architecture": "transformer",
                        "metrics": ["accuracy"],
                        "hyperparameters": ["learning_rate"],
                        "evidence_required_types": ["metric"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_benchmark_cases(fixture)
    assert len(cases) == 1
    assert cases[0].case_id == "c1"
    assert cases[0].expected_architecture == "transformer"
    assert cases[0].expected_metrics == ["accuracy"]


def test_evaluate_case_scores_evidence_and_matches():
    case = BenchmarkCase(
        case_id="c1",
        title="x",
        expected_architecture="transformer",
        expected_metrics=["accuracy"],
        expected_hyperparams=["learning_rate"],
        evidence_required_types=["metric", "hyperparameter"],
    )
    pack = ReproContextPack(
        context_pack_id="ctx",
        paper=PaperIdentity(paper_id="p", title="x"),
        observations=[
            ExtractionObservation(
                id="o1",
                stage="blueprint_extract",
                type="architecture",
                title="arch",
                narrative="n",
                structured_data={"architecture_type": "transformer"},
                confidence=0.8,
            ),
            ExtractionObservation(
                id="o2",
                stage="spec_extract",
                type="hyperparameter",
                title="h",
                narrative="n",
                structured_data={"learning_rate": "1e-4"},
                evidence=[
                    EvidenceLink(
                        type="paper_span",
                        ref="method#char:1-3",
                        supports=["learning_rate"],
                        confidence=0.9,
                    )
                ],
                confidence=0.8,
            ),
            ExtractionObservation(
                id="o3",
                stage="success_criteria",
                type="metric",
                title="m",
                narrative="n",
                structured_data={"metrics": ["accuracy"]},
                evidence=[
                    EvidenceLink(
                        type="paper_span",
                        ref="results#char:1-3",
                        supports=["metrics"],
                        confidence=0.9,
                    )
                ],
                confidence=0.8,
            ),
        ],
    )

    result = evaluate_case(case, pack)
    assert result["metric_f1"] == 1.0
    assert result["hyperparam_f1"] == 1.0
    assert result["architecture_hit"] == 1.0
    assert result["evidence_hit_rate"] == 1.0


@pytest.mark.asyncio
async def test_run_module1_benchmark_returns_summary():
    cases = [
        BenchmarkCase(
            case_id="bench1",
            title="Transformer Benchmark Study",
            abstract="We propose a transformer model.",
            full_text=(
                "Method\n"
                "Learning rate 1e-4, batch size 32.\n"
                "Results\n"
                "Accuracy reaches 90.0.\n"
            ),
            expected_architecture="transformer",
            expected_metrics=["accuracy"],
            expected_hyperparams=["learning_rate", "batch_size"],
            evidence_required_types=["metric", "hyperparameter"],
        )
    ]

    result = await run_module1_benchmark(cases)
    assert "summary" in result
    assert result["cases"]
    summary = result["summary"]
    assert set(summary.keys()) == {
        "metric_f1",
        "hyperparam_f1",
        "architecture_hit_rate",
        "evidence_hit_rate",
        "avg_warnings",
    }
    assert summary["architecture_hit_rate"] >= 0.0
    assert aggregate_results(result["cases"]) == summary
