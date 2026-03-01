from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import GenerateContextRequest, RawPaperData, ReproContextPack
from .orchestrator import ExtractionOrchestrator


@dataclass
class BenchmarkCase:
    case_id: str
    title: str
    abstract: str = ""
    full_text: str = ""
    year: int = 0
    expected_metrics: List[str] = field(default_factory=list)
    expected_hyperparams: List[str] = field(default_factory=list)
    expected_architecture: Optional[str] = None
    evidence_required_types: List[str] = field(default_factory=list)


def _f1_score(expected: Sequence[str], predicted: Sequence[str]) -> Dict[str, float]:
    exp = {item.strip().lower() for item in expected if item}
    pred = {item.strip().lower() for item in predicted if item}
    if not exp and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not exp:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    tp = len(exp & pred)
    precision = tp / len(pred)
    recall = tp / len(exp)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def load_benchmark_cases(path: str | Path) -> List[BenchmarkCase]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases: List[BenchmarkCase] = []
    for row in payload:
        cases.append(
            BenchmarkCase(
                case_id=str(row.get("case_id") or ""),
                title=str(row.get("title") or ""),
                abstract=str(row.get("abstract") or ""),
                full_text=str(row.get("full_text") or ""),
                year=int(row.get("year") or 0),
                expected_metrics=list(row.get("expected", {}).get("metrics", []) or []),
                expected_hyperparams=list(row.get("expected", {}).get("hyperparameters", []) or []),
                expected_architecture=(row.get("expected", {}) or {}).get("architecture"),
                evidence_required_types=list(
                    row.get("expected", {}).get("evidence_required_types", []) or []
                ),
            )
        )
    return cases


def evaluate_case(case: BenchmarkCase, pack: ReproContextPack) -> Dict[str, Any]:
    metric_candidates: List[str] = []
    for obs in pack.get_by_type("metric"):
        metric_candidates.extend((obs.structured_data.get("metrics") or []))
    metric_scores = _f1_score(case.expected_metrics, metric_candidates)

    hyperparam_candidates: List[str] = []
    for obs in pack.get_by_type("hyperparameter"):
        hyperparam_candidates.extend((obs.structured_data or {}).keys())
    hyperparam_scores = _f1_score(case.expected_hyperparams, hyperparam_candidates)

    architecture = ""
    architecture_obs = pack.get_by_type("architecture")
    if architecture_obs:
        architecture = str(architecture_obs[0].structured_data.get("architecture_type") or "")

    architecture_hit = (
        1.0
        if not case.expected_architecture
        else float(architecture.lower() == str(case.expected_architecture).lower())
    )

    required = case.evidence_required_types
    if required:
        evidence_hit_count = 0
        for obs_type in required:
            observations = pack.get_by_type(obs_type)
            if any(item.evidence for item in observations):
                evidence_hit_count += 1
        evidence_hit_rate = evidence_hit_count / len(required)
    else:
        evidence_hit_rate = 1.0

    return {
        "case_id": case.case_id,
        "metric_f1": metric_scores["f1"],
        "hyperparam_f1": hyperparam_scores["f1"],
        "architecture_hit": architecture_hit,
        "evidence_hit_rate": evidence_hit_rate,
        "warnings_count": len(pack.warnings),
    }


def aggregate_results(case_results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not case_results:
        return {
            "metric_f1": 0.0,
            "hyperparam_f1": 0.0,
            "architecture_hit_rate": 0.0,
            "evidence_hit_rate": 0.0,
            "avg_warnings": 0.0,
        }

    n = float(len(case_results))
    return {
        "metric_f1": sum(float(row["metric_f1"]) for row in case_results) / n,
        "hyperparam_f1": sum(float(row["hyperparam_f1"]) for row in case_results) / n,
        "architecture_hit_rate": sum(float(row["architecture_hit"]) for row in case_results) / n,
        "evidence_hit_rate": sum(float(row["evidence_hit_rate"]) for row in case_results) / n,
        "avg_warnings": sum(float(row["warnings_count"]) for row in case_results) / n,
    }


async def run_module1_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    orchestrator: Optional[ExtractionOrchestrator] = None,
) -> Dict[str, Any]:
    orch = orchestrator or ExtractionOrchestrator()
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        request = GenerateContextRequest(paper_id=f"benchmark:{case.case_id}", depth="standard")
        raw_paper = RawPaperData(
            paper_id=request.paper_id,
            title=case.title,
            abstract=case.abstract,
            year=case.year,
            full_text=case.full_text,
            source_adapter="benchmark_fixture",
        )
        pack = await orch.run(request, raw_paper=raw_paper)
        case_results.append(evaluate_case(case, pack))

    summary = aggregate_results(case_results)
    return {
        "cases": case_results,
        "summary": summary,
    }
