from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.services.document_evidence_benchmark import (  # noqa: E402
    format_document_evidence_benchmark_report,
    load_document_evidence_benchmark_fixture,
    run_document_evidence_benchmark,
)


FIXTURE_PATH = REPO_ROOT / "evals" / "fixtures" / "document_evidence" / "bench_v1.json"
THRESHOLDS = {
    "recall_at_5": 0.5,
    "evidence_hit_rate": 0.5,
}


def main() -> dict:
    fixture = load_document_evidence_benchmark_fixture(FIXTURE_PATH)
    result = run_document_evidence_benchmark(fixture)
    hybrid_summary = (result.get("summary") or {}).get("hybrid", {}).get("overall", {})
    print(format_document_evidence_benchmark_report(result))

    for metric_name, threshold in THRESHOLDS.items():
        value = float(hybrid_summary.get(metric_name, 0.0))
        assert (
            value >= threshold
        ), f"{metric_name} dropped below threshold: {value:.3f} < {threshold:.3f}"

    return {
        "status": "ok",
        "fixture": str(FIXTURE_PATH),
        "summary": hybrid_summary,
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
