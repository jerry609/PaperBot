from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.services.retrieval_benchmark import (  # noqa: E402
    format_benchmark_report,
    load_retrieval_benchmark_cases,
    run_retrieval_benchmark,
)


FIXTURE_PATH = REPO_ROOT / "evals" / "fixtures" / "retrieval" / "bench_v2.jsonl"
THRESHOLDS = {
    "ndcg_at_10": 0.95,
    "mrr_at_10": 0.95,
    "recall_at_50": 1.0,
}


async def main() -> dict:
    cases = load_retrieval_benchmark_cases(FIXTURE_PATH)
    result = await run_retrieval_benchmark(cases, ndcg_k=10, mrr_k=10, recall_k=50)
    summary = result["summary"]["overall"]
    print(format_benchmark_report(result))

    for metric_name, threshold in THRESHOLDS.items():
        value = float(summary.get(metric_name, 0.0))
        assert (
            value >= threshold
        ), f"{metric_name} dropped below threshold: {value:.3f} < {threshold:.3f}"

    return {
        "status": "ok",
        "fixture": str(FIXTURE_PATH),
        "summary": summary,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main()), ensure_ascii=False, indent=2))
