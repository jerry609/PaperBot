from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.context_engine.benchmark import (  # noqa: E402
    format_context_benchmark_report,
    load_context_benchmark_cases,
    run_context_benchmark,
)


FIXTURE_PATH = REPO_ROOT / "evals" / "fixtures" / "context" / "bench_v1.json"
THRESHOLDS = {
    "layer_precision": 0.95,
    "token_guard_accuracy": 1.0,
    "router_coverage": 1.0,
    "router_accuracy": 1.0,
}


async def main() -> dict:
    cases = load_context_benchmark_cases(FIXTURE_PATH)
    result = await run_context_benchmark(cases)
    print(format_context_benchmark_report(result))
    summary = result["summary"]["overall"]
    for metric_name, threshold in THRESHOLDS.items():
        value = float(summary.get(metric_name, 0.0))
        assert (
            value >= threshold
        ), f"{metric_name} dropped below threshold: {value:.3f} < {threshold:.3f}"
    return {"status": "ok", "fixture": str(FIXTURE_PATH), "summary": summary}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main()), ensure_ascii=False, indent=2))
