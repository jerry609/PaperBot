from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from paperbot.application.services.p2c import load_benchmark_cases, run_module1_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Paper2Context Module 1 benchmark.")
    parser.add_argument(
        "--fixtures",
        default="evals/fixtures/p2c/module1_gold.json",
        help="Path to benchmark fixture JSON.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for benchmark result JSON.",
    )
    parser.add_argument(
        "--fail-under-metric-f1",
        type=float,
        default=0.0,
        help="Exit with code 1 if summary metric_f1 is below this value.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    cases = load_benchmark_cases(args.fixtures)
    result = await run_module1_benchmark(cases)

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    metric_f1 = float(result["summary"].get("metric_f1", 0.0))
    if metric_f1 < args.fail_under_metric_f1:
        return 1
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
