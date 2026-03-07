from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.context_engine.benchmark import (  # noqa: E402
    format_context_benchmark_report,
    load_context_benchmark_cases,
    run_context_benchmark,
)


DEFAULT_FIXTURE = "evals/fixtures/context/bench_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline context-engine benchmark.")
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURE, help="Fixture JSON/JSONL path.")
    parser.add_argument("--output", default="", help="Optional path for JSON report.")
    parser.add_argument(
        "--fail-under-layer-precision",
        type=float,
        default=0.0,
        help="Exit non-zero when overall layer precision drops below this threshold.",
    )
    parser.add_argument(
        "--fail-under-token-guard-accuracy",
        type=float,
        default=0.0,
        help="Exit non-zero when token-guard accuracy drops below this threshold.",
    )
    parser.add_argument(
        "--fail-under-router-accuracy",
        type=float,
        default=0.0,
        help="Exit non-zero when router accuracy drops below this threshold.",
    )
    parser.add_argument(
        "--fail-under-router-coverage",
        type=float,
        default=0.0,
        help="Exit non-zero when router coverage drops below this threshold.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    cases = load_context_benchmark_cases(args.fixtures)
    result = await run_context_benchmark(cases)
    print(format_context_benchmark_report(result))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    overall = result["summary"]["overall"]
    if float(overall.get("layer_precision", 0.0)) < float(args.fail_under_layer_precision):
        return 1
    if float(overall.get("token_guard_accuracy", 0.0)) < float(
        args.fail_under_token_guard_accuracy
    ):
        return 1
    if float(overall.get("router_accuracy", 0.0)) < float(args.fail_under_router_accuracy):
        return 1
    if float(overall.get("router_coverage", 0.0)) < float(args.fail_under_router_coverage):
        return 1
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
