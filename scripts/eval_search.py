from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.services.retrieval_benchmark import (  # noqa: E402
    format_benchmark_report,
    load_retrieval_benchmark_cases,
    run_retrieval_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline retrieval benchmark.")
    parser.add_argument(
        "--fixtures",
        default="evals/fixtures/retrieval/bench_v2.jsonl",
        help="Path to the retrieval benchmark fixture JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument("--ndcg-k", type=int, default=10, help="Cutoff K for nDCG.")
    parser.add_argument("--mrr-k", type=int, default=10, help="Cutoff K for MRR.")
    parser.add_argument("--recall-k", type=int, default=50, help="Cutoff K for recall.")
    parser.add_argument(
        "--fail-under-ndcg",
        type=float,
        default=0.0,
        help="Exit non-zero when overall nDCG falls below this threshold.",
    )
    parser.add_argument(
        "--fail-under-mrr",
        type=float,
        default=0.0,
        help="Exit non-zero when overall MRR falls below this threshold.",
    )
    parser.add_argument(
        "--fail-under-recall",
        type=float,
        default=0.0,
        help="Exit non-zero when overall recall falls below this threshold.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    cases = load_retrieval_benchmark_cases(args.fixtures)
    result = await run_retrieval_benchmark(
        cases,
        ndcg_k=args.ndcg_k,
        mrr_k=args.mrr_k,
        recall_k=args.recall_k,
    )

    print(format_benchmark_report(result))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary = result["summary"]["overall"]
    ndcg_value = float(summary.get(f"ndcg_at_{args.ndcg_k}", 0.0))
    mrr_value = float(summary.get(f"mrr_at_{args.mrr_k}", 0.0))
    recall_value = float(summary.get(f"recall_at_{args.recall_k}", 0.0))

    if ndcg_value < float(args.fail_under_ndcg):
        return 1
    if mrr_value < float(args.fail_under_mrr):
        return 1
    if recall_value < float(args.fail_under_recall):
        return 1
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
