from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.memory.eval.performance_benchmark import (  # noqa: E402
    MemoryPerformanceConfig,
    run_memory_performance_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline memory performance benchmark.")
    parser.add_argument(
        "--sizes",
        default="10000,100000,1000000",
        help="Comma-separated dataset sizes to benchmark.",
    )
    parser.add_argument("--query-count", type=int, default=25, help="Queries per benchmark slice.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Seed batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser


def _format_report(result: dict) -> str:
    lines = ["Memory Performance Benchmark"]
    for report in result.get("reports", []):
        lines.append(
            (
                f"size={report['size']} rows | seed_ms={report['seed']['seed_time_ms']:.2f} | "
                f"db_size={report['seed']['db_size_bytes']} bytes | "
                f"unscoped_p95={report['search_unscoped']['p95_ms']:.2f}ms | "
                f"track_p95={report['search_track_scoped']['p95_ms']:.2f}ms | "
                f"batch_track_p95={report['search_batch_track']['p95_ms']:.2f}ms"
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    sizes = [int(chunk.strip()) for chunk in str(args.sizes).split(",") if chunk.strip()]
    result = run_memory_performance_benchmark(
        MemoryPerformanceConfig(
            sizes=sizes,
            query_count=args.query_count,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    )
    print(_format_report(result))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
