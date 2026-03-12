from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.services.document_evidence_benchmark import (  # noqa: E402
    format_document_evidence_benchmark_report,
    load_document_evidence_benchmark_fixture,
    run_document_evidence_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the document evidence benchmark.")
    parser.add_argument(
        "--fixtures",
        default="evals/fixtures/document_evidence/bench_v1.json",
        help="Path to the document evidence benchmark fixture JSON.",
    )
    parser.add_argument(
        "--output",
        default="output/reports/document_evidence_bench_v1.json",
        help="Optional output path for the benchmark report JSON.",
    )
    parser.add_argument(
        "--modes",
        default="fts_only,embedding_only,hybrid",
        help="Comma-separated retrieval modes to run.",
    )
    parser.add_argument(
        "--fail-under-hybrid-recall",
        type=float,
        default=0.0,
        help="Fail if hybrid recall drops below this threshold.",
    )
    parser.add_argument(
        "--fail-under-hybrid-hit-rate",
        type=float,
        default=0.0,
        help="Fail if hybrid evidence hit rate drops below this threshold.",
    )
    args = parser.parse_args()

    fixture = load_document_evidence_benchmark_fixture(REPO_ROOT / args.fixtures)
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    result = run_document_evidence_benchmark(fixture, modes=modes)
    print(format_document_evidence_benchmark_report(result))

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    hybrid = (result.get("summary") or {}).get("hybrid", {}).get("overall", {})
    top_k = 5
    for key in hybrid.keys():
        if key.startswith("recall_at_"):
            top_k = int(key.rsplit("_", 1)[-1])
            break
    hybrid_recall = float(hybrid.get(f"recall_at_{top_k}", 0.0))
    hybrid_hit_rate = float(hybrid.get("evidence_hit_rate", 0.0))

    if hybrid_recall < float(args.fail_under_hybrid_recall):
        raise SystemExit(
            f"hybrid recall dropped below threshold: {hybrid_recall:.3f} < {args.fail_under_hybrid_recall:.3f}"
        )
    if hybrid_hit_rate < float(args.fail_under_hybrid_hit_rate):
        raise SystemExit(
            f"hybrid evidence hit rate dropped below threshold: {hybrid_hit_rate:.3f} < {args.fail_under_hybrid_hit_rate:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
