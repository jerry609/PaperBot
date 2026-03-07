from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperbot.memory.eval.roi_benchmark import (
    ReproAgentROIBenchmarkRunner,
    has_configured_llm_api_key,
    load_repro_experience_seeds,
    load_roi_cases,
    run_roi_benchmark_sync,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ROI benchmark for seeded repro experiences."
    )
    parser.add_argument(
        "--cases",
        default="evals/memory/fixtures/roi_cases.json",
        help="Path to ROI paper-set fixture JSON.",
    )
    parser.add_argument(
        "--experiences",
        default="evals/memory/fixtures/repro_experiences.json",
        help="Path to repro experience seed JSON.",
    )
    parser.add_argument(
        "--runs-per-case",
        type=int,
        default=3,
        help="How many repeated runs to execute for each paper in each arm.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Optional limit for a smaller manual spot-check.",
    )
    parser.add_argument(
        "--output",
        default="evals/reports/memory_roi_benchmark.json",
        help="Optional output path for the benchmark report JSON.",
    )
    parser.add_argument(
        "--use-legacy",
        action="store_true",
        help="Use the legacy Paper2Code pipeline instead of the orchestrator path.",
    )
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=3,
        help="Maximum repair loops for each Paper2Code run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not has_configured_llm_api_key():
        parser.error(
            "ROI bench requires a configured API key via OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "OPENROUTER_API_KEY, NVIDIA_MINIMAX_API_KEY, or NVIDIA_GLM_API_KEY."
        )

    cases = load_roi_cases(args.cases)
    if args.limit_cases:
        cases = cases[: max(1, int(args.limit_cases))]
    experiences = load_repro_experience_seeds(args.experiences)

    runner = ReproAgentROIBenchmarkRunner(
        use_orchestrator=not args.use_legacy,
        max_repair_attempts=max(0, int(args.max_repair_attempts)),
    )
    report = run_roi_benchmark_sync(
        cases,
        experiences,
        runner=runner,
        runs_per_case=max(1, int(args.runs_per_case)),
    )

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
