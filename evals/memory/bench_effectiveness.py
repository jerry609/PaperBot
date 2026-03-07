from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperbot.memory.eval.effectiveness_benchmark import (
    HeuristicMemoryAnswerRunner,
    LLMMemoryAnswerRunner,
    load_effectiveness_cases,
    run_effectiveness_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multi-session memory effectiveness benchmark."
    )
    parser.add_argument(
        "--fixtures",
        default="evals/memory/fixtures/multi_session_effectiveness.json",
        help="Path to effectiveness benchmark fixture JSON.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Retrieved memories per question.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-backed short-answering.")
    parser.add_argument(
        "--output",
        default="evals/reports/memory_effectiveness_benchmark.json",
        help="Optional output path for the benchmark report JSON.",
    )
    parser.add_argument(
        "--fail-under-answer-accuracy",
        type=float,
        default=0.0,
        help="Exit with code 1 if answer accuracy is below this threshold.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cases = load_effectiveness_cases(args.fixtures)
    runner = LLMMemoryAnswerRunner() if args.use_llm else HeuristicMemoryAnswerRunner()
    report = run_effectiveness_benchmark(cases, runner=runner, top_k=max(1, int(args.top_k)))
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if float(report["summary"].get("answer_accuracy") or 0.0) < float(
        args.fail_under_answer_accuracy
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
