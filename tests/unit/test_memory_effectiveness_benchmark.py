from __future__ import annotations

import json
from pathlib import Path

from paperbot.memory.eval.effectiveness_benchmark import (
    HeuristicMemoryAnswerRunner,
    load_effectiveness_cases,
    run_effectiveness_benchmark,
)


def test_load_effectiveness_cases_parses_fixture(tmp_path: Path):
    fixture = tmp_path / "effectiveness.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "case_id": "case_1",
                    "user_id": "u1",
                    "sessions": [
                        {
                            "session_id": "s1",
                            "writes": [
                                {
                                    "kind": "fact",
                                    "content": "dataset choice: CIFAR-10",
                                    "scope_type": "paper",
                                    "scope_id": "p1",
                                }
                            ],
                        }
                    ],
                    "questions": [
                        {
                            "question_id": "q1",
                            "query": "Which dataset?",
                            "expected_answer": "CIFAR-10",
                            "category": "fact",
                            "scope_type": "paper",
                            "scope_id": "p1",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_effectiveness_cases(fixture)
    assert len(cases) == 1
    assert cases[0].case_id == "case_1"
    assert cases[0].sessions[0].writes[0].content == "dataset choice: CIFAR-10"


def test_run_effectiveness_benchmark_with_heuristic_runner():
    cases = load_effectiveness_cases("evals/memory/fixtures/multi_session_effectiveness.json")
    report = run_effectiveness_benchmark(cases, runner=HeuristicMemoryAnswerRunner(), top_k=4)

    assert report["summary"]["question_count"] == 4
    assert report["summary"]["retrieval_hit_rate"] >= 0.75
    assert report["summary"]["answer_accuracy"] >= 0.75
    assert report["summary"]["temporal_accuracy"] >= 1.0
    assert report["summary"]["update_accuracy"] >= 1.0
    assert report["summary"]["abstention_accuracy"] >= 1.0
