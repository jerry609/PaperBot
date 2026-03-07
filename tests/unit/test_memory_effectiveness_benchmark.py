from __future__ import annotations

import json
from pathlib import Path

from paperbot.memory.eval.effectiveness_benchmark import (
    EffectivenessQuestion,
    HeuristicMemoryAnswerRunner,
    _is_answer_match,
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

    assert report["summary"]["question_count"] == 32
    assert report["summary"]["retrieval_hit_rate"] >= 0.9
    assert report["summary"]["answer_accuracy"] >= 0.9
    assert report["summary"]["temporal_accuracy"] >= 0.9
    assert report["summary"]["update_accuracy"] >= 0.9
    assert report["summary"]["abstention_accuracy"] >= 1.0
    assert report["summary"]["scope_accuracy"] >= 1.0
    assert report["summary"]["multi_session_accuracy"] >= 0.8
    assert report["summary"]["category_breakdown"]["scope"]["question_count"] == 4
    assert report["summary"]["category_breakdown"]["temporal_previous"]["question_count"] == 4
    assert report["summary"]["case_breakdown"]["track_alpha_longitudinal"]["question_count"] == 7


def test_heuristic_runner_prefers_oldest_memory_for_original_queries():
    question = EffectivenessQuestion(
        question_id="q-oldest",
        query="What was the original focus?",
        expected_answer="retrieval models",
        category="temporal_previous",
    )

    answer = HeuristicMemoryAnswerRunner.answer(
        question,
        retrieved_memories=[
            {"content": "current focus: retrieval models", "created_at": "2026-03-01T10:00:00Z"},
            {"content": "current focus: multimodal agents", "created_at": "2026-03-02T10:00:00Z"},
        ],
    )

    assert answer == "retrieval models"


def test_heuristic_runner_prefers_latest_memory_for_current_queries():
    question = EffectivenessQuestion(
        question_id="q-latest",
        query="What is the current focus now?",
        expected_answer="multimodal agents",
        category="temporal",
    )

    answer = HeuristicMemoryAnswerRunner.answer(
        question,
        retrieved_memories=[
            {"content": "current focus: retrieval models", "created_at": "2026-03-01T10:00:00Z"},
            {"content": "current focus: multimodal agents", "created_at": "2026-03-02T10:00:00Z"},
        ],
    )

    assert answer == "multimodal agents"


def test_answer_match_accepts_semantically_equivalent_short_forms():
    assert _is_answer_match(
        "The validation leakage was fixed by isolating the validation loader.",
        ["isolate validation loader"],
    )
    assert _is_answer_match(
        "Pinning sentencepiece to version 0.1.99.",
        ["pin sentencepiece 0.1.99"],
    )
