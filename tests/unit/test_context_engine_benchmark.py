from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbot.context_engine.benchmark import (
    ContextBenchmarkCase,
    aggregate_context_benchmark_results,
    evaluate_context_case,
    load_context_benchmark_cases,
    run_context_benchmark,
)


def test_load_context_benchmark_cases_parses_fixture(tmp_path):
    fixture = tmp_path / "context_fixture.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "case_id": "c1",
                    "query": "clinical rag",
                    "query_type": "short",
                    "stage": "survey",
                    "active_track_id": 1,
                    "expected": {
                        "layers": {
                            "layer0_profile": True,
                            "layer1_track": True,
                            "layer2_query": False,
                            "layer3_paper": False,
                        },
                        "token_guard": False,
                        "router_track_id": 2,
                    },
                    "state": {
                        "tracks": [
                            {"id": 1, "name": "Track A", "keywords": ["agents"]},
                            {"id": 2, "name": "Track B", "keywords": ["clinical retrieval"]},
                        ],
                        "global_memories": [{"id": 1, "content": "global pref"}],
                        "track_memories": {"1": [{"id": 10, "content": "agent note"}]},
                        "paper_memories": {},
                        "tasks_by_track": {"1": [{"id": 11, "title": "agent harness"}]},
                        "milestones_by_track": {},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_context_benchmark_cases(fixture)

    assert len(cases) == 1
    assert cases[0].case_id == "c1"
    assert cases[0].expected_router_track_id == 2
    assert cases[0].expected_layers["layer1_track"] is True


def test_evaluate_context_case_scores_layers_guard_and_router():
    case = ContextBenchmarkCase(
        case_id="c1",
        query="clinical rag",
        query_type="short",
        stage="survey",
        expected_layers={
            "layer0_profile": True,
            "layer1_track": False,
            "layer2_query": True,
            "layer3_paper": False,
        },
        expected_token_guard=True,
        expected_router_track_id=2,
    )
    pack = {
        "user_prefs": [{"id": 1}],
        "progress_state": {"tasks": [], "milestones": []},
        "relevant_memories": [{"id": 2}],
        "cross_track_memories": [],
        "paper_memories": [],
        "routing": {"token_guard": {"enabled": True}, "suggestion": {"track_id": 2}},
        "context_layers": {},
    }

    result = evaluate_context_case(case, pack)

    assert result["layer_precision"] == pytest.approx(1.0)
    assert result["layer_recall"] == pytest.approx(1.0)
    assert result["token_guard_correct"] == pytest.approx(1.0)
    assert result["router_correct"] == pytest.approx(1.0)


def test_aggregate_context_benchmark_results_groups_by_stage_and_query_type():
    summary = aggregate_context_benchmark_results(
        [
            {
                "stage": "survey",
                "query_type": "short",
                "layer_precision": 1.0,
                "layer_recall": 1.0,
                "token_guard_correct": 1.0,
                "token_guard_enabled": False,
                "router_evaluable": True,
                "router_covered": 1.0,
                "router_correct": 1.0,
            },
            {
                "stage": "writing",
                "query_type": "long",
                "layer_precision": 0.5,
                "layer_recall": 1.0,
                "token_guard_correct": 1.0,
                "token_guard_enabled": True,
                "router_evaluable": False,
                "router_covered": None,
                "router_correct": None,
            },
        ]
    )

    assert summary["overall"]["case_count"] == 2.0
    assert summary["by_stage"]["survey"]["router_accuracy"] == pytest.approx(1.0)
    assert summary["by_query_type"]["long"]["token_guard_trigger_rate"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_context_benchmark_uses_repo_fixture():
    cases = load_context_benchmark_cases(Path("evals/fixtures/context/bench_v1.json"))

    result = await run_context_benchmark(cases)

    assert result["summary"]["overall"]["layer_precision"] == pytest.approx(1.0)
    assert result["summary"]["overall"]["token_guard_accuracy"] == pytest.approx(1.0)
    assert result["summary"]["overall"]["router_coverage"] == pytest.approx(1.0)
    assert any(case["token_guard_enabled"] for case in result["cases"])
