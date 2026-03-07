from __future__ import annotations

import json
from pathlib import Path

from paperbot.infrastructure.stores.repro_experience_store import ReproExperienceStore
from paperbot.memory.eval.roi_benchmark import (
    DEFAULT_ARMS,
    ROIBenchmarkArm,
    ROIBenchmarkCase,
    ROIRunSample,
    ReproExperienceSeed,
    build_delta_report,
    has_configured_llm_api_key,
    load_repro_experience_seeds,
    load_roi_cases,
    run_roi_benchmark_sync,
    seed_repro_experience_store,
    summarize_arm_samples,
)


class FakeRunner:
    async def run_case(self, case, *, arm, run_index, seed_experiences):
        base_success = (run_index + len(case.case_id)) % 2 == 0
        if arm.name == "A":
            first_pass_success = base_success
            repair_loops = 2 + (run_index % 2)
            time_to_pass_sec = 20.0 + run_index + (len(case.case_id) % 3)
            token_cost_usd = 0.18 + (run_index * 0.01)
        else:
            first_pass_success = True
            repair_loops = run_index % 2
            time_to_pass_sec = 12.0 + run_index + (len(case.case_id) % 2)
            token_cost_usd = 0.09 + (run_index * 0.01)
            assert len(seed_experiences) == 10
        return ROIRunSample(
            arm=arm.name,
            case_id=case.case_id,
            paper_id=case.paper_id,
            run_index=run_index,
            first_pass_success=first_pass_success,
            repair_loops=repair_loops,
            time_to_pass_sec=time_to_pass_sec,
            token_cost_usd=token_cost_usd,
        )


def test_load_roi_cases_parses_fixture(tmp_path: Path):
    fixture = tmp_path / "roi_cases.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "case_id": "c1",
                    "paper_id": "paper:c1",
                    "title": "Case 1",
                    "abstract": "Abstract",
                    "method_section": "Method",
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_roi_cases(fixture)
    assert len(cases) == 1
    assert cases[0].case_id == "c1"
    assert cases[0].paper_id == "paper:c1"
    assert cases[0].method_section == "Method"


def test_load_repro_experience_seeds_parses_fixture(tmp_path: Path):
    fixture = tmp_path / "repro_experiences.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "user_id": "roi_bench",
                    "paper_id": "paper:c1",
                    "pattern_type": "success_pattern",
                    "content": "Generated model.py",
                }
            ]
        ),
        encoding="utf-8",
    )

    seeds = load_repro_experience_seeds(fixture)
    assert len(seeds) == 1
    assert seeds[0].paper_id == "paper:c1"
    assert seeds[0].pattern_type == "success_pattern"


def test_seed_repro_experience_store_persists_rows():
    store = ReproExperienceStore(db_url="sqlite://", auto_create_schema=True)
    inserted = seed_repro_experience_store(
        store,
        [
            ReproExperienceSeed(
                user_id="roi_bench",
                paper_id="paper:1",
                pattern_type="success_pattern",
                content="Generated trainer.py",
            ),
            ReproExperienceSeed(
                user_id="roi_bench",
                paper_id="paper:1",
                pattern_type="verified_structure",
                content="Verified structure with model.py and train.py",
            ),
        ],
    )

    rows = store.get_by_paper_id("paper:1", user_id="roi_bench")
    assert inserted == 2
    assert len(rows) == 2


def test_run_roi_benchmark_builds_ab_report_with_significance():
    cases = [
        ROIBenchmarkCase(
            case_id=f"case_{index}",
            paper_id=f"paper:{index}",
            title=f"Case {index}",
            abstract="Abstract",
            method_section="Method",
        )
        for index in range(5)
    ]
    seeds = [
        ReproExperienceSeed(
            user_id="roi_bench",
            paper_id=f"paper:{(index // 2)}",
            pattern_type="success_pattern" if index % 2 == 0 else "verified_structure",
            content=f"seed_{index}",
        )
        for index in range(10)
    ]

    report = run_roi_benchmark_sync(
        cases,
        seeds,
        runner=FakeRunner(),
        runs_per_case=3,
    )

    assert report["config"]["samples_per_arm"] == 15
    assert set(report["arms"].keys()) == {"A", "B"}
    assert (
        report["arms"]["B"]["first_pass_success_rate"]
        > report["arms"]["A"]["first_pass_success_rate"]
    )
    assert report["delta"]["repair_loops"]["direction"] == "improved"
    assert report["delta"]["time_to_pass_sec"]["direction"] == "improved"
    assert report["delta"]["token_cost_usd"]["significance"]["status"] == "computed"


def test_delta_report_marks_insufficient_samples_when_pairs_too_small():
    baseline = [
        ROIRunSample(
            arm="A",
            case_id="case_1",
            paper_id="paper:1",
            run_index=0,
            first_pass_success=False,
            repair_loops=2,
            time_to_pass_sec=25.0,
            token_cost_usd=0.2,
        )
    ]
    treatment = [
        ROIRunSample(
            arm="B",
            case_id="case_1",
            paper_id="paper:1",
            run_index=0,
            first_pass_success=True,
            repair_loops=1,
            time_to_pass_sec=18.0,
            token_cost_usd=0.1,
        )
    ]
    baseline_summary = summarize_arm_samples(baseline, description="baseline")
    treatment_summary = summarize_arm_samples(treatment, description="seeded")

    report = build_delta_report(
        baseline_summary,
        treatment_summary,
        baseline,
        treatment,
        min_significance_samples=15,
    )

    assert report["first_pass_success_rate"]["significance"]["status"] == "insufficient_samples"


def test_has_configured_llm_api_key_false_when_env_missing(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_MINIMAX_API_KEY",
        "NVIDIA_GLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert has_configured_llm_api_key() is False
