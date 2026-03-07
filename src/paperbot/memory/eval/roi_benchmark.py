from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Protocol, Sequence

from paperbot.infrastructure.stores.llm_usage_store import LLMUsageStore
from paperbot.infrastructure.stores.repro_experience_store import ReproExperienceStore
from paperbot.repro import PaperContext, ReproAgent


@dataclass(frozen=True)
class ROIBenchmarkCase:
    case_id: str
    paper_id: str
    title: str
    abstract: str
    method_section: str = ""
    user_id: str = "roi_bench"


@dataclass(frozen=True)
class ReproExperienceSeed:
    user_id: str
    paper_id: Optional[str]
    pattern_type: str
    content: str
    pack_id: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass(frozen=True)
class ROIBenchmarkArm:
    name: str
    description: str
    enable_seeded_memory: bool


@dataclass
class ROIRunSample:
    arm: str
    case_id: str
    paper_id: str
    run_index: int
    first_pass_success: bool
    repair_loops: int
    time_to_pass_sec: float
    token_cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ROIBenchmarkRunner(Protocol):
    async def run_case(
        self,
        case: ROIBenchmarkCase,
        *,
        arm: ROIBenchmarkArm,
        run_index: int,
        seed_experiences: Sequence[ReproExperienceSeed],
    ) -> ROIRunSample: ...


DEFAULT_ARMS: Sequence[ROIBenchmarkArm] = (
    ROIBenchmarkArm(
        name="A",
        description="memory/context bridge disabled",
        enable_seeded_memory=False,
    ),
    ROIBenchmarkArm(
        name="B",
        description="seed 10 verified_structure / success_pattern experiences",
        enable_seeded_memory=True,
    ),
)

_DIRECTION_BY_METRIC = {
    "first_pass_success_rate": "higher",
    "repair_loops": "lower",
    "time_to_pass_sec": "lower",
    "token_cost_usd": "lower",
}

_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "NVIDIA_MINIMAX_API_KEY",
    "NVIDIA_GLM_API_KEY",
)


@contextmanager
def temporary_env(name: str, value: str):
    old_value = os.getenv(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


class ReproAgentROIBenchmarkRunner:
    def __init__(
        self,
        *,
        use_orchestrator: bool = True,
        use_rag: bool = True,
        max_repair_attempts: int = 3,
        use_project_llm_service: bool = True,
        prepare_requirements: bool = True,
        runtime_cache_dir: str | Path | None = None,
        verification_install_timeout: int = 900,
        prefer_cpu_torch: bool = True,
    ) -> None:
        self.use_orchestrator = use_orchestrator
        self.use_rag = use_rag
        self.max_repair_attempts = max_repair_attempts
        self.use_project_llm_service = use_project_llm_service
        self.prepare_requirements = prepare_requirements
        self.runtime_cache_dir = (
            Path(runtime_cache_dir) if runtime_cache_dir else Path("output/runtime_envs/roi_bench")
        )
        self.verification_install_timeout = verification_install_timeout
        self.prefer_cpu_torch = prefer_cpu_torch

    async def run_case(
        self,
        case: ROIBenchmarkCase,
        *,
        arm: ROIBenchmarkArm,
        run_index: int,
        seed_experiences: Sequence[ReproExperienceSeed],
    ) -> ROIRunSample:
        with tempfile.TemporaryDirectory(
            prefix=f"roi_{arm.name.lower()}_{case.case_id}_{run_index}_"
        ) as temp_dir:
            temp_path = Path(temp_dir)
            db_url = f"sqlite:///{temp_path / 'paperbot_roi.db'}"
            with temporary_env("PAPERBOT_DB_URL", db_url):
                if arm.enable_seeded_memory:
                    store = ReproExperienceStore(db_url=db_url, auto_create_schema=True)
                    seed_repro_experience_store(store, seed_experiences)

                agent = ReproAgent(
                    {
                        "use_orchestrator": self.use_orchestrator,
                        "use_rag": self.use_rag,
                        "max_repair_attempts": self.max_repair_attempts,
                        "use_project_llm_service": self.use_project_llm_service,
                        "verification_prepare_requirements": self.prepare_requirements,
                        "verification_runtime_cache_dir": str(self.runtime_cache_dir),
                        "verification_install_timeout": self.verification_install_timeout,
                        "verification_prefer_cpu_torch": self.prefer_cpu_torch,
                    }
                )
                if not arm.enable_seeded_memory:
                    self._disable_seeded_memory(agent)

                paper_context = PaperContext(
                    title=case.title,
                    abstract=case.abstract,
                    method_section=case.method_section,
                )
                setattr(paper_context, "paper_id", case.paper_id)

                output_dir = temp_path / "output"
                result = await agent.reproduce_from_paper(
                    paper_context,
                    output_dir=output_dir,
                    user_id=case.user_id,
                )

                usage_store = LLMUsageStore(db_url=db_url, auto_create_schema=True)
                usage_totals = usage_store.summarize(days=3650).get("totals", {})
                status = getattr(result.status, "value", result.status)
                status_text = str(status or "")

                return ROIRunSample(
                    arm=arm.name,
                    case_id=case.case_id,
                    paper_id=case.paper_id,
                    run_index=run_index,
                    first_pass_success=bool(
                        status_text == "completed" and int(result.retry_count or 0) == 0
                    ),
                    repair_loops=int(result.retry_count or 0),
                    time_to_pass_sec=float(result.total_duration_sec or 0.0),
                    token_cost_usd=float(usage_totals.get("total_cost_usd") or 0.0),
                    metadata={
                        "status": status_text,
                        "overall_score": int(result.overall_score or 0),
                        "generated_files": len(result.generated_files or {}),
                        "error": result.error,
                        "verification": result.verification or {},
                        "verification_runtime": (
                            getattr(agent, "_orchestrator", None).context.get(
                                "verification_runtime"
                            )
                            if getattr(agent, "_orchestrator", None)
                            and getattr(agent._orchestrator, "context", None)
                            else None
                        ),
                        "verification_runtime_error": (
                            getattr(agent, "_orchestrator", None).context.get(
                                "verification_runtime_error"
                            )
                            if getattr(agent, "_orchestrator", None)
                            and getattr(agent._orchestrator, "context", None)
                            else None
                        ),
                    },
                )

    @staticmethod
    def _disable_seeded_memory(agent: ReproAgent) -> None:
        agent.experience_store = None
        agent.generation_node.memory._experience_store = None
        agent.verification_node._experience_store = None
        if agent._orchestrator is not None:
            agent._orchestrator.coding_agent.generation_node.memory._experience_store = None
            agent._orchestrator.verification_agent._experience_store = None
            agent._orchestrator.debugging_agent._experience_store = None


def has_configured_llm_api_key() -> bool:
    return any(bool(str(os.getenv(name) or "").strip()) for name in _API_KEY_ENV_VARS)


def load_roi_cases(path: str | Path) -> List[ROIBenchmarkCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: List[ROIBenchmarkCase] = []
    for row in rows:
        cases.append(
            ROIBenchmarkCase(
                case_id=str(row.get("case_id") or ""),
                paper_id=str(row.get("paper_id") or row.get("case_id") or ""),
                title=str(row.get("title") or ""),
                abstract=str(row.get("abstract") or ""),
                method_section=str(row.get("method_section") or ""),
                user_id=str(row.get("user_id") or "roi_bench"),
            )
        )
    return cases


def load_repro_experience_seeds(path: str | Path) -> List[ReproExperienceSeed]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds: List[ReproExperienceSeed] = []
    for row in rows:
        seeds.append(
            ReproExperienceSeed(
                user_id=str(row.get("user_id") or "roi_bench"),
                paper_id=(str(row.get("paper_id")) if row.get("paper_id") else None),
                pack_id=(str(row.get("pack_id")) if row.get("pack_id") else None),
                pattern_type=str(row.get("pattern_type") or ""),
                content=str(row.get("content") or ""),
                code_snippet=(str(row.get("code_snippet")) if row.get("code_snippet") else None),
            )
        )
    return seeds


def seed_repro_experience_store(
    store: ReproExperienceStore,
    seeds: Sequence[ReproExperienceSeed],
) -> int:
    inserted = 0
    for seed in seeds:
        store.add(
            user_id=seed.user_id,
            paper_id=seed.paper_id,
            pack_id=seed.pack_id,
            pattern_type=seed.pattern_type,
            content=seed.content,
            code_snippet=seed.code_snippet,
        )
        inserted += 1
    return inserted


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def summarize_arm_samples(
    samples: Sequence[ROIRunSample],
    *,
    description: str,
) -> Dict[str, Any]:
    return {
        "description": description,
        "sample_count": len(samples),
        "case_count": len({sample.case_id for sample in samples}),
        "first_pass_success_rate": _average(
            [1.0 if sample.first_pass_success else 0.0 for sample in samples]
        ),
        "repair_loops": _average([float(sample.repair_loops) for sample in samples]),
        "time_to_pass_sec": _average([float(sample.time_to_pass_sec) for sample in samples]),
        "token_cost_usd": _average([float(sample.token_cost_usd) for sample in samples]),
    }


def _pair_metric_differences(
    baseline_samples: Sequence[ROIRunSample],
    treatment_samples: Sequence[ROIRunSample],
    metric_name: str,
) -> List[float]:
    baseline_by_key = {(sample.case_id, sample.run_index): sample for sample in baseline_samples}
    treatment_by_key = {(sample.case_id, sample.run_index): sample for sample in treatment_samples}

    differences: List[float] = []
    for key in sorted(set(baseline_by_key) & set(treatment_by_key)):
        baseline = baseline_by_key[key]
        treatment = treatment_by_key[key]
        if metric_name == "first_pass_success_rate":
            base_value = 1.0 if baseline.first_pass_success else 0.0
            treatment_value = 1.0 if treatment.first_pass_success else 0.0
        else:
            base_value = float(getattr(baseline, metric_name))
            treatment_value = float(getattr(treatment, metric_name))

        if _DIRECTION_BY_METRIC[metric_name] == "higher":
            differences.append(treatment_value - base_value)
        else:
            differences.append(base_value - treatment_value)
    return differences


def _significance_report(
    baseline_samples: Sequence[ROIRunSample],
    treatment_samples: Sequence[ROIRunSample],
    metric_name: str,
    *,
    min_samples: int = 15,
) -> Dict[str, Any]:
    differences = _pair_metric_differences(baseline_samples, treatment_samples, metric_name)
    sample_count = len(differences)
    if sample_count < min_samples:
        return {
            "status": "insufficient_samples",
            "sample_count": sample_count,
            "min_samples": min_samples,
        }

    mean_diff = _average(differences)
    if sample_count == 1:
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        variance = sum((value - mean_diff) ** 2 for value in differences) / max(1, sample_count - 1)
        if variance == 0:
            p_value = 0.0 if mean_diff != 0 else 1.0
        else:
            stderr = (variance**0.5) / (sample_count**0.5)
            z_score = mean_diff / stderr if stderr > 0 else 0.0
            p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_score)))

    return {
        "status": "computed",
        "sample_count": sample_count,
        "paired_improvement_mean": mean_diff,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


def build_delta_report(
    baseline_summary: Dict[str, Any],
    treatment_summary: Dict[str, Any],
    baseline_samples: Sequence[ROIRunSample],
    treatment_samples: Sequence[ROIRunSample],
    *,
    min_significance_samples: int = 15,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for metric_name in (
        "first_pass_success_rate",
        "repair_loops",
        "time_to_pass_sec",
        "token_cost_usd",
    ):
        baseline_value = float(baseline_summary.get(metric_name) or 0.0)
        treatment_value = float(treatment_summary.get(metric_name) or 0.0)
        absolute_delta = treatment_value - baseline_value
        relative_delta_pct = 0.0
        if baseline_value != 0:
            relative_delta_pct = (absolute_delta / abs(baseline_value)) * 100.0

        if _DIRECTION_BY_METRIC[metric_name] == "higher":
            improved = treatment_value > baseline_value
        else:
            improved = treatment_value < baseline_value

        if treatment_value == baseline_value:
            direction = "flat"
        else:
            direction = "improved" if improved else "regressed"

        report[metric_name] = {
            "baseline": baseline_value,
            "treatment": treatment_value,
            "absolute_delta": absolute_delta,
            "relative_delta_pct": relative_delta_pct,
            "direction": direction,
            "significance": _significance_report(
                baseline_samples,
                treatment_samples,
                metric_name,
                min_samples=min_significance_samples,
            ),
        }
    return report


async def run_roi_benchmark(
    cases: Sequence[ROIBenchmarkCase],
    seed_experiences: Sequence[ReproExperienceSeed],
    *,
    runner: ROIBenchmarkRunner,
    runs_per_case: int = 3,
    arms: Sequence[ROIBenchmarkArm] = DEFAULT_ARMS,
    min_significance_samples: int = 15,
) -> Dict[str, Any]:
    all_samples: List[ROIRunSample] = []
    for arm in arms:
        arm_seeds = seed_experiences if arm.enable_seeded_memory else []
        for case in cases:
            for run_index in range(max(1, int(runs_per_case))):
                sample = await runner.run_case(
                    case,
                    arm=arm,
                    run_index=run_index,
                    seed_experiences=arm_seeds,
                )
                all_samples.append(sample)

    arms_summary: Dict[str, Any] = {}
    arm_samples: Dict[str, List[ROIRunSample]] = {}
    for arm in arms:
        samples = [sample for sample in all_samples if sample.arm == arm.name]
        arm_samples[arm.name] = samples
        arms_summary[arm.name] = summarize_arm_samples(samples, description=arm.description)

    baseline_arm = arms[0]
    treatment_arm = arms[1]
    delta_report = build_delta_report(
        arms_summary[baseline_arm.name],
        arms_summary[treatment_arm.name],
        arm_samples[baseline_arm.name],
        arm_samples[treatment_arm.name],
        min_significance_samples=min_significance_samples,
    )

    return {
        "config": {
            "runs_per_case": max(1, int(runs_per_case)),
            "cases": len(cases),
            "samples_per_arm": len(cases) * max(1, int(runs_per_case)),
            "min_significance_samples": min_significance_samples,
            "arms": [asdict(arm) for arm in arms],
        },
        "cases": [asdict(case) for case in cases],
        "arms": arms_summary,
        "delta": delta_report,
        "samples": [asdict(sample) for sample in all_samples],
    }


def run_roi_benchmark_sync(
    cases: Sequence[ROIBenchmarkCase],
    seed_experiences: Sequence[ReproExperienceSeed],
    *,
    runner: ROIBenchmarkRunner,
    runs_per_case: int = 3,
    arms: Sequence[ROIBenchmarkArm] = DEFAULT_ARMS,
    min_significance_samples: int = 15,
) -> Dict[str, Any]:
    return asyncio.run(
        run_roi_benchmark(
            cases,
            seed_experiences,
            runner=runner,
            runs_per_case=runs_per_case,
            arms=arms,
            min_significance_samples=min_significance_samples,
        )
    )
