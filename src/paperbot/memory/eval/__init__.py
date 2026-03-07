"""
Memory evaluation helpers and benchmarks.

Provides:
- metric collection for memory quality gates
- ROI benchmarking for seeded repro experiences
"""

from paperbot.memory.eval.collector import MemoryMetricCollector
from paperbot.memory.eval.roi_benchmark import (
    DEFAULT_ARMS,
    ROIBenchmarkArm,
    ROIBenchmarkCase,
    ROIBenchmarkRunner,
    ROIRunSample,
    ReproAgentROIBenchmarkRunner,
    ReproExperienceSeed,
    has_configured_llm_api_key,
    load_repro_experience_seeds,
    load_roi_cases,
    run_roi_benchmark,
    run_roi_benchmark_sync,
    seed_repro_experience_store,
)

__all__ = [
    "DEFAULT_ARMS",
    "MemoryMetricCollector",
    "ROIBenchmarkArm",
    "ROIBenchmarkCase",
    "ROIBenchmarkRunner",
    "ROIRunSample",
    "ReproAgentROIBenchmarkRunner",
    "ReproExperienceSeed",
    "has_configured_llm_api_key",
    "load_repro_experience_seeds",
    "load_roi_cases",
    "run_roi_benchmark",
    "run_roi_benchmark_sync",
    "seed_repro_experience_store",
]
