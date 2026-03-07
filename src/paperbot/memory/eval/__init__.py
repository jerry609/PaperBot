"""
Memory evaluation helpers and benchmarks.

Provides:
- metric collection for memory quality gates
- offline injection-pattern detection helpers
- ROI benchmarking for seeded repro experiences
"""

from paperbot.memory.eval.collector import MemoryMetricCollector
from paperbot.memory.eval.injection_guard import (
    InjectionDetectionResult,
    detect_injection_patterns,
    normalize_injection_text,
)
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
    "InjectionDetectionResult",
    "MemoryMetricCollector",
    "ROIBenchmarkArm",
    "ROIBenchmarkCase",
    "ROIBenchmarkRunner",
    "ROIRunSample",
    "ReproAgentROIBenchmarkRunner",
    "ReproExperienceSeed",
    "detect_injection_patterns",
    "has_configured_llm_api_key",
    "load_repro_experience_seeds",
    "load_roi_cases",
    "normalize_injection_text",
    "run_roi_benchmark",
    "run_roi_benchmark_sync",
    "seed_repro_experience_store",
]
