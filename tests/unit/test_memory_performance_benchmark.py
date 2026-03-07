from __future__ import annotations

from paperbot.memory.eval.performance_benchmark import (
    MemoryPerformanceConfig,
    percentile,
    run_memory_performance_benchmark,
)


def test_percentile_returns_expected_cut_points() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 50.0) == 2.0
    assert percentile(values, 95.0) == 4.0
    assert percentile([], 95.0) == 0.0


def test_run_memory_performance_benchmark_smoke() -> None:
    result = run_memory_performance_benchmark(
        MemoryPerformanceConfig(
            sizes=[200, 400],
            query_count=2,
            batch_size=100,
            seed=7,
        )
    )

    assert result["config"]["sizes"] == [200, 400]
    assert len(result["reports"]) == 2
    for report in result["reports"]:
        assert report["seed"]["rows_seeded"] in (200, 400)
        assert report["search_unscoped"]["count"] == 2.0
        assert report["search_unscoped"]["p95_ms"] >= 0.0
        assert report["search_batch_track"]["p95_ms"] >= 0.0
