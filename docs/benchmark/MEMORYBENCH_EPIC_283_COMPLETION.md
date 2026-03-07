# MemoryBench Epic #283 Completion Report

Date: 2026-03-07
Epic: [#283](https://github.com/jerry609/PaperBot/issues/283)
Integration branch: `feat/epic-283-memorybench-suite`

## Summary

Epic #283 is fully implemented locally and integrated into a single branch from `dev`.
The benchmark suite now covers retrieval quality, context extraction quality, scope isolation,
injection robustness, latency/performance baselines, and ROI measurement for seeded repro memory.

## Delivered Workstreams

| Issue | Area | Branch | Commit | Status |
|------:|------|--------|--------|--------|
| #284 | Retrieval Bench v2 | `feat/memorybench-284-retrieval-bench` | `cf8297c` | Done |
| #285 | Scope Isolation Bench | `feat/memorybench-285-scope-isolation` | `e4c3511` | Done |
| #286 | Context Extraction Bench | `feat/contextbench-286-context-extraction-bench` | `4d9fa7a` | Done |
| #287 | Injection Robustness L1 | `feat/memorybench-287-injection-robustness` | `37641a9` | Done |
| #288 | Performance Bench | `feat/memorybench-288-performance-bench` | `5676f31` | Done |
| #289 | ROI Bench | `feat/p2cbench-289-roi-bench` | `f9c08a2` | Done |

## What Was Added

### Retrieval and Context Quality

- Retrieval benchmark service + CLI runner for Recall@K / MRR / nDCG
- Context extraction benchmark for layered assembly, token guard, and router coverage/accuracy
- Synthetic fixtures and smoke runners for both benchmark paths

### Memory Safety

- Scope isolation regression checks for cross-user and cross-scope leakage
- Injection robustness guard with offline pattern detection fixtures
- CI coverage for memory safety offline checks

### Performance and ROI

- Multi-scale latency baseline runner for memory search paths
- Manual ROI benchmark for seeded repro experiences with A/B reporting
- Significance output when paired sample count reaches the configured threshold

## Validation Snapshot

### Targeted tests and smokes

- `PYTHONPATH=src python -m pytest -q tests/unit/test_retrieval_benchmark.py`
- `PYTHONPATH=src python evals/runners/run_retrieval_benchmark_smoke.py`
- `PYTHONPATH=src python -m pytest -q tests/unit/test_context_engine_benchmark.py`
- `PYTHONPATH=src python evals/runners/run_context_engine_benchmark_smoke.py`
- `PYTHONPATH=src python -m pytest -q tests/unit/test_memory_metric_collector.py`
- `PYTHONPATH=src python evals/memory/test_scope_isolation.py`
- `PYTHONPATH=src python -m pytest -q tests/unit/test_injection_guard.py`
- `PYTHONPATH=src python evals/memory/test_injection_robustness.py`
- `PYTHONPATH=src python -m pytest -q tests/unit/test_memory_performance_benchmark.py`
- `PYTHONPATH=src python -m pytest -q tests/unit/test_roi_benchmark.py`

### Local performance baseline

From `output/reports/memory_performance_baseline_10k_100k.json`:

| Rows | Seed Time | Unscoped p95 | Track Scoped p95 | Batch Track p95 | Batch Paper p95 |
|-----:|----------:|-------------:|-----------------:|----------------:|----------------:|
| 10k | 1.04s | 39.00ms | 25.26ms | 20.46ms | 11.66ms |
| 100k | 13.12s | 23.04ms | 56.81ms | 15.82ms | 11.73ms |

Current takeaway:

- Read latency is healthy for the tested local SQLite baseline.
- The hottest path at larger scale is `track`-scoped search.
- `batch_track` and `batch_paper` remain comparatively stable and are the best current shape for larger fan-out retrieval.

## Operator Notes

- ROI benchmark is manual-only and intentionally not wired into CI.
- Run ROI benchmark with `make bench-roi` after configuring one supported LLM API key.
- Integration branch merges the six issue branches in milestone order from `dev`.

## Follow-ups

- Add a scheduled or manually triggered GitHub Action for large-scale performance runs outside the default CI path.
- Extend ROI benchmark with a runner that consumes full P2C context packs directly when that execution path is mature enough.
- Track 1M-row memory baselines separately because they are too slow for the current local loop used in day-to-day validation.
