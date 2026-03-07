# MemoryBench Runtime Report — 2026-03-07

## Executive Summary

This report captures two live runs executed on `dev` after Epic #283 landed:

1. A real `make bench-roi` run
2. A `10k / 100k / 1M` memory performance baseline focused on the `track`-scoped path

It also summarizes how memory-focused systems typically evaluate memory **effectiveness**, which is different from raw SQL or database load testing.

## Branch Cleanup

The temporary Epic #283 feature branches and the integration branch were deleted locally and remotely after merge.
Only long-lived repository branches were left in place.

## Live ROI Benchmark

### Command

```bash
make bench-roi
```

### Artifacts

- JSON report: `evals/reports/memory_roi_benchmark.json`
- Console log: `output/reports/memory_roi_benchmark_run.log`

### Result Summary

| Arm | Description | Samples | First Pass Success | Avg Repair Loops | Avg Time to Pass | Token Cost |
|-----|-------------|--------:|-------------------:|-----------------:|-----------------:|-----------:|
| A | memory/context bridge disabled | 15 | 0.00 | 3.00 | 0.4875s | 0.0 |
| B | seeded repro experiences enabled | 15 | 0.00 | 3.00 | 0.4883s | 0.0 |

### Delta

- `first_pass_success_rate`: flat
- `repair_loops`: flat
- `time_to_pass_sec`: +0.0008s for B, not significant
- `token_cost_usd`: flat at `0.0`

### Validity Assessment

This run **did execute successfully**, but it does **not yet measure memory effectiveness well**.

Dominant failure mode across all 30 samples:

- `ModuleNotFoundError: No module named 'torch'`

Implications:

- The benchmark is currently saturated by **environment/runtime dependency failure**, not by memory quality.
- Because both arms fail at the same external import boundary, the memory delta is washed out.
- `token_cost_usd` stays `0.0`, which indicates the current execution path is not yet emitting usable token accounting into `LLMUsageStore` for this benchmark.

### What to Fix Before Trusting ROI

To make ROI meaningful, the benchmark should move from “did generated code import in a bare environment” to “did memory improve task success under a realistic execution harness”.

Recommended fixes:

1. Create a per-sample venv and install generated `requirements.txt` before verification.
2. Or split verification into:
   - syntax + local-import correctness
   - optional dependency-aware execution
3. Route generation through a provider path that records usage/cost consistently.
4. Add case-level expectations where memory should help with structure reuse or debugging reuse.

## 1M Memory Performance Baseline

### Command

```bash
PYTHONPATH=src python scripts/benchmark_memory_performance.py \
  --sizes 10000,100000,1000000 \
  --query-count 10 \
  --batch-size 20000 \
  --output output/reports/memory_performance_track_curve_1m.json
```

### Artifacts

- JSON report: `output/reports/memory_performance_track_curve_1m.json`
- Console log: `output/reports/memory_performance_track_curve_1m.log`

### Curve Summary

| Rows | Seed Time | DB Size | Unscoped p95 | Track Scoped p95 | Batch Track p95 |
|-----:|----------:|--------:|-------------:|-----------------:|----------------:|
| 10k | 1.04s | 10.93 MB | 24.96ms | 25.86ms | 16.27ms |
| 100k | 13.12s | 100.96 MB | 22.91ms | 55.03ms | 11.67ms |
| 1M | 158.15s | 1004.87 MB | 24.05ms | 475.49ms | 11.58ms |

### Interpretation

- The general unscoped path stays roughly flat in this synthetic local baseline.
- The **single `track`-scoped query path degrades sharply** at `1M`, rising to `475.49ms p95`.
- The **batch track** path stays stable around `11–16ms p95`, which suggests the main issue is likely in the single-item `track` retrieval path rather than the storage layer as a whole.

### Immediate Optimization Target

If we want the next highest-impact optimization target, it is:

- `track`-scoped single-query search at high cardinality

That path should be profiled first for:

- query plan / index usage
- FTS candidate set size before scope filtering
- sort/rerank cost after candidate selection
- whether scope filtering happens too late in the pipeline

## How Memory Projects Usually Test Effectiveness

Memory-heavy systems generally do **not** stop at SQL-style pressure tests. They usually combine **task outcome**, **retrieval quality**, and **long-horizon reasoning** evaluation.

### Common external patterns

- **LoCoMo** evaluates long-term conversational memory with question answering over long dialogues, including single-hop, multi-hop, temporal, and open-domain questions. It is designed to measure whether an agent can use accumulated memories correctly over time.
  - Source: https://snap-research.github.io/locomo/

- **LongMemEval** focuses on longer-horizon agent memory abilities such as information extraction, question answering, temporal reasoning, multi-session reasoning, and knowledge updates.
  - Source: https://github.com/xiaowu0162/LongMemEval

- **Mem0** frames evaluation around downstream assistant quality and reports gains on benchmarks such as LoCoMo and other assistant-style tasks, emphasizing response quality lift rather than storage speed alone.
  - Source: https://mem0.ai/research

- **Letta** emphasizes memory-aware agent benchmarking with cost/performance trade-offs and task-level evaluation rather than only backend latency.
  - Source: https://www.letta.com/blog/memory-agent-benchmark

## What That Means for PaperBot

A useful memory evaluation stack for PaperBot should have **two distinct layers**:

### 1. Systems performance

This is what SQL-style benchmarking is good at:

- latency
- throughput
- tail latency (`p95` / `p99`)
- storage size growth
- write amplification / seed time

### 2. Memory effectiveness

This is the part that decides whether memory is actually helping the product:

- retrieval relevance (`Recall@K`, `MRR`, `nDCG`)
- context assembly correctness
- write quality: did we store the right thing?
- update quality: can newer memories overwrite stale ones correctly?
- temporal reasoning: can the system answer based on *when* something happened?
- personalization lift: does memory improve first-pass success or reduce retries?
- abstention: does the agent avoid hallucinating when memory is missing?
- safety: isolation, deletion compliance, prompt-injection resistance

## Recommendation

For the next phase, treat the current suite as:

- **good on safety + retrieval + performance foundations**
- **not yet complete on true memory usefulness measurement**

The highest-value next additions would be:

1. A LoCoMo / LongMemEval-style multi-session benchmark for remembered facts and temporal updates
2. A write/update benchmark for stale-memory overwrite and contradiction resolution
3. A fixed ROI harness with dependency-aware verification so memory lift is not hidden by missing runtime packages


## Follow-up Update — March 7, 2026

### Dependency-ready ROI smoke

Artifact:

- `output/reports/memory_roi_benchmark_smoke.json`

What changed since the earlier ROI section in this report:

- ROI now prepares a cached verification runtime from generated `requirements.txt`
- runtime install failures are recorded with `pip` stdout/stderr and a persisted failure-log path
- failed partial runtime caches are rebuilt automatically
- torch-family installs prefer the CPU wheel index during verification runtime prep
- invalid repair output such as `Unknown module` is no longer written into `requirements.txt`

Current smoke result on **March 7, 2026**:

- runtime preparation succeeded and reused a cached env (`req_ef8ee04c13655c4b`)
- the dominant failure moved from missing runtime dependencies to a real code defect:
  - `ImportError: cannot import name 'DataLoader' from 'data'`
- `token_cost_usd` still remained `0.0` in this environment because the configured OpenAI-compatible relay returned repeated `500` overload errors for both `reasoning` and `code` routes

Interpretation:

- this is a meaningful improvement over the earlier invalid ROI state because the benchmark is now blocked by generated-code correctness rather than dependency bootstrap
- a trustable LLM-backed ROI number still requires a healthy provider endpoint so the generation path can actually use `LLMService` and emit usage accounting

### Multi-session effectiveness prototype

Artifacts:

- heuristic: `output/reports/memory_effectiveness_benchmark_heuristic.json`
- LLM path prototype: `output/reports/memory_effectiveness_benchmark_llm.json`

Implementation:

- core benchmark: `src/paperbot/memory/eval/effectiveness_benchmark.py`
- CLI: `evals/memory/bench_effectiveness.py`
- fixture: `evals/memory/fixtures/multi_session_effectiveness.json`

Current heuristic summary on **March 7, 2026**:

- `question_count = 4`
- `retrieval_hit_rate = 0.75`
- `answer_accuracy = 1.0`
- `temporal_accuracy = 1.0`
- `update_accuracy = 1.0`
- `abstention_accuracy = 1.0`

What this prototype covers:

- stale-memory overwrite / update quality
- temporal state tracking across sessions
- scope-aware retrieval
- abstention when memory is missing

That makes it a better fit for “is the memory module effective?” than pure SQL pressure testing alone.
