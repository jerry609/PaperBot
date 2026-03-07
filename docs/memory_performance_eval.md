# Memory Performance Evaluation

Offline baseline harness for issue `#288`.

## What it measures

For each configured dataset size, the harness seeds a synthetic SQLite memory store and reports:

- seed time
- database size on disk
- `search_memories()` unscoped latency
- `search_memories()` track-scoped latency
- `search_memories_batch()` track latency
- `search_memories_batch()` paper latency

Latency summaries include `avg_ms`, `p50_ms`, `p95_ms`, and `p99_ms`.

## Default sizes

The CLI defaults to the three baseline scales requested by the issue:

- `10,000`
- `100,000`
- `1,000,000`

## Run locally

```bash
PYTHONPATH=src python scripts/benchmark_memory_performance.py \
  --sizes 10000,100000,1000000 \
  --query-count 25 \
  --output output/reports/memory_performance_baseline.json
```

For a faster smoke run during development:

```bash
PYTHONPATH=src python scripts/benchmark_memory_performance.py \
  --sizes 1000,5000 \
  --query-count 5
```

## Notes

- The benchmark uses deterministic synthetic data and disables embedding generation so the baseline is stable and offline.
- It is intended as a reporting baseline, not a strict CI latency gate, because runtime varies across machines.
