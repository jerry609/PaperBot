# P2C ROI Benchmark

## Purpose

This benchmark estimates the ROI of the current memory bridge on Paper2Code runs by comparing:

- **Group A**: seeded repro-memory disabled
- **Group B**: preload 10 `verified_structure` / `success_pattern` records into `ReproExperienceStore`

Each arm runs the same paper set three times and reports A/B deltas for:

- `first_pass_success_rate`
- `repair_loops`
- `time_to_pass_sec`
- `token_cost_usd`

If there are at least 15 paired samples, the report also includes an approximate paired significance test.

## Fixtures

- Cases: `evals/memory/fixtures/roi_cases.json`
- Seeded experiences: `evals/memory/fixtures/repro_experiences.json`

The default setup uses 5 papers × 3 runs = 15 paired samples, which is enough to emit significance output.

## Run

Manual only. Do **not** add this benchmark to CI.

```bash
make bench-roi
```

Direct script invocation:

```bash
PYTHONPATH=src python evals/memory/bench_roi.py \
  --cases evals/memory/fixtures/roi_cases.json \
  --experiences evals/memory/fixtures/repro_experiences.json \
  --runs-per-case 3 \
  --output evals/reports/memory_roi_benchmark.json
```

For a smaller local spot-check:

```bash
PYTHONPATH=src python evals/memory/bench_roi.py --limit-cases 2
```

## API Key Requirement

The manual runner requires at least one configured provider key:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `NVIDIA_MINIMAX_API_KEY`
- `NVIDIA_GLM_API_KEY`

Without one of these, `evals/memory/bench_roi.py` exits before starting.

## Notes

- Each sample uses a fresh SQLite database so the benchmark only measures the fixed seeded-memory lift, not cumulative self-learning across earlier runs.
- `token_cost_usd` comes from the existing `LLMUsageStore` records captured during the run.
- `time_to_pass_sec` is the end-to-end runtime proxy for the sample and is tracked even when a run fails, so the A/B comparison still reflects wasted time.
- The current live runner exercises the memory path already wired into end-to-end reproduction. If a future runner consumes full P2C context packs directly, it can plug into the same benchmark protocol without changing the report format.
