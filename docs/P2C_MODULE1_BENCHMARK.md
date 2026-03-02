# P2C Module 1 Benchmark

## Purpose

This benchmark tracks extraction quality changes for P2C Module 1 so refactors can be compared with reproducible numbers.

## Fixture

- Source: `evals/fixtures/p2c/module1_gold.json`
- Scope:
  - architecture hit
  - metrics extraction F1
  - hyperparameter extraction F1
  - evidence hit rate for required observation types
  - warnings count

## Run

```bash
PYTHONPATH=src python scripts/p2c_module1_benchmark.py \
  --fixtures evals/fixtures/p2c/module1_gold.json \
  --output evals/reports/p2c_module1_baseline.json
```

Optional CI gate:

```bash
PYTHONPATH=src python scripts/p2c_module1_benchmark.py \
  --fail-under-metric-f1 0.90
```

## Current Baseline (2026-02-26)

- metric_f1: `1.0000`
- hyperparam_f1: `1.0000`
- architecture_hit_rate: `1.0000`
- evidence_hit_rate: `0.8889`
- avg_warnings: `1.3333`

Raw report:
- `evals/reports/p2c_module1_baseline.json`

## Known Limits

- Dataset is intentionally small and synthetic; it is best for regression checks, not real-world coverage.
- Scores can be inflated by keyword-heavy fixtures and should be paired with manual spot checks.
- Evidence hit rate is strict and can remain below 1.0 when environment evidence is missing.
