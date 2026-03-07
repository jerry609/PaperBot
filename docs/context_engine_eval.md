# Context Engine Evaluation

Offline context-engine evaluation for issue `#286` is fixture-driven and deterministic. It targets
three behaviors already present in `ContextEngine`:

- layered context assembly
- token-guard trimming
- advisory track routing

## Files

- Fixture: `evals/fixtures/context/bench_v1.json`
- Benchmark library: `src/paperbot/context_engine/benchmark.py`
- CLI: `scripts/eval_context_engine.py`
- Smoke runner: `evals/runners/run_context_engine_benchmark_smoke.py`

## Fixture shape

Each case includes:

- `query`, `query_type`, `stage`
- routing setup: `active_track_id`, optional `track_id`
- optional `paper_id`
- optional `context_token_budget`
- `expected.layers`
- `expected.token_guard`
- optional `expected.router_track_id`
- deterministic store state under `state`

The benchmark builds a real `ContextEngine` with fake research/memory stores and runs
`build_context_pack()` directly, so the scorer observes the same layer assembly and token-guard
logic that production code uses.

## Metrics

- `layer_precision`: expected populated layers vs actual populated layers
- `layer_recall`: expected populated layers recovered by the engine
- `token_guard_accuracy`: whether guard triggering matches expectation
- `token_guard_trigger_rate`: observed guard trigger rate across cases
- `router_coverage`: share of router-evaluable cases that produced a suggestion
- `router_accuracy`: share of router-evaluable cases whose suggested track matched expectation

## Run locally

```bash
PYTHONPATH=src python scripts/eval_context_engine.py \
  --fixtures evals/fixtures/context/bench_v1.json \
  --output output/reports/context_bench_v1.json \
  --fail-under-layer-precision 0.95 \
  --fail-under-token-guard-accuracy 1.0 \
  --fail-under-router-coverage 1.0 \
  --fail-under-router-accuracy 1.0
```

## Seed coverage

The seed cases cover the combinations called out in the issue:

- stages: `survey`, `writing`, `rebuttal`
- query types: `short`, `long`, `track_query`, `paper_targeted`
- one explicit token-guard overflow case
- two router-evaluable cases with deterministic switch expectations
