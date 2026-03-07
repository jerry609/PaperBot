# P2C ROI Benchmark

## Purpose

This benchmark estimates the ROI of the current memory bridge on Paper2Code runs by comparing:

- **Group A**: seeded repro-memory disabled
- **Group B**: preload 10 `verified_structure` / `success_pattern` records into `ReproExperienceStore`

Each arm runs the same paper set repeatedly and reports A/B deltas for:

- `first_pass_success_rate`
- `repair_loops`
- `time_to_pass_sec`
- `token_cost_usd`

If there are at least 15 paired samples, the report also includes an approximate paired significance test.

## Fixtures

- Cases: `evals/memory/fixtures/roi_cases.json`
- Seeded experiences: `evals/memory/fixtures/repro_experiences.json`

The default setup uses 5 papers × 3 runs = 15 paired samples, which is enough to emit significance output.

## What changed in this follow-up

As of **March 7, 2026**, the manual ROI runner now does three things it did not do before:

1. prepares a cached per-requirements runtime before verification
2. surfaces dependency-install failures directly in the report metadata
3. can route generation through the project `LLMService` path so usage/cost can be captured when the provider is healthy

Key implementation points:

- cached runtime prep: `src/paperbot/repro/verification_runtime.py`
- ROI runner: `src/paperbot/memory/eval/roi_benchmark.py`
- manual CLI: `evals/memory/bench_roi.py`

## Run

Manual only. Do **not** add this benchmark to CI.

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
LLM_REASONING_PROVIDER=openai \
LLM_REASONING_MODEL=gpt-4o-mini \
LLM_REASONING_API_KEY_ENV=OPENAI_API_KEY \
LLM_CODE_MODEL=gpt-4o-mini \
LLM_CODE_API_KEY_ENV=OPENAI_API_KEY \
make bench-roi
```

Direct script invocation:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
LLM_REASONING_PROVIDER=openai \
LLM_REASONING_MODEL=gpt-4o-mini \
LLM_REASONING_API_KEY_ENV=OPENAI_API_KEY \
LLM_CODE_MODEL=gpt-4o-mini \
LLM_CODE_API_KEY_ENV=OPENAI_API_KEY \
PYTHONPATH=src python evals/memory/bench_roi.py \
  --cases evals/memory/fixtures/roi_cases.json \
  --experiences evals/memory/fixtures/repro_experiences.json \
  --runs-per-case 3 \
  --output evals/reports/memory_roi_benchmark.json
```

For a smaller spot-check:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
LLM_REASONING_PROVIDER=openai \
LLM_REASONING_MODEL=gpt-4o-mini \
LLM_REASONING_API_KEY_ENV=OPENAI_API_KEY \
LLM_CODE_MODEL=gpt-4o-mini \
LLM_CODE_API_KEY_ENV=OPENAI_API_KEY \
PYTHONPATH=src python evals/memory/bench_roi.py \
  --limit-cases 1 \
  --runs-per-case 1 \
  --max-repair-attempts 1 \
  --output output/reports/memory_roi_benchmark_smoke.json
```

## Useful flags

- `--no-project-llm` — disable the project `LLMService` path and fall back to legacy node heuristics / SDKs
- `--no-prepare-requirements` — skip cached dependency preparation
- `--runtime-cache-dir` — choose where prepared runtimes are stored
- `--verification-install-timeout` — change dependency install timeout
- `--no-prefer-cpu-torch` — disable the CPU-only torch wheel index optimization during verification runtime prep

## API key requirement

The manual runner requires at least one configured provider key:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `NVIDIA_MINIMAX_API_KEY`
- `NVIDIA_GLM_API_KEY`

Without one of these, `evals/memory/bench_roi.py` exits before starting.

## Current smoke status on March 7, 2026

Artifact:

- `output/reports/memory_roi_benchmark_smoke.json`

What is fixed:

- dependency installation is now attempted in a prepared runtime instead of failing immediately on missing `torch`
- runtime install failures now surface exact `pip` stdout/stderr and failure-log paths in the JSON report
- failed half-built runtime caches are rebuilt automatically on the next run
- torch-family installs prefer the CPU wheel index during verification runtime prep
- invalid dependency repair like `Unknown module` is blocked instead of poisoning `requirements.txt`

What is still blocking a trustworthy ROI number in this environment:

- the local OpenAI-compatible relay returned repeated `500` overload errors for both `reasoning` and `code` routes, so `token_cost_usd` remained `0.0`
- once dependency prep succeeded, the dominant failure moved to real generated-code quality:
  - `ImportError: cannot import name 'DataLoader' from 'data'`

That is still progress: the benchmark is now failing on an actual code-generation issue instead of an environment bootstrap issue.

## Notes

- Each sample uses a fresh SQLite database so the two arms do not share seeded memory.
- The verification runtime cache is keyed by generated `requirements.txt` content hash.
- The benchmark is still manual-only because it depends on live provider access, runtime package install, and local machine conditions.
- If your environment injects SOCKS proxies, clear them for the benchmark command unless your provider client explicitly supports them.
