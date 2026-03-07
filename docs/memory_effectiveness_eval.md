# Memory Effectiveness Evaluation

## Why this is different from SQL pressure testing

Memory systems need two different validation layers:

1. **Systems performance** — latency, throughput, storage growth, and tail latency
2. **Task effectiveness** — whether stored memory actually improves answers, context assembly, first-pass success, and abstention quality

For PaperBot, the second layer matters more when we ask questions like:

- Did the memory store retrieve the right fact?
- Did newer memories overwrite stale ones correctly?
- Did scoped memories stay isolated?
- Did the agent abstain when memory was insufficient?
- Did memory reduce retries or repair loops in Paper2Code?

That is why memory evaluation should not stop at SQL-style load testing.

## External benchmark patterns

PaperBot's effectiveness prototype follows the same broad direction used by long-horizon memory benchmarks such as:

- `LoCoMo` — long-dialog memory QA with temporal and multi-hop questions
- `LongMemEval` — multi-session memory evaluation across extraction, QA, updates, and temporal reasoning
- `Mem0` research reporting — emphasizes downstream assistant quality, not only storage speed
- `Letta` memory benchmark writeup — emphasizes task success, recall quality, cost, and practical agent behavior

References:

- https://snap-research.github.io/locomo/
- https://github.com/xiaowu0162/LongMemEval
- https://mem0.ai/research
- https://www.letta.com/blog/memory-agent-benchmark

## PaperBot effectiveness prototype

PaperBot now includes a lightweight multi-session benchmark prototype:

- Runner: `evals/memory/bench_effectiveness.py`
- Core logic: `src/paperbot/memory/eval/effectiveness_benchmark.py`
- Fixture: `evals/memory/fixtures/multi_session_effectiveness.json`

The expanded prototype now includes **5 cases / 32 questions** and covers seven practical memory behaviors:

- **Update accuracy** — the latest memory should win over stale memory
- **Temporal accuracy** — the current state should reflect later sessions
- **Temporal-previous accuracy** — earliest / original facts should still be recoverable
- **Scoped retrieval** — the correct paper/track scope should be used
- **Multi-session synthesis** — recap-style answers should combine facts across sessions
- **Abstention accuracy** — the system should return `INSUFFICIENT_MEMORY` when evidence is missing
- **Per-case / per-category reporting** — the report now breaks down accuracy by question type and case

## Run it

Heuristic runner:

```bash
PYTHONPATH=src python evals/memory/bench_effectiveness.py \
  --top-k 4 \
  --output output/reports/memory_effectiveness_benchmark_heuristic.json
```

LLM-backed runner:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
LLM_REASONING_PROVIDER=openai \
LLM_REASONING_MODEL=gpt-4o-mini \
LLM_REASONING_API_KEY_ENV=OPENAI_API_KEY \
PYTHONPATH=src python evals/memory/bench_effectiveness.py \
  --use-llm \
  --top-k 4 \
  --output output/reports/memory_effectiveness_benchmark_llm.json
```

NVIDIA Integrate endpoint example for the answering model only:

```bash
bash -ic 'cd /home/master1/PaperBot && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  PAPERBOT_EMBEDDING_PROVIDER_CHAIN=none \
  LLM_REASONING_PROVIDER=openai \
  LLM_REASONING_MODEL=minimaxai/minimax-m2.5 \
  LLM_REASONING_API_KEY_ENV=NVIDIA_API_KEY \
  LLM_REASONING_BASE_URL=https://integrate.api.nvidia.com/v1 \
  PYTHONPATH=src python evals/memory/bench_effectiveness.py \
    --use-llm \
    --top-k 4 \
    --output output/reports/memory_effectiveness_benchmark_nvidia.json'
```

## Current results on March 7, 2026

Expanded heuristic benchmark output from `output/reports/memory_effectiveness_benchmark_heuristic.json`:

- `question_count = 32`
- `retrieval_hit_rate = 1.0`
- `answer_accuracy = 1.0`
- `temporal_accuracy = 1.0`
- `update_accuracy = 1.0`
- `abstention_accuracy = 1.0`
- `scope_accuracy = 1.0`
- `multi_session_accuracy = 1.0`

This means the benchmark fixture itself is now broad enough to cover:

- longitudinal updates
- original-vs-current temporal recall
- paper/track scope isolation
- recap-style multi-session synthesis
- abstention under missing evidence

LLM-backed execution is also wired up, including NVIDIA's OpenAI-compatible endpoint for reasoning. In this environment, two caveats still apply:

- the answering model can hit `429 Too Many Requests` under a full 32-question run
- NVIDIA's chat endpoint is fine for answer generation, but it is **not** an embeddings endpoint, so `PAPERBOT_EMBEDDING_PROVIDER_CHAIN=none` is required for this benchmark path

A previously completed LLM-backed run on the NVIDIA route produced `retrieval_hit_rate = 1.0` but only `answer_accuracy = 0.625`, which suggests the remaining bottleneck is now the answering model / prompting path rather than retrieval itself.

## How to use this with the rest of MemoryBench

Use the effectiveness benchmark together with the other suites:

- retrieval relevance for `Recall@K`, `MRR`, `nDCG`
- scope isolation for zero-leak guarantees
- injection robustness for offline detection
- performance baselines for `10k / 100k / 1M`
- ROI benchmarking for end-to-end Paper2Code lift

A healthy memory system needs all of these together:

- **fast enough**
- **retrieves the right thing**
- **does not leak across scopes**
- **updates stale memory correctly**
- **improves downstream agent behavior**
