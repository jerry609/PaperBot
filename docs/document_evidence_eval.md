# Document Evidence Evaluation

Offline evaluation for document intelligence should answer one concrete question:

Can evidence retrieval beat a strong FTS-only baseline enough to justify keeping
embedding-based retrieval in the stack?

This benchmark is the guardrail for the document evidence pipeline introduced in
`feat(document): add explicit evidence indexing pipeline`.

## Scope

The benchmark targets evidence retrieval over indexed paper content, not global
paper search. The evaluated path is:

`explicit ingest -> document indexing -> evidence retrieval -> /research/context evidence_hits`

Current v1 indexing source:

- paper overview metadata
- abstract
- structured card fields

Future PDF or markdown parsing can plug into the same benchmark as long as the
fixture and scorer contract stay stable.

## Files

- Seed fixture: `evals/fixtures/document_evidence/bench_v1.json`
- Benchmark library: `src/paperbot/application/services/document_evidence_benchmark.py`
- CLI: `scripts/eval_document_evidence.py`
- Smoke runner: `evals/runners/run_document_evidence_benchmark_smoke.py`
- Benchmark report target: `output/reports/document_evidence_bench_v1.json`
- Context sampling target: `/research/context`

The seed benchmark is intentionally deterministic and offline-capable so it can
run without an embedding API in CI and local regression checks.

## Retrieval modes

Every benchmark run must compare three modes on the same fixture:

- `fts_only`
- `embedding_only`
- `hybrid`

The purpose is not to prove embeddings are useful in theory. The purpose is to
prove whether they add measurable value on PaperBot's evidence retrieval cases.

## Fixture shape

The benchmark fixture is document-centric and deterministic. It contains:

- a small seeded corpus of canonical papers
- the indexed sections or section-like inputs used to produce chunks
- labeled queries
- expected paper hits and chunk hits
- graded judgments for ranking metrics

Seed schema:

```json
{
  "version": "v1",
  "description": "Seed benchmark for document evidence retrieval",
  "papers": [
    {
      "paper_id": 101,
      "title": "Sparse Retrieval for Transformer Agents",
      "abstract": "...",
      "structured_card": {
        "method": "...",
        "findings": ["..."],
        "limitations": "..."
      }
    }
  ],
  "cases": [
    {
      "case_id": "doc_evi_001",
      "query": "retrieval-aware memory routing latency",
      "query_type": "paraphrase",
      "top_k": 5,
      "judgments": [
        {
          "paper_id": 101,
          "chunk_ref": "101:method:0",
          "relevance": 3
        },
        {
          "paper_id": 101,
          "chunk_ref": "101:findings:0",
          "relevance": 2
        }
      ],
      "expected_paper_ids": [101],
      "expected_chunk_refs": ["101:method:0"]
    }
  ]
}
```

### Field notes

- `query_type` should cover at least:
  - `exact`
  - `paraphrase`
  - `term_mismatch`
  - `paper_targeted`
  - `cross_field`
- `chunk_ref` must be stable across runs.
  - Use `paper_id:section:section_chunk_index`.
  - Do not use transient database row ids in fixtures.
- `judgments.relevance` is graded, expected range `0..3`.

## Metrics

Each run must report:

- `recall_at_k`
- `mrr_at_k`
- `ndcg_at_k`
- `evidence_hit_rate`
- `avg_latency_ms`
- `p95_latency_ms`

### Metric intent

- `recall_at_k`: whether the retriever actually brings the right evidence back
- `mrr_at_k`: whether the first truly useful hit appears early enough
- `ndcg_at_k`: whether graded relevance is ranked well, not just binary hits
- `evidence_hit_rate`: whether each case has at least one acceptable evidence hit
- `latency`: whether the improvement is cheap enough for the `/research/context` fast path

## Manual review pass

Offline metrics are necessary but not sufficient. Every major retrieval change
should also run a small manual sampling pass on `/research/context`.

Sample process:

1. Select 20 representative queries from active research workflows.
2. Capture returned `evidence_hits` for each retrieval mode.
3. Label each sample on:
   - `grounded`: directly supports the query
   - `broad`: related but too generic
   - `miss`: does not support the query
4. Compare whether embedding-based modes improve grounding or just broaden the
   surface area.

The specific manual question is:

Does `evidence_hits` become more relevant, or only more semantically vague?

If the answer is "more vague", that mode should not be promoted.

## Decision policy

Embeddings are optional infrastructure, not a product requirement.

Decision rule:

- If `hybrid` clearly improves ranking and hit rate over `fts_only` while
  staying within acceptable latency, keep it.
- If `embedding_only` underperforms `fts_only`, that is acceptable as long as
  `hybrid` wins.
- If embedding-backed modes do not improve the benchmark, fall back to
  `fts_only`.

The benchmark exists to justify the complexity, not to excuse it.

## Do we need an embedding API?

Not for the benchmark contract itself.

Two benchmark tiers should exist:

- `offline deterministic`
  - uses `HashEmbeddingProvider`
  - requires no external API
  - suitable for CI and local regression checks
- `live shadow benchmark`
  - uses a real embedding provider such as OpenAI
  - optional, manual, and used only to validate whether production embeddings
    outperform the deterministic hash baseline

### Dedicated embedding configuration

Live shadow runs should not assume the chat endpoint also supports embeddings.

PaperBot now resolves embedding configuration in this order:

- `PAPERBOT_EMBEDDING_API_KEY`
- `PAPERBOT_EMBEDDING_BASE_URL`
- `PAPERBOT_EMBEDDING_MODEL`

If those are unset, it falls back to:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_EMBEDDING_MODEL`

This separation matters because many OpenAI-compatible relays expose chat
completions but do not expose `/embeddings`.

This means the answer is:

- benchmark coverage does not require an embedding API
- production-quality semantic retrieval may still benefit from one

## Acceptance criteria

Before enabling embedding retrieval by default for document evidence, require:

- a fixed judged dataset with `query -> expected paper/chunk ids`
- side-by-side `fts_only`, `embedding_only`, and `hybrid` results
- reported `Recall@k`, `MRR`, `nDCG`, `evidence_hit_rate`, and `latency`
- one manual `/research/context` sampling pass confirming hits are more grounded
- a documented rollback decision if the numbers do not improve

## Run locally

```bash
PYTHONPATH=src python scripts/eval_document_evidence.py \
  --fixtures evals/fixtures/document_evidence/bench_v1.json \
  --output output/reports/document_evidence_bench_v1.json \
  --embedding-provider hash \
  --fail-under-hybrid-recall 0.5 \
  --fail-under-hybrid-hit-rate 0.5
```

Optional live shadow run:

```bash
PAPERBOT_EMBEDDING_API_KEY=... \
PAPERBOT_EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1 \
PAPERBOT_EMBEDDING_MODEL=text-embedding-3-small \
PYTHONPATH=src python scripts/eval_document_evidence.py \
  --fixtures evals/fixtures/document_evidence/bench_v1.json \
  --output output/reports/document_evidence_bench_live.json \
  --embedding-provider openai
```

## Next implementation steps

1. Expand the judged fixture beyond metadata-only examples.
2. Add a manual sampling report template for `/research/context evidence_hits`.
3. Introduce a live shadow mode with a real embedding provider for optional
   provider-vs-hash comparisons.
4. Add PDF-derived evidence cases once fulltext indexing lands.
