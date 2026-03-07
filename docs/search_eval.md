# Search Evaluation

Offline retrieval evaluation for PaperBot now lives in the same `evals/` workflow as the
existing smoke suites.

## What ships in `#284`

- Seed fixture: `evals/fixtures/retrieval/bench_v2.jsonl`
- Benchmark library: `src/paperbot/application/services/retrieval_benchmark.py`
- CLI entry: `scripts/eval_search.py`
- CI smoke gate: `evals/runners/run_retrieval_benchmark_smoke.py`

The fixture is intentionally offline and deterministic: every case includes the judged relevant
documents plus adapter-level stub results so the benchmark exercises `PaperSearchService`
fusion, deduplication, source weighting, and latency accounting without network access.

## Fixture schema

Each JSONL row represents one query case.

```json
{
  "query_id": "q_short_rag",
  "query": "rag",
  "query_type": "short_acronym",
  "source": "semantic_scholar",
  "sources": ["semantic_scholar"],
  "max_results": 10,
  "judgments": [
    {"doc_id": "id:doi:10.1000/rag-primer", "relevance": 3}
  ],
  "results_by_source": {
    "semantic_scholar": [
      {
        "title": "RAG Primer for Practitioners",
        "abstract": "...",
        "year": 2025,
        "citation_count": 120,
        "identities": [{"source": "doi", "external_id": "10.1000/rag-primer"}]
      }
    ]
  }
}
```

### Field notes

- `query_type`: powers grouped reporting such as `short_acronym`, `track_query`, `long_tail`
- `source`: human-readable grouping label for reports
- `sources`: actual adapter names passed into `PaperSearchService.search()`
- `judgments.relevance`: graded relevance, expected range `0..3`
- `results_by_source`: deterministic adapter outputs used by the offline benchmark runner

## Metrics

- `ndcg_at_10`: graded ranking quality for the top 10 results
- `mrr_at_10`: first relevant hit position in the top 10
- `recall_at_50`: fraction of judged relevant docs recovered in the top 50
- `avg_latency_ms` / `p95_latency_ms`: per-case wall time for the offline search pipeline

Reports are grouped in three views:

- overall
- by `query_type`
- by `source`

## Run locally

```bash
PYTHONPATH=src python scripts/eval_search.py \
  --fixtures evals/fixtures/retrieval/bench_v2.jsonl \
  --output output/reports/retrieval_bench_v2.json \
  --fail-under-ndcg 0.95 \
  --fail-under-mrr 0.95 \
  --fail-under-recall 1.0
```

## CI gate

CI runs `evals/runners/run_retrieval_benchmark_smoke.py`, which loads the same fixture and
enforces the current regression guard thresholds:

- `ndcg_at_10 >= 0.95`
- `mrr_at_10 >= 0.95`
- `recall_at_50 >= 1.0`

To scale this seed set into a larger labeled collection, extend the same JSONL schema with more
cases under `evals/fixtures/retrieval/` or move the annotations into a dedicated data directory
once the set grows beyond smoke/regression usage.
