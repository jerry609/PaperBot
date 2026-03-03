# HuggingFace Daily Papers Source

This note documents the production-facing entry points of the HF Daily source used by PaperBot.

## Connector APIs

- `HFDailyPapersConnector.fetch_daily_papers(limit, page, use_cache=True)`
  - Raw API fetch against `https://huggingface.co/api/daily_papers`
  - Includes in-memory TTL cache to reduce repeat calls
- `HFDailyPapersConnector.get_daily(limit, page_size, max_pages)`
  - Parsed records (`HFDailyPaperRecord`) with graceful degradation on transient failures
- `HFDailyPapersConnector.get_trending(mode, limit, page_size, max_pages)`
  - Supports `mode=hot|rising|new`
  - `hot`: primarily upvotes
  - `rising`: upvotes weighted by recency
  - `new`: latest submitted first

## Adapter APIs

- `HFDailyAdapter.search(query, ...)`
  - Query-based search and ranking
- `HFDailyAdapter.get_daily(limit=100)`
- `HFDailyAdapter.get_trending(mode="hot", limit=30)`

`HFDailyAdapter` emits both `hf_daily` identity and inferred `arxiv` identity (when available),
so multi-source dedupe can merge HF/S2/OpenAlex copies by arXiv id.

## Cache and runtime metrics

`HFDailyPapersConnector.metrics` returns:

- `requests`: outbound API calls
- `cache_hits`: in-memory cache hits
- `errors`: request/parse errors
- `degraded`: fallbacks triggered after transient API failure
