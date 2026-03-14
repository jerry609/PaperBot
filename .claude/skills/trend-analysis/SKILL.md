---
name: trend-analysis
description: This skill should be used when the user asks to "analyze trends in a
  research area", "what is trending in X", "research landscape for topic Y", "topic
  trend analysis", "emerging themes in machine learning", "what are researchers working
  on in Z", or wants to survey a field and identify emerging patterns across multiple
  papers using PaperBot MCP tools.
tools:
  - paper_search
  - analyze_trends
  - get_research_context
  - save_to_memory
---

# Trend Analysis Workflow

Identify research trends across a topic by collecting papers, analyzing patterns, and
saving a synthesis of emerging themes.

## Workflow

### Step 1: Load research context (optional)

If a research track exists for the topic, call `get_research_context` to retrieve
existing memories and previously found papers.

- Parameters: `query` (the research topic), `user_id` (default `"default"`),
  `track_id` (optional; pass if a specific track ID is known)
- Returns: dict with `papers` (list), `memories` (list), `stage` (workflow stage string)
- Use the existing memories as context when synthesizing results in Step 4
- Skip this step if no prior research context exists for the topic

### Step 2: Search for papers

Call `paper_search` with the topic. Use a broader corpus for trend analysis.

- Parameters: `query` (required), `max_results` (use 20–50 for trend analysis — a
  larger corpus improves trend signal quality), `sources` (optional)
- Returns: list of paper dicts with `title`, `abstract`, `authors`, `year`, `venue`
- If `track_id` context was loaded in Step 1, merge the existing papers with new results
  (deduplicate by `arxiv_id` or `doi`)

### Step 3: Analyze trends

Call `analyze_trends` with the topic and the list of papers from Step 2.

- Parameters: `topic` (the research area string), `papers` (list of paper dicts from
  `paper_search`; pass the full list for best results)
- Returns: dict with `trend_analysis` (natural language narrative), `topic`, `paper_count`
- Check for `degraded=True` — `analyze_trends` requires a configured LLM API key

### Step 4: Save synthesis

Call `save_to_memory` with the trend analysis narrative and any additional observations.

- Parameters: `content` (the `trend_analysis` text from Step 3, optionally enhanced with
  your own observations), `kind` (`"note"` for factual observations, `"hypothesis"` for
  directional predictions), `user_id` (default `"default"`),
  `scope_type` (`"global"` for broad field trends, `"track"` if scoping to a research area),
  `scope_id` (track ID if `scope_type="track"`), `confidence` (0.0–1.0)
- Returns: dict with `created` or `skipped` status

## Degraded Mode

`analyze_trends` requires a configured LLM API key. `paper_search` and `get_research_context`
work without LLM.

When `analyze_trends` returns `degraded=True`:
- The response also contains an `error` key describing the issue
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and restart the MCP server
- In degraded mode, present the raw search results grouped by year or venue as a
  manual trend signal; skip Step 3 or surface the paper list to the user directly

## Notes

- For fast trend snapshots, use `max_results=20` and skip Step 1
- For deep research landscape maps, use `max_results=50` and integrate prior context
  from `get_research_context`
- When analyzing sub-field trends (e.g., "sparse attention mechanisms"), narrow the
  query rather than broadening `max_results`
- Multiple calls with different `topic` variants (e.g., "mixture of experts" vs.
  "sparse expert models") can be combined for a richer landscape view
