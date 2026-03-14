---
name: literature-review
description: This skill should be used when the user asks to "do a literature review",
  "survey papers on a topic", "search and summarize research on X", "find papers about
  attention mechanisms", "systematic review of the literature", "what papers exist on Y",
  or wants a multi-step workflow to search, filter by relevance, score quality, and
  summarize academic papers using PaperBot MCP tools.
tools:
  - paper_search
  - relevance_assess
  - paper_judge
  - paper_summarize
  - export_to_obsidian
  - save_to_memory
---

# Literature Review Workflow

Conduct a systematic literature review: search, filter by relevance, judge quality,
summarize top papers, and save findings to memory.

## Workflow

### Step 1: Search for papers

Call `paper_search` with the research question or topic.

- Parameters: `query` (required), `max_results` (default 10; use 20–50 for broad surveys),
  `sources` (optional; omit for all sources, or specify `["arxiv", "semantic_scholar"]`)
- Returns: list of paper dicts with `title`, `abstract`, `authors`, `year`, `venue`,
  `arxiv_id`, `doi`

### Step 2: Filter by relevance

For each paper, call `relevance_assess` with `title`, `abstract`, and the same `query`.

- Parameters: `title`, `abstract`, `query`, `keywords` (optional comma-separated terms)
- Returns: dict with `score` (0–100) and `reason`
- Suggested threshold: discard papers with `score` below 40
- If `degraded=True`, token-overlap scoring is used (less accurate but functional)

### Step 3: Judge quality of relevant papers

For papers above the relevance threshold, call `paper_judge`.

- Parameters: `title`, `abstract`, `full_text` (optional), `rubric` (default `"default"`;
  pass the research question for context-aware judging)
- Returns: dimension scores (1–5), `overall_score`, `recommendation`
  (`must_read` / `worth_reading` / `skim` / `skip`)
- Prioritize papers with `must_read` and `worth_reading` recommendations

### Step 4: Summarize top papers

Call `paper_summarize` for papers recommended as `must_read` or `worth_reading`.

- Parameters: `title`, `abstract`
- Returns: dict with `summary` key (concise string)
- If `degraded=True`, generate a manual summary from the abstract text

### Step 5: Export to Obsidian (optional)

Call `export_to_obsidian` for papers to save as permanent Obsidian notes.

- Parameters: `title`, `abstract`, `authors` (list), `year`, `venue`, `arxiv_id`, `doi`
  (provide whichever identifiers are available)
- Returns: dict with `markdown` key — YAML-frontmattered note ready to write to vault

### Step 6: Save synthesis to memory

Call `save_to_memory` with a synthesis of findings across all reviewed papers.

- Parameters: `content` (synthesis text), `kind` (`"note"` for general observations,
  `"hypothesis"` for research directions), `user_id` (default `"default"`),
  `scope_type` (`"global"` unless scoping to a specific research track),
  `scope_id` (required if `scope_type="track"`), `confidence` (0.0–1.0)
- Returns: dict with `created` or `skipped` status

## Degraded Mode

`paper_judge`, `paper_summarize`, and `relevance_assess` require a configured LLM API key.
`paper_search` works without LLM and returns raw search results in all cases.

When any LLM-backed tool returns `degraded=True`:
- The response also contains an `error` key describing the issue
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and restart the MCP server
- In degraded mode, proceed with `paper_search` results only; skip Steps 2–4

## Notes

- For broad surveys (>30 papers), consider running `relevance_assess` in bulk before
  `paper_judge` to reduce LLM calls
- Use `rubric="reproducibility"` in `paper_judge` if the review goal is identifying
  reproducible papers for implementation
- The `export_to_obsidian` step is optional — skip it if the user has not set up an
  Obsidian vault or does not need persistent notes
