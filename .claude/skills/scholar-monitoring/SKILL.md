---
name: scholar-monitoring
description: This skill should be used when the user asks to "monitor a scholar",
  "check researcher activity", "track publications from author X", "follow author Y",
  "scholar update for Z", "what has researcher X published recently", or wants to
  retrieve and synthesize a researcher's recent publication activity using PaperBot
  MCP tools.
tools:
  - check_scholar
  - analyze_trends
  - save_to_memory
---

# Scholar Monitoring Workflow

Monitor a researcher's recent publication activity: fetch their profile and papers,
optionally analyze output trends, and save a monitoring note.

## Workflow

### Step 1: Check scholar activity

Call `check_scholar` with the researcher's name.

- Parameters: `scholar_name` (required; use the researcher's full name as commonly
  published), `max_papers` (default 10; increase to 20–30 for career-wide coverage)
- Returns: dict with:
  - `scholar`: profile dict with `name`, `hIndex`, `citationCount`, `affiliations`,
    `paperCount`, `url`
  - `recent_papers`: list of paper dicts (title, abstract, year, venue, citation count)
  - `candidates`: list of top-3 candidate matches (inspect if the top result is wrong)
- If `degraded=True`, the scholar was not found on Semantic Scholar or the API is
  unavailable

### Step 2: Analyze paper trends (optional)

If `recent_papers` is non-empty and the user wants thematic analysis, call `analyze_trends`.

- Parameters: `topic` (use the scholar's name or primary research area as the topic),
  `papers` (the `recent_papers` list from Step 1)
- Returns: dict with `trend_analysis` (natural language narrative of the scholar's
  research focus and evolution)
- Skip this step if the user only needs raw paper metadata (no LLM API key required
  for Step 1 alone)

### Step 3: Save monitoring note

Call `save_to_memory` with a summary of the scholar's recent activity.

- Parameters: `content` (monitoring summary — include scholar name, hIndex,
  recent paper titles, and trend analysis if available), `kind="note"`,
  `user_id` (default `"default"`), `scope_type="global"`,
  `confidence` (0.0–1.0; suggest 0.9 for factual publication data)
- Returns: dict with `created` or `skipped` status

## Note on Scholar Lookup

`check_scholar` searches Semantic Scholar by name. Common issues:

- **Name diacritics:** Names with accents (e.g., "Müller", "Bengio") may need the
  ASCII variant ("Muller", "Yoshua Bengio") if exact-match fails
- **New researchers:** Very new researchers may have limited or no Semantic Scholar
  records — check `paperCount` in the returned profile
- **Name ambiguity:** The `candidates` field in the response lists the top 3 matches;
  inspect these if the top result appears to be the wrong person (wrong affiliation,
  wrong research area)
- **Name format:** Use "First Last" format; middle names are generally not needed but
  can help disambiguate common names

## Degraded Mode

`analyze_trends` (Step 2) requires a configured LLM API key. `check_scholar` (Step 1)
and `save_to_memory` (Step 3) do not require LLM.

When `analyze_trends` returns `degraded=True`:
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and restart the MCP server
- Skip Step 2 and proceed directly to Step 3 with a summary based on raw paper metadata

When `check_scholar` itself returns `degraded=True`:
- This indicates the scholar was not found or the Semantic Scholar API is unavailable
- Try an alternate name spelling or abbreviation
- Check `candidates` in the response for close matches
