---
name: paper-reproduction
description: This skill should be used when the user asks to "reproduce a paper",
  "implement paper code", "paper2code", "replicate research results", "run experiment
  from paper", "implement the algorithm from this paper", or wants to locate, understand,
  and plan implementation of a specific academic paper using PaperBot MCP tools.
tools:
  - paper_search
  - paper_judge
  - paper_summarize
  - export_to_obsidian
  - save_to_memory
---

# Paper Reproduction Workflow

Reproduce or implement a paper: locate it, assess reproducibility, understand its
contributions, save an implementation plan, and export a paper note.

## Workflow

### Step 1: Find the paper

Call `paper_search` with the paper title, topic, or known identifier.

- Parameters: `query` (required; include ArXiv ID or DOI if known for direct lookup),
  `max_results` (default 10; use 3–5 for a known paper to minimize noise)
- Returns: list of paper dicts with `title`, `abstract`, `authors`, `year`, `venue`,
  `arxiv_id`, `doi`
- Select the most specific match if multiple results are returned

### Step 2: Judge reproducibility

Call `paper_judge` with `rubric="reproducibility"` to assess implementation feasibility.

- Parameters: `title`, `abstract`, `full_text` (optional; include if available for
  richer analysis), `rubric="reproducibility"`
- Returns: dimension scores (1–5) including `rigor`, `clarity`, `novelty`, `reproducibility`,
  `overall_score`, and `recommendation`
- Favorable signals: high `rigor` and `clarity` scores
- Unfavorable signals: low `clarity` score may indicate reproduction difficulty; low
  `reproducibility` score indicates missing implementation details (pseudocode, datasets)

### Step 3: Summarize paper contributions

Call `paper_summarize` to extract key contributions, methods, and findings.

- Parameters: `title`, `abstract`
- Returns: dict with `summary` key (concise string covering contributions and approach)
- Use the summary to inform the implementation plan in Step 4

### Step 4: Save reproduction plan to memory

Call `save_to_memory` with an outline of the planned implementation steps.

- Parameters: `content` (implementation plan text), `kind` (`"project"` for structured
  plans or `"decision"` for approach decisions), `user_id` (default `"default"`),
  `scope_type` (`"global"` or `"track"` if this paper belongs to a research track),
  `scope_id` (track ID if `scope_type="track"`), `confidence` (0.0–1.0)
- Include: key algorithms to implement, datasets needed, evaluation metrics, dependencies

### Step 5: Export paper note

Call `export_to_obsidian` to create a structured Obsidian note for the paper.

- Parameters: `title`, `abstract`, `authors` (list), `year`, `venue`, `arxiv_id`, `doi`
  (provide all available identifiers)
- Returns: dict with `markdown` key — YAML-frontmattered note ready for Obsidian vault
- The note provides a permanent reference alongside the implementation

## Implementation Guidance

After completing the above workflow, proceed with code implementation using available
tools (Bash, Write, etc.). The Paper2Code pipeline in PaperBot
(`src/paperbot/repro/`) provides deeper multi-stage analysis (Planning → Blueprint →
Environment → Generation → Verification) for complex reproductions requiring the full
PaperBot backend.

For simpler reproductions:
1. Use the summary from Step 3 and the plan from Step 4 as starting context
2. Implement iteratively, checking against paper details in the Obsidian note
3. Store implementation decisions in memory with `kind="decision"` as the work progresses

## Degraded Mode

`paper_judge` and `paper_summarize` require a configured LLM API key.
`paper_search` works without LLM.

When LLM-backed tools return `degraded=True`:
- The response also contains an `error` key describing the issue
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and restart the MCP server
- In degraded mode, use `paper_search` to locate the paper and proceed to implementation
  using the raw abstract and metadata; skip Steps 2 and 3
