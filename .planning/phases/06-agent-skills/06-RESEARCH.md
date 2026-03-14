# Phase 6: Agent Skills - Research

**Researched:** 2026-03-14
**Domain:** Claude Code SKILL.md format, agent skill authoring conventions, `.claude/skills/` directory structure
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCP-13 | Agent can discover and load PaperBot workflow skills via `.claude/skills/` SKILL.md files (literature-review, paper-reproduction, trend-analysis, scholar-monitoring) | SKILL.md format fully documented from canonical `skill-development` skill in Claude Code plugins; tool names verified from implemented MCP server |
</phase_requirements>

---

## Summary

Phase 6 is purely a content-creation phase. No Python code is written. The deliverable is four SKILL.md files placed in `.claude/skills/{skill-name}/SKILL.md`. Each file is a self-contained markdown document with YAML frontmatter (name, description, and optionally tools) followed by a workflow body that instructs the agent how to use PaperBot MCP tools to accomplish a multi-step research workflow.

The SKILL.md format is governed by the Claude Code plugin system. The agent runtime reads skills from `.claude/skills/`, scans subdirectories for SKILL.md, loads metadata always, loads the body when the skill triggers, and loads any `references/` files on demand. All four skills should be lean (under 500 lines of body content), use imperative/infinitive writing style (not second person), and reference specific MCP tool names that PaperBot has already implemented.

All nine MCP tools and four MCP resources are complete and verified from phases 2–5. The skill files reference these by their exact Python-registered names. Testing for this phase is structural (file presence, YAML parse, required frontmatter fields) — no unit tests need to exercise the skill content at runtime.

**Primary recommendation:** Create four SKILL.md files under `.claude/skills/` using the established frontmatter schema; reference PaperBot MCP tool names exactly as registered; keep bodies lean and workflow-focused.

---

## Standard Stack

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| SKILL.md files | Claude Code convention | Agent skill discovery and loading | Native Claude Code/Codex skill format — no library needed |
| YAML frontmatter | standard YAML | Metadata (name, description, tools) | Required by Claude Code skill loader |
| Markdown body | CommonMark | Workflow instructions | Human and agent readable; supports headers, code blocks, lists |

### No Dependencies
This phase requires zero new Python packages, zero npm packages. It is plain file authoring.

---

## Architecture Patterns

### Recommended Directory Structure
```
.claude/
└── skills/
    ├── literature-review/
    │   └── SKILL.md
    ├── paper-reproduction/
    │   └── SKILL.md
    ├── trend-analysis/
    │   └── SKILL.md
    └── scholar-monitoring/
        └── SKILL.md
```

Optional (per-skill, if body exceeds ~300 lines or detail is heavy):
```
.claude/skills/literature-review/
├── SKILL.md
└── references/
    └── workflow-detail.md   # loaded by agent on demand
```

For the four PaperBot skills, the bodies are unlikely to exceed 300 lines, so `references/` subdirectories are optional. Keep it minimal.

### Pattern 1: SKILL.md Frontmatter Schema

**What:** YAML block at top of file, delimited by `---`. Required fields: `name`, `description`. Optional: `tools`.

**Critical rules:**
- `name`: lowercase, hyphens only, matches the skill directory name
- `description`: third-person, includes specific trigger phrases the user or agent would say. This is the primary discovery mechanism — the agent reads the description to decide whether to load the skill body.
- `tools`: optional list of MCP tool names or Claude Code built-in tools the skill uses. Informational — helps agents pre-allow the right tools.

**Template:**
```yaml
---
name: literature-review
description: This skill should be used when the user asks to "do a literature review",
  "survey papers on a topic", "find and summarize research on X", "search papers and
  judge quality", or wants a multi-step workflow to search, score, and summarize
  academic papers using PaperBot MCP tools.
tools:
  - paper_search
  - paper_judge
  - paper_summarize
  - relevance_assess
  - save_to_memory
  - get_research_context
---
```

### Pattern 2: SKILL.md Body — Workflow Instructions

**What:** Markdown body following the frontmatter. Written in imperative/infinitive form. Describes multi-step workflow using specific tool names.

**Rules:**
- Write in imperative form: "Search for papers using `paper_search`." NOT "You should search..."
- Reference MCP tools by their exact registered names (verified below)
- Keep body under 500 lines (ideally 100–250 lines for these focused workflows)
- Use numbered steps for sequential workflows
- Call out degraded states (all PaperBot tools return `degraded=True` when LLM is unavailable)

**Example body structure:**
```markdown
# Literature Review Workflow

Conduct a multi-step academic literature review using PaperBot MCP tools.

## Workflow

### Step 1: Search for papers
Call `paper_search` with the research topic.
- Parameters: `query` (required), `max_results` (default 10), `sources` (optional)
- Returns: list of paper dicts with title, abstract, authors, year, venue

### Step 2: Assess relevance
For each paper, call `relevance_assess` to score relevance (0–100).
- Parameters: `title`, `abstract`, `query`, `keywords` (optional)
- Filter out papers with score below threshold (suggest 40)

### Step 3: Judge quality
For high-relevance papers, call `paper_judge` to assess quality dimensions.
- Parameters: `title`, `abstract`, `full_text` (optional), `rubric` (default "default")
- Returns: dimension scores (1–5), overall score, recommendation (must_read/worth_reading/skim/skip)

### Step 4: Summarize selected papers
Call `paper_summarize` for papers recommended as must_read or worth_reading.
- Parameters: `title`, `abstract`
- Returns: concise summary string

### Step 5: Save findings to memory
Call `save_to_memory` with synthesized findings.
- Parameters: `content` (the synthesis), `kind` ("note" or "hypothesis"), `scope_type` ("global" or "track")

## Degraded Mode
If tools return `degraded=True`, LLM API keys are not configured.
Set OPENAI_API_KEY or ANTHROPIC_API_KEY and restart the MCP server.
```

### Pattern 3: Tool Name Reference (Verified)

The exact names as registered via `@mcp.tool()` decorators in PaperBot:

| Tool Name | Parameters | Returns | Phase |
|-----------|-----------|---------|-------|
| `paper_search` | `query`, `max_results=10`, `sources=None` | list of paper dicts | Phase 2 |
| `paper_judge` | `title`, `abstract`, `full_text=""`, `rubric="default"` | dict with scores + recommendation | Phase 2 |
| `paper_summarize` | `title`, `abstract` | dict with `summary` key | Phase 2 |
| `relevance_assess` | `title`, `abstract`, `query`, `keywords=""` | dict with `score` (0–100) + `reason` | Phase 2 |
| `analyze_trends` | `topic`, `papers` (list of dicts) | dict with `trend_analysis` string | Phase 3 |
| `check_scholar` | `scholar_name`, `max_papers=10` | dict with `scholar` + `recent_papers` | Phase 3 |
| `get_research_context` | `query`, `user_id="default"`, `track_id=None` | dict with papers, memories, stage | Phase 3 |
| `save_to_memory` | `content`, `kind="note"`, `user_id`, `scope_type`, `scope_id`, `confidence` | dict with created/skipped | Phase 3 |
| `export_to_obsidian` | `title`, `abstract`, `authors=[]`, `year=None`, `venue=""`, `arxiv_id=""`, `doi=""` | dict with `markdown` key | Phase 3 |

MCP Resources (read-only, referenced by URI not tool call):

| Resource URI | Returns |
|-------------|---------|
| `paperbot://track/{id}` | Track metadata |
| `paperbot://track/{id}/papers` | Papers in a track |
| `paperbot://track/{id}/memory` | Track memory items |
| `paperbot://scholars` | Scholar subscriptions |

### Pattern 4: Workflow Mapping to Skills

| Skill Name | Primary Tools | Description Trigger Phrases |
|-----------|-------------|---------------------------|
| `literature-review` | paper_search, relevance_assess, paper_judge, paper_summarize, save_to_memory | "literature review", "survey papers on", "search and summarize research", "find papers about" |
| `paper-reproduction` | paper_search, paper_summarize, paper_judge, export_to_obsidian, save_to_memory | "reproduce paper", "implement paper code", "paper2code", "replicate research", "run experiment from paper" |
| `trend-analysis` | paper_search, analyze_trends, save_to_memory, get_research_context | "analyze trends", "what's trending in", "research landscape", "topic trend analysis" |
| `scholar-monitoring` | check_scholar, save_to_memory, analyze_trends | "monitor scholar", "check researcher activity", "track publications", "follow author" |

### Anti-Patterns to Avoid

- **Wrong tool names:** Any typo in a tool name means the agent calls a non-existent tool. Use the exact names from the table above — they come directly from `@mcp.tool()` function definitions in the Python source.
- **Second-person writing in body:** "You should call `paper_search`" is wrong. "Call `paper_search`" is correct. Imperative form throughout.
- **Vague description field:** "Helps with paper research" will not trigger reliably. Include concrete user phrases ("do a literature review on transformers") in the description.
- **Referencing unimplemented tools:** Do not reference tools not in the verified list above. The four skills cover everything with the 9 existing tools.
- **Body over 500 lines:** If workflow detail is too long, move it to `references/workflow.md` and link from SKILL.md body.
- **Missing `name` or `description` in frontmatter:** The skill loader requires both. Omitting either causes the skill to be silently ignored.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Skill discovery mechanism | Custom skill scanner | `.claude/skills/` directory convention | Claude Code has built-in skill loader |
| Tool routing logic | Custom tool dispatcher | MCP tool calling (already in MCP server) | Phase 3–5 implementation handles dispatch |
| Workflow engine | State machine for multi-step workflows | Skill body instructions + agent reasoning | Agent reasons through steps; no code needed |
| Paper metadata schemas | Custom Paper class in skill | References existing tool return shapes | Tools already define their output format |

**Key insight:** This phase is content, not code. The "engine" is the agent itself reading the SKILL.md and calling MCP tools. Zero Python is written.

---

## Common Pitfalls

### Pitfall 1: Tool Name Typos
**What goes wrong:** Skill body says `search_papers` instead of `paper_search`. Agent attempts to call a non-existent tool and fails.
**Why it happens:** Skill authors write from memory, not from verified source.
**How to avoid:** Copy tool names directly from the verified table above, which was extracted from the actual `@mcp.tool()` source files.
**Warning signs:** Agent logs show "Tool not found" or similar MCP errors during skill execution.

### Pitfall 2: Over-specifying Parameters in Skill Body
**What goes wrong:** Skill hardcodes `max_results=5` in every step, preventing the agent from adjusting for the user's actual query scope.
**Why it happens:** Trying to be too prescriptive.
**How to avoid:** Specify defaults as suggestions, not mandates. "Call `paper_search` with `max_results=10` (adjust as needed)."
**Warning signs:** Users complain that reviews always return the same number of papers regardless of topic breadth.

### Pitfall 3: Weak Description Trigger Phrases
**What goes wrong:** Skill description says "helps with academic research" — too vague to trigger reliably. Agent loads a different skill or no skill.
**Why it happens:** Generic language in the description field.
**How to avoid:** Include at least 4–6 specific user-utterance trigger phrases in the description. Cover both casual ("find papers about attention mechanisms") and formal ("conduct a systematic literature review on X") phrasings.
**Warning signs:** Agent does not load the skill when the user asks an obvious workflow question.

### Pitfall 4: Missing Degraded-Mode Guidance
**What goes wrong:** User runs a literature review with no LLM API key configured. Tools return `degraded=True`. Agent silently produces empty results.
**Why it happens:** Skill body doesn't mention the degraded mode pattern.
**How to avoid:** Add a "Degraded Mode" section to each skill body. All PaperBot LLM-backed tools (`paper_judge`, `paper_summarize`, `relevance_assess`, `analyze_trends`) return `degraded=True` plus an `error` key when the LLM is unavailable. `paper_search` and `check_scholar` are degraded-resilient (no LLM needed).
**Warning signs:** Agent returns empty summaries without error messages.

### Pitfall 5: Directory Name Mismatch
**What goes wrong:** Skill directory named `lit-review/` but frontmatter `name: literature-review`. Some loaders use the directory name as the identifier.
**How to avoid:** Match the directory name to the `name` field in frontmatter exactly. Use the four names defined in the requirement: `literature-review`, `paper-reproduction`, `trend-analysis`, `scholar-monitoring`.
**Warning signs:** Skill is listed under wrong name in `/skills` command output.

---

## Code Examples

Verified patterns from canonical skill format and PaperBot MCP tool signatures:

### Minimal Valid SKILL.md Frontmatter
```yaml
---
name: trend-analysis
description: This skill should be used when the user asks to "analyze trends in a
  research area", "what is trending in X", "research landscape for topic Y", "topic
  trend analysis", or wants to survey a field and identify emerging themes across
  multiple papers using PaperBot.
tools:
  - paper_search
  - analyze_trends
  - save_to_memory
  - get_research_context
---
```

### Trend Analysis Workflow Body (Example)
```markdown
# Trend Analysis Workflow

Identify research trends across a topic by searching, collecting, and analyzing papers.

## Workflow

### Step 1: Load research context (optional)
Call `get_research_context` with the topic to retrieve existing memories and papers.
- If `track_id` is known, pass it to scope the context.

### Step 2: Search for papers
Call `paper_search` with the topic. Use `max_results=20–50` for trend analysis
(broader corpus improves trend signal).

### Step 3: Analyze trends
Call `analyze_trends` with `topic` and the list of paper dicts from Step 2.
- Returns `trend_analysis` (natural language), `topic`, `paper_count`.
- Check for `degraded=True` — requires LLM API key.

### Step 4: Save synthesis
Call `save_to_memory` with the trend analysis text.
- Use `kind="note"` or `kind="hypothesis"` as appropriate.
- Use `scope_type="global"` unless scoping to a specific track.

## Degraded Mode
`analyze_trends` returns `degraded=True` and an `error` key when LLM is unavailable.
Configure OPENAI_API_KEY or ANTHROPIC_API_KEY before using this workflow.
```

### Scholar Monitoring Workflow Body (Example)
```markdown
# Scholar Monitoring Workflow

Monitor a researcher's recent publication activity and synthesize their output.

## Workflow

### Step 1: Check scholar activity
Call `check_scholar` with the scholar's name.
- Returns `scholar` (profile with hIndex, citationCount) and `recent_papers` (list).
- If `degraded=True`, the scholar was not found on Semantic Scholar. Try alternate name spellings.

### Step 2: Analyze paper trends (optional)
If recent_papers is non-empty, call `analyze_trends` with the scholar's name as topic
and the recent_papers list.

### Step 3: Save monitoring note
Call `save_to_memory` with a summary of the scholar's recent activity.
- Use `kind="note"`, `scope_type="global"`.

## Note on Scholar Lookup
`check_scholar` searches Semantic Scholar by name. Common issues:
- Names with diacritics may need ASCII variant.
- Very new researchers may have limited Semantic Scholar records.
- The tool returns top 3 candidates in `candidates` — inspect these if top match is wrong.
```

### Literature Review Workflow Body (Example)
```markdown
# Literature Review Workflow

Conduct a systematic literature review: search, filter by relevance, judge quality,
summarize, and save findings.

## Workflow

### Step 1: Search for papers
Call `paper_search` with the research question.
- `max_results`: 10–20 for focused reviews, up to 50 for broad surveys.
- `sources`: omit for all sources, or specify `["arxiv", "semantic_scholar"]`.

### Step 2: Filter by relevance
For each paper, call `relevance_assess` with `title`, `abstract`, and the same `query`.
- `score` is 0–100. Threshold suggestion: discard papers below 40.
- If `degraded=True`, fallback to token-overlap scoring (less accurate but functional).

### Step 3: Judge quality of relevant papers
For papers above relevance threshold, call `paper_judge`.
- `rubric`: pass the research question as the rubric for context-aware judging.
- `recommendation`: use to prioritize (must_read > worth_reading > skim > skip).

### Step 4: Summarize top papers
Call `paper_summarize` for must_read and worth_reading papers.
- Returns `summary` text.
- If `degraded=True`, generate manual summary from abstract.

### Step 5: Export to Obsidian (optional)
Call `export_to_obsidian` for papers to save as permanent notes.
- Provide `title`, `abstract`, `authors`, `year`, `venue`, `arxiv_id`/`doi` as available.
- Returns `markdown` string with YAML frontmatter ready to write to vault.

### Step 6: Save synthesis to memory
Call `save_to_memory` with a synthesis of findings.
- `kind="note"` for general observations, `kind="hypothesis"` for research directions.

## Degraded Mode
`paper_judge`, `paper_summarize`, `relevance_assess` require LLM API keys.
`paper_search` works without LLM. When degraded, search results are returned but
quality scoring and summaries are unavailable.
```

### Paper Reproduction Workflow Body (Example)
```markdown
# Paper Reproduction Workflow

Reproduce or implement a paper: locate it, understand its contributions, and guide
implementation or code generation.

## Workflow

### Step 1: Find the paper
Call `paper_search` with the paper title or topic.
- If ArXiv ID or DOI is known, include it in the query for direct lookup.

### Step 2: Judge reproducibility
Call `paper_judge` with `rubric="reproducibility"` to assess implementation feasibility.
- High rigor score and clear methodology are favorable signals.
- Low clarity score may indicate reproduction difficulty.

### Step 3: Summarize paper contributions
Call `paper_summarize` to get a concise summary of key contributions, methods, findings.

### Step 4: Save reproduction plan to memory
Call `save_to_memory` with an outline of implementation steps.
- `kind="project"` or `kind="decision"` for planned implementation approach.

### Step 5: Export paper note
Call `export_to_obsidian` to create an Obsidian note for the paper.
- Provides structured YAML frontmatter and body for the research notebook.

## Implementation Guidance
After completing the above workflow, proceed with code implementation using available
coding tools (Bash, Write, etc.). The Paper2Code pipeline in PaperBot
(`src/paperbot/repro/`) provides deeper analysis for complex reproductions but
requires the full PaperBot backend.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ad-hoc agent prompts | SKILL.md convention | 2024–2025 (Claude Code skill system) | Structured discovery, progressive loading |
| Monolithic agent instructions | Progressive disclosure (metadata → body → references) | Claude Code plugin system | Context-efficient skill loading |
| Tool documentation in CLAUDE.md | Skill-specific workflow files in `.claude/skills/` | 2025 (agent skills ecosystem) | Domain-scoped, discoverable, reloadable |

**Current convention (HIGH confidence):**
- Skills live in `.claude/skills/{name}/SKILL.md`
- YAML frontmatter with `name` and `description` (required), `tools` (optional)
- Body in imperative/infinitive form
- Progressive disclosure: put detail in `references/` subdirectory, not SKILL.md body

---

## Open Questions

1. **`tools` frontmatter field: enforced or advisory?**
   - What we know: The `skill-development` SKILL.md from Claude Code plugins does not include a `tools` field in its examples. The GSD phase-researcher agent file includes a `tools:` field (e.g., `tools: Read, Write, Bash, Grep`).
   - What's unclear: Whether the Claude Code skill loader enforces tool restrictions based on the `tools` field, or whether it is purely advisory/documentation.
   - Recommendation: Include `tools` field as documentation of which MCP tools the skill uses. It signals to the agent what tools to pre-allow. Do not rely on it as a security boundary.

2. **Skill body length for these four workflows**
   - What we know: Canonical recommendation is 1,500–2,000 words for plugin skills, under 500 lines for Codex skills.
   - What's unclear: PaperBot's skills are MCP-tool-calling workflows, not general knowledge domains. The bodies may be naturally compact (100–200 lines).
   - Recommendation: Target 80–200 lines per skill body. All workflow steps fit within this range without needing `references/` subdirectories.

3. **`version` field in frontmatter**
   - What we know: The `skill-development` SKILL.md includes `version: 0.1.0`. The `frontend-design` SKILL.md does not include a version field.
   - What's unclear: Whether `version` is required or advisory.
   - Recommendation: Omit `version` for simplicity. It is not required by the discovery mechanism.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest with pytest-asyncio (asyncio_mode = "strict") |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` |
| Quick run command | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -q` |
| Full suite command | `PYTHONPATH=src pytest -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCP-13 | `.claude/skills/` directory exists | static/file check | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py::test_skills_directory_exists -x` | ❌ Wave 0 |
| MCP-13 | `literature-review/SKILL.md` exists | static/file check | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py::test_skill_files_exist -x` | ❌ Wave 0 |
| MCP-13 | `paper-reproduction/SKILL.md` exists | static/file check | same | ❌ Wave 0 |
| MCP-13 | `trend-analysis/SKILL.md` exists | static/file check | same | ❌ Wave 0 |
| MCP-13 | `scholar-monitoring/SKILL.md` exists | static/file check | same | ❌ Wave 0 |
| MCP-13 | Each SKILL.md has valid YAML frontmatter | unit (yaml parse) | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py::test_skill_frontmatter_valid -x` | ❌ Wave 0 |
| MCP-13 | Each SKILL.md frontmatter has `name` field | unit | same | ❌ Wave 0 |
| MCP-13 | Each SKILL.md frontmatter has `description` field | unit | same | ❌ Wave 0 |
| MCP-13 | Each SKILL.md body references at least one PaperBot MCP tool by name | unit (grep) | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py::test_skill_references_tools -x` | ❌ Wave 0 |
| MCP-13 | Skills `name` field matches directory name | unit | `PYTHONPATH=src pytest tests/unit/test_agent_skills.py::test_skill_name_matches_directory -x` | ❌ Wave 0 |

### Test Implementation Notes

SKILL.md files are plain text — no async needed. Tests use `pathlib.Path` to read files and `yaml` (PyYAML, already a project transitive dep) to parse frontmatter.

```python
# Example test pattern (no async needed — file I/O only):
import pathlib
import yaml

SKILLS_DIR = pathlib.Path(".claude/skills")
EXPECTED_SKILLS = [
    "literature-review",
    "paper-reproduction",
    "trend-analysis",
    "scholar-monitoring",
]
KNOWN_TOOLS = {
    "paper_search", "paper_judge", "paper_summarize", "relevance_assess",
    "analyze_trends", "check_scholar", "get_research_context",
    "save_to_memory", "export_to_obsidian",
}

def _parse_skill(skill_name: str):
    path = SKILLS_DIR / skill_name / "SKILL.md"
    content = path.read_text()
    # Strip leading/trailing --- delimiters
    parts = content.split("---", 2)
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2] if len(parts) > 2 else ""
    return frontmatter, body

def test_skills_directory_exists():
    assert SKILLS_DIR.is_dir()

def test_skill_files_exist():
    for name in EXPECTED_SKILLS:
        assert (SKILLS_DIR / name / "SKILL.md").is_file(), f"Missing: {name}/SKILL.md"

def test_skill_frontmatter_valid():
    for name in EXPECTED_SKILLS:
        fm, _ = _parse_skill(name)
        assert "name" in fm, f"{name}: missing 'name' in frontmatter"
        assert "description" in fm, f"{name}: missing 'description' in frontmatter"

def test_skill_name_matches_directory():
    for name in EXPECTED_SKILLS:
        fm, _ = _parse_skill(name)
        assert fm["name"] == name, f"name mismatch: {fm['name']} != {name}"

def test_skill_references_tools():
    for name in EXPECTED_SKILLS:
        _, body = _parse_skill(name)
        found = any(tool in body for tool in KNOWN_TOOLS)
        assert found, f"{name}: body does not reference any PaperBot MCP tool"
```

### Sampling Rate
- **Per task commit:** `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -q`
- **Per wave merge:** `PYTHONPATH=src pytest tests/unit/test_agent_skills.py -q`
- **Phase gate:** Full CI offline suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `.claude/skills/` directory — must be created (does not exist yet)
- [ ] `.claude/skills/literature-review/SKILL.md` — covers MCP-13
- [ ] `.claude/skills/paper-reproduction/SKILL.md` — covers MCP-13
- [ ] `.claude/skills/trend-analysis/SKILL.md` — covers MCP-13
- [ ] `.claude/skills/scholar-monitoring/SKILL.md` — covers MCP-13
- [ ] `tests/unit/test_agent_skills.py` — covers all MCP-13 structural assertions

---

## Sources

### Primary (HIGH confidence)
- `/home/master1/PaperBot/src/paperbot/mcp/server.py` — verified all 9 tool and 4 resource registrations
- `/home/master1/PaperBot/src/paperbot/mcp/tools/*.py` — verified exact function names and parameter signatures for all tools
- `/home/master1/.claude/plugins/marketplaces/claude-plugins-official/plugins/plugin-dev/skills/skill-development/SKILL.md` — canonical SKILL.md format, frontmatter schema, progressive disclosure rules, writing style
- `/home/master1/.codex/skills/.system/skill-creator/SKILL.md` — Codex-side skill format (same structure, confirms conventions)
- `/home/master1/.claude/agents/gsd-phase-researcher.md` — confirms `skills:` field in `.claude/agents/*.md` files; shows how tools/skills fields relate

### Secondary (MEDIUM confidence)
- `/home/master1/.claude/plugins/marketplaces/claude-plugins-official/plugins/plugin-dev/skills/mcp-integration/SKILL.md` — shows how MCP tool names are referenced in skills; `tools` field format
- `/home/master1/.claude/plugins/marketplaces/claude-plugins-official/plugins/frontend-design/skills/frontend-design/SKILL.md` — minimal SKILL.md (no `tools`, no `version` field) — shows fields are optional beyond `name` and `description`

### Tertiary (LOW confidence)
- None — all findings are verified from codebase and canonical skill files on this machine.

---

## Metadata

**Confidence breakdown:**
- SKILL.md format: HIGH — read canonical `skill-development` SKILL.md directly from installed Claude Code plugins; cross-verified with Codex skill-creator
- MCP tool names: HIGH — extracted directly from Python source `@mcp.tool()` function definitions in `src/paperbot/mcp/tools/`
- Workflow content per skill: MEDIUM — workflow steps are authored judgments based on tool capabilities; agent may adapt steps at runtime. Content is correct but not verified against user testing.
- Test approach: HIGH — standard pytest + pathlib + yaml pattern; no async needed

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (SKILL.md format is stable; MCP tool signatures are locked by Phase 2–5 implementation)
