# Phase 3: Remaining MCP Tools - Research

**Researched:** 2026-03-14
**Domain:** FastMCP tool registration, PaperBot application services (TrendAnalyzer, SemanticScholarClient, ContextEngine, MemoryStore, ObsidianExporter)
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCP-01 | Agent can analyze trends across a set of papers via `analyze_trends` MCP tool | `TrendAnalyzer.analyze()` wraps `LLMService.analyze_trends()` — sync, needs anyio wrap; accepts `topic: str` + `items: Sequence[Dict]` |
| MCP-02 | Agent can check a scholar's recent publications via `check_scholar` MCP tool | `SemanticScholarClient.search_authors()` + `get_author_papers()` — async native; fallback to name-based search when no S2 ID |
| MCP-03 | Agent can retrieve research context for a track via `get_research_context` MCP tool | `ContextEngine.build_context_pack()` — async native; requires `user_id`, `query`, optional `track_id` |
| MCP-04 | Agent can save research findings to memory via `save_to_memory` MCP tool | `SqlAlchemyMemoryStore.add_memories()` — sync; accepts `MemoryCandidate` objects; needs anyio wrap |
| MCP-05 | Agent can export papers/notes to Obsidian format via `export_to_obsidian` MCP tool | `ObsidianFilesystemExporter._render_paper_note()` returns markdown string without filesystem I/O; wrap as in-memory renderer |
</phase_requirements>

---

## Summary

Phase 3 adds five MCP tools (`analyze_trends`, `check_scholar`, `get_research_context`, `save_to_memory`, `export_to_obsidian`) to bring the total registered tool count to 9. All five tools follow the same pattern established in Phase 2: a module file under `src/paperbot/mcp/tools/`, an async `_<tool>_impl()` function for testability, a `register(mcp)` function for FastMCP, and audit logging via `log_tool_call()`.

Each tool wraps an existing application-layer service. The services vary in calling convention: `TrendAnalyzer.analyze()` and `SqlAlchemyMemoryStore.add_memories()` are synchronous (use `anyio.to_thread.run_sync()`). `ContextEngine.build_context_pack()` and `SemanticScholarClient` methods are already async. `ObsidianFilesystemExporter._render_paper_note()` is synchronous but can be called directly via thread to produce a markdown string without touching the filesystem.

The key design choice for `export_to_obsidian` is to avoid requiring a real vault path at MCP call time. Instead, render the markdown in-memory and return it as a string — consistent with how an AI agent consumes the tool (it reads the markdown, it doesn't manage a filesystem). This is a clean divergence from the `ObsidianFilesystemExporter.export_library_snapshot()` API which requires an existing directory.

**Primary recommendation:** Follow Phase 2 patterns exactly. One file per tool, `_impl` function + `register()`, anyio wrapping for sync services, degraded detection for LLM-dependent tools, log_tool_call() in all paths.

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mcp[fastmcp]` | `>=1.8.0,<2.0.0` | FastMCP server + `@mcp.tool()` decorator | Established in Phase 1 |
| `anyio` | existing | Sync-to-async bridge via `to_thread.run_sync()` | Established in Phase 2 |
| `jinja2` | existing | Template rendering in `ObsidianFilesystemExporter._render_paper_note()` | Already a project dep |
| `yaml` (PyYAML) | existing | YAML frontmatter generation in `_yaml_frontmatter()` | Already a project dep |

### No New Dependencies
All five tools wrap existing services. Zero new packages required.

**Installation:** No changes needed.

---

## Architecture Patterns

### Recommended File Structure (Phase 3 additions)
```
src/paperbot/mcp/tools/
├── _audit.py              # existing: shared audit helper
├── paper_search.py        # existing (Phase 2)
├── paper_judge.py         # existing (Phase 2)
├── paper_summarize.py     # existing (Phase 2)
├── relevance.py           # existing (Phase 2)
├── analyze_trends.py      # NEW: wraps TrendAnalyzer
├── check_scholar.py       # NEW: wraps SemanticScholarClient
├── get_research_context.py# NEW: wraps ContextEngine
├── save_to_memory.py      # NEW: wraps SqlAlchemyMemoryStore
└── export_to_obsidian.py  # NEW: wraps ObsidianFilesystemExporter renderer

tests/unit/
├── test_mcp_analyze_trends.py
├── test_mcp_check_scholar.py
├── test_mcp_get_research_context.py
├── test_mcp_save_to_memory.py
└── test_mcp_export_to_obsidian.py

tests/integration/
└── test_mcp_tool_calls.py  # existing: add Phase 3 tools to discovery test
```

### Pattern 1: Sync service wrapped with anyio (established by Phase 2)
**What:** Wrap a blocking synchronous service call in `anyio.to_thread.run_sync(lambda: ...)` inside an async `_impl` function.
**When to use:** `TrendAnalyzer.analyze()`, `SqlAlchemyMemoryStore.add_memories()`

```python
# Source: src/paperbot/mcp/tools/paper_judge.py (established pattern)
async def _analyze_trends_impl(topic: str, papers: List[Dict[str, Any]], _run_id: str = "") -> dict:
    start = time.monotonic()
    analyzer = _get_analyzer()
    try:
        result = await anyio.to_thread.run_sync(
            lambda: analyzer.analyze(topic=topic, items=papers)
        )
        # ... audit log + return
    except Exception as exc:
        # ... audit log with error + raise
```

### Pattern 2: Native async service (no wrapping needed)
**What:** Call an already-async service method directly with `await`.
**When to use:** `ContextEngine.build_context_pack()`, `SemanticScholarClient.search_authors()`, `SemanticScholarClient.get_author_papers()`

```python
# Source: src/paperbot/context_engine/engine.py line 799
# build_context_pack() is already async
async def _get_research_context_impl(user_id: str, query: str, track_id: Optional[int] = None, _run_id: str = "") -> dict:
    engine = _get_context_engine()
    result = await engine.build_context_pack(user_id=user_id, query=query, track_id=track_id)
    # result is a Dict[str, Any] with keys: papers, memories, track, etc.
```

### Pattern 3: Module-level lazy singleton
**What:** Module-level `_service = None` + `_get_service()` helper that constructs on first call.
**When to use:** All 5 new tools (mirrors Phase 2 pattern).
**Why:** Enables test injection via `mod._service = fake_service` without FastMCP.

### Pattern 4: In-memory Obsidian rendering (new for Phase 3)
**What:** Call `ObsidianFilesystemExporter._render_paper_note()` without writing to disk.
**When to use:** `export_to_obsidian` tool — agent needs the markdown string, not a file.

```python
# Source: src/paperbot/infrastructure/exporters/obsidian_exporter.py line 519
# _render_paper_note() returns a str; no filesystem I/O
async def _export_to_obsidian_impl(title: str, abstract: str, authors: List[str] = [], ...) -> dict:
    exporter = _get_exporter()
    paper = {"title": title, "abstract": abstract, "authors": authors, ...}
    # Call _render_paper_note via thread (it is sync)
    markdown = await anyio.to_thread.run_sync(
        lambda: exporter._render_paper_note(
            template_path=None,
            title=title,
            abstract=abstract,
            metadata_rows=[...],
            track_link=None,
            external_links=[],
            related_links=[],
            reference_links=[],
            cited_by_links=[],
            paper=paper,
            track=None,
            related_titles=[],
        )
    )
    frontmatter = _yaml_frontmatter({...})  # from obsidian_exporter module
    return {"markdown": frontmatter + markdown}
```

### Pattern 5: MemoryCandidate construction for save_to_memory
**What:** Accept flat string inputs from MCP caller, construct `MemoryCandidate` dataclass internally.
**When to use:** `save_to_memory` tool.

```python
# Source: src/paperbot/memory/schema.py
# MemoryKind is a Literal with allowed values
from paperbot.memory.schema import MemoryCandidate, MemoryKind

candidate = MemoryCandidate(
    kind=kind,         # e.g. "note", "fact", "hypothesis", "decision"
    content=content,
    confidence=confidence,  # float 0.0-1.0, default 0.6
    scope_type=scope_type,  # "global", "track", "project", "paper"
    scope_id=scope_id,      # optional: track_id as str
)
memory_store.add_memories(user_id=user_id, memories=[candidate])
```

### Anti-Patterns to Avoid

- **Calling `ObsidianFilesystemExporter.export_library_snapshot()` from the MCP tool**: requires a real vault_path directory on disk — not appropriate for an MCP tool that returns a value to an agent.
- **Constructing `ContextEngine` without defaults**: `ContextEngine()` with no arguments builds `SqlAlchemyResearchStore()` and `SqlAlchemyMemoryStore()` which opens the DB. Fine for production, but tests need to inject fakes via constructor.
- **Passing papers as raw JSON strings to analyze_trends**: `TrendAnalyzer.analyze()` expects `Sequence[Dict[str, Any]]`. The MCP tool should accept a list of dicts (FastMCP will serialize JSON inputs as Python dicts automatically via type hints).
- **Using `Container.instance().resolve()` inside the lazy singleton**: The lazy singleton pattern (module-level `_service`) is simpler and test-friendlier. Reserve `Container` resolution for the `_audit.py` helper only (established pattern).
- **Making `check_scholar` require a Semantic Scholar author_id**: Agents calling from Claude/Codex won't always have an S2 ID — implement a name-search fallback (`SemanticScholarClient.search_authors()` → pick first match → `get_author_papers()`).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Trend analysis LLM prompt | Custom prompt logic | `TrendAnalyzer.analyze(topic, items)` | Prompt already in `application/prompts/trend_detection.py` |
| Scholar lookup | Custom HTTP to S2 API | `SemanticScholarClient.search_authors()` + `get_author_papers()` | Rate limiting, error handling already implemented |
| Research context assembly | Custom track/memory queries | `ContextEngine.build_context_pack()` | Multi-layer loading, routing, embedding all handled |
| Memory persistence | Direct SQLAlchemy inserts | `SqlAlchemyMemoryStore.add_memories()` | Dedup, PII detection, audit log, hash generation all built in |
| Obsidian markdown | Custom template | `ObsidianFilesystemExporter._render_paper_note()` | Jinja2 template, frontmatter, wikilinks, metadata rows all handled |

**Key insight:** Every tool in Phase 3 wraps a fully-featured existing service. The MCP tool layer is thin: parameter translation + anyio wrapping + audit logging.

---

## Common Pitfalls

### Pitfall 1: `check_scholar` makes live HTTP calls in tests
**What goes wrong:** `SemanticScholarClient` makes real HTTP requests to `api.semanticscholar.org` if not faked out.
**Why it happens:** Unlike Phase 2 tools that wrap local services (LLM, search adapters), `check_scholar` uses a network client.
**How to avoid:** Module-level `_client = None` + `_get_client()` pattern; tests inject a fake client (`mod._client = _FakeS2Client()`) before calling `_impl`. Never hit real S2 in unit tests.
**Warning signs:** Tests passing locally but failing in CI offline (`PAPERBOT_OFFLINE=true`).

### Pitfall 2: `save_to_memory` creates real DB rows if MemoryStore not faked
**What goes wrong:** Unit tests calling `_save_to_memory_impl()` without injecting a fake store will attempt to open `data/paperbot.db`.
**Why it happens:** `SqlAlchemyMemoryStore()` defaults to `get_db_url()` which reads `PAPERBOT_DB_URL` env var or falls back to `sqlite:///data/paperbot.db`.
**How to avoid:** Use `tmp_path` fixture for test DB, or inject a `_FakeMemoryStore` that records calls without DB access. Follow existing pattern from `tests/unit/test_memory_module.py`.
**Warning signs:** `OperationalError: unable to open database file` in test output.

### Pitfall 3: `ContextEngine` offline mode conflict
**What goes wrong:** `ContextEngine.build_context_pack()` will attempt external search if `config.offline=False` and `config.paper_limit > 0`.
**Why it happens:** Default `ContextEngineConfig()` has `offline=False`.
**How to avoid:** Pass `config=ContextEngineConfig(offline=True, paper_limit=0)` in the MCP tool's `_get_context_engine()` lazy constructor, or accept that the tool returns cached/stored results only when offline. For tests, inject a mock `ContextEngine` directly.
**Warning signs:** `httpx.ConnectError` or network errors during unit tests.

### Pitfall 4: MemoryCandidate `kind` validation
**What goes wrong:** `MemoryKind` is a `Literal[...]` type — passing an unsupported kind (e.g. `"research_note"`) will cause a type error at runtime in strict-type contexts.
**Why it happens:** The `kind` field is defined as a Literal with fixed values: `"profile"`, `"preference"`, `"goal"`, `"project"`, `"constraint"`, `"todo"`, `"fact"`, `"note"`, `"decision"`, `"hypothesis"`, `"keyword_set"`.
**How to avoid:** Default the MCP tool's `kind` parameter to `"note"` (safe general-purpose kind). Validate or sanitize against the allowed set before constructing `MemoryCandidate`.
**Warning signs:** `TypeError` or silent DB store with unrecognized kind.

### Pitfall 5: `export_to_obsidian` private method call
**What goes wrong:** Calling `_render_paper_note()` (a private method by Python convention) may break if `ObsidianFilesystemExporter` is refactored.
**Why it happens:** The `ObsidianFilesystemExporter` doesn't have a public "render to string" API — only `export_library_snapshot()` which writes to disk.
**How to avoid:** Accept this as a known coupling point. In the tool, wrap the private call minimally and add a comment noting the dependency. This is the correct trade-off: avoid reimplementing the template renderer, accept the internal coupling.

### Pitfall 6: `analyze_trends` LLM degraded detection
**What goes wrong:** When no API key is configured, `TrendAnalyzer.analyze()` returns an empty string (same as `PaperSummarizer`).
**Why it happens:** `LLMService.complete()` returns `""` when no provider is available.
**How to avoid:** Mirror `paper_summarize`'s degraded detection: check `if not result or not result.strip()` → set `degraded=True` + error message.

---

## Code Examples

Verified patterns from existing codebase:

### analyze_trends: TrendAnalyzer API
```python
# Source: src/paperbot/application/workflows/analysis/trend_analyzer.py
from paperbot.application.workflows.analysis.trend_analyzer import TrendAnalyzer

analyzer = TrendAnalyzer()  # uses get_llm_service() internally
result: str = analyzer.analyze(topic="large language models", items=[
    {"title": "...", "abstract": "..."},
    {"title": "...", "abstract": "..."},
])
# result is a raw LLM text string; empty string when LLM unavailable
```

### check_scholar: SemanticScholarClient API
```python
# Source: src/paperbot/infrastructure/api_clients/semantic_scholar.py
from paperbot.infrastructure.api_clients.semantic_scholar import SemanticScholarClient

client = SemanticScholarClient()  # optional api_key from env

# Step 1: find author by name
authors = await client.search_authors("Yoshua Bengio", limit=3,
    fields=["name", "authorId", "hIndex", "paperCount", "citationCount"])
# returns [{"authorId": "...", "name": "...", "hIndex": N, ...}]

# Step 2: get recent papers
if authors:
    papers = await client.get_author_papers(authors[0]["authorId"], limit=10,
        fields=["title", "year", "citationCount", "venue", "abstract"])
# returns [{"title": "...", "year": ..., ...}]
```

### get_research_context: ContextEngine API
```python
# Source: src/paperbot/context_engine/engine.py line 799
from paperbot.context_engine import ContextEngine, ContextEngineConfig

engine = ContextEngine(config=ContextEngineConfig(offline=True, paper_limit=0))
result = await engine.build_context_pack(
    user_id="default",
    query="attention mechanisms",
    track_id=None,  # uses active track
)
# result keys: papers, memories, track, routing_suggestion, stage, ...
```

### save_to_memory: MemoryStore + MemoryCandidate
```python
# Source: src/paperbot/memory/schema.py + src/paperbot/infrastructure/stores/memory_store.py
from paperbot.memory.schema import MemoryCandidate
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore

store = SqlAlchemyMemoryStore()
candidate = MemoryCandidate(
    kind="note",           # one of: note, fact, decision, hypothesis, etc.
    content="Key finding: ...",
    confidence=0.8,
    scope_type="track",
    scope_id="42",         # track_id as string
)
created, skipped, rows = store.add_memories(
    user_id="default",
    memories=[candidate],
)
# returns (created_count, skipped_count, created_rows)
```

### export_to_obsidian: In-memory rendering
```python
# Source: src/paperbot/infrastructure/exporters/obsidian_exporter.py line 519
from paperbot.infrastructure.exporters.obsidian_exporter import (
    ObsidianFilesystemExporter,
    _yaml_frontmatter,
)

exporter = ObsidianFilesystemExporter()
body = exporter._render_paper_note(
    template_path=None,
    title="Paper Title",
    abstract="Paper abstract...",
    metadata_rows=["Authors: A, B", "Year: 2024", "Venue: NeurIPS"],
    track_link=None,
    external_links=[],
    related_links=[],
    reference_links=[],
    cited_by_links=[],
    paper={"title": "Paper Title", "abstract": "..."},
    track=None,
    related_titles=[],
)
frontmatter = _yaml_frontmatter({"title": "Paper Title", "paperbot_type": "paper"})
markdown = frontmatter + body
```

### server.py registration pattern (what Plan 03 adds)
```python
# Source: src/paperbot/mcp/server.py (current state after Phase 2)
# Phase 3 adds these 5 lines:
from paperbot.mcp.tools import analyze_trends
from paperbot.mcp.tools import check_scholar
from paperbot.mcp.tools import get_research_context
from paperbot.mcp.tools import save_to_memory
from paperbot.mcp.tools import export_to_obsidian

analyze_trends.register(mcp)
check_scholar.register(mcp)
get_research_context.register(mcp)
save_to_memory.register(mcp)
export_to_obsidian.register(mcp)
```

---

## Tool Signatures (recommended)

These are the exact parameter signatures each `_impl` function should expose:

### `_analyze_trends_impl`
```python
async def _analyze_trends_impl(
    topic: str,
    papers: List[Dict[str, Any]],
    _run_id: str = "",
) -> dict:
    # Returns: {"trend_analysis": str, "topic": str, "paper_count": int}
    # Degraded: {"degraded": True, "error": "...", "trend_analysis": ""}
```

### `_check_scholar_impl`
```python
async def _check_scholar_impl(
    scholar_name: str,
    max_papers: int = 10,
    _run_id: str = "",
) -> dict:
    # Returns: {"scholar": {name, authorId, hIndex, paperCount}, "recent_papers": [...]}
    # Degraded: {"degraded": True, "error": "Scholar not found", "scholar": None, "recent_papers": []}
```

### `_get_research_context_impl`
```python
async def _get_research_context_impl(
    query: str,
    user_id: str = "default",
    track_id: Optional[int] = None,
    _run_id: str = "",
) -> dict:
    # Returns: ContextEngine.build_context_pack() result dict
    # Keys include: papers, memories, track, stage, routing_suggestion
```

### `_save_to_memory_impl`
```python
async def _save_to_memory_impl(
    content: str,
    kind: str = "note",
    user_id: str = "default",
    scope_type: str = "global",
    scope_id: str = "",
    confidence: float = 0.8,
    _run_id: str = "",
) -> dict:
    # Returns: {"saved": True, "created": N, "skipped": N}
    # Error: {"saved": False, "error": "..."}
```

### `_export_to_obsidian_impl`
```python
async def _export_to_obsidian_impl(
    title: str,
    abstract: str,
    authors: List[str] = [],
    year: Optional[int] = None,
    venue: str = "",
    arxiv_id: str = "",
    doi: str = "",
    _run_id: str = "",
) -> dict:
    # Returns: {"markdown": str}  — complete Obsidian note with YAML frontmatter
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct tool functions registered inline | `register(mcp)` pattern per module | Phase 2 | Avoids circular imports, enables isolated testing |
| `unittest.mock` | Fake classes (`_FakeLLMService`) | Phase 1 | Matches project test policy (CLAUDE.md) |
| `asyncio.to_thread` | `anyio.to_thread.run_sync()` | Phase 2 | anyio is MCP-compatible event loop agnostic |

**Current state:** Phase 2 established all patterns. Phase 3 applies them to 5 more services. No new patterns needed.

---

## Open Questions

1. **`ContextEngine` offline behavior in production**
   - What we know: `ContextEngineConfig(offline=True, paper_limit=0)` disables external search
   - What's unclear: Should the MCP tool always run offline (return stored context only), or should it optionally trigger a live search?
   - Recommendation: Default to offline for the MCP tool to keep tool calls fast and side-effect-free. Agents that want live papers can call `paper_search` first then pass results to `analyze_trends`.

2. **`save_to_memory` user_id**
   - What we know: `MemoryStore.add_memories()` requires `user_id: str`
   - What's unclear: MCP callers don't have a user identity system — what `user_id` to use?
   - Recommendation: Default `user_id="default"`. This matches how the context engine uses `"default"` for single-user deployments.

3. **`check_scholar` S2 author name disambiguation**
   - What we know: `search_authors()` returns multiple matches for common names
   - What's unclear: Should the tool return the first match or all candidates?
   - Recommendation: Return the top match (highest hIndex), and include all raw candidates in the response as `"candidates"` key so agents can inspect.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest with pytest-asyncio (asyncio_mode = "strict") |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` |
| Quick run command | `PYTHONPATH=src pytest tests/unit/test_mcp_analyze_trends.py tests/unit/test_mcp_check_scholar.py tests/unit/test_mcp_get_research_context.py tests/unit/test_mcp_save_to_memory.py tests/unit/test_mcp_export_to_obsidian.py -q` |
| Full suite command | `PYTHONPATH=src pytest -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCP-01 | `analyze_trends` returns trend analysis string for papers list | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_analyze_trends.py -x` | ❌ Wave 0 |
| MCP-01 | `analyze_trends` returns degraded=True when LLM unavailable | unit | same | ❌ Wave 0 |
| MCP-01 | `analyze_trends` logs tool call via audit helper | unit | same | ❌ Wave 0 |
| MCP-02 | `check_scholar` returns scholar info + recent papers | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_check_scholar.py -x` | ❌ Wave 0 |
| MCP-02 | `check_scholar` returns degraded when scholar not found | unit | same | ❌ Wave 0 |
| MCP-02 | `check_scholar` logs tool call | unit | same | ❌ Wave 0 |
| MCP-03 | `get_research_context` returns context pack dict | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_get_research_context.py -x` | ❌ Wave 0 |
| MCP-03 | `get_research_context` accepts user_id and track_id | unit | same | ❌ Wave 0 |
| MCP-03 | `get_research_context` logs tool call | unit | same | ❌ Wave 0 |
| MCP-04 | `save_to_memory` persists content and returns saved=True | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_save_to_memory.py -x` | ❌ Wave 0 |
| MCP-04 | `save_to_memory` handles invalid kind gracefully | unit | same | ❌ Wave 0 |
| MCP-04 | `save_to_memory` logs tool call | unit | same | ❌ Wave 0 |
| MCP-05 | `export_to_obsidian` returns dict with `markdown` key | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_export_to_obsidian.py -x` | ❌ Wave 0 |
| MCP-05 | markdown contains YAML frontmatter + paper title + abstract | unit | same | ❌ Wave 0 |
| MCP-05 | `export_to_obsidian` logs tool call | unit | same | ❌ Wave 0 |
| All 9 | All 9 tools listed in tools/list (server.py discovery) | integration | `PYTHONPATH=src pytest tests/integration/test_mcp_tool_calls.py -x` | ✅ (needs update) |
| All 9 | All 9 tools log via audit helper | integration | same | ✅ (needs update) |

### Sampling Rate
- **Per task commit:** Run the specific tool's unit tests (e.g., `pytest tests/unit/test_mcp_analyze_trends.py -q`)
- **Per wave merge:** `PYTHONPATH=src pytest tests/unit/test_mcp_*.py tests/integration/test_mcp_tool_calls.py -q`
- **Phase gate:** Full CI offline suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_mcp_analyze_trends.py` — covers MCP-01 (3 tests)
- [ ] `tests/unit/test_mcp_check_scholar.py` — covers MCP-02 (3 tests)
- [ ] `tests/unit/test_mcp_get_research_context.py` — covers MCP-03 (3 tests)
- [ ] `tests/unit/test_mcp_save_to_memory.py` — covers MCP-04 (3 tests)
- [ ] `tests/unit/test_mcp_export_to_obsidian.py` — covers MCP-05 (3 tests)
- [ ] `tests/integration/test_mcp_tool_calls.py` — update EXPECTED_TOOLS from 4 to 9

---

## Sources

### Primary (HIGH confidence)
- `src/paperbot/mcp/server.py` — FastMCP instance, registration pattern
- `src/paperbot/mcp/tools/_audit.py` — `log_tool_call()` API
- `src/paperbot/mcp/tools/paper_judge.py` — anyio wrapping pattern, lazy singleton, degraded detection
- `src/paperbot/mcp/tools/paper_search.py` — async native pattern, `register(mcp)` signature
- `src/paperbot/application/workflows/analysis/trend_analyzer.py` — `TrendAnalyzer.analyze()` API
- `src/paperbot/infrastructure/api_clients/semantic_scholar.py` — `search_authors()`, `get_author_papers()` API
- `src/paperbot/context_engine/engine.py` — `ContextEngine.build_context_pack()` API (async, line 799)
- `src/paperbot/context_engine/engine.py` — `ContextEngineConfig` (line 497)
- `src/paperbot/infrastructure/stores/memory_store.py` — `SqlAlchemyMemoryStore.add_memories()` (line 560)
- `src/paperbot/memory/schema.py` — `MemoryCandidate`, `MemoryKind`
- `src/paperbot/application/ports/memory_port.py` — `MemoryPort` protocol
- `src/paperbot/infrastructure/exporters/obsidian_exporter.py` — `ObsidianFilesystemExporter._render_paper_note()`, `_yaml_frontmatter()`
- `tests/integration/test_mcp_tool_calls.py` — existing test structure to extend
- `tests/unit/test_mcp_paper_judge.py` — fake class pattern, `setup_method` reset
- `.planning/phases/02-core-paper-tools/02-02-SUMMARY.md` — Phase 2 decisions

### Secondary (MEDIUM confidence)
- `pyproject.toml` — `asyncio_mode = "strict"` confirmed; `@pytest.mark.asyncio` required on all async tests
- `CLAUDE.md` — test patterns: use stub/fake classes, not `unittest.mock`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages already installed and in use
- Architecture (tool patterns): HIGH — directly reading Phase 2 source
- Service APIs: HIGH — directly reading service source files
- Pitfalls: HIGH — derived from reading actual code (DB URL defaults, S2 HTTP calls, etc.)

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable application code)
