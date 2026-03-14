# Phase 4: MCP Resources - Research

**Researched:** 2026-03-14
**Domain:** FastMCP resource registration, URI template resources, PaperBot data layer (ResearchStore, MemoryStore, SubscriptionService)
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCP-06 | Agent can read track metadata via `paperbot://track/{id}` resource | `SqlAlchemyResearchStore.get_track_by_id(track_id=int)` returns `Dict[str, Any]` with id, name, description, keywords, venues, methods, is_active, timestamps; wrap as JSON string resource |
| MCP-07 | Agent can read track paper list via `paperbot://track/{id}/papers` resource | `SqlAlchemyResearchStore.list_track_feed(user_id, track_id, limit)` returns `{"items": [...], "total": N}`; papers have full metadata via `_paper_to_dict()` |
| MCP-08 | Agent can read track memory via `paperbot://track/{id}/memory` resource | `SqlAlchemyMemoryStore.list_memories(user_id, scope_type="track", scope_id=str(id))` returns `List[Dict]` of approved, non-deleted memories |
| MCP-09 | Agent can read scholar subscriptions via `paperbot://scholars` resource | `SubscriptionService.get_scholar_configs()` returns `List[Dict]` of raw config dicts with name, semantic_scholar_id, keywords, etc. |
</phase_requirements>

---

## Summary

Phase 4 adds four MCP resources to the existing server (`paperbot://track/{id}`, `paperbot://track/{id}/papers`, `paperbot://track/{id}/memory`, `paperbot://scholars`). These are read-only data endpoints that agents read without tool calls — they appear in the MCP `resources/templates/list` (URI templates with `{id}`) and `resources/list` (static scholars resource). Resources return JSON strings serialized from existing store/service APIs.

The implementation follows the same `register(mcp)` module pattern as tools, but uses `@mcp.resource(uri)` instead of `@mcp.tool()`. Resources are organized in a new `src/paperbot/mcp/resources/` subdirectory, registered in `server.py` alongside the 9 existing tools. Each resource module exposes a `_<name>_impl` async function for testability and a `register(mcp)` function for FastMCP binding.

The critical distinction from tools: resources with `{id}` URI template parameters appear in `resources/templates/list`, not in `resources/list`. The static `paperbot://scholars` resource appears in `resources/list`. All 4 must appear in the combined `resources/list` + `resources/templates/list` output, satisfying the success criterion "All 4 resources appear in MCP resources/list."

**Primary recommendation:** Create `src/paperbot/mcp/resources/` directory with one module per resource, use `@mcp.resource("paperbot://...")` decorator inside `register(mcp)` functions, serialize all data to JSON strings, use `user_id="default"` for single-user deployment consistency with existing tool pattern.

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mcp[fastmcp]` | `>=1.8.0,<2.0.0` | `@mcp.resource()` decorator, URI template routing | Established in Phase 1; used for all 9 tools |
| `anyio` | existing | `to_thread.run_sync()` for sync store calls | Established in Phase 2 |
| `json` | stdlib | Serialize Dict/List to JSON string for resource content | No additional dep needed |
| `PyYAML` | existing | Already dep; SubscriptionService uses it | No additional dep needed |

### No New Dependencies
All 4 resources wrap existing stores and services. Zero new packages required.

**Installation:** No changes needed.

---

## Architecture Patterns

### Recommended File Structure (Phase 4 additions)
```
src/paperbot/mcp/
├── server.py              # existing: add resource imports + register calls
├── tools/                 # existing: 9 tool modules
└── resources/             # NEW directory
    ├── __init__.py        # NEW: empty
    ├── track_metadata.py  # NEW: paperbot://track/{id}
    ├── track_papers.py    # NEW: paperbot://track/{id}/papers
    ├── track_memory.py    # NEW: paperbot://track/{id}/memory
    └── scholars.py        # NEW: paperbot://scholars

tests/unit/
├── test_mcp_track_metadata.py     # NEW: covers MCP-06
├── test_mcp_track_papers.py       # NEW: covers MCP-07
├── test_mcp_track_memory.py       # NEW: covers MCP-08
└── test_mcp_scholars.py           # NEW: covers MCP-09

tests/integration/
└── test_mcp_tool_calls.py         # existing: add resource discovery checks
```

### Pattern 1: Resource module structure (mirrors tool pattern)
**What:** One file per resource, `_impl` async function + `register(mcp)` function. `_impl` is called by the `@mcp.resource` handler and also directly from tests.
**When to use:** All 4 resources in Phase 4.

```python
# Source: mirrors src/paperbot/mcp/tools/save_to_memory.py pattern
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

import anyio

logger = logging.getLogger(__name__)

_store = None  # module-level lazy singleton

def _get_store():
    global _store
    if _store is None:
        from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
        _store = SqlAlchemyResearchStore()
    return _store

async def _track_metadata_impl(track_id: str) -> str:
    """Fetch track metadata dict and return as JSON string."""
    tid = int(track_id)
    store = _get_store()
    track = await anyio.to_thread.run_sync(
        lambda: store.get_track_by_id(track_id=tid)
    )
    if track is None:
        return json.dumps({"error": f"Track {tid} not found"})
    return json.dumps(track)

def register(mcp) -> None:
    @mcp.resource("paperbot://track/{track_id}")
    async def track_metadata(track_id: str) -> str:
        """Read PaperBot research track metadata by ID."""
        return await _track_metadata_impl(track_id)
```

### Pattern 2: URI templates vs static resources
**What:** Resources with `{param}` in their URI are registered as resource templates (appear in `resources/templates/list`). Static URIs appear in `resources/list`.
**When to use:**
- `paperbot://track/{id}` → template (MCP-06)
- `paperbot://track/{id}/papers` → template (MCP-07)
- `paperbot://track/{id}/memory` → template (MCP-08)
- `paperbot://scholars` → static resource (MCP-09)

**Important:** The Phase 4 success criterion states "All 4 resources appear in MCP resources/list." In MCP protocol terms, URI templates appear in `list_resource_templates`, not `list_resources`. Both collectively satisfy "available to agents." For integration tests, verify by checking source code references (same approach as Phase 2/3 with Python 3.9 constraint).

### Pattern 3: JSON string return type
**What:** Resource functions return `str` (JSON-serialized). FastMCP sends this as `TextResourceContents` with `mime_type="text/plain"` by default. Specify `mime_type="application/json"` for clarity.
**When to use:** All 4 resources (all return structured data).

```python
# Recommended: explicit mime_type for JSON responses
@mcp.resource("paperbot://scholars", mime_type="application/json")
async def scholars() -> str:
    return await _scholars_impl()
```

### Pattern 4: anyio wrapping for sync store calls
**What:** All store methods (`SqlAlchemyResearchStore`, `SqlAlchemyMemoryStore`, `SubscriptionService`) are synchronous. Use `anyio.to_thread.run_sync(lambda: ...)`.
**When to use:** All 4 resources — none of the backing services are async.

```python
# Source: established in Phase 2 (paper_judge.py), Phase 3 (save_to_memory.py)
result = await anyio.to_thread.run_sync(
    lambda: store.get_track_by_id(track_id=int(track_id))
)
```

### Pattern 5: `user_id="default"` for memory and track_feed queries
**What:** `SqlAlchemyResearchStore.list_track_feed()` and `SqlAlchemyMemoryStore.list_memories()` require `user_id`. MCP resources don't have a caller identity. Use `"default"` consistently.
**When to use:** `track_papers` (MCP-07) and `track_memory` (MCP-08).
**Why:** Matches the convention established by `get_research_context` and `save_to_memory` tools.

### Anti-Patterns to Avoid
- **Using `list_track_feed` for MCP-07 with default `user_id="default"` when tracks belong to different users:** `get_track_by_id()` does NOT require `user_id` (it's a global lookup by ID). But `list_track_feed()` requires `user_id`. Use `user_id="default"` consistently since PaperBot is a single-user deployment for MCP use.
- **Returning Python dicts directly from resource handlers:** FastMCP accepts `str` return for resources, not arbitrary Python objects. Always `json.dumps(data)` before returning.
- **Registering resources in `tools/` directory:** Keep resources in `resources/` subdirectory for clear separation of MCP primitives.
- **Calling `SubscriptionService().get_scholars()` instead of `get_scholar_configs()`:** `get_scholars()` returns `Scholar` domain objects which need `.to_dict()` conversion. `get_scholar_configs()` returns raw dicts directly from config — simpler and no domain object dependency.
- **Assuming `get_track_by_id()` scope matches `list_track_feed()` scope:** `get_track_by_id()` has no user scoping (global lookup). `list_track_feed()` is user-scoped. For MCP resources serving a single-user deployment, use `user_id="default"` for `list_track_feed`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Track metadata lookup | Custom SQLAlchemy query | `SqlAlchemyResearchStore.get_track_by_id(track_id=int)` | Returns pre-formatted dict with `_track_to_dict()` — all fields properly typed/serialized |
| Track paper listing | Custom paper JOIN query | `SqlAlchemyResearchStore.list_track_feed(user_id, track_id, limit)` | Handles feedback scoring, dedup, cap logic already |
| Track memory listing | Custom memory query | `SqlAlchemyMemoryStore.list_memories(user_id, scope_type="track", scope_id=str(id))` | Handles deleted/expired/pending filtering, status="approved" default |
| Scholar config reading | Direct YAML parse | `SubscriptionService.get_scholar_configs()` | Handles config path resolution, validation, caching |

**Key insight:** Phase 4 resources are thin JSON serializers over existing store methods. The MCP resource layer is: parameter extraction → anyio thread call → json.dumps().

---

## Common Pitfalls

### Pitfall 1: `track_id` comes in as `str` from URI template
**What goes wrong:** FastMCP extracts URI template parameters as strings. `store.get_track_by_id(track_id="42")` will fail because `track_id` is typed as `int` in the store API.
**Why it happens:** URI template parameters are always strings in the MCP protocol.
**How to avoid:** Always cast `int(track_id)` before calling store methods. Wrap in try/except for invalid (non-numeric) IDs.
**Warning signs:** `TypeError: int() argument must be a string or number, not 'str'` or SQLAlchemy type errors.

### Pitfall 2: Track not found returns empty response instead of useful message
**What goes wrong:** `get_track_by_id()` returns `None` when the track doesn't exist. If the resource handler returns `None` or crashes, the agent gets an unhelpful error.
**Why it happens:** MCP resources are read-only; returning `None` to FastMCP is not a valid resource content.
**How to avoid:** Always check for `None` and return a JSON error object: `json.dumps({"error": f"Track {track_id} not found"})`.

### Pitfall 3: `list_track_feed` requires `user_id` — must default to "default"
**What goes wrong:** `list_track_feed(user_id=None, track_id=42)` raises AttributeError or returns empty because there are no tracks owned by `None`.
**Why it happens:** `list_track_feed` filters by `user_id` in the WHERE clause.
**How to avoid:** Hardcode `user_id="default"` in the resource handler. Document this assumption in the module docstring.

### Pitfall 4: `SubscriptionService` FileNotFoundError on missing config
**What goes wrong:** If `config/scholar_subscriptions.yaml` doesn't exist, `SubscriptionService.load_config()` raises `FileNotFoundError`.
**Why it happens:** The service uses a hardcoded path relative to the project root. In test environments without the config file, this will fail.
**How to avoid:** Wrap the `SubscriptionService` call in try/except, return `json.dumps({"error": "Scholar config not found", "scholars": []})` on `FileNotFoundError`. In tests, inject a fake service via the module-level singleton.

### Pitfall 5: `list_memories` returns ALL scope types unless filtered
**What goes wrong:** Without `scope_type="track"` and `scope_id=str(track_id)`, `list_memories` returns all memories for the user (global + track + project). For `paperbot://track/{id}/memory`, only track-scoped memories should be returned.
**Why it happens:** `list_memories()` signature accepts optional `scope_type` and `scope_id` — they're not required.
**How to avoid:** Always pass both `scope_type="track"` and `scope_id=str(int(track_id))` to `list_memories()` in the track memory resource.

### Pitfall 6: URI template resources appear in `list_resource_templates`, not `list_resources`
**What goes wrong:** Integration tests checking `resources/list` exclusively will not find template resources. The success criterion says "All 4 resources appear in MCP resources/list" — this likely means the combined resource discovery surface.
**Why it happens:** MCP protocol distinction: static URIs → `list_resources`; URI templates → `list_resource_templates`.
**How to avoid:** For integration tests (which don't invoke FastMCP directly due to Python 3.9 constraint), verify by checking server.py source code for all 4 `register()` calls, same pattern as tool discovery tests in Phase 2/3. Name the test class `TestMCPResourceListing` and document the templates-vs-list distinction clearly.

---

## Code Examples

Verified patterns from codebase:

### Track metadata resource
```python
# Source: src/paperbot/infrastructure/stores/research_store.py line 314
# get_track_by_id() - no user_id required, global ID lookup
# Returns: {"id": int, "name": str, "description": str, "keywords": List[str],
#           "venues": List[str], "methods": List[str], "is_active": bool,
#           "archived_at": str|None, "created_at": str|None, "updated_at": str|None}

store = SqlAlchemyResearchStore()
track = store.get_track_by_id(track_id=42)
# track is None if not found
```

### Track papers resource
```python
# Source: src/paperbot/infrastructure/stores/research_store.py line 930
# list_track_feed() - requires user_id; returns {"items": [...], "total": int}
# items are paper dicts from _paper_to_dict() (id, title, abstract, authors, etc.)

store = SqlAlchemyResearchStore()
feed = store.list_track_feed(user_id="default", track_id=42, limit=50)
# feed["items"] is a List[Dict] of papers
# feed["total"] is total count
```

### Track memory resource
```python
# Source: src/paperbot/infrastructure/stores/memory_store.py line 706
# list_memories() with scope filtering -- returns approved, non-deleted, non-expired items

store = SqlAlchemyMemoryStore()
memories = store.list_memories(
    user_id="default",
    scope_type="track",
    scope_id=str(42),  # scope_id must be str, not int
    limit=100,
)
# returns List[Dict] with id, content, kind, confidence, scope_type, scope_id, etc.
```

### Scholar subscriptions resource
```python
# Source: src/paperbot/infrastructure/services/subscription_service.py line 123
# get_scholar_configs() - returns raw config dicts (no domain object conversion)
# Each dict has: name, semantic_scholar_id, keywords (optional), affiliations (optional), etc.

svc = SubscriptionService()  # defaults to config/scholar_subscriptions.yaml
scholars = svc.get_scholar_configs()
# returns [{"name": "Dawn Song", "semantic_scholar_id": "1741101", "keywords": [...], ...}, ...]
```

### server.py resource registration pattern
```python
# Source: src/paperbot/mcp/server.py (current state -- to be extended)
# Add after existing tool registrations:

from paperbot.mcp.resources import track_metadata
from paperbot.mcp.resources import track_papers
from paperbot.mcp.resources import track_memory
from paperbot.mcp.resources import scholars

track_metadata.register(mcp)
track_papers.register(mcp)
track_memory.register(mcp)
scholars.register(mcp)
```

### Full resource module example (track_metadata)
```python
# Recommended implementation pattern
from __future__ import annotations

import json
import logging
from typing import Any

import anyio

logger = logging.getLogger(__name__)

_store = None


def _get_store():
    global _store
    if _store is None:
        from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
        _store = SqlAlchemyResearchStore()
    return _store


async def _track_metadata_impl(track_id: str) -> str:
    """Fetch track metadata and return as JSON string.

    Args:
        track_id: Track ID as string (URI template extracts strings).

    Returns:
        JSON-encoded track metadata dict, or error dict if not found.
    """
    try:
        tid = int(track_id)
    except (ValueError, TypeError):
        return json.dumps({"error": f"Invalid track_id: {track_id!r}"})

    store = _get_store()
    track = await anyio.to_thread.run_sync(
        lambda: store.get_track_by_id(track_id=tid)
    )
    if track is None:
        return json.dumps({"error": f"Track {tid} not found"})
    return json.dumps(track)


def register(mcp) -> None:
    """Register paperbot://track/{track_id} resource on the given FastMCP instance."""

    @mcp.resource("paperbot://track/{track_id}", mime_type="application/json")
    async def track_metadata(track_id: str) -> str:
        """Read PaperBot research track metadata by ID.

        Returns track name, description, keywords, venues, methods,
        and status for the given track ID.
        """
        return await _track_metadata_impl(track_id)
```

---

## Resource URI Scheme

| Resource | URI | Type | Backing API |
|----------|-----|------|-------------|
| Track metadata | `paperbot://track/{track_id}` | Template | `SqlAlchemyResearchStore.get_track_by_id(track_id=int)` |
| Track papers | `paperbot://track/{track_id}/papers` | Template | `SqlAlchemyResearchStore.list_track_feed(user_id="default", track_id=int, limit=50)` |
| Track memory | `paperbot://track/{track_id}/memory` | Template | `SqlAlchemyMemoryStore.list_memories(user_id="default", scope_type="track", scope_id=str(id))` |
| Scholar subscriptions | `paperbot://scholars` | Static | `SubscriptionService.get_scholar_configs()` |

**URI scheme chosen:** `paperbot://` prefix is consistent with the requirement spec. It avoids collision with standard schemes (`file://`, `data://`, `config://`).

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tools for data retrieval (requires parameters, side effects possible) | Resources for read-only data access | Phase 4 | Agents can use `read_resource` instead of `call_tool` for structured data |
| `@mcp.tool()` only | `@mcp.tool()` + `@mcp.resource()` | Phase 4 | MCP server now surfaces both primitives |
| Static URIs only | URI templates with `{param}` | Phase 4 | Enables per-track resource access without separate static resources per track ID |

**Deprecated/outdated:**
- None from Phase 3 — all tool patterns continue unchanged.

---

## Open Questions

1. **`list_track_feed` vs direct paper query for MCP-07**
   - What we know: `list_track_feed()` is the only existing "papers for a track" API in `SqlAlchemyResearchStore`. It does fuzzy term matching (keywords/venues/methods), not a strict FK join.
   - What's unclear: Does it return papers explicitly associated with the track, or papers that match track keywords? The answer: it matches papers by track keywords/methods/venues, NOT by explicit track-paper FK. There is no explicit assignment of papers to tracks.
   - Recommendation: Use `list_track_feed()` as-is — it's the intended API for "papers relevant to a track." Document this in the resource module docstring. The `limit=50` default is reasonable.

2. **`paperbot://track/{id}/memory` — which `user_id`?**
   - What we know: `list_memories()` requires `user_id`. PaperBot MCP tools use `"default"` universally.
   - What's unclear: Could there be multi-user deployments where memories belong to non-default users?
   - Recommendation: Use `user_id="default"` for Phase 4, consistent with `save_to_memory` and `get_research_context` tools. Document this assumption.

3. **Should `paperbot://scholars` reflect live config or a cached snapshot?**
   - What we know: `SubscriptionService.get_scholar_configs()` caches after first load (`self._config` is set once). If the YAML file is edited at runtime, the cached version will be stale until process restart.
   - What's unclear: Does the agent use case require fresh reads on each access?
   - Recommendation: Accept the caching behavior (consistent with how `SubscriptionService` works throughout the codebase). For Phase 4, instantiate a fresh `SubscriptionService()` per MCP call (no module-level singleton for `scholars.py`) to ensure fresh reads. Cost is negligible (YAML parse is fast).

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest with pytest-asyncio (asyncio_mode = "strict") |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` |
| Quick run command | `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py tests/unit/test_mcp_track_papers.py tests/unit/test_mcp_track_memory.py tests/unit/test_mcp_scholars.py -q` |
| Full suite command | `PYTHONPATH=src pytest -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCP-06 | `_track_metadata_impl("42")` returns JSON with id, name, description, keywords | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py -x` | ❌ Wave 0 |
| MCP-06 | `_track_metadata_impl("99")` returns JSON error when track not found | unit | same | ❌ Wave 0 |
| MCP-06 | `_track_metadata_impl("abc")` returns JSON error for non-integer track_id | unit | same | ❌ Wave 0 |
| MCP-07 | `_track_papers_impl("42")` returns JSON with items list | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_track_papers.py -x` | ❌ Wave 0 |
| MCP-07 | Returns empty items list when track has no matching papers | unit | same | ❌ Wave 0 |
| MCP-08 | `_track_memory_impl("42")` returns JSON list of memories | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_track_memory.py -x` | ❌ Wave 0 |
| MCP-08 | Returns empty list when no memories exist for track | unit | same | ❌ Wave 0 |
| MCP-09 | `_scholars_impl()` returns JSON list of scholar dicts with name and semantic_scholar_id | unit | `PYTHONPATH=src pytest tests/unit/test_mcp_scholars.py -x` | ❌ Wave 0 |
| MCP-09 | Returns empty list when config file not found | unit | same | ❌ Wave 0 |
| All 4 | All 4 resource modules expose `register()` and `_impl` functions | integration | `PYTHONPATH=src pytest tests/integration/test_mcp_tool_calls.py -x -k resource` | ✅ (needs update) |
| All 4 | server.py imports and calls `register()` for all 4 resource modules | integration | same | ✅ (needs update) |

### Sampling Rate
- **Per task commit:** Run the specific resource's unit tests (e.g., `pytest tests/unit/test_mcp_track_metadata.py -q`)
- **Per wave merge:** `PYTHONPATH=src pytest tests/unit/test_mcp_track_*.py tests/unit/test_mcp_scholars.py tests/integration/test_mcp_tool_calls.py -q`
- **Phase gate:** Full CI offline suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/paperbot/mcp/resources/__init__.py` — new directory marker
- [ ] `src/paperbot/mcp/resources/track_metadata.py` — covers MCP-06
- [ ] `src/paperbot/mcp/resources/track_papers.py` — covers MCP-07
- [ ] `src/paperbot/mcp/resources/track_memory.py` — covers MCP-08
- [ ] `src/paperbot/mcp/resources/scholars.py` — covers MCP-09
- [ ] `tests/unit/test_mcp_track_metadata.py` — 3+ tests for MCP-06
- [ ] `tests/unit/test_mcp_track_papers.py` — 2+ tests for MCP-07
- [ ] `tests/unit/test_mcp_track_memory.py` — 2+ tests for MCP-08
- [ ] `tests/unit/test_mcp_scholars.py` — 2+ tests for MCP-09
- [ ] `tests/integration/test_mcp_tool_calls.py` — add `TestMCPResourceListing` class

---

## Sources

### Primary (HIGH confidence)
- `src/paperbot/mcp/server.py` — FastMCP instance, established `register(mcp)` pattern
- `src/paperbot/mcp/tools/save_to_memory.py` — module-level lazy singleton, anyio wrapping
- `src/paperbot/infrastructure/stores/research_store.py` line 314 — `get_track_by_id(track_id=int)` API
- `src/paperbot/infrastructure/stores/research_store.py` line 930 — `list_track_feed(user_id, track_id, limit)` API
- `src/paperbot/infrastructure/stores/research_store.py` line 1881 — `_track_to_dict()` field spec
- `src/paperbot/infrastructure/stores/research_store.py` line 2003 — `_paper_to_dict()` field spec
- `src/paperbot/infrastructure/stores/memory_store.py` line 706 — `list_memories(user_id, scope_type, scope_id)` API
- `src/paperbot/infrastructure/services/subscription_service.py` line 123 — `get_scholar_configs()` API
- `config/scholar_subscriptions.yaml` — actual scholar config structure (name, semantic_scholar_id, keywords)
- `tests/integration/test_mcp_tool_calls.py` — existing test structure to extend
- `tests/unit/test_mcp_analyze_trends.py` — fake injection + `@pytest.mark.asyncio` unit test pattern
- `src/paperbot/mcp/tools/_audit.py` — audit helper (resources may optionally log, but no requirement to)

### Secondary (MEDIUM confidence)
- [FastMCP Resources & Templates](https://gofastmcp.com/servers/resources) — `@mcp.resource()` decorator API, URI template syntax, return types
- [MCP Python SDK Issue #141](https://github.com/modelcontextprotocol/python-sdk/issues/141) — URI templates appear in `list_resource_templates`, NOT `list_resources`
- `pyproject.toml` — `asyncio_mode = "strict"` confirmed; `@pytest.mark.asyncio` required

### Tertiary (LOW confidence)
- FastMCP 3.x standalone package docs — Phase 4 uses `mcp.server.fastmcp` (v1.x SDK), not standalone `fastmcp` package; API is compatible but version-specific behavior unverified

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — MCP package already in use, anyio already in use, no new deps
- Architecture (register pattern): HIGH — directly extends established Phase 2/3 tool pattern
- Backend APIs: HIGH — directly reading store source code, method signatures verified
- FastMCP resource decorator API: MEDIUM — verified from official docs + GitHub issue; actual `@mcp.resource` behavior with Python 3.9 not testable (same constraint as tools)
- Pitfalls: HIGH — derived from reading actual store code (user_id requirements, scope_id str cast, None returns)

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable application code; MCP API docs valid until major version bump)
