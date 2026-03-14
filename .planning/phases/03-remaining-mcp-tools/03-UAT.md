---
status: testing
phase: 03-remaining-mcp-tools
source: [03-01-SUMMARY.md, 03-02-SUMMARY.md, 03-03-SUMMARY.md]
started: 2026-03-14T05:00:00Z
updated: 2026-03-14T05:00:00Z
---

## Current Test

number: 1
name: All 9 MCP tools registered in server
expected: |
  Run `PYTHONPATH=src python -c "import paperbot.mcp.server; print([t for t in dir(paperbot.mcp.server.mcp) if not t.startswith('_')])"` or inspect server.py — all 9 tools should be importable without errors: paper_search, paper_judge, paper_summarize, relevance, analyze_trends, check_scholar, get_research_context, save_to_memory, export_to_obsidian.
awaiting: user response

## Tests

### 1. All 9 MCP tools registered in server
expected: Running `PYTHONPATH=src python -c "import paperbot.mcp.server"` succeeds without import errors. server.py contains 9 `.register(mcp)` calls for all tools.
result: [pending]

### 2. analyze_trends returns trend analysis
expected: Calling `_analyze_trends_impl(topic="llms", papers=[{"title": "Paper A"}])` with a valid TrendAnalyzer returns a dict with keys `trend_analysis`, `topic`, `paper_count`. Unit test `test_mcp_analyze_trends.py` passes.
result: [pending]

### 3. analyze_trends degrades gracefully when LLM unavailable
expected: When TrendAnalyzer returns empty string, `_analyze_trends_impl` returns `{"degraded": True, ...}` instead of raising an error. Unit test covers this path.
result: [pending]

### 4. check_scholar returns scholar info and papers
expected: Calling `_check_scholar_impl(scholar_name="Test")` with a valid S2 client returns a dict with `scholar` (name, authorId, hIndex) and `recent_papers` list. Unit test passes.
result: [pending]

### 5. check_scholar degrades when scholar not found
expected: When SemanticScholarClient returns empty authors list, `_check_scholar_impl` returns `{"degraded": True, "scholar": None, "recent_papers": []}` instead of crashing.
result: [pending]

### 6. get_research_context returns context pack
expected: Calling `_get_research_context_impl(query="transformers")` with a ContextEngine returns a dict with `papers`, `memories`, `track`, `stage` keys. Defaults to offline mode.
result: [pending]

### 7. save_to_memory persists content with kind validation
expected: Calling `_save_to_memory_impl(content="Finding X", kind="note")` returns `{"saved": True, "created": 1, ...}`. Invalid kind (e.g. "research_note") defaults to "note" with a warning instead of erroring.
result: [pending]

### 8. export_to_obsidian returns markdown with frontmatter
expected: Calling `_export_to_obsidian_impl(title="Paper A", abstract="...")` returns `{"markdown": str}` where the markdown contains YAML frontmatter delimiters `---` and the paper title. No filesystem writes.
result: [pending]

### 9. All tools log calls via audit helper
expected: Each of the 5 new tools calls `log_tool_call()` in both success and exception paths. Integration test `test_all_tool_events_have_consistent_structure` validates all 9 tools emit `workflow="mcp"`, `stage="tool_call"`.
result: [pending]

### 10. Full test suite passes
expected: Running `PYTHONPATH=src pytest tests/unit/test_mcp_analyze_trends.py tests/unit/test_mcp_check_scholar.py tests/unit/test_mcp_get_research_context.py tests/unit/test_mcp_save_to_memory.py tests/unit/test_mcp_export_to_obsidian.py tests/integration/test_mcp_tool_calls.py -v` passes all 46 tests (15 unit + 31 integration).
result: [pending]

## Summary

total: 10
passed: 0
issues: 0
pending: 10
skipped: 0

## Gaps

[none yet]
