---
status: testing
phase: 04-mcp-resources
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md]
started: 2026-03-14T05:15:00Z
updated: 2026-03-14T05:15:00Z
---

## Current Test

number: 2
name: Track Resource Unit Tests Pass
expected: |
  Run `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py tests/unit/test_mcp_track_papers.py tests/unit/test_mcp_track_memory.py -x -q`. All 10 tests pass covering metadata (normal, not-found, invalid-id), papers (normal, empty, invalid-id), and memory (normal, empty, scope-filter, invalid-id).
awaiting: user response

## Tests

### 1. Cold Start Smoke Test
expected: Run `PYTHONPATH=src python -c "import paperbot.mcp.server; print('server imports OK')"`. Server module imports without errors, confirming all 4 resource modules and 9 tool modules load cleanly.
result: pass

### 2. Track Resource Unit Tests Pass
expected: Run `PYTHONPATH=src pytest tests/unit/test_mcp_track_metadata.py tests/unit/test_mcp_track_papers.py tests/unit/test_mcp_track_memory.py -x -q`. All 10 tests pass covering metadata (normal, not-found, invalid-id), papers (normal, empty, invalid-id), and memory (normal, empty, scope-filter, invalid-id).
result: [pending]

### 3. Scholars Resource Unit Tests Pass
expected: Run `PYTHONPATH=src pytest tests/unit/test_mcp_scholars.py -x -q`. Both tests pass covering normal scholar list return and FileNotFoundError handling.
result: [pending]

### 4. Integration Tests Pass (Resources + Tools)
expected: Run `PYTHONPATH=src pytest tests/integration/test_mcp_tool_calls.py -x -q`. All 34 tests pass (31 tool tests + 3 new resource tests). TestMCPResourceListing confirms all 4 resources are listed, registered in server.py, and have correct signatures.
result: [pending]

### 5. Resource Module Pattern Consistency
expected: Run `PYTHONPATH=src python -c "from paperbot.mcp.resources import track_metadata, track_papers, track_memory, scholars; assert all(hasattr(m, 'register') and callable(m.register) for m in [track_metadata, track_papers, track_memory, scholars]); print('All 4 resources have register()')"`. All 4 resource modules export a callable `register()` function following the established pattern.
result: [pending]

## Summary

total: 5
passed: 1
issues: 0
pending: 4
skipped: 0

## Gaps

[none yet]
