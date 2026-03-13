# E2E Execution Bugfix Repair Plan

> Generated: 2026-03-11
> Branch: `feature/AgentSwarm`
> Scope: `src/paperbot/infrastructure/swarm/`

---

## Overview

Analysis of the Agent Board's E2E execution pipeline revealed **6 bugs** across 4 files. This plan specifies the fix for each bug, ordered by priority, with affected files, test coverage requirements, and acceptance criteria.

---

## Bug #1 (Critical): Dependencies not reinstalled after repair

**File:** `e2e_execution.py:220`
**Root cause:** `attempt == 0` guard skips `pip install` on retry attempts, so newly added dependencies are never installed.

### Fix

```python
# BEFORE
if policy.install_deps and attempt == 0:
    deps_cmd = _build_deps_command(sandbox, paper_slug)

# AFTER
if policy.install_deps:
    deps_cmd = _build_deps_command(sandbox, paper_slug)
```

Remove the `attempt == 0` condition entirely. `pip install -q` is idempotent — already-installed packages are skipped quickly, so the cost of re-running is negligible.

### Tests

- `test_e2e_execution.py`: Add `test_deps_reinstalled_on_repair_attempt` — mock sandbox, run `run_e2e` with `attempt=1`, assert `run_in_paper` is called with the deps command.

### Acceptance

- `run_e2e(attempt=1)` calls `pip install -r requirements.txt` when `install_deps=True`.

---

## Bug #2 (Medium): Entry point detection matches comments

**File:** `e2e_execution.py:157-162`
**Root cause:** Substring match `'if __name__' in content` matches commented-out lines.

### Fix

```python
import re

# BEFORE
if content and 'if __name__' in content:
    return f

# AFTER
if content and re.search(r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', content, re.MULTILINE):
    return f
```

### Tests

- `test_e2e_execution.py`: Add `test_detect_entry_point_ignores_commented_main_guard` — file with `# if __name__ == "__main__":` should NOT be detected. File with actual guard should be detected.

### Acceptance

- Files containing only commented `if __name__` guards are skipped.
- Files with real guards (any indentation) are still detected.

---

## Bug #3 (Medium): Shell injection in find/grep commands

**Files:**
- `shared_sandbox.py:89-91` — `list_files()`
- `shared_sandbox.py:107-108` — `list_files_recursive()`
- `shared_sandbox.py:123-124` — `search_files()`

**Root cause:** `paper_slug` flows into shell commands without escaping. Slugs with spaces or metacharacters can break or exploit the command.

### Fix

Add `import shlex` to `shared_sandbox.py` and quote all interpolated values:

```python
# list_files (line 91)
# BEFORE
f"find {target} -maxdepth 1 -not -name '.*' 2>/dev/null | head -100"
# AFTER
f"find {shlex.quote(target)} -maxdepth 1 -not -name '.*' 2>/dev/null | head -100"

# list_files_recursive (line 108)
# BEFORE
f"find {root} -type f 2>/dev/null"
# AFTER
f"find {shlex.quote(root)} -type f 2>/dev/null"

# search_files (line 124)
# BEFORE
f"grep -rn --include='{glob}' '{pattern}' {root} 2>/dev/null | head -50"
# AFTER
f"grep -rn --include={shlex.quote(glob)} {shlex.quote(pattern)} {shlex.quote(root)} 2>/dev/null | head -50"
```

Also apply to `ensure_paper_dir` (line 195):
```python
f"mkdir -p {shlex.quote(self.paper_root(slug))}"
```

### Tests

- `test_shared_sandbox.py`: Add `test_list_files_with_special_chars_in_slug` — slug containing spaces and single quotes should not break the command.

### Acceptance

- All shell-interpolated paths are wrapped in `shlex.quote()`.
- Slugs with spaces, quotes, semicolons produce correct commands (no injection).

---

## Bug #4 (Medium): Output truncation loses error info

**File:** `e2e_execution.py:231-232`
**Root cause:** Truncation keeps the first N chars (progress logs), discards the tail (actual errors).

### Fix

Add a helper function and use it for both stdout and stderr:

```python
def _tail_truncate(text: str, max_chars: int) -> str:
    """Keep the LAST max_chars characters, where errors typically appear."""
    if len(text) <= max_chars:
        return text
    return "...[truncated]\n" + text[-max_chars:]
```

```python
# BEFORE
stdout = (result.logs or "")[:_MAX_OUTPUT_CHARS]
stderr = (result.error or "")[:_MAX_OUTPUT_CHARS]

# AFTER
stdout = _tail_truncate(result.logs or "", _MAX_OUTPUT_CHARS)
stderr = _tail_truncate(result.error or "", _MAX_OUTPUT_CHARS)
```

Also update `_build_diagnosis_prompt` (line 263) to use the same tail-truncation:

```python
# BEFORE
f"```\n{output_section[:6000]}\n```\n\n"
# AFTER
f"```\n{_tail_truncate(output_section, 6000)}\n```\n\n"
```

### Tests

- `test_e2e_execution.py`: Add `test_output_truncation_keeps_tail` — 20K-char output with error at the end should preserve the error portion.

### Acceptance

- Errors at the end of output are preserved in the diagnosis prompt.
- Short outputs (< max) are returned unchanged.

---

## Bug #5 (Low): No exception handling in repair dispatch

**Files:**
- `verification.py:149-154`
- `e2e_execution.py:342-346`

**Root cause:** If `dispatch_with_sandbox_tools` raises (network error, API timeout), the coroutine crashes instead of gracefully returning the last result.

### Fix

**verification.py:**

```python
# BEFORE
await dispatcher.dispatch_with_sandbox_tools(
    task_id=f"repair-{attempt + 1}",
    prompt=repair_prompt,
    tool_executor=tool_executor,
    on_step=on_step,
)

# AFTER
try:
    await dispatcher.dispatch_with_sandbox_tools(
        task_id=f"repair-{attempt + 1}",
        prompt=repair_prompt,
        tool_executor=tool_executor,
        on_step=on_step,
    )
except Exception:
    log.exception("Repair dispatch failed at attempt %d", attempt + 1)
    break
```

**e2e_execution.py:**

```python
# BEFORE
tool_exec = tool_executor_factory()
await dispatcher.dispatch_with_sandbox_tools(
    task_id=f"e2e-repair-{attempt + 1}",
    prompt=diagnosis_prompt,
    tool_executor=tool_exec,
    on_step=on_step,
)

# AFTER
tool_exec = tool_executor_factory()
try:
    await dispatcher.dispatch_with_sandbox_tools(
        task_id=f"e2e-repair-{attempt + 1}",
        prompt=diagnosis_prompt,
        tool_executor=tool_exec,
        on_step=on_step,
    )
except Exception:
    log.exception("E2E repair dispatch failed at attempt %d", attempt)
    break
```

### Tests

- `test_e2e_execution.py`: Add `test_repair_dispatch_exception_returns_last_result` — mock dispatcher to raise `RuntimeError`, assert function returns the failed E2EResult without crashing.
- `test_verification.py`: Same pattern for `verify_and_repair`.

### Acceptance

- API errors during repair are logged and the loop exits gracefully.
- The last execution/verification result is returned (not an unhandled exception).

---

## Bug #6 (Low): `run_e2e_with_repair` can return undefined

**File:** `e2e_execution.py:349`
**Root cause:** If `max_repair_attempts + 1 == 0` (impossible with current defaults but possible via env override), `result` is never assigned.

### Fix

```python
# BEFORE
repair_history: List[Dict[str, Any]] = []
for attempt in range(policy.max_repair_attempts + 1):
    result = run_e2e(...)
    ...
return result  # type: ignore[possibly-undefined]

# AFTER
repair_history: List[Dict[str, Any]] = []
result: Optional[E2EResult] = None
for attempt in range(policy.max_repair_attempts + 1):
    result = run_e2e(...)
    ...
if result is None:
    return E2EResult(
        success=False,
        entry_point=policy.entry_point or "unknown",
        command=build_run_command(policy),
        exit_code=1,
        stderr="No execution attempts were made (max_repair_attempts < 0?).",
    )
return result
```

### Tests

- Not strictly needed (edge case), but can add a defensive test with `max_repair_attempts=-1`.

### Acceptance

- The `# type: ignore` comment is removed.
- Function always returns a valid `E2EResult`.

---

## Implementation Order

| Priority | Bug | File(s) | Estimated Effort |
|----------|-----|---------|-----------------|
| 1 | #1 — Deps reinstall | `e2e_execution.py` | 1 line change + 1 test |
| 2 | #4 — Tail truncation | `e2e_execution.py` | ~15 lines + 1 test |
| 3 | #3 — Shell escaping | `shared_sandbox.py` | ~8 line changes + 1 test |
| 4 | #2 — Regex guard | `e2e_execution.py` | 1 line change + 1 test |
| 5 | #5 — Exception handling | `e2e_execution.py`, `verification.py` | ~10 lines + 2 tests |
| 6 | #6 — Undefined result | `e2e_execution.py` | ~6 lines |

Total: ~40 lines of production code, ~6 new test cases.

---

## Validation

After all fixes, run:

```bash
PYTHONPATH=src pytest -q \
  tests/unit/test_e2e_execution.py \
  tests/unit/test_verification.py \
  tests/unit/test_sandbox_tool_executor.py \
  tests/unit/test_shared_sandbox.py
```

All existing tests must continue to pass. New tests must cover each fix.
