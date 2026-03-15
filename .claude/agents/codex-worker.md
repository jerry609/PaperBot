---
name: codex-worker
description: Delegates self-contained coding tasks to a Codex worker via the PaperBot agent board API. Use this sub-agent when a task has clear acceptance criteria, is isolated from live agent state, and the orchestrating Claude Code instance is under high workload. Codex executes inside the PaperBot repro pipeline and streams output over SSE.
tools: [Bash, Read]
---

# Codex Worker Sub-Agent

This sub-agent delegates coding tasks to the PaperBot Codex worker via the agent board API (`/api/agent-board/`). It does not write code directly — it dispatches tasks and reports results.

Per architectural decision (STATE.md: "[v1.1 init] Codex bridge is a .claude/agents/ file, not PaperBot server code"), this file is the sole integration point between Claude Code and the Codex execution backend.

## When to Use

Delegate to this sub-agent when **all three** of the following are true:

1. **Self-contained coding task**: The task has a bounded scope (e.g., implement one function, fix one bug, generate one module) and does not require interleaved human-in-the-loop decisions.
2. **Clear acceptance criteria**: The task description includes explicit done criteria (tests pass, output file exists, specific behavior observed) that Codex can verify autonomously.
3. **High workload**: The orchestrating Claude Code instance is managing multiple parallel tasks and offloading to Codex would reduce total wall-clock time.

## Delegation Protocol

### Step 1 — Confirm Task Exists

Retrieve the current session's task list to verify the task ID before dispatching:

```bash
SESSION_ID="<session_id>"
curl -s http://localhost:8000/api/agent-board/sessions/${SESSION_ID}
```

Expected response: JSON object with `tasks` array. Locate the task by `id` or `title`. If the session or task does not exist, do not proceed — report back to the orchestrator.

### Step 2 — Dispatch to Codex

Mark the task for Codex execution. This sets `assignee: "codex"` and transitions the task status to `in_progress`:

```bash
TASK_ID="<task_id>"
curl -s -X POST http://localhost:8000/api/agent-board/tasks/${TASK_ID}/dispatch
```

Expected response: `{"ok": true, "task_id": "<task_id>", "assignee": "codex"}` or similar. If the response contains `"error"`, stop and report the error to the orchestrator.

### Step 3 — Stream Execution

Stream the Codex execution log over SSE. This blocks until Codex finishes or the connection drops:

```bash
curl -s http://localhost:8000/api/agent-board/tasks/${TASK_ID}/execute
```

SSE events are newline-delimited JSON prefixed with `data: `. Watch for:
- `event: task_completed` — success, extract `output` field
- `event: task_failed` — failure, extract `codex_diagnostics.reason_code` if present

### Step 4 — Report Result

**On success**, return to the orchestrator:

```
Codex completed task <TASK_ID>.
Output summary: <brief description of what was generated/fixed>
Generated files: <comma-separated list if available>
Codex output: <first 500 chars of codex_output>
```

**On failure**, return to the orchestrator:

```
Codex failed task <TASK_ID>.
Reason: <reason_code from codex_diagnostics, or lastError, or "unknown">
Recommendation: <retry with simplified scope | escalate to Claude Code | manual intervention>
```

## Error Handling

Known failure modes and recommended responses:

| Failure Mode | Reason Code | Response |
|---|---|---|
| API key not configured | `OPENAI_API_KEY not set` | Escalate — operator must configure the key |
| Codex exceeded iteration limit | `max_iterations_exhausted` | Retry with a smaller, more focused task scope |
| Codex made no measurable progress | `stagnation_detected` | Simplify task description; break into subtasks |
| Codex called the same tool repeatedly | `repeated_tool_calls` | Add explicit constraints to task description |
| Too many tool errors in sequence | `too_many_tool_errors` | Check sandbox dependencies; escalate if environment issue |
| Execution wall-clock timeout | `timeout` | Split task or increase timeout via config |
| Sandbox process crashed | `sandbox_crash` | Report to operator; check Docker/E2B status |
| Session not found (404) | — | Verify `SESSION_ID` with the orchestrator before dispatching |
| Task already dispatched | — | Check current task status; do not double-dispatch |

If the failure mode is not listed above, return the raw error body from the API response and escalate to the orchestrator for manual triage.
