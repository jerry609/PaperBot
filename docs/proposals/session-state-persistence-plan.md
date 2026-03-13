# Session State Persistence — Implementation Plan (Approach A)

> **Goal**: After page refresh, restore Progress (agent tasks) and Context Pack state from the backend, rather than losing them.

## Problem

The Zustand store (`studio-store.ts`) only persists `papers` to `localStorage`. These fields are **in-memory only** and reset on refresh:

| Field | What the user sees disappear |
|-------|------------------------------|
| `boardSessionId` | Entire agent board disconnects |
| `agentTasks` | Task cards / progress bars |
| `pipelinePhase` | Pipeline status indicator |
| `contextPack` | Context Pack panel content |
| `sandboxFiles` | Sidebar file tree |
| `e2eState` | E2E execution results |

## Design: Persist IDs → Reload from Backend

```
┌──────────────────────────────────────────────────────────┐
│                     Page Refresh                          │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│ 1. Read localStorage:                                     │
│    - boardSessionId                                       │
│    - contextPackId (per selected paper)                   │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│ 2. If boardSessionId exists:                              │
│    GET /api/agent-board/sessions/{id}         ← NEW      │
│    → { session, tasks[], pipelinePhase, sandboxFiles[] }  │
│    → Hydrate: agentTasks, pipelinePhase, sandboxFiles,    │
│              e2eState, boardSessionId                     │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│ 3. If contextPackId exists:                               │
│    GET /api/research/repro/context/{id}      ← EXISTS    │
│    → Hydrate: contextPack                                 │
└──────────────────────────────────────────────────────────┘
```

## Changes

### 1. Backend: Add `GET /api/agent-board/sessions/{session_id}` endpoint

**File**: `src/paperbot/api/routes/agent_board.py`

Add a new GET endpoint that returns the full session snapshot:

```python
@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Return full session state for frontend hydration."""
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Derive pipeline phase from session state
    phase = _derive_pipeline_phase(session)

    # Build sandbox file tree (if sandbox is active)
    sandbox_files = []
    if session.sandbox_id:
        try:
            sandbox_files = await _get_sandbox_tree(session)
        except Exception:
            pass  # Non-critical — sidebar just stays empty

    return {
        "session": session.to_dict(),
        "tasks": [t.model_dump() for t in session.tasks],
        "pipeline_phase": phase,
        "sandbox_files": sandbox_files,
    }
```

**Helper — `_derive_pipeline_phase(session)`**:

Infer pipeline phase from task statuses + lifecycle events:

```python
def _derive_pipeline_phase(session: BoardSession) -> str:
    events = session.lifecycle_events
    if not session.tasks:
        return "idle"

    # Check lifecycle events (most recent first)
    event_types = [e.get("event") for e in reversed(events)]

    if "pipeline_completed" in event_types or "pipeline_failed" in event_types:
        return "completed" if "pipeline_completed" in event_types else "failed"
    if "e2e_start" in event_types:
        return "e2e"
    if any(e.startswith("task_") for e in event_types if e):
        return "executing"
    if "plan_done" in event_types:
        return "planned"

    # Fallback: infer from task statuses
    statuses = {t.status for t in session.tasks}
    if statuses <= {"done"}:
        return "completed"
    if "in_progress" in statuses or "ai_review" in statuses:
        return "executing"
    return "planned"
```

**Helper — `_get_sandbox_tree(session)`**:

Reuse the existing sandbox tree logic from the `/sandbox/tree` endpoint:

```python
async def _get_sandbox_tree(session: BoardSession) -> list:
    """Fetch file tree from active sandbox."""
    # Reuse existing tree-building logic from the sandbox/tree endpoint
    mgr = PersistentSandboxManager()
    # ... (delegate to existing implementation)
```

### 2. Frontend: Persist `boardSessionId` and `contextPackId` to localStorage

**File**: `web/src/lib/store/studio-store.ts`

Add a second localStorage key for session IDs:

```typescript
const SESSION_STORAGE_KEY = "studio-active-session"

interface PersistedSessionIds {
  boardSessionId: string | null
  contextPackId: string | null
  selectedPaperId: string | null
}
```

**Save on change** — update `setBoardSessionId` and `setContextPack`:

```typescript
setBoardSessionId: (id) => {
  set({ boardSessionId: id })
  // Persist
  const prev = _loadSessionIds()
  _saveSessionIds({ ...prev, boardSessionId: id })
},
```

Similarly, when `contextPack` is set, persist `contextPack.context_pack_id`.

**Helpers**:

```typescript
function _loadSessionIds(): PersistedSessionIds {
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY)
    return raw ? JSON.parse(raw) : { boardSessionId: null, contextPackId: null, selectedPaperId: null }
  } catch {
    return { boardSessionId: null, contextPackId: null, selectedPaperId: null }
  }
}

function _saveSessionIds(ids: PersistedSessionIds): void {
  localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(ids))
}
```

### 3. Frontend: Add `restoreSession()` action to store

**File**: `web/src/lib/store/studio-store.ts`

```typescript
restoreSession: async () => {
  const { boardSessionId, contextPackId, selectedPaperId } = _loadSessionIds()

  if (selectedPaperId) {
    set({ selectedPaperId })
  }

  // Restore context pack
  if (contextPackId) {
    try {
      const res = await fetch(backendUrl(`/api/research/repro/context/${contextPackId}`))
      if (res.ok) {
        const payload = await res.json()
        const pack = normalizePack(payload)
        if (pack) set({ contextPack: pack })
      }
    } catch { /* non-critical */ }
  }

  // Restore agent board session
  if (boardSessionId) {
    try {
      const res = await fetch(backendUrl(`/api/agent-board/sessions/${boardSessionId}`))
      if (res.ok) {
        const data = await res.json()
        set({
          boardSessionId,
          agentTasks: data.tasks ?? [],
          pipelinePhase: data.pipeline_phase ?? "idle",
          sandboxFiles: data.sandbox_files ?? [],
        })
      } else if (res.status === 404) {
        // Session expired or deleted — clear persisted IDs
        _saveSessionIds({ boardSessionId: null, contextPackId: null, selectedPaperId: null })
      }
    } catch { /* non-critical */ }
  }
},
```

### 4. Frontend: Call `restoreSession()` on page mount

**File**: `web/src/app/studio/page.tsx`

```typescript
useEffect(() => {
  loadPapers()
  restoreSession()  // ← ADD
}, [])
```

### 5. Clear persisted IDs on session reset

When the user starts a new session or clears the board:

```typescript
clearAgentTasks: () => {
  set({ agentTasks: [], boardSessionId: null, pipelinePhase: "idle", sandboxFiles: [] })
  const prev = _loadSessionIds()
  _saveSessionIds({ ...prev, boardSessionId: null })
},
```

## File Change Summary

| File | Change |
|------|--------|
| `src/paperbot/api/routes/agent_board.py` | Add `GET /sessions/{id}` endpoint + `_derive_pipeline_phase()` helper |
| `web/src/lib/store/studio-store.ts` | Add localStorage helpers, persist IDs in `setBoardSessionId`/`setContextPack`, add `restoreSession()` action |
| `web/src/app/studio/page.tsx` | Call `restoreSession()` on mount |

## Edge Cases

| Case | Handling |
|------|----------|
| Session 404 (deleted/expired) | Clear persisted IDs, show clean slate |
| Backend unreachable | Silently fail, show empty state (same as fresh load) |
| Context pack deleted but session exists | Board restores without context pack panel |
| Multiple tabs | Last-write-wins on localStorage (acceptable — single-user tool) |
| Running pipeline interrupted by refresh | Phase shows as "executing" but SSE stream is lost. User can re-trigger `/run` (backend is idempotent per checkpoint). Future: add "Resume" button. |

## Non-Goals (for now)

- **Auto-reconnect SSE on refresh**: If a pipeline was mid-execution, we restore the last-known state but don't reconnect the stream. The user can check status or re-run.
- **Multi-session switching**: One active session at a time. Session history/switching is a future feature.
- **Offline cache of context pack**: We re-fetch from backend; no client-side caching beyond the ID.
