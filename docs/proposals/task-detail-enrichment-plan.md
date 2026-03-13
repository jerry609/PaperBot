# Task Detail View Enrichment — Design Proposal

> Borrow Kimi's **typed content blocks + visible reasoning** and vibe-kanban's **diff view + workspace drill-down** to replace the current flat-log task dialog with a two-layer detail experience.

## Current State (what we have)

Clicking a task card opens `TaskDetailDialog` with 4 tabs:
- **Overview**: description + AI feedback + error (plain text)
- **Subtasks**: checklist dots
- **Logs**: raw execution log (monospace terminal style)
- **Files**: flat file list (extracted from codex output)

**Problems:**
1. Logs are a wall of undifferentiated text — no structure, no reasoning visibility
2. Files tab just lists filenames — no diff, no content preview
3. No way to see *what the agent was thinking* vs *what commands it ran* vs *what code it wrote*
4. No way to drill into a specific file's changes

---

## Proposed Design: Two-Layer Detail View

### Layer 1: Task Overview Panel (click card → opens dialog)

Inspired by **vibe-kanban Figure 1**. Replace current tabs with a structured single-page overview:

```
┌─────────────────────────────────────────────────────────────────┐
│  Task-3 · "Implement data loader"                    ● Running  │
│  codex · 72% complete · 3m 42s elapsed                     ✕    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Description ─────────────────────────────────────────────┐  │
│  │ Parse CSV data files, handle missing values, normalize... │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Subtasks ────────────────────────────────────────────────┐  │
│  │ ● Read CSV with pandas          ✅                        │  │
│  │ ● Handle missing values         ✅                        │  │
│  │ ● Normalize columns             ⬜ (in progress)          │  │
│  │ ● Write unit tests              ⬜                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Workspace ───────────────────────────────── + ∞ ▾ ───┐     │
│  │  🟢 Active   "Implement data loader"                   │     │
│  │  3m ago · 3 files · +187 -12          No PR created    │     │
│  │                                           [View →]     │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌─ AI Review ───────────────────────────────────────────────┐  │
│  │ ✅ Approved — "Clean implementation, handles edge cases"  │  │
│  │ Suggestions: [Consider adding type hints to normalize()]  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Human Review ────────────────────────────────────────────┐  │
│  │  [Approve ✓]  [Reject ✗]  [Request Changes ↻]            │  │
│  │  Comments: _________________________________              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key sections:**
1. **Header** — Title, status badge, assignee, progress %, elapsed time
2. **Description** — Task description (collapsible)
3. **Subtasks** — Checkbox list with completion state
4. **Workspace card** — Shows file count, net line changes (`+187 -12`), time since last change. **Click → opens Layer 2**
5. **AI Review** — Structured review feedback (approved/rejected, suggestions list)
6. **Human Review** — Action buttons for approve/reject/request changes (existing feature, better placement)

### Layer 2: Workspace / Thinking Process View (click Workspace card → drill-in)

Inspired by **vibe-kanban Figure 2** + **Kimi's typed blocks**. This is the core innovation.

```
┌─────────────────────────────────────────────────────────────────┐
│  Task-3 / workspace                                   ⤢  ✕     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Thinking Process (typed content blocks) ─────────────────┐  │
│  │                                                           │  │
│  │  ℹ️  System initialized with model: codex-mini-latest      │  │
│  │                                                           │  │
│  │  💭 "The task asks me to implement a CSV data loader.      │  │
│  │      I'll use pandas for CSV parsing and add handling     │  │
│  │      for missing values with fillna/dropna."              │  │
│  │                                                           │  │
│  │  ⚡ ls /home/user/paper-slug/src/                          │  │
│  │  🟢 data/  models/  utils/  __init__.py                   │  │
│  │                                                           │  │
│  │  ⚡ pip list 2>/dev/null | grep pandas                     │  │
│  │  🟢 pandas 2.1.4                                          │  │
│  │                                                           │  │
│  │  💭 "pandas is available. I'll create data_loader.py      │  │
│  │      in src/data/ with the core loading logic."           │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ 📄 src/data/data_loader.py  +87               ▾     │ │  │
│  │  │  1  + import pandas as pd                            │ │  │
│  │  │  2  + import numpy as np                             │ │  │
│  │  │  3  +                                                │ │  │
│  │  │  4  + class DataLoader:                              │ │  │
│  │  │  5  +     def __init__(self, path: str):             │ │  │
│  │  │  6  +         self.path = path                       │ │  │
│  │  │  ...  (expandable)                                   │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  💭 "Now I'll add unit tests to verify the loader."       │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ 📄 tests/test_data_loader.py  +42                    │ │  │
│  │  │  1  + import pytest                                  │ │  │
│  │  │  2  + from src.data.data_loader import DataLoader    │ │  │
│  │  │  ...                                                 │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ⚡ python -m pytest tests/test_data_loader.py -v         │  │
│  │  🟢 3 passed, 0 failed                                   │  │
│  │                                                           │  │
│  │  ✅ task_done: "Implemented CSV data loader with pandas,  │  │
│  │     handles missing values, 3 tests passing."             │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  [Open Workspace]   ↑  ❋  ☰  ↻           [Latest ▾]           │
│                                                                 │
│  Continue working on this task...                     👁        │
│  ┌───────────────────────────────────────┐  ⚙ Opus·High  📎 ▸ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Typed Content Blocks (from Kimi)

The execution log entries are currently flat text. Re-parse them into **5 block types**:

| Block Type | Icon | Visual Style | Source |
|-----------|------|-------------|--------|
| **ThinkBlock** | 💭 | Italic, light indigo background, rounded | Agent reasoning text |
| **ToolBlock** | ⚡ | Monospace, dark background, with status dot (🟢/🔴) | Shell commands + output |
| **DiffBlock** | 📄 | Green/red highlighted code diff, collapsible | File write/edit operations |
| **InfoBlock** | ℹ️ | Gray text, smaller font | System messages, model init |
| **ResultBlock** | ✅/❌ | Green/red border, summary text | task_done / verification result |

### Mapping from existing `execution_log` entries

The backend already sends structured log entries with `phase` and `event` fields:

```
phase=tool, event=shell_exec    →  ToolBlock
phase=tool, event=file_write    →  DiffBlock  (need file content from sandbox)
phase=codex, event=thinking     →  ThinkBlock (NEW — need backend change)
phase=codex, event=response     →  ThinkBlock
phase=system, event=*           →  InfoBlock
phase=verify, event=result      →  ResultBlock
phase=review, event=*           →  ResultBlock
```

---

## Diff View (from vibe-kanban)

Each `DiffBlock` shows:
- **Header**: filename + net line change badge (`+87` green, `-12` red)
- **Body**: Collapsible unified diff with syntax highlighting
  - Added lines: green background (`bg-emerald-50`)
  - Removed lines: red background (`bg-red-50`)
  - Context lines: neutral
- **Click filename** → opens full file in the existing `SandboxFileViewerDialog`

Since our Codex workers write files to the sandbox, we can:
1. Track file snapshots before/after each tool call (backend enhancement)
2. Or simpler: show the full file content with all lines as "added" (green) for new files, which is what vibe-kanban does in Figure 2

---

## Bottom Action Bar (from vibe-kanban)

The bottom bar provides contextual actions:

```
┌──────────────────────────────────────────────────────────────┐
│ [Open Workspace]  ↑ ❋ ☰ ↻                    [Latest ▾]     │
├──────────────────────────────────────────────────────────────┤
│ Continue working on this task...                    👁       │
│                                         ⚙ Opus·High  📎  ▸  │
└──────────────────────────────────────────────────────────────┘
```

| Button | Function |
|--------|----------|
| **Open Workspace** | Opens sandbox file viewer for this task's working directory |
| **↑** (scroll top) | Jump to start of thinking process |
| **❋** (focus) | Toggle auto-scroll to latest block |
| **☰** (outline) | Show block outline / jump to specific step |
| **↻** (refresh) | Reload execution log from backend |
| **Latest ▾** | Filter: show all blocks / only latest / only diffs |
| **Continue working...** | Text input for human-in-the-loop interaction (future: send instructions to running agent) |
| **Model selector** | Shows which model executed this task |

For **Phase 1**, we implement: Open Workspace, scroll top, auto-scroll toggle, Latest filter.
For **Phase 2** (future): Continue working input, model selector, outline view.

---

## Implementation Plan

### Backend Changes

**File: `src/paperbot/api/routes/agent_board.py`**

1. **Enrich execution log entries** with a `block_type` field:
   ```python
   def _append_task_log(task, phase, event, message, level="info", **extra):
       block_type = _infer_block_type(phase, event)
       entry = {
           "id": ..., "timestamp": ..., "phase": phase, "event": event,
           "message": message, "level": level,
           "block_type": block_type,  # NEW: "think" | "tool" | "diff" | "info" | "result"
           **extra,
       }
   ```

2. **Capture file diffs** — When codex writes a file, include the filename and line count in the log entry:
   ```python
   # In sandbox tool executor, after file_write:
   _append_task_log(task, "tool", "file_write", f"Wrote {path}",
       block_type="diff",
       file_path=path,
       lines_added=count_lines(content),
       content_preview=content[:2000],  # first 2KB for inline diff
   )
   ```

3. **Capture agent thinking** — Parse Codex response for reasoning blocks:
   ```python
   # In codex dispatcher, extract thinking from response:
   for block in response.output:
       if block.type == "reasoning":
           _append_task_log(task, "codex", "thinking", block.summary,
               block_type="think")
   ```

4. **Add workspace stats endpoint** ✅:
   ```
   GET /api/agent-board/tasks/{task_id}/workspace-stats
   → { files_changed: 3, lines_added: 187, lines_removed: 12, last_change: "2024-..." }
   ```

### Frontend Changes

**New file: `web/src/components/studio/TaskDetailPanel.tsx`**

Replace `TaskDetailDialog` with a two-layer component:

```
TaskDetailPanel (Layer 1 - Overview)
├── TaskHeader (title, status, progress, elapsed)
├── TaskDescription (collapsible)
├── SubtaskChecklist
├── WorkspaceCard (click → opens Layer 2)
│   └── shows: file count, +/- lines, time since last change
├── AIReviewSection
└── HumanReviewSection

TaskWorkspaceView (Layer 2 - Thinking Process)
├── ThinkingTimeline
│   ├── ThinkBlock
│   ├── ToolBlock
│   ├── DiffBlock (collapsible, syntax-highlighted)
│   ├── InfoBlock
│   └── ResultBlock
└── BottomActionBar
    ├── OpenWorkspace button
    ├── ScrollTop / AutoScroll toggle
    └── Filter dropdown (All / Latest / Diffs only)
```

**New file: `web/src/components/studio/blocks/`**

```
blocks/
├── ThinkBlock.tsx      — 💭 Indigo bg, italic reasoning text
├── ToolBlock.tsx       — ⚡ Dark bg, monospace, status dot
├── DiffBlock.tsx       — 📄 Green/red diff, collapsible, click → file viewer
├── InfoBlock.tsx       — ℹ️ Gray, compact system messages
└── ResultBlock.tsx     — ✅/❌ Bordered summary card
```

### File Change Summary

| File | Change | Priority |
|------|--------|----------|
| `src/paperbot/api/routes/agent_board.py` | Add `block_type` to log entries, capture thinking + file diffs | P0 |
| `src/paperbot/infrastructure/swarm/codex_dispatcher.py` | Extract reasoning blocks from Codex response | P0 |
| `src/paperbot/infrastructure/swarm/sandbox_tool_executor.py` | Capture file content on write for diff preview | P0 |
| `web/src/components/studio/blocks/*.tsx` | 5 new block components | P0 |
| `web/src/components/studio/TaskDetailPanel.tsx` | New two-layer detail component | P0 |
| `web/src/components/studio/AgentBoard.tsx` | Replace `TaskDetailDialog` with `TaskDetailPanel` | P1 |
| `web/src/components/studio/AgentBoardNodes.tsx` | Add workspace stats to task card (file count, +/- lines) | P1 |
| `web/src/lib/store/studio-store.ts` | Add `blockType` to execution log entry type | P1 |

### Phasing

**Phase 1 (Core blocks + Layer 2): DONE**
- [x] Backend: add `block_type` field, capture file_write content
- [x] Frontend: 5 block components + ThinkingTimeline + Layer 2 workspace view
- [x] Wire up: click Workspace card in overview → opens Layer 2

**Phase 2 (Polish + interaction): DONE**
- [x] Bottom action bar with filter/scroll/auto-scroll controls
- [x] Workspace stats on task card (`+187 -12`)
- [x] Agent thinking capture (`on_think` callback from Codex dispatcher)
- [x] Block type counts in bottom bar
- [ ] Syntax highlighting in DiffBlock (use Monaco or highlight.js) — future
- [ ] Continue working input (send instructions to agent) — future

**Phase 3 (Future):**
- True unified diffs (before/after snapshots)
- Block outline / jump-to navigation
- Real-time streaming blocks during execution
