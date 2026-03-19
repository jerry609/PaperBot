# Phase 9: Three-Panel Dashboard - Research

**Researched:** 2026-03-15
**Domain:** React/Next.js resizable IDE layout, inline diff rendering, per-task file lists, Zustand state management
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DASH-01 | User can view agent orchestration in a three-panel IDE layout (tasks | activity | files) | `SplitPanels` component already implements a three-panel `ResizablePanelGroup` pattern with `localStorage` persistence; Phase 9 reuses it with agent-specific content |
| DASH-04 | User can resize panels in the three-panel layout to customize workspace | `react-resizable-panels` v4.0.11 is installed; `SplitPanels.tsx` already implements `onLayoutChange` + `localStorage` persistence; the pattern is proven and reusable |
| FILE-01 | User can view inline diffs showing what agents changed in each file | `DiffViewer` component + `computeDiff`/`DiffLine` utilities already exist in `web/src/components/studio/DiffViewer.tsx` and `web/src/lib/diff.ts`; Phase 9 wires these to the agent event store |
| FILE-02 | User can see a per-task file list showing created/modified files with status indicators | `computeWorkspaceStats` in `TaskDetailPanel.tsx` already extracts file names from `AgentTaskLog` entries; `DiffBlock` renders per-file rows; Phase 9 lifts this pattern into the agent-events store |
</phase_requirements>

---

## Summary

Phase 9 builds the three-panel IDE layout that is the primary UI for the agent orchestration dashboard. The three panels are: **tasks** (left rail — agent status summary, task list), **activity feed** (centre — scrolling real-time event feed built in Phase 8), and **files** (right — per-task file list with created/modified indicators and inline diff preview).

All the hard primitives are already present in the codebase. `SplitPanels` (`web/src/components/layout/SplitPanels.tsx`) is a production-quality three-panel layout that uses `react-resizable-panels` with `localStorage` persistence — it matches DASH-01 and DASH-04 exactly. `DiffViewer` + `computeDiff` cover FILE-01. `TaskDetailPanel`'s `computeWorkspaceStats` logic (which extracts file names from `AgentTaskLog` entries by looking for `blockType === "diff"` rows) covers the file-collection pattern for FILE-02. The Phase 8 Zustand store (`useAgentEventStore`) and SSE hook (`useAgentEvents`) feed the activity panel.

The primary new work is: (1) a new `/agent-dashboard` page that composes `SplitPanels` with the three content panels, (2) a **TasksPanel** component showing the per-agent status list from the store, (3) a **FileListPanel** component that reads `AgentAction`/`AgentTask` store data and renders a file list with status badges, and (4) an **InlineDiffPanel** component that wraps the existing `DiffViewer` for display inside the files panel.

**Primary recommendation:** Route the new page to `/agent-dashboard`. Compose `SplitPanels` with `storageKey="agent-dashboard"`, passing TasksPanel as `rail`, ActivityFeed (Phase 8) as `list`, and FileListPanel as `detail`. Mount `useAgentEvents()` once at the page root. All layout persistence is handled automatically by `SplitPanels`.

---

## Standard Stack

### Core (zero new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `react-resizable-panels` | ^4.0.11 | Three-panel resizable layout with drag handles | Already installed; already wrapped in `/components/ui/resizable.tsx`; v4 API confirmed from installed package |
| `SplitPanels` component | project (codebase) | Three-panel layout with `localStorage` persistence, collapse/expand, mobile fallback | Already battle-tested for `rail + list + detail` pattern — identical to what Phase 9 needs |
| `useAgentEventStore` | Phase 8 deliverable | Zustand store with feed, agentStatuses, toolCalls | All Phase 8 files exist and tests pass |
| `useAgentEvents` hook | Phase 8 deliverable | SSE connection to `/api/events/stream` | Mounts once at page root; child components read from store |
| `DiffViewer` component | `web/src/components/studio/DiffViewer.tsx` | Inline diff rendering with +/- stats header | Already implements LCS diff; just needs wiring |
| `computeDiff` / `DiffLine` | `web/src/lib/diff.ts` | LCS-based line diff algorithm | Already in production use by DiffViewer |
| `@radix-ui/react-scroll-area` | ^1.2.10 | Scrollable panel content | Already installed; used by Phase 8 ActivityFeed |
| `lucide-react` | ^0.562.0 | Status icons (FileEdit, FilePlus2, CheckCircle2, etc.) | Already the project's icon library |
| `zustand` | ^5.0.9 | Store for task + file state | Already used project-wide |
| `tailwindcss` | ^4 | All component styling | Project-wide CSS framework |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `@radix-ui/react-tooltip` | ^1.2.8 | Tooltip for truncated file paths | Already installed; use for long file names in FileListPanel |
| `framer-motion` | ^12.23.26 | Subtle entrance animation for new file entries | Already installed; optional, keep lightweight |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reusing `SplitPanels` | Custom ResizablePanelGroup | `SplitPanels` already handles mobile fallback, collapse state persistence, and collapse button toolbar — rebuilding duplicates proven work |
| `useAgentEventStore` for file state | New dedicated Zustand store | The existing store can be extended with file tracking fields, avoiding a second global store; keep all agent display state co-located |
| `DiffViewer` + `computeDiff` | `@monaco-editor/react` in diff mode | Monaco is installed but adds significant weight; `DiffViewer` is already present, lightweight, and sufficient for read-only diff display |
| `localStorage` via `SplitPanels` | `useDefaultLayout` from `react-resizable-panels` | Project's `SplitPanels` already persists via `localStorage` with a `storageKey` prop — consistent with existing usage in scholars/research pages |

**Installation:** No new packages needed.

---

## Architecture Patterns

### Recommended File Structure

```
web/src/
├── app/
│   └── agent-dashboard/
│       └── page.tsx                        # NEW: three-panel agent dashboard page
├── components/
│   └── agent-dashboard/                    # NEW: dashboard-specific components
│       ├── TasksPanel.tsx                  # NEW: left rail — agent status + task list
│       ├── FileListPanel.tsx               # NEW: right panel — per-task file list
│       └── InlineDiffPanel.tsx             # NEW: right panel — inline diff view (wraps DiffViewer)
└── lib/
    └── agent-events/
        ├── store.ts                        # MODIFIED: add FileTouchedEntry + file tracking actions
        └── types.ts                        # MODIFIED: add FileTouchedEntry type
```

**No changes to Phase 8 components** (ActivityFeed, AgentStatusPanel, ToolCallTimeline are reused as-is).

### Pattern 1: Three-Panel Page Layout

The page mounts `useAgentEvents()` once and composes `SplitPanels`:

```typescript
// web/src/app/agent-dashboard/page.tsx
"use client"

import { useAgentEvents } from "@/lib/agent-events/useAgentEvents"
import { SplitPanels } from "@/components/layout/SplitPanels"
import { TasksPanel } from "@/components/agent-dashboard/TasksPanel"
import { ActivityFeed } from "@/components/agent-events/ActivityFeed"
import { FileListPanel } from "@/components/agent-dashboard/FileListPanel"

export default function AgentDashboardPage() {
  // Mount SSE hook exactly once — child components read from Zustand store
  useAgentEvents()

  return (
    <div className="h-screen min-h-0 flex flex-col">
      <header className="border-b px-4 py-2 flex items-center gap-3 shrink-0">
        <h1 className="text-sm font-semibold">Agent Dashboard</h1>
      </header>
      <div className="flex-1 min-h-0">
        <SplitPanels
          storageKey="agent-dashboard"
          rail={<TasksPanel />}
          list={<ActivityFeed />}
          detail={<FileListPanel />}
        />
      </div>
    </div>
  )
}
```

**Why `SplitPanels`:** The `storageKey` prop drives all `localStorage` persistence for panel sizes and collapse state. The component already handles mobile fallback with a tab strip. DASH-04 is satisfied for free.

### Pattern 2: Extending the Agent Event Store for File Tracking

Add file-touched tracking to `useAgentEventStore` by listening for `file_change`/`tool_result` events that reference file paths. The store extension adds:
- A `filesTouched` record: `Record<run_id, FileTouchedEntry[]>` — per-run file list
- A `selectedRunId` string — which run's file list the detail panel shows
- Actions: `addFileTouched`, `setSelectedRunId`

```typescript
// web/src/lib/agent-events/types.ts — APPEND

export type FileChangeStatus = "created" | "modified"

export type FileTouchedEntry = {
  run_id: string
  path: string          // relative file path
  status: FileChangeStatus
  ts: string
  linesAdded?: number
  linesDeleted?: number
  diff?: string         // optional: unified diff string from payload
  oldContent?: string
  newContent?: string
}
```

**Why extend the existing store rather than creating a new one:** All agent display state is already co-located in `useAgentEventStore`. Adding file tracking there keeps the component tree simple — `FileListPanel` reads from the same store that `ActivityFeed` and `TasksPanel` already consume.

**How file events arrive:** The backend can emit `file_change` events via `make_event(type="file_change", payload={path, status, lines_added, diff})`. If no explicit `file_change` events flow yet, Phase 9 can derive file lists from `tool_result` events where `payload.tool == "write_file"` — the same pattern used by `DiffBlock.tsx` which checks `log.details?.tool === "write_file"`.

### Pattern 3: FileListPanel Component

```typescript
// web/src/components/agent-dashboard/FileListPanel.tsx
"use client"

import { useAgentEventStore } from "@/lib/agent-events/store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { FilePlus2, FileEdit, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { InlineDiffPanel } from "./InlineDiffPanel"
import type { FileTouchedEntry } from "@/lib/agent-events/types"

export function FileListPanel() {
  const filesTouched = useAgentEventStore((s) => s.filesTouched)
  const selectedRunId = useAgentEventStore((s) => s.selectedRunId)
  const setSelectedFile = useAgentEventStore((s) => s.setSelectedFile)
  const selectedFile = useAgentEventStore((s) => s.selectedFile)

  const files: FileTouchedEntry[] = selectedRunId
    ? (filesTouched[selectedRunId] ?? [])
    : Object.values(filesTouched).flat()

  if (files.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        No file changes yet
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {selectedFile ? (
        <InlineDiffPanel
          entry={selectedFile}
          onBack={() => setSelectedFile(null)}
        />
      ) : (
        <ScrollArea className="flex-1">
          <ul className="px-2 py-2 space-y-0.5">
            {files.map((entry) => (
              <li key={`${entry.run_id}-${entry.path}-${entry.ts}`}>
                <button
                  className="flex items-center gap-2 w-full rounded px-2 py-1.5 text-left hover:bg-accent transition-colors"
                  onClick={() => setSelectedFile(entry)}
                >
                  {entry.status === "created" ? (
                    <FilePlus2 className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
                  ) : (
                    <FileEdit className="h-3.5 w-3.5 shrink-0 text-amber-500" />
                  )}
                  <span className="font-mono text-xs truncate flex-1 text-foreground">
                    {entry.path}
                  </span>
                  {entry.linesAdded != null && entry.linesAdded > 0 && (
                    <span className="text-[10px] text-emerald-600 shrink-0">
                      +{entry.linesAdded}
                    </span>
                  )}
                  <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
                </button>
              </li>
            ))}
          </ul>
        </ScrollArea>
      )}
    </div>
  )
}
```

### Pattern 4: InlineDiffPanel (wraps DiffViewer)

```typescript
// web/src/components/agent-dashboard/InlineDiffPanel.tsx
"use client"

import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { DiffViewer } from "@/components/studio/DiffViewer"
import type { FileTouchedEntry } from "@/lib/agent-events/types"

export function InlineDiffPanel({
  entry,
  onBack,
}: {
  entry: FileTouchedEntry
  onBack: () => void
}) {
  // If no diff content available, show a message
  if (!entry.oldContent && !entry.newContent && !entry.diff) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex items-center gap-2 border-b px-3 py-2">
          <Button variant="ghost" size="icon" onClick={onBack} className="h-6 w-6">
            <ArrowLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="font-mono text-xs truncate">{entry.path}</span>
        </div>
        <div className="flex flex-1 items-center justify-center text-xs text-muted-foreground">
          Diff not available for this change
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b px-3 py-2 shrink-0">
        <Button variant="ghost" size="icon" onClick={onBack} className="h-6 w-6">
          <ArrowLeft className="h-3.5 w-3.5" />
        </Button>
        <span className="font-mono text-xs truncate">{entry.path}</span>
      </div>
      <div className="flex-1 min-h-0 overflow-hidden">
        <DiffViewer
          oldValue={entry.oldContent ?? ""}
          newValue={entry.newContent ?? entry.diff ?? ""}
          filename={entry.path}
        />
      </div>
    </div>
  )
}
```

### Pattern 5: TasksPanel (left rail)

The left rail shows the per-agent status map and a summary of active runs. It reads from `useAgentEventStore`:

```typescript
// web/src/components/agent-dashboard/TasksPanel.tsx
"use client"

import { useAgentEventStore } from "@/lib/agent-events/store"
import { AgentStatusPanel } from "@/components/agent-events/AgentStatusPanel"
import { ScrollArea } from "@/components/ui/scroll-area"

export function TasksPanel() {
  const feed = useAgentEventStore((s) => s.feed)
  const selectedRunId = useAgentEventStore((s) => s.selectedRunId)
  const setSelectedRunId = useAgentEventStore((s) => s.setSelectedRunId)

  // Derive distinct run_ids from the feed, most recent first
  const runs = Array.from(
    new Map(
      feed
        .filter((item) => item.raw.run_id)
        .map((item) => [String(item.raw.run_id), item])
    ).values()
  ).slice(0, 20)

  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-3 py-2 shrink-0">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Agents
        </p>
      </div>
      <AgentStatusPanel compact />
      <div className="border-b" />
      <div className="px-3 py-2 shrink-0">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Runs
        </p>
      </div>
      <ScrollArea className="flex-1">
        <ul className="px-2 pb-2 space-y-0.5">
          {runs.map((item) => (
            <li key={String(item.raw.run_id)}>
              <button
                className={`w-full rounded px-2 py-1.5 text-left text-xs transition-colors hover:bg-accent ${
                  selectedRunId === String(item.raw.run_id) ? "bg-accent" : ""
                }`}
                onClick={() =>
                  setSelectedRunId(
                    selectedRunId === String(item.raw.run_id)
                      ? null
                      : String(item.raw.run_id)
                  )
                }
              >
                <span className="block truncate font-mono text-[11px] text-muted-foreground">
                  {String(item.raw.run_id).slice(0, 8)}
                </span>
                <span className="block truncate text-foreground">{item.agent_name}</span>
              </button>
            </li>
          ))}
          {runs.length === 0 && (
            <li className="px-2 py-4 text-center text-xs text-muted-foreground">
              No runs yet
            </li>
          )}
        </ul>
      </ScrollArea>
    </div>
  )
}
```

### Pattern 6: `SplitPanels` Layout API (confirmed from source)

The `SplitPanels` component at `web/src/components/layout/SplitPanels.tsx` accepts:
- `storageKey: string` — drives `localStorage` key prefix for layout and collapse state
- `rail: React.ReactNode` — left panel content
- `list: React.ReactNode` — centre panel content
- `detail: React.ReactNode` — right panel content

Default sizes: rail=20%, list=50%, detail=30%. All persistence is handled internally using `onLayoutChange` and `window.localStorage`.

The `react-resizable-panels` v4.0.11 `Group` component's `onLayoutChange` callback receives a `Layout = { [panelId: string]: number }` map. `SplitPanels` already uses `panel id` props (`"rail"`, `"list"`, `"detail"`) to ensure stable associations across page navigations.

### Anti-Patterns to Avoid

- **Using `autoSaveId` from an older version:** v4 does not have `autoSaveId`. The project's `SplitPanels` uses `onLayoutChange` + manual `localStorage` — this is the correct approach for v4.
- **Re-implementing `SplitPanels`:** The existing component handles mobile breakpoints, collapse buttons, and persistence. Do not create a new ResizablePanelGroup from scratch for Phase 9.
- **Mounting `useAgentEvents` inside `SplitPanels` children:** Mount it at the page root, above `SplitPanels`. Children read from store.
- **Using `AgentStatusPanel` without a `compact` prop variant:** The existing `AgentStatusPanel` is designed for a full panel. In the `TasksPanel` rail (narrow), pass a `compact` prop (add it if not present) to suppress labels and use smaller icon sizes.
- **Duplicating file list state in a second store:** Keep all agent event display state in `useAgentEventStore`. Add `filesTouched`, `selectedRunId`, `selectedFile` as new fields in the existing store.
- **Next.js Server Component violation:** The page must have `"use client"` at top — it mounts hooks.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Resizable panel layout with persistence | Custom CSS resize handles + localStorage | `SplitPanels` (already exists) | Already implements drag handles, collapse/expand, mobile fallback, localStorage persistence with a single `storageKey` prop |
| Inline diff display | Custom diff renderer | `DiffViewer` + `computeDiff` from `web/src/lib/diff.ts` | LCS algorithm already handles add/remove/unchanged line classification; `DiffViewer` renders with +/- stats header |
| File extraction from events | Custom event parser | Extend `parsers.ts` with `parseFileTouched()` — same pure function pattern as `parseToolCall()` | Consistent with Phase 8 parser architecture; no mutation of event shape needed |
| Layout state persistence | Zustand persist middleware | `SplitPanels` storageKey → localStorage | `SplitPanels` uses direct `localStorage` for layout (not Zustand) — consistent with the existing implementation |
| SSE connection | New EventSource or fetch loop | Existing `useAgentEvents` hook (Phase 8) | Already handles reconnect, abort, feed dispatch; mounting it again would create a second SSE connection |

**Key insight:** The three-panel layout problem is already solved by `SplitPanels`. Phase 9 is primarily about: creating the new page, building the two new panel components (`TasksPanel`, `FileListPanel`/`InlineDiffPanel`), and extending the store with file-tracking state. The layout mechanics are done.

---

## Common Pitfalls

### Pitfall 1: `SplitPanels` `defaultLayout` Causes Layout Shift on Hydration

**What goes wrong:** `SplitPanels` reads from `localStorage` in a `useEffect`, but `ResizablePanelGroup` renders with `defaultLayout` prop synchronously. If the SSR layout differs from localStorage, there's a visible flash.

**Why it happens:** Next.js App Router renders the client component tree on the server with the default values, then hydrates. `localStorage` is not available server-side.

**How to avoid:** `SplitPanels` already handles this correctly — it initializes state with `DEFAULT_LAYOUT` and updates via `useEffect` on mount. The existing implementation uses `requestAnimationFrame` for collapse state too. No SSR mismatch — just accept a single-frame flicker on first load.

**Warning signs:** Panels snap from stored size to default size after a moment; console hydration warnings about layout mismatch.

### Pitfall 2: File State Unbounded Growth

**What goes wrong:** `filesTouched` accumulates entries for every run. Long-running sessions with many runs exhaust memory.

**Why it happens:** Each file change event appends to the record. Without eviction, the record grows forever.

**How to avoid:** Cap to the most recent 20 run IDs. When adding a new run's first file entry, if the record has >20 keys, drop the oldest key. Mirror the `FEED_MAX` pattern from Phase 8.

**Warning signs:** Browser memory usage grows proportionally with number of pipeline runs.

### Pitfall 3: `SplitPanels` `onResize` Fires with `{ inPixels }` Not Just a Number

**What goes wrong:** In `react-resizable-panels` v4, the `onResize` callback on `Panel` receives `{ inPixels: number }` (an object), not a plain number. The existing `SplitPanels` already handles this correctly — but if you write new `onResize` handlers, destructure correctly.

**Why it happens:** v4 changed the `onResize` signature from `(size: number) => void` to `({ inPixels }: { inPixels: number }) => void`.

**How to avoid:** Copy the existing `SplitPanels` pattern: `onResize={({ inPixels }) => setCollapsedState("rail", inPixels < 2)}`.

**Warning signs:** TypeScript type error: "Argument of type 'number' is not assignable to parameter of type '{ inPixels: number }'."

### Pitfall 4: `DiffViewer` Rendered Inside a Flex Panel Without `min-h-0`

**What goes wrong:** `DiffViewer` uses `h-full` internally. Inside a resizable panel, `h-full` without `min-h-0` on the parent flex container causes overflow that breaks layout.

**Why it happens:** Flex children do not shrink below their content height by default.

**How to avoid:** Always add `min-h-0 overflow-hidden` to the container wrapping `DiffViewer` inside the resizable panel. The `InlineDiffPanel` pattern above includes this.

**Warning signs:** The diff panel overflows the browser viewport; other panels get pushed off-screen.

### Pitfall 5: `AgentStatusPanel` Was Not Designed for Narrow Widths

**What goes wrong:** `AgentStatusPanel` (Phase 8) renders each agent as a full badge row with text labels. In the 20% left rail, long agent names cause overflow.

**Why it happens:** The component was designed for the full-page `/agent-events` test harness, not a narrow 20% panel.

**How to avoid:** Add a `compact?: boolean` prop to `AgentStatusPanel`. In compact mode, show only the status icon + agent name abbreviated (first 8 chars + ellipsis). The `TasksPanel` uses `compact`.

**Warning signs:** Agent names overflow the left rail; horizontal scroll appears on the tasks panel.

### Pitfall 6: `file_change` Events May Not Exist in Current Backend

**What goes wrong:** Phase 9 depends on `file_change` events arriving via SSE, but `EventType` class in `message_schema.py` does not yet define `FILE_CHANGE`. The Phase 8 vocabulary only covers lifecycle and tool call events.

**Why it happens:** FILE-01 and FILE-02 require new event types that were not in scope for Phase 8.

**How to avoid:** Phase 9 Plan 01 must add `EventType.FILE_CHANGE = "file_change"` to `message_schema.py` and update `parsers.ts` with a `parseFileTouched()` function. As a fallback for FILE-02, the frontend can also derive file touches from `tool_result` events where `payload.tool == "write_file"` (same logic as `DiffBlock.tsx`). This fallback ensures the file list panel shows something even before backend emits explicit `file_change` events.

**Warning signs:** FileListPanel always shows "No file changes yet" despite active agent runs.

---

## Code Examples

### Store Extension: FileTouchedEntry fields

```typescript
// web/src/lib/agent-events/store.ts — additional fields to add to AgentEventState

// In the interface:
filesTouched: Record<string, FileTouchedEntry[]>  // keyed by run_id
addFileTouched: (entry: FileTouchedEntry) => void
selectedRunId: string | null
setSelectedRunId: (id: string | null) => void
selectedFile: FileTouchedEntry | null
setSelectedFile: (file: FileTouchedEntry | null) => void

// In the create() call:
filesTouched: {},
addFileTouched: (entry) =>
  set((s) => {
    const existing = s.filesTouched[entry.run_id] ?? []
    // Avoid duplicate path in same run
    if (existing.some((e) => e.path === entry.path)) {
      return {}
    }
    const updated = { ...s.filesTouched, [entry.run_id]: [...existing, entry] }
    // Evict oldest runs beyond cap of 20
    const keys = Object.keys(updated)
    if (keys.length > 20) {
      const toDelete = keys[0]
      const { [toDelete]: _, ...rest } = updated
      return { filesTouched: rest }
    }
    return { filesTouched: updated }
  }),
selectedRunId: null,
setSelectedRunId: (id) => set({ selectedRunId: id }),
selectedFile: null,
setSelectedFile: (file) => set({ selectedFile: file }),
```

### Parser Extension: parseFileTouched

```typescript
// web/src/lib/agent-events/parsers.ts — append

const FILE_CHANGE_TYPES = new Set(["file_change"])
// Fallback: tool_result where payload.tool == "write_file"
export function parseFileTouched(raw: AgentEventEnvelopeRaw): FileTouchedEntry | null {
  const t = String(raw.type ?? "")
  const payload = (raw.payload ?? {}) as Record<string, unknown>

  const isExplicitFileChange = FILE_CHANGE_TYPES.has(t)
  const isWriteFileTool =
    (t === "tool_result") && typeof payload.tool === "string" && payload.tool === "write_file"

  if (!isExplicitFileChange && !isWriteFileTool) return null
  if (!raw.run_id || !raw.ts) return null

  const path = String(
    (isExplicitFileChange ? payload.path : payload.arguments
      ? (payload.arguments as Record<string, unknown>).path
      : undefined) ?? ""
  )
  if (!path) return null

  return {
    run_id: String(raw.run_id),
    path,
    status: (payload.status as "created" | "modified") ?? "modified",
    ts: String(raw.ts),
    linesAdded: typeof payload.lines_added === "number" ? payload.lines_added : undefined,
    linesDeleted: typeof payload.lines_deleted === "number" ? payload.lines_deleted : undefined,
    diff: typeof payload.diff === "string" ? payload.diff : undefined,
    oldContent: typeof payload.old_content === "string" ? payload.old_content : undefined,
    newContent: typeof payload.new_content === "string" ? payload.new_content : undefined,
  }
}
```

### SplitPanels API Usage (from confirmed source)

```typescript
// Source: web/src/components/layout/SplitPanels.tsx (existing project file)
// Props confirmed: storageKey, rail, list, detail, className

<SplitPanels
  storageKey="agent-dashboard"
  rail={<TasksPanel />}
  list={<ActivityFeed />}
  detail={<FileListPanel />}
  className="h-full"
/>
// Persistence: automatically stores sizes under
//   localStorage["agent-dashboard:layout"]
//   localStorage["agent-dashboard:collapsed"]
```

### react-resizable-panels v4 Layout Type (from installed package)

```typescript
// Source: node_modules/react-resizable-panels/dist/react-resizable-panels.d.ts
export declare type Layout = {
    [id: string]: number;
}
// onLayoutChange: (layout: Layout) => void
// Panel onResize: ({ inPixels }: { inPixels: number }) => void  ← v4 change
```

### DiffViewer (confirmed existing API)

```typescript
// Source: web/src/components/studio/DiffViewer.tsx
interface DiffViewerProps {
    oldValue: string;
    newValue: string;
    filename?: string;
    onApply?: () => void;    // omit for read-only
    onReject?: () => void;   // omit for read-only
    onClose?: () => void;    // omit for read-only
    splitView?: boolean;
}
// Usage in InlineDiffPanel: omit action callbacks for read-only display
<DiffViewer oldValue={entry.oldContent ?? ""} newValue={entry.newContent ?? ""} filename={entry.path} />
```

### Python: Adding FILE_CHANGE to EventType

```python
# Source: src/paperbot/application/collaboration/message_schema.py — APPEND to EventType class

# --- File change events ---
FILE_CHANGE: str = "file_change"
# Payload contract (for parsers.ts parseFileTouched):
#   path: str — relative file path
#   status: "created" | "modified"
#   lines_added: int (optional)
#   lines_deleted: int (optional)
#   old_content: str (optional — for inline diff)
#   new_content: str (optional — for inline diff)
#   diff: str (optional — unified diff string)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full-page single-panel dashboard | Three-panel IDE layout via `SplitPanels` | Phase 9 | Users can observe tasks, activity, and files simultaneously without switching views |
| File diff only in DeepCode Studio modal | Inline diff in the files panel | Phase 9 | Agent file changes visible in the agent dashboard without navigating to Studio |
| Ad-hoc panel in `/agent-events` test harness | Dedicated `/agent-dashboard` route | Phase 9 | Permanent entry point; test harness at `/agent-events` can be retained for debugging |
| No `file_change` event type in EventType | `EventType.FILE_CHANGE = "file_change"` | Phase 9 (backend wave) | Frontend can render per-task file lists from the SSE stream |

**Deprecated/outdated:**
- Manually writing `autoSaveId` on `PanelGroup` (v3 API): v4 uses `id` + `onLayoutChange` + manual storage. `SplitPanels` is the project's standard approach.

---

## Open Questions

1. **Where should `file_change` events be emitted in the Python backend?**
   - What we know: No backend code currently emits `file_change` typed events. The `repro/` pipeline (`nodes/generation.py`, etc.) writes files but does not emit SSE events for individual writes.
   - What's unclear: Which stage of the Paper2Code pipeline should emit them, and at what granularity.
   - Recommendation: For Phase 9, rely on the `write_file` tool_result fallback for FILE-02 (already works via `parsers.ts` extension). Add explicit `file_change` emission as a follow-up. This unblocks Phase 9 without a big backend change.

2. **Should `AgentStatusPanel` receive a `compact` prop, or should `TasksPanel` import a new `AgentStatusBadge` list directly?**
   - What we know: `AgentStatusPanel` currently renders badge rows with full text. The left rail is 20% wide.
   - What's unclear: Modifying `AgentStatusPanel` risks breaking the `/agent-events` test harness.
   - Recommendation: Add an optional `compact?: boolean` prop to `AgentStatusPanel`. Default `false` (backward compatible). Compact mode: icon only + short name, no status text label.

3. **Should the new `/agent-dashboard` page appear in the sidebar navigation?**
   - What we know: `LayoutShell` renders a `Sidebar` with nav links. There is currently no "Agent Dashboard" entry.
   - What's unclear: Whether this should be a top-level nav item or nested under "Studio".
   - Recommendation: Add a nav item to the sidebar. Phase 9 plan should include a task to add the link. Keep it simple — no deep integration work.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | vitest 2.1.4 (frontend); pytest + pytest-asyncio 0.21+ (backend) |
| Config file | `web/vitest.config.ts` — environment: "node", alias: "@" → "./src" |
| Quick run command | `cd web && npm test -- agent-dashboard` |
| Full suite command | `cd web && npm test -- agent-dashboard agent-events` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DASH-01 | Three-panel layout renders rail, list, and detail panels | unit (vitest — component snapshot) | `cd web && npm test -- agent-dashboard` | Wave 0 |
| DASH-01 | `SplitPanels` renders all three slots | unit (vitest) | `cd web && npm test -- SplitPanels` | Already exists (inspect) |
| DASH-04 | `SplitPanels` `onLayoutChange` writes to localStorage with `storageKey` prefix | unit (vitest) | `cd web && npm test -- SplitPanels` | Wave 0 |
| FILE-01 | `InlineDiffPanel` renders `DiffViewer` with `oldContent`/`newContent` when available | unit (vitest) | `cd web && npm test -- agent-dashboard` | Wave 0 |
| FILE-01 | `InlineDiffPanel` renders "Diff not available" when no content fields | unit (vitest) | `cd web && npm test -- agent-dashboard` | Wave 0 |
| FILE-02 | `parseFileTouched()` returns `FileTouchedEntry` for `file_change` events | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 (extend parsers.test.ts) |
| FILE-02 | `parseFileTouched()` returns entry for `tool_result` with `payload.tool == "write_file"` | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 (extend parsers.test.ts) |
| FILE-02 | `parseFileTouched()` returns null for lifecycle events | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 (extend parsers.test.ts) |
| FILE-02 | Store `addFileTouched` deduplicates same path within a run | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 (extend store.test.ts) |
| FILE-02 | Store evicts oldest run when >20 run keys | unit (vitest) | `cd web && npm test -- agent-events` | Wave 0 (extend store.test.ts) |
| DASH-01 (backend) | `EventType.FILE_CHANGE == "file_change"` | unit (pytest) | `PYTHONPATH=src pytest tests/unit/test_agent_events_vocab.py -x` | Extend existing test |

### Sampling Rate

- **Per task commit:** `cd web && npm test -- agent-dashboard agent-events 2>&1 | tail -10`
- **Per wave merge:** `cd web && npm test -- agent-dashboard agent-events`
- **Phase gate:** Full vitest suite green (`cd web && npm test`) before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `web/src/components/agent-dashboard/TasksPanel.test.tsx` — renders "No runs yet" when feed is empty
- [ ] `web/src/components/agent-dashboard/FileListPanel.test.tsx` — renders file list, handles empty state, navigates to diff on click
- [ ] `web/src/components/agent-dashboard/InlineDiffPanel.test.tsx` — renders DiffViewer, renders fallback when no content
- [ ] `web/src/lib/agent-events/parsers.ts` — MODIFIED: add `parseFileTouched()` function
- [ ] `web/src/lib/agent-events/parsers.test.ts` — EXTENDED: add parseFileTouched test cases
- [ ] `web/src/lib/agent-events/store.ts` — MODIFIED: add filesTouched, selectedRunId, selectedFile fields
- [ ] `web/src/lib/agent-events/store.test.ts` — EXTENDED: addFileTouched dedup + eviction tests

*(No new framework install needed — vitest already configured)*

---

## Sources

### Primary (HIGH confidence)

- Codebase direct read: `web/src/components/layout/SplitPanels.tsx` — confirmed `SplitPanels` API, layout persistence via `localStorage`, collapse state, mobile fallback
- Codebase direct read: `web/node_modules/react-resizable-panels/dist/react-resizable-panels.d.ts` — confirmed `Layout` type, `LayoutStorage` type, `onLayoutChange` signature, `useDefaultLayout` hook, v4 `onResize` change
- Codebase direct read: `web/node_modules/react-resizable-panels/package.json` — confirmed version 4.0.11
- Codebase direct read: `web/src/components/ui/resizable.tsx` — confirmed project wrapper exports `ResizablePanelGroup`, `ResizablePanel`, `ResizableHandle`
- Codebase direct read: `web/src/components/studio/DiffViewer.tsx` — confirmed `DiffViewerProps` interface, read-only usage (omit `onApply`/`onReject`)
- Codebase direct read: `web/src/lib/diff.ts` — confirmed `computeDiff` and `DiffLine` exports
- Codebase direct read: `web/src/lib/agent-events/store.ts` — confirmed Phase 8 Zustand store shape with `create<T>((set) => ...)` single-call form
- Codebase direct read: `web/src/lib/agent-events/types.ts` — confirmed Phase 8 TypeScript types
- Codebase direct read: `web/src/lib/agent-events/parsers.ts` — confirmed parser function signatures
- Codebase direct read: `web/src/app/agent-events/page.tsx` — confirmed Phase 8 SSE mount pattern
- Codebase direct read: `web/src/components/studio/TaskDetailPanel.tsx` — confirmed `computeWorkspaceStats` pattern for file extraction from `AgentTaskLog`
- Codebase direct read: `web/src/components/studio/blocks/DiffBlock.tsx` — confirmed `write_file` tool fallback pattern for file detection
- Codebase direct read: `web/src/app/layout.tsx` + `web/src/components/layout/LayoutShell.tsx` — confirmed app layout structure
- Codebase direct read: `web/src/app/studio/page.tsx` — confirmed existing `ResizablePanelGroup` usage pattern
- Codebase direct read: `src/paperbot/application/collaboration/message_schema.py` — confirmed `EventType` class constants; no `FILE_CHANGE` exists yet
- Codebase direct read: `web/vitest.config.ts` — confirmed test environment is "node" with "@" alias

### Secondary (MEDIUM confidence)

- `web/package.json` — all frontend dependency versions confirmed present; no new installs needed
- `web/node_modules/react-resizable-panels/README.md` — Group/Panel/Separator props confirmed for v4 API

### Tertiary (LOW confidence)

- None — all research based on direct codebase inspection

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all dependencies confirmed installed; `SplitPanels` source code read directly
- Architecture: HIGH — patterns derived from direct inspection of `SplitPanels`, `DiffViewer`, `TaskDetailPanel`, `DiffBlock`, Phase 8 store and parsers
- Pitfalls: HIGH — `onResize` v4 API change verified from TypeScript definitions; layout shift behaviour confirmed from `SplitPanels` source; `min-h-0` requirement from existing studio page layout patterns

**Research date:** 2026-03-15
**Valid until:** 2026-09-15 (all dependencies pinned; re-verify if react-resizable-panels or Next.js major version changes)
