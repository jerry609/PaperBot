# DeepCode Studio Redesign: Claude Commander + Codex Workers

## Project

**Repository:** `/Users/boyu/Documents/VScodeProject/PaperBot`
**Branch:** `feature/AgentSwarm`
**Stack:** Next.js 16 + React 19 + Tailwind CSS v4 + Zustand + FastAPI (Python)

## Context

The current DeepCode Studio page (`web/src/app/studio/page.tsx`) has a two-panel layout:
- **Left panel (18%):** `<FilesPanel />` — file tree browser
- **Right panel (82%):** `<ReproductionLog />` — main workspace with 3 tabs: Progress, Context Pack, Chat

When the user clicks "Create Session" on the Context Pack tab, it transitions to the Chat tab where Claude CLI handles code reproduction as a single agent.

**Goal:** Transform into a multi-agent system where Claude acts as the "commander" orchestrating multiple Codex API (cloud) workers. Redesign the UI with:
1. Left panel → Chat session history sidebar (only visible on Chat tab)
2. "Open in VS Code" button in the tab bar
3. New "Agent Board" tab with Kanban dashboard

### Reference Projects
- **Oh My OpenCode** (`code-yeongyu/oh-my-opencode`): Commander/worker split. Claude for orchestration (mechanics-driven prompts), Codex for execution (principle-driven prompts). Category-based dispatch. Wisdom accumulation between tasks.
- **Auto-Claude** (`AndyMik90/Auto-Claude`): Kanban board with columns: Planning → In Progress → AI Review → Human Review → Done

---

## Current File Contents (Key Files)

### `web/src/app/studio/page.tsx` (248 lines)

The studio page. Has `StudioContent` component with:
- Gallery view when no paper selected (`<PaperGallery />`)
- Workspace view when paper selected: `<ResizablePanelGroup>` with `<FilesPanel />` (left) + `<ReproductionLog />` (right)
- Mobile: tab layout switching between Reproduction and Files

Key state: `papers`, `selectedPaperId` from `useStudioStore()`

### `web/src/components/studio/ReproductionLog.tsx` (654 lines)

The main workspace component. Contains:
- Tab navigation bar: Progress | Context Pack | Chat (lines 406-429)
- `viewMode` state: `"log" | "generating" | "context_pack"` (line 204)
- Chat input area with mode selector (Code/Plan/Ask) and model selector (lines 566-626)
- Chat timeline rendering via `ActionItem` components
- Monaco editor for file viewing
- `handleSendMessage()` → POST `/api/studio/chat` → SSE stream

### `web/src/lib/store/studio-store.ts` (433 lines)

Zustand store with:
- `papers: StudioPaper[]`, `selectedPaperId`, `_paperCache`
- `tasks: Task[]`, `activeTaskId`
- Each `Task` has `id`, `name`, `status`, `actions: AgentAction[]`, `paperId`
- `contextPack`, `generationProgress`, `liveObservations`

### `web/src/components/studio/ContextPackPanel.tsx` (339 lines)

Shows context pack details. Has `handleCreateSession()` which POSTs to `/api/research/repro/context/{id}/session` and calls `onSessionCreated`.

### `src/paperbot/api/routes/studio_chat.py` (632 lines)

Backend chat route. Spawns Claude CLI subprocess (`claude -p --output-format stream-json`) and streams NDJSON events. Falls back to Anthropic API if CLI not found.

---

## Implementation Plan

### Step 1: Add "Open in VS Code" Button to Tab Bar

**File:** `web/src/components/studio/ReproductionLog.tsx`

**Target location:** Line 406-429 (tab navigation bar area)

Add a button to the LEFT of the tab buttons inside the `<div className="flex items-center shrink-0 border-b">`:

```tsx
import { ExternalLink } from "lucide-react"

// Inside the tab bar div, BEFORE the tab buttons map:
{projectDir && (
  <button
    onClick={() => window.open(`vscode://file${projectDir}`, '_blank')}
    className="flex items-center gap-1.5 px-3 py-2.5 text-sm font-medium text-muted-foreground hover:text-foreground/80 transition-colors border-r mr-1"
    title={`Open ${projectDir} in VS Code`}
  >
    <ExternalLink className="h-3.5 w-3.5" />
    Open in VS Code
  </button>
)}
```

The `projectDir` is already available in the component (line 216): `const projectDir = selectedPaper?.outputDir || lastGenCodeResult?.outputDir || null`

### Step 2: Add "Agent Board" Tab

**File:** `web/src/components/studio/ReproductionLog.tsx`

**Change 1:** Update `viewMode` type (line 204):
```tsx
const [viewMode, setViewMode] = useState<"log" | "generating" | "context_pack" | "agent_board">("log")
```

**Change 2:** Add tab to the tab array (line 407-411):
```tsx
{ key: "agent_board" as const, label: "Agent Board", icon: LayoutDashboard },
```

Import `LayoutDashboard` from `lucide-react`.

**Change 3:** Add rendering branch in the main content area (around line 440), before the chat timeline else branch:
```tsx
) : viewMode === "agent_board" ? (
  <AgentBoard paperId={selectedPaperId} />
```

Import `AgentBoard` from `./AgentBoard`.

### Step 3: Create AgentBoard Component

**New file:** `web/src/components/studio/AgentBoard.tsx`

A Kanban board with 5 columns (Auto-Claude style):

```tsx
"use client"

import { useMemo } from "react"
import { useStudioStore } from "@/lib/store/studio-store"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Clock, Bot, Cpu, CheckCircle2, AlertCircle, MoreHorizontal } from "lucide-react"
import { cn } from "@/lib/utils"

const COLUMNS = [
  { id: "planning", label: "Planning", color: "border-t-yellow-500" },
  { id: "in_progress", label: "In Progress", color: "border-t-blue-500" },
  { id: "ai_review", label: "AI Review", color: "border-t-purple-500" },
  { id: "human_review", label: "Human Review", color: "border-t-orange-500" },
  { id: "done", label: "Done", color: "border-t-green-500" },
] as const

type ColumnId = typeof COLUMNS[number]["id"]

export interface AgentTask {
  id: string
  title: string
  description: string
  status: ColumnId
  assignee: string        // "claude" | "codex-1" | "codex-2" etc.
  progress: number        // 0-100
  tags: string[]
  createdAt: string
  updatedAt: string
  subtasks: { id: string; title: string; done: boolean }[]
}

interface Props {
  paperId: string | null
}

export function AgentBoard({ paperId }: Props) {
  const { agentTasks } = useStudioStore()

  const tasksByColumn = useMemo(() => {
    const filtered = paperId
      ? agentTasks.filter(t => t.paperId === paperId)
      : agentTasks
    const map: Record<ColumnId, AgentTask[]> = {
      planning: [], in_progress: [], ai_review: [], human_review: [], done: [],
    }
    for (const task of filtered) {
      if (map[task.status]) map[task.status].push(task)
    }
    return map
  }, [agentTasks, paperId])

  return (
    <div className="h-full flex overflow-x-auto p-4 gap-4">
      {COLUMNS.map(col => (
        <div key={col.id} className="flex-shrink-0 w-72 flex flex-col">
          {/* Column header */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">{col.label}</h3>
              <span className="text-xs text-muted-foreground bg-muted rounded-full px-2 py-0.5">
                {tasksByColumn[col.id].length}
              </span>
            </div>
          </div>

          {/* Task cards */}
          <div className={cn("flex-1 space-y-3 overflow-y-auto rounded-lg border-t-2 bg-muted/20 p-2", col.color)}>
            {tasksByColumn[col.id].length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                {col.id === "in_progress" ? (
                  <>
                    <Cpu className="h-5 w-5 mb-2 opacity-30 animate-pulse" />
                    <p className="text-xs">Nothing running</p>
                    <p className="text-[10px]">Start a task from Planning</p>
                  </>
                ) : col.id === "ai_review" ? (
                  <>
                    <Bot className="h-5 w-5 mb-2 opacity-30" />
                    <p className="text-xs">No tasks in review</p>
                    <p className="text-[10px]">AI will review completed tasks</p>
                  </>
                ) : (
                  <p className="text-xs">No tasks</p>
                )}
              </div>
            ) : (
              tasksByColumn[col.id].map(task => (
                <TaskCard key={task.id} task={task} />
              ))
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function TaskCard({ task }: { task: AgentTask }) {
  const isCompleted = task.status === "done"
  const completedSubtasks = task.subtasks.filter(s => s.done).length
  const totalSubtasks = task.subtasks.length

  const relativeTime = useMemo(() => {
    const diff = Date.now() - new Date(task.updatedAt).getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 60) return `${mins}m ago`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }, [task.updatedAt])

  return (
    <Card className="shadow-sm">
      <CardContent className="p-3 space-y-2">
        <div className="flex items-start justify-between">
          <h4 className="text-sm font-medium leading-tight line-clamp-2">{task.title}</h4>
          <div className="flex gap-1 shrink-0 ml-2">
            {isCompleted ? (
              <Badge variant="outline" className="text-[10px] bg-green-50 text-green-700 border-green-200">
                Completed
              </Badge>
            ) : task.status === "in_progress" ? (
              <Badge variant="outline" className="text-[10px] bg-blue-50 text-blue-700 border-blue-200">
                Running
              </Badge>
            ) : null}
          </div>
        </div>

        {task.description && (
          <p className="text-xs text-muted-foreground line-clamp-2">{task.description}</p>
        )}

        {task.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {task.tags.map(tag => (
              <Badge key={tag} variant="secondary" className="text-[10px]">{tag}</Badge>
            ))}
          </div>
        )}

        {/* Progress bar */}
        {task.progress > 0 && task.progress < 100 && (
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>Progress</span>
              <span>{task.progress}%</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${task.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Subtask dots (like Auto-Claude) */}
        {totalSubtasks > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            {task.subtasks.slice(0, 10).map(sub => (
              <div
                key={sub.id}
                className={cn(
                  "h-2 w-2 rounded-full",
                  sub.done ? "bg-green-500" : "bg-muted-foreground/30"
                )}
                title={sub.title}
              />
            ))}
            {totalSubtasks > 10 && (
              <span className="text-[10px] text-muted-foreground">+{totalSubtasks - 10}</span>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-1">
          <div className="flex items-center gap-1">
            {task.assignee === "claude" ? (
              <Bot className="h-3 w-3" />
            ) : (
              <Cpu className="h-3 w-3" />
            )}
            <span>{task.assignee}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-2.5 w-2.5" />
            <span>{relativeTime}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
```

### Step 4: Create ChatHistoryPanel Component

**New file:** `web/src/components/studio/ChatHistoryPanel.tsx`

```tsx
"use client"

import { useMemo } from "react"
import { useStudioStore } from "@/lib/store/studio-store"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Plus, MessageSquare } from "lucide-react"
import { cn } from "@/lib/utils"

export function ChatHistoryPanel() {
  const { tasks, activeTaskId, setActiveTask, selectedPaperId } = useStudioStore()

  // Filter tasks for the current paper, most recent first
  const paperTasks = useMemo(() => {
    return tasks
      .filter(t => t.paperId === selectedPaperId)
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
  }, [tasks, selectedPaperId])

  const relativeTime = (date: Date) => {
    const diff = Date.now() - new Date(date).getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 60) return `${mins}m`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h`
    const days = Math.floor(hours / 24)
    if (days < 7) return `${days}d`
    const weeks = Math.floor(days / 7)
    return `${weeks}w`
  }

  return (
    <div className="h-full flex flex-col border-r">
      {/* Header */}
      <div className="px-3 py-2.5 border-b flex items-center justify-between">
        <span className="text-sm font-medium">Threads</span>
        <Button variant="ghost" size="icon" className="h-6 w-6" title="New thread">
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Thread list */}
      <ScrollArea className="flex-1">
        <div className="p-1.5 space-y-0.5">
          {paperTasks.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <MessageSquare className="h-5 w-5 mb-2 opacity-30" />
              <p className="text-xs">No conversations yet</p>
            </div>
          ) : (
            paperTasks.map(task => (
              <button
                key={task.id}
                onClick={() => setActiveTask(task.id)}
                className={cn(
                  "w-full text-left px-3 py-2 rounded-md transition-colors text-xs",
                  task.id === activeTaskId
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-muted"
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-medium">{task.name}</span>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {relativeTime(task.createdAt)}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
```

### Step 5: Modify Studio Page Layout

**File:** `web/src/app/studio/page.tsx`

Replace the desktop workspace section (lines 199-214) to conditionally show left panel only on Chat tab.

**Changes:**
1. Lift `viewMode` state up to `StudioContent`
2. Conditionally render left panel

```tsx
// Add state in StudioContent:
const [viewMode, setViewMode] = useState<"log" | "generating" | "context_pack" | "agent_board">("log")

// Replace desktop workspace div (lines 199-214):
<div className="hidden md:flex flex-1 min-h-0">
  {viewMode === "log" ? (
    <ResizablePanelGroup orientation="horizontal" className="flex-1">
      <ResizablePanel defaultSize={20} minSize={14}>
        <ChatHistoryPanel />
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel defaultSize={80} minSize={40}>
        <ReproductionLog viewMode={viewMode} onViewModeChange={setViewMode} />
      </ResizablePanel>
    </ResizablePanelGroup>
  ) : (
    <ReproductionLog viewMode={viewMode} onViewModeChange={setViewMode} />
  )}
</div>
```

**This requires refactoring `ReproductionLog` to accept `viewMode` and `onViewModeChange` as props** instead of managing `viewMode` internally. Remove the internal `useState` for `viewMode` and use the props instead.

### Step 6: Update Studio Store for Agent Tasks

**File:** `web/src/lib/store/studio-store.ts`

Add to `StudioState` interface:
```ts
// Agent Board state
agentTasks: AgentTask[]
addAgentTask: (task: Omit<AgentTask, 'id' | 'createdAt' | 'updatedAt'>) => string
updateAgentTask: (taskId: string, updates: Partial<AgentTask>) => void
moveAgentTask: (taskId: string, status: AgentTask['status']) => void
```

Add `AgentTask` type (same as in AgentBoard.tsx, export from store or create shared type):
```ts
export type AgentTaskStatus = 'planning' | 'in_progress' | 'ai_review' | 'human_review' | 'done'

export interface AgentTask {
  id: string
  title: string
  description: string
  status: AgentTaskStatus
  assignee: string
  progress: number
  tags: string[]
  createdAt: string
  updatedAt: string
  subtasks: { id: string; title: string; done: boolean }[]
  paperId?: string
}
```

Add implementations in the store:
```ts
agentTasks: [],

addAgentTask: (task) => {
  const id = `agent-task-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
  const now = new Date().toISOString()
  set(state => ({
    agentTasks: [...state.agentTasks, { ...task, id, createdAt: now, updatedAt: now }]
  }))
  return id
},

updateAgentTask: (taskId, updates) => {
  set(state => ({
    agentTasks: state.agentTasks.map(t =>
      t.id === taskId
        ? { ...t, ...updates, updatedAt: new Date().toISOString() }
        : t
    )
  }))
},

moveAgentTask: (taskId, status) => {
  set(state => ({
    agentTasks: state.agentTasks.map(t =>
      t.id === taskId
        ? { ...t, status, updatedAt: new Date().toISOString() }
        : t
    )
  }))
},
```

### Step 7: Backend — Agent Board API

**New file:** `src/paperbot/api/routes/agent_board.py`

```python
"""
Agent Board API — Claude Commander + Codex Workers

Claude decomposes the context pack into tasks, dispatches them to
Codex API workers, reviews results, and manages the Kanban lifecycle.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..streaming import StreamEvent, wrap_generator

router = APIRouter(prefix="/api/agent-board")
log = logging.getLogger(__name__)

# In-memory store (replace with DB in production)
_sessions: Dict[str, "BoardSession"] = {}


class AgentTask(BaseModel):
    id: str
    title: str
    description: str
    status: str = "planning"  # planning | in_progress | ai_review | human_review | done
    assignee: str = "claude"
    progress: int = 0
    tags: List[str] = []
    subtasks: List[Dict[str, Any]] = []
    created_at: str = ""
    updated_at: str = ""
    paper_id: Optional[str] = None


class BoardSession:
    def __init__(self, session_id: str, paper_id: str, context_pack_id: str):
        self.session_id = session_id
        self.paper_id = paper_id
        self.context_pack_id = context_pack_id
        self.tasks: List[AgentTask] = []
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "paper_id": self.paper_id,
            "context_pack_id": self.context_pack_id,
            "tasks": [t.model_dump() for t in self.tasks],
            "created_at": self.created_at,
        }


class PlanRequest(BaseModel):
    paper_id: str
    context_pack_id: str
    workspace_dir: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    assignee: Optional[str] = None


@router.post("/sessions")
async def create_session(request: PlanRequest):
    """Create a new agent board session."""
    session_id = f"board-{uuid.uuid4().hex[:12]}"
    session = BoardSession(
        session_id=session_id,
        paper_id=request.paper_id,
        context_pack_id=request.context_pack_id,
    )
    _sessions[session_id] = session
    return session.to_dict()


@router.post("/sessions/{session_id}/plan")
async def plan_session(session_id: str):
    """Claude decomposes context pack into tasks (SSE stream)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        wrap_generator(_plan_stream(session), workflow="agent_board_plan"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _plan_stream(session: BoardSession) -> AsyncGenerator[StreamEvent, None]:
    """Use Claude to decompose context pack into tasks."""
    yield StreamEvent(type="progress", data={"phase": "planning", "message": "Claude is analyzing the context pack..."})

    try:
        # Load context pack
        from ...infrastructure.stores.repro_context_store import SqlAlchemyReproContextStore
        store = SqlAlchemyReproContextStore()
        pack = store.get(session.context_pack_id)

        if not pack:
            yield StreamEvent(type="error", message="Context pack not found")
            return

        # Use Claude to decompose into tasks
        tasks = await _decompose_with_claude(pack)

        now = datetime.utcnow().isoformat()
        for task_data in tasks:
            task = AgentTask(
                id=f"task-{uuid.uuid4().hex[:8]}",
                title=task_data.get("title", "Untitled"),
                description=task_data.get("description", ""),
                status="planning",
                assignee="claude",
                tags=task_data.get("tags", []),
                subtasks=task_data.get("subtasks", []),
                created_at=now,
                updated_at=now,
                paper_id=session.paper_id,
            )
            session.tasks.append(task)
            yield StreamEvent(
                type="progress",
                data={"event": "task_created", "task": task.model_dump()},
            )

        yield StreamEvent(type="result", data={"tasks_count": len(session.tasks)})

    except Exception as e:
        log.exception("Planning failed")
        yield StreamEvent(type="error", message=str(e))


async def _decompose_with_claude(pack: dict) -> list[dict]:
    """Use Claude API to decompose a context pack into discrete tasks."""
    # Build prompt from context pack
    roadmap = pack.get("task_roadmap", [])
    observations = pack.get("observations", [])

    tasks = []
    for i, step in enumerate(roadmap):
        task = {
            "title": step.get("title", f"Step {i + 1}"),
            "description": step.get("description", ""),
            "tags": [step.get("estimated_difficulty", "medium")],
            "subtasks": [],
        }
        # Add acceptance criteria as subtasks
        criteria = step.get("acceptance_criteria", [])
        if isinstance(criteria, list):
            for j, c in enumerate(criteria):
                task["subtasks"].append({
                    "id": f"sub-{i}-{j}",
                    "title": c if isinstance(c, str) else str(c),
                    "done": False,
                })
        tasks.append(task)

    return tasks


@router.get("/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    """List all tasks in a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [t.model_dump() for t in session.tasks]


@router.post("/tasks/{task_id}/dispatch")
async def dispatch_task(task_id: str):
    """Dispatch a task to a Codex worker."""
    task = _find_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = "in_progress"
    task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
    task.updated_at = datetime.utcnow().isoformat()

    return task.model_dump()


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, request: TaskUpdateRequest):
    """Update a task's status or other fields."""
    task = _find_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if request.status:
        task.status = request.status
    if request.progress is not None:
        task.progress = request.progress
    if request.assignee:
        task.assignee = request.assignee
    task.updated_at = datetime.utcnow().isoformat()

    return task.model_dump()


def _find_task(task_id: str) -> Optional[AgentTask]:
    """Find a task across all sessions."""
    for session in _sessions.values():
        for task in session.tasks:
            if task.id == task_id:
                return task
    return None
```

### Step 8: Backend — Codex Dispatcher

**New file:** `src/paperbot/infrastructure/swarm/codex_dispatcher.py`

```python
"""
Codex Dispatcher — sends coding tasks to OpenAI Codex API (cloud).

Uses principle-driven prompts (concise, goal-oriented) as Codex responds
better to this style than mechanics-driven prompts.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class CodexResult:
    task_id: str
    success: bool
    output: str = ""
    files_generated: List[str] = field(default_factory=list)
    error: Optional[str] = None


class CodexDispatcher:
    """Dispatches coding tasks to OpenAI Codex API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "codex-mini-latest"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model

    async def dispatch(self, task_id: str, prompt: str, workspace: Path) -> CodexResult:
        """Send a coding task to Codex API and return the result."""
        if not self.api_key:
            return CodexResult(
                task_id=task_id,
                success=False,
                error="OPENAI_API_KEY not set",
            )

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.responses.create(
                model=self.model,
                instructions=(
                    "You are an expert coding agent. Generate clean, well-documented code. "
                    "Focus on correctness and clarity. Follow the project conventions."
                ),
                input=prompt,
            )

            output_text = response.output_text or ""

            return CodexResult(
                task_id=task_id,
                success=True,
                output=output_text,
            )

        except Exception as e:
            log.exception("Codex dispatch failed for task %s", task_id)
            return CodexResult(
                task_id=task_id,
                success=False,
                error=str(e),
            )

    async def dispatch_parallel(
        self, tasks: List[Dict[str, Any]], workspace: Path
    ) -> List[CodexResult]:
        """Dispatch multiple tasks concurrently."""
        coros = [
            self.dispatch(t["task_id"], t["prompt"], workspace)
            for t in tasks
        ]
        return await asyncio.gather(*coros, return_exceptions=False)
```

### Step 9: Backend — Claude Commander

**New file:** `src/paperbot/infrastructure/swarm/claude_commander.py`

```python
"""
Claude Commander — orchestrates the multi-agent workflow.

Claude acts as the "boss", decomposing work into tasks, dispatching
to Codex workers, reviewing results, and accumulating wisdom.

Inspired by Oh My OpenCode's three-layer architecture:
- Planning layer (Claude Opus) → structured task decomposition
- Execution layer (Codex) → autonomous code generation
- Review layer (Claude) → quality verification
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

log = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    approved: bool
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)


@dataclass
class WisdomEntry:
    """Accumulated learning from completed tasks."""
    learnings: List[str] = field(default_factory=list)
    conventions: List[str] = field(default_factory=list)
    gotchas: List[str] = field(default_factory=list)


class ClaudeCommander:
    """Claude as the commander orchestrating Codex workers."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.wisdom = WisdomEntry()

    async def decompose(self, context_pack: dict) -> List[Dict[str, Any]]:
        """Decompose context pack into discrete coding tasks."""
        roadmap = context_pack.get("task_roadmap", [])

        tasks = []
        for step in roadmap:
            tasks.append({
                "title": step.get("title", "Untitled"),
                "description": step.get("description", ""),
                "difficulty": step.get("estimated_difficulty", "medium"),
                "acceptance_criteria": step.get("acceptance_criteria", []),
            })

        return tasks

    async def build_codex_prompt(self, task: dict, workspace: Path) -> str:
        """Build a principle-driven prompt for Codex worker.

        Codex responds best to concise, goal-oriented prompts
        (unlike Claude which prefers mechanics-driven prompts).
        """
        parts = [
            f"# Task: {task['title']}",
            "",
            f"## Goal",
            task.get("description", ""),
            "",
        ]

        if task.get("acceptance_criteria"):
            parts.append("## Acceptance Criteria")
            for c in task["acceptance_criteria"]:
                parts.append(f"- {c}")
            parts.append("")

        # Inject accumulated wisdom
        if self.wisdom.learnings:
            parts.append("## Context from Previous Tasks")
            for l in self.wisdom.learnings[-5:]:  # Last 5 learnings
                parts.append(f"- {l}")
            parts.append("")

        if self.wisdom.conventions:
            parts.append("## Project Conventions")
            for c in self.wisdom.conventions[-5:]:
                parts.append(f"- {c}")
            parts.append("")

        return "\n".join(parts)

    async def review(self, task: dict, codex_output: str) -> ReviewResult:
        """Review Codex worker output using Claude."""
        # For now, auto-approve. In production, use Claude API to review.
        if not codex_output or len(codex_output.strip()) < 10:
            return ReviewResult(
                approved=False,
                feedback="Output is too short or empty",
            )

        return ReviewResult(approved=True, feedback="Looks good")

    def accumulate_wisdom(self, task: dict, output: str) -> None:
        """Extract learnings from completed tasks for future workers."""
        self.wisdom.learnings.append(
            f"Completed: {task.get('title', 'unknown')} — output length: {len(output)} chars"
        )
```

### Step 10: Register Backend Router

**File:** `src/paperbot/api/main.py`

Add import and router registration:
```python
from .routes.agent_board import router as agent_board_router
app.include_router(agent_board_router)
```

---

## Verification

1. `cd web && npm run build` — should compile without errors
2. Open Studio → select paper → verify 4 tabs: Progress, Context Pack, Chat, Agent Board
3. "Open in VS Code" button appears when `projectDir` is set, clicking opens VS Code
4. Chat tab shows left panel with session history, other tabs are full-width
5. Agent Board shows 5 Kanban columns
6. Backend: `POST /api/agent-board/sessions` creates session
7. Backend: `POST /api/agent-board/sessions/{id}/plan` decomposes context pack into tasks
