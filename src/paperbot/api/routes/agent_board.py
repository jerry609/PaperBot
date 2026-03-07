"""
Agent Board API -- Claude Commander + Codex Workers

Claude decomposes the context pack into tasks, dispatches them to
Codex API workers, reviews results, and manages the Kanban lifecycle.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..streaming import StreamEvent, wrap_generator

router = APIRouter(prefix="/api/agent-board")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store (replace with DB in production)
# ---------------------------------------------------------------------------
_sessions: Dict[str, "BoardSession"] = {}

# Shared commander instance (preserves wisdom across tasks)
_commander = None


def _get_commander():
    global _commander
    if _commander is None:
        from ...infrastructure.swarm import ClaudeCommander

        _commander = ClaudeCommander()
    return _commander


def _get_dispatcher():
    from ...infrastructure.swarm import CodexDispatcher

    return CodexDispatcher()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


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
    # Execution metadata
    codex_output: Optional[str] = None
    review_feedback: Optional[str] = None
    generated_files: List[str] = Field(default_factory=list)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    human_reviews: List[Dict[str, Any]] = Field(default_factory=list)


class BoardSession:
    def __init__(
        self,
        session_id: str,
        paper_id: str,
        context_pack_id: str,
        workspace_dir: Optional[str] = None,
    ):
        self.session_id = session_id
        self.paper_id = paper_id
        self.context_pack_id = context_pack_id
        self.workspace_dir = workspace_dir
        self.tasks: List[AgentTask] = []
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "paper_id": self.paper_id,
            "context_pack_id": self.context_pack_id,
            "workspace_dir": self.workspace_dir,
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


class RunAllRequest(BaseModel):
    workspace_dir: Optional[str] = None


class HumanReviewRequest(BaseModel):
    decision: Literal["approve", "request_changes"]
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_session(request: PlanRequest):
    """Create a new agent board session."""
    session_id = f"board-{uuid.uuid4().hex[:12]}"
    session = BoardSession(
        session_id=session_id,
        paper_id=request.paper_id,
        context_pack_id=request.context_pack_id,
        workspace_dir=request.workspace_dir,
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


@router.post("/sessions/{session_id}/run")
async def run_all_tasks(session_id: str, request: RunAllRequest):
    """Dispatch all planning tasks to Codex workers (SSE stream)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.workspace_dir:
        session.workspace_dir = request.workspace_dir

    return StreamingResponse(
        wrap_generator(_run_all_stream(session), workflow="agent_board_run"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    """List all tasks in a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [task.model_dump() for task in session.tasks]


@router.post("/tasks/{task_id}/dispatch")
async def dispatch_task(task_id: str):
    """Dispatch a single task to a Codex worker."""
    task = _find_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = "in_progress"
    task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
    task.updated_at = datetime.utcnow().isoformat()

    return task.model_dump()


@router.post("/tasks/{task_id}/execute")
async def execute_task(task_id: str):
    """Execute a single task: Codex runs, Claude reviews (SSE stream)."""
    task = _find_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    session = _find_session_for_task(task_id)

    return StreamingResponse(
        wrap_generator(_execute_task_stream(task, session), workflow="agent_board_execute"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


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


@router.post("/tasks/{task_id}/human-review")
async def human_review_task(task_id: str, request: HumanReviewRequest):
    """Allow a user to approve or request changes for a human-review task."""
    task = _find_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    notes = (request.notes or "").strip()
    review_entry = {
        "id": f"hr-{uuid.uuid4().hex[:8]}",
        "decision": request.decision,
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat(),
    }
    task.human_reviews.append(review_entry)

    if request.decision == "approve":
        task.status = "done"
        task.progress = 100
        task.assignee = "human"
        for sub in task.subtasks:
            sub["done"] = True
        _append_task_log(
            task,
            event="human_approved",
            phase="human_review",
            level="success",
            message="Human approved task.",
            details={"notes": notes},
        )
    else:
        task.status = "planning"
        task.progress = 0
        task.assignee = "claude"
        for sub in task.subtasks:
            sub["done"] = False
        _append_task_log(
            task,
            event="human_requested_changes",
            phase="human_review",
            level="warning",
            message="Human requested changes; task moved back to planning.",
            details={"notes": notes},
        )

    if notes:
        task.review_feedback = _merge_review_feedback(task.review_feedback, notes)

    task.updated_at = datetime.utcnow().isoformat()
    return task.model_dump()


# ---------------------------------------------------------------------------
# SSE Streams
# ---------------------------------------------------------------------------


async def _plan_stream(session: BoardSession) -> AsyncGenerator[StreamEvent, None]:
    """Use Claude to decompose context pack into tasks."""
    yield StreamEvent(
        type="progress",
        data={
            "phase": "planning",
            "message": "Claude is analyzing the context pack...",
        },
    )

    try:
        pack = _load_context_pack(session.context_pack_id)
        if not pack:
            yield StreamEvent(type="error", message="Context pack not found")
            return

        commander = _get_commander()
        tasks_data = await commander.decompose(pack)

        now = datetime.utcnow().isoformat()
        for i, task_data in enumerate(tasks_data):
            task = AgentTask(
                id=f"task-{uuid.uuid4().hex[:8]}",
                title=task_data.get("title", "Untitled"),
                description=task_data.get("description", ""),
                status="planning",
                assignee="claude",
                tags=_build_tags(task_data),
                subtasks=_build_subtasks(i, task_data),
                created_at=now,
                updated_at=now,
                paper_id=session.paper_id,
            )
            _append_task_log(
                task,
                event="planned",
                phase="planning",
                level="info",
                message="Task created from context pack decomposition.",
            )
            session.tasks.append(task)
            yield StreamEvent(
                type="progress",
                data={"event": "task_created", "task": task.model_dump()},
            )

        yield StreamEvent(
            type="result",
            data={
                "tasks_count": len(session.tasks),
                "session_id": session.session_id,
            },
        )

    except Exception as exc:
        log.exception("Planning failed")
        yield StreamEvent(type="error", message=str(exc))


async def _run_all_stream(
    session: BoardSession,
) -> AsyncGenerator[StreamEvent, None]:
    """Execute all planning tasks sequentially: Codex runs, Claude reviews."""
    planning_tasks = [t for t in session.tasks if t.status == "planning"]
    if not planning_tasks:
        yield StreamEvent(type="error", message="No tasks in planning status")
        return

    total = len(planning_tasks)
    yield StreamEvent(
        type="progress",
        data={
            "phase": "dispatching",
            "message": f"Dispatching {total} tasks to Codex workers...",
            "total": total,
        },
    )

    commander = _get_commander()
    dispatcher = _get_dispatcher()
    workspace = Path(session.workspace_dir or "/tmp/paperbot-workspace")

    for i, task in enumerate(planning_tasks):
        # --- Dispatch to Codex ---
        task.status = "in_progress"
        task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
        task.updated_at = datetime.utcnow().isoformat()
        task.progress = 10
        _append_task_log(
            task,
            event="task_dispatched",
            phase="dispatching",
            level="info",
            message=f"Dispatched to {task.assignee}.",
        )

        yield StreamEvent(
            type="progress",
            data={
                "event": "task_dispatched",
                "task_id": task.id,
                "task": task.model_dump(),
                "index": i,
                "total": total,
            },
        )

        # Build prompt and execute
        task_dict = {
            "title": task.title,
            "description": task.description,
            "acceptance_criteria": [s["title"] for s in task.subtasks],
        }
        prompt = await commander.build_codex_prompt(task_dict, workspace)

        # Run dispatch with periodic heartbeats so the SSE connection stays alive
        result = None
        dispatch_coro = dispatcher.dispatch(task.id, prompt, workspace)
        dispatch_task = asyncio.ensure_future(dispatch_coro)
        while not dispatch_task.done():
            await asyncio.sleep(5)
            if not dispatch_task.done():
                task.progress = min(task.progress + 5, 60)
                task.updated_at = datetime.utcnow().isoformat()
                _append_task_log(
                    task,
                    event="heartbeat",
                    phase="codex_running",
                    level="info",
                    message=f"Codex still running ({task.progress}%).",
                )
                yield StreamEvent(
                    type="progress",
                    data={
                        "event": "heartbeat",
                        "task_id": task.id,
                        "task": task.model_dump(),
                        "phase": "codex_running",
                    },
                )
        result = dispatch_task.result()

        task.progress = 70
        task.codex_output = result.output if result.success else result.error
        task.generated_files = result.files_generated if result.success else []
        task.updated_at = datetime.utcnow().isoformat()

        if not result.success:
            task.status = "human_review"
            task.progress = 100
            _append_task_log(
                task,
                event="task_failed",
                phase="codex_running",
                level="error",
                message=result.error or "Codex execution failed.",
            )
            yield StreamEvent(
                type="progress",
                data={
                    "event": "task_failed",
                    "task_id": task.id,
                    "task": task.model_dump(),
                    "error": result.error,
                },
            )
            continue

        yield StreamEvent(
            type="progress",
            data={
                "event": "task_codex_done",
                "task_id": task.id,
                "task": task.model_dump(),
                "output_length": len(result.output),
            },
        )
        _append_task_log(
            task,
            event="task_codex_done",
            phase="codex_running",
            level="success",
            message=f"Codex output received ({len(result.output)} chars).",
        )
        if task.generated_files:
            _append_task_log(
                task,
                event="files_written",
                phase="codex_running",
                level="success",
                message=f"Wrote {len(task.generated_files)} file(s) to workspace.",
                details={"files": task.generated_files},
            )

        # --- Claude reviews ---
        task.status = "ai_review"
        task.assignee = "claude"
        task.progress = 85
        task.updated_at = datetime.utcnow().isoformat()
        _append_task_log(
            task,
            event="task_reviewing",
            phase="ai_review",
            level="info",
            message="Claude started review.",
        )

        yield StreamEvent(
            type="progress",
            data={
                "event": "task_reviewing",
                "task_id": task.id,
                "task": task.model_dump(),
            },
        )

        # Run review with periodic heartbeats
        review = None
        review_coro = commander.review(task_dict, result.output)
        review_task = asyncio.ensure_future(review_coro)
        while not review_task.done():
            await asyncio.sleep(5)
            if not review_task.done():
                _append_task_log(
                    task,
                    event="heartbeat",
                    phase="review_running",
                    level="info",
                    message="Review still running.",
                )
                yield StreamEvent(
                    type="progress",
                    data={
                        "event": "heartbeat",
                        "task_id": task.id,
                        "task": task.model_dump(),
                        "phase": "review_running",
                    },
                )
        review = review_task.result()
        task.review_feedback = review.feedback

        if review.approved:
            task.status = "done"
            task.progress = 100
            # Mark all subtasks as done
            for sub in task.subtasks:
                sub["done"] = True
            commander.accumulate_wisdom(task_dict, result.output)
            _append_task_log(
                task,
                event="task_reviewed",
                phase="ai_review",
                level="success",
                message="Claude approved task.",
                details={"feedback": review.feedback},
            )
        else:
            task.status = "human_review"
            task.progress = 90
            _append_task_log(
                task,
                event="task_reviewed",
                phase="ai_review",
                level="warning",
                message="Claude requested human review.",
                details={"feedback": review.feedback},
            )

        task.updated_at = datetime.utcnow().isoformat()

        yield StreamEvent(
            type="progress",
            data={
                "event": "task_reviewed",
                "task_id": task.id,
                "task": task.model_dump(),
                "approved": review.approved,
                "feedback": review.feedback,
            },
        )

    done_count = sum(1 for t in session.tasks if t.status == "done")
    yield StreamEvent(
        type="result",
        data={
            "completed": done_count,
            "total": total,
            "session_id": session.session_id,
        },
    )


async def _execute_task_stream(
    task: AgentTask,
    session: Optional["BoardSession"],
) -> AsyncGenerator[StreamEvent, None]:
    """Execute a single task through Codex + Claude review."""
    commander = _get_commander()
    dispatcher = _get_dispatcher()
    workspace = Path((session.workspace_dir if session else None) or "/tmp/paperbot-workspace")

    # Dispatch
    task.status = "in_progress"
    task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
    task.progress = 10
    task.updated_at = datetime.utcnow().isoformat()
    _append_task_log(
        task,
        event="task_dispatched",
        phase="dispatching",
        level="info",
        message=f"Dispatched to {task.assignee}.",
    )

    yield StreamEvent(
        type="progress",
        data={"event": "task_dispatched", "task": task.model_dump()},
    )

    task_dict = {
        "title": task.title,
        "description": task.description,
        "acceptance_criteria": [s["title"] for s in task.subtasks],
    }
    prompt = await commander.build_codex_prompt(task_dict, workspace)
    result = await dispatcher.dispatch(task.id, prompt, workspace)

    task.progress = 70
    task.codex_output = result.output if result.success else result.error
    task.generated_files = result.files_generated if result.success else []
    task.updated_at = datetime.utcnow().isoformat()

    if not result.success:
        task.status = "human_review"
        task.progress = 100
        _append_task_log(
            task,
            event="task_failed",
            phase="codex_running",
            level="error",
            message=result.error or "Codex execution failed.",
        )
        yield StreamEvent(
            type="progress",
            data={
                "event": "task_failed",
                "task": task.model_dump(),
                "error": result.error,
            },
        )
        yield StreamEvent(type="result", data={"success": False})
        return

    if task.generated_files:
        _append_task_log(
            task,
            event="files_written",
            phase="codex_running",
            level="success",
            message=f"Wrote {len(task.generated_files)} file(s) to workspace.",
            details={"files": task.generated_files},
        )

    # Review
    task.status = "ai_review"
    task.assignee = "claude"
    task.progress = 85
    task.updated_at = datetime.utcnow().isoformat()
    _append_task_log(
        task,
        event="task_reviewing",
        phase="ai_review",
        level="info",
        message="Claude started review.",
    )

    yield StreamEvent(
        type="progress",
        data={"event": "task_reviewing", "task": task.model_dump()},
    )

    review = await commander.review(task_dict, result.output)
    task.review_feedback = review.feedback

    if review.approved:
        task.status = "done"
        task.progress = 100
        for sub in task.subtasks:
            sub["done"] = True
        commander.accumulate_wisdom(task_dict, result.output)
        _append_task_log(
            task,
            event="task_reviewed",
            phase="ai_review",
            level="success",
            message="Claude approved task.",
            details={"feedback": review.feedback},
        )
    else:
        task.status = "human_review"
        task.progress = 90
        _append_task_log(
            task,
            event="task_reviewed",
            phase="ai_review",
            level="warning",
            message="Claude requested human review.",
            details={"feedback": review.feedback},
        )

    task.updated_at = datetime.utcnow().isoformat()

    yield StreamEvent(
        type="progress",
        data={
            "event": "task_reviewed",
            "task": task.model_dump(),
            "approved": review.approved,
            "feedback": review.feedback,
        },
    )
    yield StreamEvent(type="result", data={"success": review.approved})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_task_log(
    task: AgentTask,
    *,
    event: str,
    phase: str,
    message: str,
    level: str = "info",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    entry: Dict[str, Any] = {
        "id": f"log-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "phase": phase,
        "level": level,
        "message": message,
    }
    if details:
        entry["details"] = details
    task.execution_log.append(entry)
    # Keep bounded to avoid unbounded memory growth in long sessions.
    if len(task.execution_log) > 500:
        task.execution_log = task.execution_log[-500:]


def _merge_review_feedback(existing: Optional[str], notes: str) -> str:
    if not existing:
        return f"Human feedback: {notes}"
    return f"{existing}\n\nHuman feedback: {notes}"


def _load_context_pack(context_pack_id: str) -> Optional[dict]:
    try:
        from ...infrastructure.stores.repro_context_store import (
            SqlAlchemyReproContextStore,
        )

        store = SqlAlchemyReproContextStore()
        return store.get(context_pack_id)
    except Exception:
        log.exception("Failed to load context pack %s", context_pack_id)
        return None


def _build_tags(task_data: dict) -> List[str]:
    tags = []
    difficulty = task_data.get("difficulty", task_data.get("estimated_difficulty"))
    if difficulty:
        tags.append(difficulty)
    return tags


def _build_subtasks(index: int, task_data: dict) -> List[Dict[str, Any]]:
    subtasks = []
    criteria = task_data.get("acceptance_criteria", [])
    if isinstance(criteria, list):
        for j, criterion in enumerate(criteria):
            subtasks.append(
                {
                    "id": f"sub-{index}-{j}",
                    "title": criterion if isinstance(criterion, str) else str(criterion),
                    "done": False,
                }
            )
    return subtasks


def _find_task(task_id: str) -> Optional[AgentTask]:
    """Find a task across all sessions."""
    for session in _sessions.values():
        for task in session.tasks:
            if task.id == task_id:
                return task
    return None


def _find_session_for_task(task_id: str) -> Optional[BoardSession]:
    """Find the session containing a task."""
    for session in _sessions.values():
        for task in session.tasks:
            if task.id == task_id:
                return session
    return None
