"""
Agent Board API -- Claude Commander + Codex Workers

Claude decomposes the context pack into tasks, dispatches them to
Codex API workers, reviews results, and manages the Kanban lifecycle.
"""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...infrastructure.stores.pipeline_session_store import PipelineSessionStore
from ..streaming import StreamEvent, sse_response

router = APIRouter(prefix="/api/agent-board")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent session store
# ---------------------------------------------------------------------------
_DEFAULT_WORKSPACE_DIR = Path("/tmp/paperbot-workspace")
_SAFE_WORKSPACE_PATH_RE = re.compile(r"^[A-Za-z0-9._/\\~ -]+$")

_DANGEROUS_WORKSPACE_ROOTS = (
    Path("/etc"),
    Path("/root"),
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/lib"),
    Path("/lib64"),
    Path("/System"),
    Path("/Library"),
    Path("/Applications"),
)
_board_store: Optional[PipelineSessionStore] = None

# Shared commander instance (preserves wisdom across tasks)
_commander = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_board_store() -> PipelineSessionStore:
    global _board_store
    if _board_store is None:
        _board_store = PipelineSessionStore(auto_create_schema=True)
    return _board_store


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
    status: Literal["planning", "in_progress", "ai_review", "human_review", "done"] = "planning"
    assignee: str = "claude"
    progress: int = 0
    tags: List[str] = []
    subtasks: List[Dict[str, Any]] = []
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoardSession":
        session = cls(
            session_id=str(data.get("session_id", "")),
            paper_id=str(data.get("paper_id", "")),
            context_pack_id=str(data.get("context_pack_id", "")),
            workspace_dir=data.get("workspace_dir"),
        )
        session.created_at = str(data.get("created_at", session.created_at))
        tasks_raw = data.get("tasks", [])
        if isinstance(tasks_raw, list):
            session.tasks = [AgentTask.model_validate(item) for item in tasks_raw]
        return session


class PlanRequest(BaseModel):
    paper_id: str
    context_pack_id: str
    workspace_dir: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    status: Optional[Literal["planning", "in_progress", "ai_review", "human_review", "done"]] = None
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
    workspace_dir = (
        str(_sanitize_workspace_dir(request.workspace_dir))
        if request.workspace_dir and request.workspace_dir.strip()
        else None
    )
    session_id = f"board-{uuid.uuid4().hex[:12]}"
    session = BoardSession(
        session_id=session_id,
        paper_id=request.paper_id,
        context_pack_id=request.context_pack_id,
        workspace_dir=workspace_dir,
    )
    _persist_session(session, checkpoint="created", status="running")
    return session.to_dict()


@router.post("/sessions/{session_id}/plan")
async def plan_session(session_id: str):
    """Claude decomposes context pack into tasks (SSE stream)."""
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return sse_response(_plan_stream(session), workflow="agent_board_plan")


@router.post("/sessions/{session_id}/run")
async def run_all_tasks(session_id: str, request: RunAllRequest):
    """Dispatch all planning tasks to Codex workers (SSE stream)."""
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.workspace_dir and request.workspace_dir.strip():
        session.workspace_dir = str(_sanitize_workspace_dir(request.workspace_dir))
    elif session.workspace_dir:
        session.workspace_dir = str(_sanitize_workspace_dir(session.workspace_dir))
    else:
        session.workspace_dir = str(_DEFAULT_WORKSPACE_DIR)
    _persist_session(session, checkpoint="run_requested", status="running")

    return sse_response(_run_all_stream(session), workflow="agent_board_run")


@router.get("/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    """List all tasks in a session."""
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [task.model_dump() for task in session.tasks]


@router.post("/tasks/{task_id}/dispatch")
async def dispatch_task(task_id: str):
    """Dispatch a single task to a Codex worker."""
    match = _find_task_with_session(task_id)
    if not match:
        raise HTTPException(status_code=404, detail="Task not found")
    session, task = match

    task.status = "in_progress"
    task.assignee = f"codex-{uuid.uuid4().hex[:4]}"
    task.updated_at = datetime.utcnow().isoformat()
    _persist_session(session, checkpoint="task_dispatched", status="running")

    return task.model_dump()


@router.post("/tasks/{task_id}/execute")
async def execute_task(task_id: str):
    """Execute a single task: Codex runs, Claude reviews (SSE stream)."""
    match = _find_task_with_session(task_id)
    if not match:
        raise HTTPException(status_code=404, detail="Task not found")
    session, task = match

    if session and session.workspace_dir:
        session.workspace_dir = str(_sanitize_workspace_dir(session.workspace_dir))

    return sse_response(_execute_task_stream(task, session), workflow="agent_board_execute")


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, request: TaskUpdateRequest):
    """Update a task's status or other fields."""
    match = _find_task_with_session(task_id)
    if not match:
        raise HTTPException(status_code=404, detail="Task not found")
    session, task = match

    if request.status:
        task.status = request.status
    if request.progress is not None:
        task.progress = request.progress
    if request.assignee:
        task.assignee = request.assignee
    task.updated_at = datetime.utcnow().isoformat()
    _persist_session(session, checkpoint="task_updated", status="running")

    return task.model_dump()


@router.post("/tasks/{task_id}/human-review")
async def human_review_task(task_id: str, request: HumanReviewRequest):
    """Allow a user to approve or request changes for a human-review task."""
    match = _find_task_with_session(task_id)
    if not match:
        raise HTTPException(status_code=404, detail="Task not found")
    session, task = match

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
    _persist_session(session, checkpoint="human_reviewed", status="running")
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
            _persist_session(session, checkpoint="planned", status="running")
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
        _persist_session(session, checkpoint="plan_complete", status="running")

    except Exception as exc:
        log.exception("Planning failed")
        _persist_session(session, checkpoint="plan_failed", status="failed")
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
    workspace = _sanitize_workspace_dir(session.workspace_dir or str(_DEFAULT_WORKSPACE_DIR))

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
        _persist_session(session, checkpoint="task_dispatched", status="running")

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
        _persist_session(session, checkpoint="task_codex_done", status="running")

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
            _persist_session(session, checkpoint="task_failed", status="running")
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
        _persist_session(session, checkpoint="task_reviewed", status="running")

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
    final_status = "completed" if done_count == total else "running"
    _persist_session(session, checkpoint="run_complete", status=final_status)


async def _execute_task_stream(
    task: AgentTask,
    session: Optional["BoardSession"],
) -> AsyncGenerator[StreamEvent, None]:
    """Execute a single task through Codex + Claude review."""
    commander = _get_commander()
    dispatcher = _get_dispatcher()
    workspace = _sanitize_workspace_dir(
        (session.workspace_dir if session else None) or str(_DEFAULT_WORKSPACE_DIR)
    )

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
    if session:
        _persist_session(session, checkpoint="task_dispatched", status="running")

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
    if session:
        _persist_session(session, checkpoint="task_codex_done", status="running")

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
        if session:
            _persist_session(session, checkpoint="task_failed", status="running")
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
    if session:
        _persist_session(session, checkpoint="task_reviewed", status="running")
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
    if session:
        _persist_session(session, checkpoint="task_execute_complete", status="running")


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


def _session_payload(session: BoardSession) -> Dict[str, Any]:
    return {
        "paper_id": session.paper_id,
        "context_pack_id": session.context_pack_id,
        "workspace_dir": session.workspace_dir,
    }


def _persist_session(
    session: BoardSession,
    *,
    checkpoint: str,
    status: str,
) -> None:
    store = _get_board_store()
    payload = _session_payload(session)
    snapshot = session.to_dict()

    try:
        existing = store.get_session(session.session_id)
        if existing is None:
            store.start_session(
                workflow="agent_board",
                payload=payload,
                session_id=session.session_id,
                resume=False,
            )

        store.update_status(
            session_id=session.session_id,
            status=status,
            checkpoint=checkpoint,
            state_patch={"board_session": snapshot},
            result={"board_session": snapshot},
        )
    except Exception:
        log.exception("Failed to persist agent board session %s", session.session_id)


def _load_session(session_id: str) -> Optional[BoardSession]:
    row = _get_board_store().get_session(session_id)
    if not row or row.get("workflow") != "agent_board":
        return None
    return _session_from_store_row(row)


def _session_from_store_row(row: Dict[str, Any]) -> Optional[BoardSession]:
    if not isinstance(row, dict):
        return None

    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    snapshot = state.get("board_session") if isinstance(state, dict) else None

    if isinstance(snapshot, dict):
        try:
            return BoardSession.from_dict(snapshot)
        except Exception:
            log.exception("Failed to parse board session snapshot")
            return None

    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    session_id = str(row.get("session_id", "")).strip()
    if not session_id:
        return None
    return BoardSession(
        session_id=session_id,
        paper_id=str(payload.get("paper_id", "")),
        context_pack_id=str(payload.get("context_pack_id", "")),
        workspace_dir=payload.get("workspace_dir"),
    )


def _find_task_with_session(task_id: str) -> Optional[tuple[BoardSession, AgentTask]]:
    scan_limit = _session_scan_limit()
    rows = _get_board_store().list_sessions(workflow="agent_board", limit=scan_limit)
    for row in rows:
        session = _session_from_store_row(row)
        if not session:
            continue
        for task in session.tasks:
            if task.id == task_id:
                return session, task
    return None


def _session_scan_limit() -> int:
    raw = os.getenv("PAPERBOT_AGENT_BOARD_SCAN_LIMIT", "500").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 500
    return max(50, min(value, 5000))


def _sanitize_workspace_dir(raw_path: str) -> Path:
    candidate_text = (raw_path or "").strip()
    if not candidate_text:
        raise HTTPException(status_code=400, detail="workspace_dir is required")
    if "\x00" in candidate_text:
        raise HTTPException(status_code=400, detail="workspace_dir contains invalid characters")

    expanded_text = os.path.expanduser(candidate_text)
    drive, tail = os.path.splitdrive(expanded_text)
    if drive:
        if len(drive) != 2 or not drive[0].isalpha() or drive[1] != ":":
            raise HTTPException(status_code=400, detail="workspace_dir contains an invalid drive prefix")
        if ":" in tail:
            raise HTTPException(status_code=400, detail="workspace_dir contains invalid characters")
        path_to_validate = drive[0] + tail
    else:
        path_to_validate = expanded_text

    if not _SAFE_WORKSPACE_PATH_RE.fullmatch(path_to_validate):
        raise HTTPException(status_code=400, detail="workspace_dir contains invalid characters")

    segments = _workspace_segments(tail or expanded_text)
    if _looks_absolute_workspace_path(expanded_text):
        candidate = _workspace_from_absolute_input(expanded_text)
    else:
        candidate = _workspace_join(Path.cwd().resolve(strict=False), segments)

    for blocked_root in _DANGEROUS_WORKSPACE_ROOTS:
        blocked_root_resolved = blocked_root.resolve(strict=False)
        if _is_within(candidate, blocked_root_resolved):
            raise HTTPException(
                status_code=400,
                detail=f"workspace_dir is not allowed: {candidate}",
            )

    allowed_roots = _workspace_allowed_roots()
    if not any(_is_within(candidate, root) for root in allowed_roots):
        raise HTTPException(
            status_code=400,
            detail=(
                "workspace_dir must be under an allowed root: "
                + ", ".join(str(root) for root in allowed_roots)
            ),
        )

    return candidate


def _workspace_allowed_roots() -> List[Path]:
    configured = os.getenv("PAPERBOT_WORKSPACE_ALLOWED_ROOTS", "").strip()
    roots: List[Path] = []

    if configured:
        raw_items = [item.strip() for item in configured.split(os.pathsep)]
        for item in raw_items:
            if item:
                roots.append(Path(item).expanduser().resolve(strict=False))
    else:
        roots.extend(
            [
                Path.home().resolve(strict=False),
                Path.cwd().resolve(strict=False),
                Path(tempfile.gettempdir()).resolve(strict=False),
            ]
        )

    roots.append(_DEFAULT_WORKSPACE_DIR.resolve(strict=False))

    unique: List[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _workspace_segments(raw_path: str) -> List[str]:
    segments = [segment for segment in re.split(r"[\\/]+", raw_path) if segment not in ("", ".")]
    if any(segment == ".." for segment in segments):
        raise HTTPException(status_code=400, detail="workspace_dir must not contain parent traversal")
    return segments


def _workspace_join(root: Path, segments: List[str]) -> Path:
    root_real = os.path.realpath(str(root))
    candidate_real = os.path.realpath(os.path.join(root_real, *segments))
    if os.path.commonpath([root_real, candidate_real]) != root_real:
        raise HTTPException(status_code=400, detail="workspace_dir must stay within an allowed root")
    return Path(candidate_real)


def _looks_absolute_workspace_path(value: str) -> bool:
    return os.path.isabs(value) or value.startswith(("/", "\\"))


def _normalized_workspace_text(value: str) -> str:
    normalized = value.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        normalized = normalized[0].upper() + normalized[1:]
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def _workspace_root_variants(root: Path) -> List[str]:
    normalized = _normalized_workspace_text(str(root))
    variants = [normalized]
    if len(normalized) >= 2 and normalized[1] == ":":
        drive_less = normalized[2:] or "/"
        variants.append(drive_less)
    unique: List[str] = []
    for value in variants:
        if value not in unique:
            unique.append(value)
    return unique


def _workspace_relative_suffix(normalized_input: str, normalized_root: str) -> Optional[str]:
    if normalized_input == normalized_root:
        return ""
    prefix = normalized_root.rstrip("/") + "/"
    if normalized_input.startswith(prefix):
        return normalized_input[len(prefix):]
    return None


def _workspace_from_absolute_input(raw_path: str) -> Path:
    normalized_input = _normalized_workspace_text(raw_path)

    for blocked_root in _DANGEROUS_WORKSPACE_ROOTS:
        for normalized_blocked in _workspace_root_variants(blocked_root):
            if _workspace_relative_suffix(normalized_input, normalized_blocked) is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"workspace_dir is not allowed: {raw_path}",
                )

    allowed_roots = _workspace_allowed_roots()
    for root in allowed_roots:
        for normalized_root in _workspace_root_variants(root):
            suffix = _workspace_relative_suffix(normalized_input, normalized_root)
            if suffix is None:
                continue
            suffix_segments = _workspace_segments(suffix)
            return _workspace_join(root, suffix_segments)
    raise HTTPException(
        status_code=400,
        detail=(
            "workspace_dir must be under an allowed root: "
            + ", ".join(str(root) for root in allowed_roots)
        ),
    )


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
