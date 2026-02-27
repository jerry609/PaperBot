"""
P2C (Paper-to-Context) API Route

Endpoints:
  POST   /api/research/repro/context/generate          — generate context pack (SSE)
  GET    /api/research/repro/context                   — list packs for a user
  GET    /api/research/repro/context/{pack_id}         — get full pack detail
  POST   /api/research/repro/context/{pack_id}/session — create repro session from pack
  DELETE /api/research/repro/context/{pack_id}         — soft-delete pack
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict as _asdict
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from paperbot.api.streaming import StreamEvent, wrap_generator
from paperbot.application.services.p2c.models import (
    GenerateContextRequest as P2CRequest,
    new_context_pack_id,
)
from paperbot.application.services.p2c.orchestrator import ExtractionOrchestrator
from paperbot.infrastructure.stores.repro_context_store import SqlAlchemyReproContextStore
from paperbot.utils.logging_config import Logger, LogFiles, set_trace_id

router = APIRouter()

_store = SqlAlchemyReproContextStore()


# ------------------------------------------------------------------ #
# Request / Response schemas                                           #
# ------------------------------------------------------------------ #

class GenerateContextPackRequest(BaseModel):
    paper_id: str
    user_id: str = "default"
    project_id: Optional[str] = None
    track_id: Optional[int] = None
    depth: Literal["fast", "standard", "deep"] = "standard"


class CreateSessionRequest(BaseModel):
    executor_preference: Literal["auto", "claude_code", "codex", "local"] = "auto"
    override_env: Optional[dict] = None
    override_roadmap: Optional[list] = None


# ------------------------------------------------------------------ #
# POST /generate  (SSE)                                               #
# ------------------------------------------------------------------ #

async def _generate_stream(request: GenerateContextPackRequest):
    """SSE generator for context pack generation via Module 1 ExtractionOrchestrator."""
    pack_id = new_context_pack_id()
    Logger.info(
        f"[M2] start generate pack_id={pack_id} paper_id={request.paper_id} depth={request.depth}",
        file=LogFiles.API,
    )

    # Persist initial "running" record so the frontend can poll status.
    _store.save(
        pack_id=pack_id,
        user_id=request.user_id,
        paper_id=request.paper_id,
        depth=request.depth,
        pack_data={},
        project_id=request.project_id,
        confidence_overall=0.0,
        warning_count=0,
    )

    yield StreamEvent(type="status", data={"pack_id": pack_id, "status": "running"})

    # asyncio.Queue bridges the on_stage_complete callback → SSE generator.
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()
    _ERROR = object()

    total_stages = 2 if request.depth == "fast" else 6
    stages_done = 0
    last_stage_name: Optional[str] = None

    async def on_stage_complete(stage_name: str, observations: list, warnings: list) -> None:
        nonlocal stages_done
        nonlocal last_stage_name
        stages_done += 1
        last_stage_name = stage_name
        confidence = (
            sum(o.confidence for o in observations) / len(observations)
            if observations
            else 0.0
        )
        observation_summaries = [
            {
                "id": o.id,
                "type": o.type,
                "title": o.title,
                "confidence": o.confidence,
            }
            for o in observations
        ]
        _store.save_stage_result(
            pack_id=pack_id,
            stage_name=stage_name,
            status="completed",
            result_data={"observations": [o.to_full() for o in observations]},
            confidence=confidence,
            duration_ms=0,
        )
        Logger.info(
            f"[M2] stage_complete pack_id={pack_id} stage={stage_name} obs={len(observations)} warnings={len(warnings)}",
            file=LogFiles.API,
        )
        await queue.put(
            StreamEvent(
                type="stage_observations",
                data={
                    "stage": stage_name,
                    "observations": observation_summaries,
                },
            )
        )
        await queue.put(
            StreamEvent(
                type="progress",
                data={
                    "stage": stage_name,
                    "progress": stages_done / total_stages,
                    "message": f"Completed {stage_name}",
                },
            )
        )

    p2c_request = P2CRequest(
        paper_id=request.paper_id,
        user_id=request.user_id,
        project_id=request.project_id,
        track_id=request.track_id,
        depth=request.depth,
    )
    orchestrator = ExtractionOrchestrator()
    result_holder: list = []

    async def _run() -> None:
        try:
            pack = await orchestrator.run(p2c_request, on_stage_complete=on_stage_complete)
            result_holder.append(pack)
            await queue.put(_DONE)
        except Exception as exc:  # noqa: BLE001
            Logger.error(
                f"[M2] generation_failed pack_id={pack_id} error={exc}",
                file=LogFiles.ERROR,
            )
            result_holder.append(exc)
            await queue.put(_ERROR)

    asyncio.create_task(_run())

    # Drain queue, yielding progress events until the orchestrator finishes.
    while True:
        item = await queue.get()
        if item is _DONE:
            break
        elif item is _ERROR:
            _store.update_status(pack_id, status="failed")
            yield StreamEvent(type="error", data={"message": str(result_holder[0])})
            return
        else:
            yield item  # StreamEvent from on_stage_complete

    if last_stage_name:
        yield StreamEvent(
            type="progress",
            data={
                "stage": last_stage_name,
                "progress": 1.0,
                "message": f"Completed {last_stage_name}",
            },
        )

    # Serialize and persist the final pack.
    pack = result_holder[0]
    pack_dict = _asdict(pack)
    pack_dict["context_pack_id"] = pack_id  # align with our DB record

    _store.update_status(
        pack_id,
        status="completed",
        pack_data=pack_dict,
        confidence_overall=pack.confidence.overall,
        warning_count=len(pack.warnings),
        objective=pack.objective,
    )
    Logger.info(
        f"[M2] generation_completed pack_id={pack_id} observations={len(pack.observations)} warnings={len(pack.warnings)}",
        file=LogFiles.API,
    )

    yield StreamEvent(
        type="result",
        data={
            "context_pack_id": pack_id,
            "status": "completed",
            "summary": f"Context pack created for {request.paper_id}",
            "confidence": _asdict(pack.confidence),
            "warnings": pack.warnings,
            "next_action": "create_repro_session",
        },
    )


@router.post("/generate")
async def generate_context_pack(request: GenerateContextPackRequest):
    """Generate a P2C context pack for the given paper. Returns SSE stream."""
    trace_id = set_trace_id()
    Logger.info(
        f"[M2] generate_request trace_id={trace_id} paper_id={request.paper_id} user_id={request.user_id}",
        file=LogFiles.API,
    )
    return StreamingResponse(
        wrap_generator(
            _generate_stream(request),
            workflow="p2c_generate",
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ------------------------------------------------------------------ #
# GET /  (list)                                                        #
# ------------------------------------------------------------------ #

@router.get("")
async def list_context_packs(
    user_id: str = "default",
    paper_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """List context packs for a user, with optional filters."""
    set_trace_id()
    Logger.info(
        f"[M2] list_packs user_id={user_id} paper_id={paper_id} project_id={project_id} limit={limit} offset={offset}",
        file=LogFiles.API,
    )
    items, total = _store.list_by_user(
        user_id=user_id,
        paper_id=paper_id,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    return {"items": items, "total": total}


# ------------------------------------------------------------------ #
# GET /{pack_id}                                                       #
# ------------------------------------------------------------------ #

@router.get("/{pack_id}")
async def get_context_pack(pack_id: str):
    """Return the full context pack detail."""
    set_trace_id()
    Logger.info(f"[M2] get_pack pack_id={pack_id}", file=LogFiles.API)
    pack = _store.get(pack_id)
    if pack is None:
        Logger.warning(f"[M2] pack_not_found pack_id={pack_id}", file=LogFiles.API)
        raise HTTPException(status_code=404, detail="Context pack not found.")
    return pack


@router.get("/{pack_id}/observation/{observation_id}")
async def get_observation_detail(pack_id: str, observation_id: str):
    """Return a single observation detail by ID."""
    set_trace_id()
    Logger.info(
        f"[M2] get_observation pack_id={pack_id} observation_id={observation_id}",
        file=LogFiles.API,
    )
    observation = _store.get_observation(pack_id, observation_id)
    if observation is None:
        Logger.warning(
            f"[M2] observation_not_found pack_id={pack_id} observation_id={observation_id}",
            file=LogFiles.API,
        )
        raise HTTPException(status_code=404, detail="Observation not found.")
    return observation


# ------------------------------------------------------------------ #
# POST /{pack_id}/session                                             #
# ------------------------------------------------------------------ #

@router.post("/{pack_id}/session")
async def create_repro_session(pack_id: str, request: CreateSessionRequest):
    """
    Convert a context pack into a runbook session.

    TODO: integrate with existing runbook creation once Module 1 is wired.
    """
    pack = _store.get(pack_id)
    if pack is None:
        Logger.warning(f"[M2] session_pack_not_found pack_id={pack_id}", file=LogFiles.API)
        raise HTTPException(status_code=404, detail="Context pack not found.")
    Logger.info(
        f"[M2] create_session pack_id={pack_id} executor={request.executor_preference}",
        file=LogFiles.API,
    )

    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    runbook_id = f"rb_{uuid.uuid4().hex[:12]}"

    roadmap = pack.get("pack", {}).get("task_roadmap", [])
    initial_steps = [
        {
            "step_id": f"S{i + 1}",
            "title": cp.get("title", f"Step {i + 1}"),
            "command": "",
            "status": "pending",
        }
        for i, cp in enumerate(roadmap)
    ] or [{"step_id": "S1", "title": "Setup environment", "command": "", "status": "pending"}]

    return {
        "session_id": session_id,
        "runbook_id": runbook_id,
        "initial_steps": initial_steps,
        "initial_prompt": (
            f"Based on the reproduction context pack for paper {pack.get('paper_id', '')}, "
            "implement the code step by step following the roadmap."
        ),
    }


# ------------------------------------------------------------------ #
# DELETE /{pack_id}                                                   #
# ------------------------------------------------------------------ #

@router.delete("/{pack_id}")
async def delete_context_pack(pack_id: str):
    """Soft-delete a context pack."""
    set_trace_id()
    Logger.info(f"[M2] delete_pack pack_id={pack_id}", file=LogFiles.API)
    deleted = _store.soft_delete(pack_id)
    if not deleted:
        Logger.warning(f"[M2] delete_pack_not_found pack_id={pack_id}", file=LogFiles.API)
        raise HTTPException(status_code=404, detail="Context pack not found.")
    return {"status": "deleted", "context_pack_id": pack_id}
