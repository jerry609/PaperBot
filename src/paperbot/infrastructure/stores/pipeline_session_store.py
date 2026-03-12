from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import desc, select

from paperbot.application.ports.pipeline_session_port import PipelineSessionPort
from paperbot.infrastructure.stores.models import Base, PipelineSessionModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PipelineSessionStore(PipelineSessionPort):
    """Persist lightweight workflow checkpoints for resume support."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    def start_session(
        self,
        *,
        workflow: str,
        payload: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        resolved_id = (session_id or "").strip() or uuid4().hex
        now = _utcnow()

        with self._provider.session() as session:
            row = session.execute(
                select(PipelineSessionModel).where(PipelineSessionModel.session_id == resolved_id)
            ).scalar_one_or_none()

            if row and resume:
                row.updated_at = now
                session.add(row)
                session.commit()
                return self._to_dict(row)

            if row is None:
                row = PipelineSessionModel(
                    session_id=resolved_id,
                    workflow=(workflow or "")[:64],
                    status="running",
                    checkpoint="init",
                    payload_json=json.dumps(payload or {}, ensure_ascii=False),
                    state_json="{}",
                    result_json="{}",
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.workflow = (workflow or row.workflow or "")[:64]
                row.status = "running"
                row.checkpoint = "init"
                row.payload_json = json.dumps(payload or {}, ensure_ascii=False)
                row.state_json = "{}"
                row.result_json = "{}"
                row.updated_at = now
                session.add(row)

            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        with self._provider.session() as session:
            row = session.execute(
                select(PipelineSessionModel).where(PipelineSessionModel.session_id == sid)
            ).scalar_one_or_none()
            return self._to_dict(row) if row else None

    def list_sessions(
        self,
        *,
        workflow: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> list[Dict[str, Any]]:
        with self._provider.session() as session:
            stmt = select(PipelineSessionModel)
            if workflow:
                stmt = stmt.where(PipelineSessionModel.workflow == str(workflow)[:64])
            if status:
                stmt = stmt.where(PipelineSessionModel.status == str(status)[:32])
            stmt = stmt.order_by(desc(PipelineSessionModel.updated_at)).limit(max(1, int(limit)))
            rows = session.execute(stmt).scalars().all()
            return [self._to_dict(row) for row in rows]

    def save_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return None

        with self._provider.session() as session:
            row = session.execute(
                select(PipelineSessionModel).where(PipelineSessionModel.session_id == sid)
            ).scalar_one_or_none()
            if row is None:
                return None

            row.checkpoint = (checkpoint or "")[:64] or row.checkpoint
            row.state_json = json.dumps(state or {}, ensure_ascii=False)
            row.status = "running"
            row.updated_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def save_result(
        self,
        *,
        session_id: str,
        result: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return None

        with self._provider.session() as session:
            row = session.execute(
                select(PipelineSessionModel).where(PipelineSessionModel.session_id == sid)
            ).scalar_one_or_none()
            if row is None:
                return None

            row.status = (status or "completed")[:32]
            row.checkpoint = "result"
            row.result_json = json.dumps(result or {}, ensure_ascii=False)
            row.updated_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def mark_failed(self, *, session_id: str, error: str) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return None

        with self._provider.session() as session:
            row = session.execute(
                select(PipelineSessionModel).where(PipelineSessionModel.session_id == sid)
            ).scalar_one_or_none()
            if row is None:
                return None

            state = _safe_json_dict(row.state_json)
            state["error"] = str(error or "")
            row.status = "failed"
            row.state_json = json.dumps(state, ensure_ascii=False)
            row.updated_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def update_status(
        self,
        *,
        session_id: str,
        status: str,
        checkpoint: Optional[str] = None,
        state_patch: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return None

        with self._provider.session() as session:
            row = session.execute(
                select(PipelineSessionModel).where(PipelineSessionModel.session_id == sid)
            ).scalar_one_or_none()
            if row is None:
                return None

            row.status = (status or row.status or "running")[:32]
            if checkpoint:
                row.checkpoint = str(checkpoint)[:64]

            if state_patch:
                merged = _safe_json_dict(row.state_json)
                merged.update(state_patch)
                row.state_json = json.dumps(merged, ensure_ascii=False)

            if result is not None:
                row.result_json = json.dumps(result, ensure_ascii=False)

            row.updated_at = _utcnow()
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    @staticmethod
    def _to_dict(row: PipelineSessionModel) -> Dict[str, Any]:
        return {
            "session_id": row.session_id,
            "workflow": row.workflow,
            "status": row.status,
            "checkpoint": row.checkpoint,
            "payload": _safe_json_dict(row.payload_json),
            "state": _safe_json_dict(row.state_json),
            "result": _safe_json_dict(row.result_json),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass


def _safe_json_dict(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
