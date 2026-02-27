"""SqlAlchemyReproContextStore — P2C context pack persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func

from paperbot.infrastructure.stores.models import ReproContextPackModel, ReproContextStageResultModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SqlAlchemyReproContextStore:
    """SQLAlchemy implementation of ReproContextPort."""

    def __init__(self, db_url: Optional[str] = None):
        self._provider = SessionProvider(db_url)

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def save(
        self,
        *,
        pack_id: str,
        user_id: str,
        paper_id: str,
        depth: str,
        pack_data: Dict[str, Any],
        paper_title: Optional[str] = None,
        project_id: Optional[str] = None,
        objective: Optional[str] = None,
        confidence_overall: float = 0.0,
        warning_count: int = 0,
    ) -> str:
        now = _utcnow()
        row = ReproContextPackModel(
            id=pack_id,
            user_id=user_id,
            project_id=project_id,
            paper_id=paper_id,
            paper_title=paper_title,
            depth=depth,
            status="running",
            objective=objective,
            confidence_overall=confidence_overall,
            warning_count=warning_count,
            created_at=now,
            updated_at=now,
        )
        row.set_pack(pack_data)
        with self._provider.session() as session:
            session.add(row)
            session.commit()
        return pack_id

    def update_status(
        self,
        pack_id: str,
        *,
        status: str,
        pack_data: Optional[Dict[str, Any]] = None,
        confidence_overall: Optional[float] = None,
        warning_count: Optional[int] = None,
        objective: Optional[str] = None,
    ) -> None:
        with self._provider.session() as session:
            row = session.get(ReproContextPackModel, pack_id)
            if row is None:
                return
            row.status = status
            row.updated_at = _utcnow()
            if pack_data is not None:
                row.set_pack(pack_data)
            if confidence_overall is not None:
                row.confidence_overall = confidence_overall
            if warning_count is not None:
                row.warning_count = warning_count
            if objective is not None:
                row.objective = objective
            session.commit()

    def soft_delete(self, pack_id: str) -> bool:
        with self._provider.session() as session:
            row = session.get(ReproContextPackModel, pack_id)
            if row is None or row.deleted_at is not None:
                return False
            row.deleted_at = _utcnow()
            row.updated_at = _utcnow()
            session.commit()
            return True

    def save_stage_result(
        self,
        *,
        pack_id: str,
        stage_name: str,
        status: str,
        result_data: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        duration_ms: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        row = ReproContextStageResultModel(
            context_pack_id=pack_id,
            stage_name=stage_name,
            status=status,
            result_json=json.dumps(result_data or {}, ensure_ascii=False),
            confidence=confidence,
            duration_ms=duration_ms,
            error_message=error_message,
            created_at=now,
        )
        with self._provider.session() as session:
            session.add(row)
            session.commit()

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def get(self, pack_id: str) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.get(ReproContextPackModel, pack_id)
            if row is None or row.deleted_at is not None:
                return None
            return self._row_to_full_dict(row)

    def get_observation(self, pack_id: str, observation_id: str) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            stage_rows = session.scalars(
                select(ReproContextStageResultModel)
                .where(ReproContextStageResultModel.context_pack_id == pack_id)
                .order_by(ReproContextStageResultModel.created_at.asc())
            ).all()

            for row in stage_rows:
                if not row.result_json:
                    continue
                try:
                    payload = json.loads(row.result_json)
                except Exception:
                    continue
                for obs in payload.get("observations", []) if isinstance(payload, dict) else []:
                    if isinstance(obs, dict) and obs.get("id") == observation_id:
                        return obs

            pack_row = session.get(ReproContextPackModel, pack_id)
            if pack_row is None or pack_row.deleted_at is not None:
                return None
            pack = pack_row.get_pack() if pack_row else {}
            for obs in pack.get("observations", []) if isinstance(pack, dict) else []:
                if isinstance(obs, dict) and obs.get("id") == observation_id:
                    return obs
        return None

    def list_by_user(
        self,
        *,
        user_id: str,
        paper_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._provider.session() as session:
            base = (
                select(ReproContextPackModel)
                .where(
                    ReproContextPackModel.user_id == user_id,
                    ReproContextPackModel.deleted_at.is_(None),
                )
                .order_by(ReproContextPackModel.created_at.desc())
            )
            if paper_id:
                base = base.where(ReproContextPackModel.paper_id == paper_id)
            if project_id:
                base = base.where(ReproContextPackModel.project_id == project_id)

            total = session.scalar(
                select(func.count()).select_from(base.subquery())
            ) or 0

            rows = session.scalars(base.limit(limit).offset(offset)).all()
            summaries = [self._row_to_summary_dict(r) for r in rows]
            return summaries, total

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _row_to_summary_dict(self, row: ReproContextPackModel) -> Dict[str, Any]:
        return {
            "context_pack_id": row.id,
            "paper_id": row.paper_id,
            "paper_title": row.paper_title,
            "depth": row.depth,
            "status": row.status,
            "confidence_overall": row.confidence_overall,
            "warning_count": row.warning_count,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    def _row_to_full_dict(self, row: ReproContextPackModel) -> Dict[str, Any]:
        base = self._row_to_summary_dict(row)
        base.update({
            "user_id": row.user_id,
            "project_id": row.project_id,
            "objective": row.objective,
            "version": row.version,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "pack": row.get_pack(),
        })
        return base
