from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Union, List

from sqlalchemy import select, desc, asc

from paperbot.application.collaboration.message_schema import AgentEventEnvelope
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.stores.models import AgentRunModel, AgentEventModel, Base
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url

logger = logging.getLogger(__name__)


def _parse_ts(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            pass
    return datetime.now(timezone.utc)


class SqlAlchemyEventLog(EventLogPort):
    """
    Persist events into SQLite via SQLAlchemy.

    - append(): upsert run row, insert event row
    - stream(run_id): yield events ordered by ts
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            # Safety net for local dev/evals. In production, prefer Alembic migrations.
            Base.metadata.create_all(self._provider.engine)

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        evt: Dict[str, Any]
        if isinstance(event, AgentEventEnvelope):
            evt = event.to_dict()
        else:
            evt = dict(event)

        run_id = str(evt.get("run_id") or "")
        if not run_id:
            raise ValueError("Event missing run_id")

        workflow = str(evt.get("workflow") or "")
        ts = _parse_ts(evt.get("ts"))

        with self._provider.session() as session:
            # Upsert run row
            run = session.get(AgentRunModel, run_id)
            if run is None:
                run = AgentRunModel(
                    run_id=run_id,
                    workflow=workflow,
                    started_at=ts,
                    status="running",
                )
                run.set_metadata({"db_url": self.db_url})
                session.add(run)
            else:
                # fill workflow if missing
                if not run.workflow and workflow:
                    run.workflow = workflow

            ev = AgentEventModel(
                run_id=run_id,
                trace_id=str(evt.get("trace_id") or ""),
                span_id=str(evt.get("span_id") or ""),
                parent_span_id=evt.get("parent_span_id"),
                workflow=workflow,
                stage=str(evt.get("stage") or ""),
                attempt=int(evt.get("attempt") or 0),
                agent_name=str(evt.get("agent_name") or ""),
                role=str(evt.get("role") or ""),
                type=str(evt.get("type") or ""),
                ts=ts,
            )
            ev.payload_json = json.dumps(evt.get("payload") or {}, ensure_ascii=False)
            ev.metrics_json = json.dumps(evt.get("metrics") or {}, ensure_ascii=False)
            ev.tags_json = json.dumps(evt.get("tags") or {}, ensure_ascii=False)
            session.add(ev)
            session.commit()

    def stream(self, run_id: str) -> Iterable[dict]:
        with self._provider.session() as session:
            rows = session.execute(
                select(AgentEventModel).where(AgentEventModel.run_id == run_id).order_by(asc(AgentEventModel.ts))
            ).scalars()
            for row in rows:
                yield {
                    "run_id": row.run_id,
                    "trace_id": row.trace_id,
                    "span_id": row.span_id,
                    "parent_span_id": row.parent_span_id,
                    "workflow": row.workflow,
                    "stage": row.stage,
                    "attempt": row.attempt,
                    "agent_name": row.agent_name,
                    "role": row.role,
                    "type": row.type,
                    "payload": row.get_payload(),
                    "metrics": row.get_metrics(),
                    "tags": row.get_tags(),
                    "ts": row.ts.isoformat(),
                }

    def list_runs(self, limit: int = 50) -> List[dict]:
        with self._provider.session() as session:
            rows = session.execute(select(AgentRunModel).order_by(desc(AgentRunModel.started_at)).limit(limit)).scalars()
            return [
                {
                    "run_id": r.run_id,
                    "workflow": r.workflow,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                    "status": r.status,
                    "metadata": r.get_metadata(),
                }
                for r in rows
            ]

    def list_events(self, run_id: str, *, trace_id: Optional[str] = None, limit: int = 1000) -> List[dict]:
        with self._provider.session() as session:
            stmt = select(AgentEventModel).where(AgentEventModel.run_id == run_id)
            if trace_id:
                stmt = stmt.where(AgentEventModel.trace_id == trace_id)
            stmt = stmt.order_by(asc(AgentEventModel.ts)).limit(limit)
            rows = session.execute(stmt).scalars()
            return [
                {
                    "run_id": row.run_id,
                    "trace_id": row.trace_id,
                    "span_id": row.span_id,
                    "parent_span_id": row.parent_span_id,
                    "workflow": row.workflow,
                    "stage": row.stage,
                    "attempt": row.attempt,
                    "agent_name": row.agent_name,
                    "role": row.role,
                    "type": row.type,
                    "payload": row.get_payload(),
                    "metrics": row.get_metrics(),
                    "tags": row.get_tags(),
                    "ts": row.ts.isoformat(),
                }
                for row in rows
            ]

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass


