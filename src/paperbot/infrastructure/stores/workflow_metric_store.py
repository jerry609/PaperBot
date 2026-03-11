from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from paperbot.application.ports.workflow_metric_port import WorkflowMetricPort
from paperbot.infrastructure.stores.models import Base, WorkflowEvalMetricModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowMetricStore(WorkflowMetricPort):
    """Persist and aggregate workflow quality metrics (coverage/latency/status)."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    def record_metric(
        self,
        *,
        workflow: str,
        stage: str = "",
        status: str = "completed",
        track_id: Optional[int] = None,
        claim_count: int = 0,
        evidence_count: int = 0,
        elapsed_ms: float = 0.0,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        claims = max(0, int(claim_count or 0))
        evidences = max(0, int(evidence_count or 0))
        coverage_rate = (float(evidences) / float(claims)) if claims > 0 else 0.0

        row = WorkflowEvalMetricModel(
            ts=_utcnow(),
            workflow=(workflow or "unknown")[:64],
            stage=(stage or "")[:64],
            status=(status or "completed")[:32],
            track_id=int(track_id) if track_id is not None else None,
            claim_count=claims,
            evidence_count=evidences,
            coverage_rate=max(0.0, min(1.0, float(coverage_rate))),
            elapsed_ms=max(0.0, float(elapsed_ms or 0.0)),
            detail_json=json.dumps(detail or {}, ensure_ascii=False),
        )

        with self._provider.session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    def summarize(
        self,
        *,
        days: int = 7,
        workflow: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        window_days = max(1, min(int(days), 90))
        since = _utcnow() - timedelta(days=window_days)

        stmt = select(WorkflowEvalMetricModel).where(WorkflowEvalMetricModel.ts >= since)
        if workflow:
            stmt = stmt.where(WorkflowEvalMetricModel.workflow == str(workflow)[:64])
        if track_id is not None:
            stmt = stmt.where(WorkflowEvalMetricModel.track_id == int(track_id))

        with self._provider.session() as session:
            rows = session.execute(stmt).scalars().all()

        totals = {
            "runs": 0,
            "success_runs": 0,
            "failed_runs": 0,
            "avg_elapsed_ms": 0.0,
            "claim_count": 0,
            "evidence_count": 0,
            "coverage_rate": 0.0,
        }

        by_day: Dict[str, Dict[str, Any]] = {}
        by_workflow: Dict[str, Dict[str, Any]] = {}
        by_track: Dict[str, Dict[str, Any]] = {}
        failure_stages: Dict[str, int] = defaultdict(int)

        for row in rows:
            totals["runs"] += 1
            if row.status == "completed":
                totals["success_runs"] += 1
            else:
                totals["failed_runs"] += 1
                failure_stages[row.stage or "unknown"] += 1

            totals["avg_elapsed_ms"] += float(row.elapsed_ms or 0.0)
            totals["claim_count"] += int(row.claim_count or 0)
            totals["evidence_count"] += int(row.evidence_count or 0)

            date_key = (row.ts or _utcnow()).date().isoformat()
            day = by_day.setdefault(
                date_key,
                {
                    "date": date_key,
                    "runs": 0,
                    "success_runs": 0,
                    "failed_runs": 0,
                    "claim_count": 0,
                    "evidence_count": 0,
                    "avg_elapsed_ms": 0.0,
                    "coverage_rate": 0.0,
                },
            )
            self._add_to_bucket(day, row)

            wf = by_workflow.setdefault(
                row.workflow or "unknown",
                {
                    "workflow": row.workflow or "unknown",
                    "runs": 0,
                    "success_runs": 0,
                    "failed_runs": 0,
                    "claim_count": 0,
                    "evidence_count": 0,
                    "avg_elapsed_ms": 0.0,
                    "coverage_rate": 0.0,
                },
            )
            self._add_to_bucket(wf, row)

            track_key = str(row.track_id) if row.track_id is not None else "none"
            track_bucket = by_track.setdefault(
                track_key,
                {
                    "track_id": row.track_id,
                    "runs": 0,
                    "success_runs": 0,
                    "failed_runs": 0,
                    "claim_count": 0,
                    "evidence_count": 0,
                    "avg_elapsed_ms": 0.0,
                    "coverage_rate": 0.0,
                },
            )
            self._add_to_bucket(track_bucket, row)

        if totals["runs"] > 0:
            totals["avg_elapsed_ms"] = round(totals["avg_elapsed_ms"] / totals["runs"], 2)
        totals["coverage_rate"] = self._coverage(totals["claim_count"], totals["evidence_count"])

        return {
            "window_days": window_days,
            "totals": totals,
            "by_day": [self._finalize_bucket(by_day[k]) for k in sorted(by_day.keys())],
            "by_workflow": sorted(
                (self._finalize_bucket(v) for v in by_workflow.values()),
                key=lambda x: int(x.get("runs") or 0),
                reverse=True,
            ),
            "by_track": sorted(
                (self._finalize_bucket(v) for v in by_track.values()),
                key=lambda x: int(x.get("runs") or 0),
                reverse=True,
            ),
            "failure_stages": dict(
                sorted(failure_stages.items(), key=lambda kv: kv[1], reverse=True)
            ),
        }

    def _add_to_bucket(self, bucket: Dict[str, Any], row: WorkflowEvalMetricModel) -> None:
        bucket["runs"] = int(bucket.get("runs") or 0) + 1
        if row.status == "completed":
            bucket["success_runs"] = int(bucket.get("success_runs") or 0) + 1
        else:
            bucket["failed_runs"] = int(bucket.get("failed_runs") or 0) + 1
        bucket["claim_count"] = int(bucket.get("claim_count") or 0) + int(row.claim_count or 0)
        bucket["evidence_count"] = int(bucket.get("evidence_count") or 0) + int(
            row.evidence_count or 0
        )
        bucket["avg_elapsed_ms"] = float(bucket.get("avg_elapsed_ms") or 0.0) + float(
            row.elapsed_ms or 0.0
        )

    def _finalize_bucket(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        runs = max(0, int(bucket.get("runs") or 0))
        if runs > 0:
            bucket["avg_elapsed_ms"] = round(float(bucket.get("avg_elapsed_ms") or 0.0) / runs, 2)
        else:
            bucket["avg_elapsed_ms"] = 0.0
        bucket["coverage_rate"] = self._coverage(
            int(bucket.get("claim_count") or 0), int(bucket.get("evidence_count") or 0)
        )
        return bucket

    @staticmethod
    def _coverage(claim_count: int, evidence_count: int) -> float:
        claims = max(0, int(claim_count or 0))
        evidences = max(0, int(evidence_count or 0))
        if claims <= 0:
            return 0.0
        return round(max(0.0, min(1.0, evidences / claims)), 4)

    @staticmethod
    def _to_dict(row: WorkflowEvalMetricModel) -> Dict[str, Any]:
        try:
            detail = json.loads(row.detail_json or "{}")
            if not isinstance(detail, dict):
                detail = {}
        except Exception:
            detail = {}
        return {
            "id": int(row.id),
            "ts": row.ts.isoformat() if row.ts else None,
            "workflow": row.workflow,
            "stage": row.stage,
            "status": row.status,
            "track_id": row.track_id,
            "claim_count": int(row.claim_count or 0),
            "evidence_count": int(row.evidence_count or 0),
            "coverage_rate": float(row.coverage_rate or 0.0),
            "elapsed_ms": float(row.elapsed_ms or 0.0),
            "detail": detail,
        }

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass
