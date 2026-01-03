"""
Memory metric collector for Scope and Acceptance criteria.

Collects and stores evaluation metrics defined in docs/memory_types.md:
- extraction_precision: >= 85%
- false_positive_rate: <= 5%
- retrieval_hit_rate: >= 80%
- injection_pollution_rate: <= 2%
- deletion_compliance: 100%
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func

from paperbot.infrastructure.stores.models import Base, MemoryEvalMetricModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


class MemoryMetricCollector:
    """
    Collects and stores memory system evaluation metrics.

    Usage:
        collector = MemoryMetricCollector(db_provider)

        # Record extraction results
        collector.record_extraction_precision(
            correct_count=85,
            total_count=100,
            evaluator_id="human:reviewer1"
        )

        # Get summary
        summary = collector.get_metrics_summary()
    """

    # P0 target thresholds
    TARGETS = {
        "extraction_precision": 0.85,
        "false_positive_rate": 0.05,
        "retrieval_hit_rate": 0.80,
        "injection_pollution_rate": 0.02,
        "deletion_compliance": 1.00,
    }

    def __init__(self, db_url: Optional[str] = None):
        self._provider = SessionProvider(db_url or get_db_url())
        # Ensure table exists
        Base.metadata.create_all(self._provider.engine)

    def record_extraction_precision(
        self,
        correct_count: int,
        total_count: int,
        evaluator_id: str = "system",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record extraction precision metric (correct / total extracted)."""
        if total_count == 0:
            return
        value = correct_count / total_count
        self._record_metric(
            metric_name="extraction_precision",
            metric_value=value,
            sample_size=total_count,
            evaluator_id=evaluator_id,
            detail=detail,
        )

    def record_false_positive_rate(
        self,
        false_positive_count: int,
        total_approved_count: int,
        evaluator_id: str = "system",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record false positive rate (incorrect approved / total approved)."""
        if total_approved_count == 0:
            return
        value = false_positive_count / total_approved_count
        self._record_metric(
            metric_name="false_positive_rate",
            metric_value=value,
            sample_size=total_approved_count,
            evaluator_id=evaluator_id,
            detail=detail,
        )

    def record_retrieval_hit_rate(
        self,
        hits: int,
        expected: int,
        evaluator_id: str = "system",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record retrieval hit rate (retrieved relevant / total relevant)."""
        if expected == 0:
            return
        value = hits / expected
        self._record_metric(
            metric_name="retrieval_hit_rate",
            metric_value=value,
            sample_size=expected,
            evaluator_id=evaluator_id,
            detail=detail,
        )

    def record_injection_pollution_rate(
        self,
        polluted_count: int,
        total_injections: int,
        evaluator_id: str = "system",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record injection pollution rate (polluted responses / total with memory)."""
        if total_injections == 0:
            return
        value = polluted_count / total_injections
        self._record_metric(
            metric_name="injection_pollution_rate",
            metric_value=value,
            sample_size=total_injections,
            evaluator_id=evaluator_id,
            detail=detail,
        )

    def record_deletion_compliance(
        self,
        deleted_retrieved_count: int,
        deleted_total_count: int,
        evaluator_id: str = "system",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record deletion compliance (should be 1.0 = 100% compliant).

        Compliance = 1.0 - (deleted_retrieved / deleted_total)
        If any deleted item was retrieved, compliance < 1.0.
        """
        if deleted_total_count == 0:
            return
        value = 1.0 - (deleted_retrieved_count / deleted_total_count)
        self._record_metric(
            metric_name="deletion_compliance",
            metric_value=value,
            sample_size=deleted_total_count,
            evaluator_id=evaluator_id,
            detail=detail,
        )

    def _record_metric(
        self,
        metric_name: str,
        metric_value: float,
        sample_size: int,
        evaluator_id: str,
        detail: Optional[Dict[str, Any]],
    ) -> None:
        """Store a metric record in the database."""
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = MemoryEvalMetricModel(
                metric_name=metric_name,
                metric_value=metric_value,
                sample_size=sample_size,
                evaluated_at=now,
                evaluator_id=evaluator_id,
                detail_json=json.dumps(detail or {}, ensure_ascii=False),
            )
            session.add(row)
            session.commit()

    def get_latest_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get the most recent value for each metric type."""
        result = {}
        with self._provider.session() as session:
            for metric_name in self.TARGETS.keys():
                stmt = (
                    select(MemoryEvalMetricModel)
                    .where(MemoryEvalMetricModel.metric_name == metric_name)
                    .order_by(MemoryEvalMetricModel.evaluated_at.desc())
                    .limit(1)
                )
                row = session.execute(stmt).scalar_one_or_none()
                if row:
                    result[metric_name] = {
                        "value": row.metric_value,
                        "target": self.TARGETS[metric_name],
                        "meets_target": self._meets_target(metric_name, row.metric_value),
                        "sample_size": row.sample_size,
                        "evaluated_at": row.evaluated_at.isoformat() if row.evaluated_at else None,
                        "evaluator_id": row.evaluator_id,
                    }
        return result

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics with pass/fail status."""
        latest = self.get_latest_metrics()
        all_pass = all(m.get("meets_target", False) for m in latest.values())
        return {
            "status": "pass" if all_pass else "fail",
            "metrics": latest,
            "targets": self.TARGETS,
        }

    def _meets_target(self, metric_name: str, value: float) -> bool:
        """Check if a metric value meets its target."""
        target = self.TARGETS.get(metric_name)
        if target is None:
            return True
        # For rates that should be LOW (false_positive, pollution), lower is better
        if metric_name in ("false_positive_rate", "injection_pollution_rate"):
            return value <= target
        # For rates that should be HIGH (precision, hit_rate, compliance), higher is better
        return value >= target

    def get_metric_history(
        self, metric_name: str, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical values for a specific metric."""
        result = []
        with self._provider.session() as session:
            stmt = (
                select(MemoryEvalMetricModel)
                .where(MemoryEvalMetricModel.metric_name == metric_name)
                .order_by(MemoryEvalMetricModel.evaluated_at.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
            for row in rows:
                result.append({
                    "value": row.metric_value,
                    "sample_size": row.sample_size,
                    "evaluated_at": row.evaluated_at.isoformat() if row.evaluated_at else None,
                    "evaluator_id": row.evaluator_id,
                })
        return result
