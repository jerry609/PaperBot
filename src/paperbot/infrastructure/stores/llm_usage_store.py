from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from paperbot.application.ports.llm_usage_port import LLMUsagePort
from paperbot.infrastructure.stores.models import Base, LLMUsageModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LLMUsageStore(LLMUsagePort):
    """Persist and aggregate LLM token/cost usage records."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    def record_usage(
        self,
        *,
        task_type: str,
        provider_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ts = _utcnow()
        total_tokens = max(0, int(prompt_tokens) + int(completion_tokens))
        row = LLMUsageModel(
            ts=ts,
            task_type=(task_type or "default")[:32],
            provider_name=(provider_name or "unknown")[:64],
            model_name=(model_name or "")[:128],
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
            total_tokens=total_tokens,
            estimated_cost_usd=max(0.0, float(estimated_cost_usd or 0.0)),
            metadata_json="{}",
        )

        import json

        row.metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._provider.session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return {
                "id": int(row.id),
                "ts": row.ts.isoformat() if row.ts else None,
                "task_type": row.task_type,
                "provider_name": row.provider_name,
                "model_name": row.model_name,
                "prompt_tokens": int(row.prompt_tokens or 0),
                "completion_tokens": int(row.completion_tokens or 0),
                "total_tokens": int(row.total_tokens or 0),
                "estimated_cost_usd": float(row.estimated_cost_usd or 0.0),
            }

    def summarize(self, *, days: int = 7) -> Dict[str, Any]:
        window_days = max(1, min(int(days), 90))
        since = _utcnow() - timedelta(days=window_days)

        with self._provider.session() as session:
            rows = (
                session.execute(select(LLMUsageModel).where(LLMUsageModel.ts >= since))
                .scalars()
                .all()
            )

        daily_map: Dict[str, Dict[str, Any]] = {}
        provider_model_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for row in rows:
            date_key = (row.ts or _utcnow()).date().isoformat()
            day = daily_map.setdefault(
                date_key,
                {
                    "date": date_key,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "providers": defaultdict(int),
                },
            )
            provider_key = (row.provider_name or "unknown").strip() or "unknown"
            model_key = (row.model_name or "").strip() or "unknown"
            total_tokens = int(row.total_tokens or 0)
            total_cost = float(row.estimated_cost_usd or 0.0)

            day["total_tokens"] += total_tokens
            day["total_cost_usd"] += total_cost
            day["providers"][provider_key] += total_tokens

            key = (provider_key, model_key)
            bucket = provider_model_map.setdefault(
                key,
                {
                    "provider_name": provider_key,
                    "model_name": model_key,
                    "calls": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                },
            )
            bucket["calls"] += 1
            bucket["total_tokens"] += total_tokens
            bucket["total_cost_usd"] += total_cost

        daily_rows: List[Dict[str, Any]] = []
        for date_key in sorted(daily_map.keys()):
            row = daily_map[date_key]
            daily_rows.append(
                {
                    "date": row["date"],
                    "total_tokens": int(row["total_tokens"]),
                    "total_cost_usd": round(float(row["total_cost_usd"]), 8),
                    "providers": dict(row["providers"]),
                }
            )

        provider_model_rows = sorted(
            provider_model_map.values(),
            key=lambda x: int(x["total_tokens"]),
            reverse=True,
        )

        totals = {
            "calls": int(sum(x["calls"] for x in provider_model_rows)),
            "total_tokens": int(sum(x["total_tokens"] for x in provider_model_rows)),
            "total_cost_usd": round(
                float(sum(x["total_cost_usd"] for x in provider_model_rows)), 8
            ),
        }

        return {
            "window_days": window_days,
            "daily": daily_rows,
            "provider_models": provider_model_rows,
            "totals": totals,
        }

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass
