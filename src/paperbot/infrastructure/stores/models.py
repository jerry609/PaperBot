from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AgentRunModel(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workflow: Mapped[str] = mapped_column(String(64), default="", index=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")

    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    events = relationship("AgentEventModel", back_populates="run", cascade="all, delete-orphan")

    def set_metadata(self, data: Dict[str, Any]) -> None:
        self.metadata_json = json.dumps(data or {}, ensure_ascii=False)

    def get_metadata(self) -> Dict[str, Any]:
        try:
            return json.loads(self.metadata_json or "{}")
        except Exception:
            return {}


class AgentEventModel(Base):
    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("agent_runs.run_id"), index=True)
    trace_id: Mapped[str] = mapped_column(String(64), default="", index=True)

    span_id: Mapped[str] = mapped_column(String(32), default="")
    parent_span_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    workflow: Mapped[str] = mapped_column(String(64), default="")
    stage: Mapped[str] = mapped_column(String(64), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=0)

    agent_name: Mapped[str] = mapped_column(String(128), default="")
    role: Mapped[str] = mapped_column(String(32), default="")
    type: Mapped[str] = mapped_column(String(64), default="")

    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    tags_json: Mapped[str] = mapped_column(Text, default="{}")

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    run = relationship("AgentRunModel", back_populates="events")

    def set_payload(self, payload: Dict[str, Any]) -> None:
        self.payload_json = json.dumps(payload or {}, ensure_ascii=False)

    def get_payload(self) -> Dict[str, Any]:
        try:
            return json.loads(self.payload_json or "{}")
        except Exception:
            return {}

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metrics_json = json.dumps(metrics or {}, ensure_ascii=False)

    def get_metrics(self) -> Dict[str, Any]:
        try:
            return json.loads(self.metrics_json or "{}")
        except Exception:
            return {}

    def set_tags(self, tags: Dict[str, Any]) -> None:
        self.tags_json = json.dumps(tags or {}, ensure_ascii=False)

    def get_tags(self) -> Dict[str, Any]:
        try:
            return json.loads(self.tags_json or "{}")
        except Exception:
            return {}


