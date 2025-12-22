from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
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

    # Sandbox-specific fields
    executor_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # e2b/docker/local
    timeout_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    paper_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    paper_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    events = relationship("AgentEventModel", back_populates="run", cascade="all, delete-orphan")
    logs = relationship("ExecutionLogModel", back_populates="run", cascade="all, delete-orphan")
    metrics = relationship("ResourceMetricModel", back_populates="run", cascade="all, delete-orphan")
    runbook_steps = relationship("RunbookStepModel", back_populates="run", cascade="all, delete-orphan")

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


class ExecutionLogModel(Base):
    """Execution log entries for sandbox runs"""
    __tablename__ = "execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("agent_runs.run_id"), index=True)

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    level: Mapped[str] = mapped_column(String(16), default="info")  # debug/info/warning/error
    message: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="system")  # stdout/stderr/executor/system

    run = relationship("AgentRunModel", back_populates="logs")


class ResourceMetricModel(Base):
    """Resource usage metrics for sandbox runs"""
    __tablename__ = "resource_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("agent_runs.run_id"), index=True)

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    cpu_percent: Mapped[float] = mapped_column(Float, default=0.0)
    memory_mb: Mapped[float] = mapped_column(Float, default=0.0)
    memory_limit_mb: Mapped[float] = mapped_column(Float, default=4096.0)

    run = relationship("AgentRunModel", back_populates="metrics")


class RunbookStepModel(Base):
    """
    Structured Runbook step records.

    Logs and metrics are stored separately (execution_logs/resource_metrics) keyed by run_id.
    This table provides step-level lifecycle and configuration for long-term evidence tracking.
    """
    __tablename__ = "runbook_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("agent_runs.run_id"), index=True)

    step_name: Mapped[str] = mapped_column(String(64), index=True)  # smoke/install/data/train/eval/report/...
    status: Mapped[str] = mapped_column(String(32), default="running")  # running/success/failed/error
    executor_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # docker/e2b/ssh_docker/...

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    command: Mapped[str] = mapped_column(Text, default="")
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    run = relationship("AgentRunModel", back_populates="runbook_steps")


