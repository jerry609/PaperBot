from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return uuid.uuid4().hex


def new_trace_id() -> str:
    # Keep same format as run_id for simplicity.
    return uuid.uuid4().hex


def new_span_id() -> str:
    # Span IDs can be shorter; keep it compact but unique enough.
    return uuid.uuid4().hex[:16]


@dataclass
class EvidenceRef:
    """Reference to evidence supporting a claim/event."""

    uri: str
    note: str = ""
    span: Optional[str] = None
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"uri": self.uri}
        if self.note:
            d["note"] = self.note
        if self.span:
            d["span"] = self.span
        if self.hash:
            d["hash"] = self.hash
        return d


@dataclass
class ArtifactRef:
    """Reference to an artifact produced during a run (file/report/patch)."""

    uri: str
    kind: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"uri": self.uri}
        if self.kind:
            d["kind"] = self.kind
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class AgentEventEnvelope:
    """
    Unified event envelope (Phase-0).

    This is intentionally vendor/framework-agnostic and can be persisted to DB later.
    """

    # Identity & tracing
    run_id: str
    trace_id: str
    span_id: str = field(default_factory=new_span_id)
    parent_span_id: Optional[str] = None

    # Orchestration semantics
    workflow: str = ""
    stage: str = ""
    attempt: int = 0

    # Actor
    agent_name: str = ""
    role: str = ""  # orchestrator/worker/evaluator/system

    # Payload
    type: str = ""  # tool_call/tool_result/insight/fact/score/error/stage_event/...
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceRef] = field(default_factory=list)
    artifacts: List[ArtifactRef] = field(default_factory=list)

    # Metrics & tags
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)

    # Timestamp
    ts: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "workflow": self.workflow,
            "stage": self.stage,
            "attempt": self.attempt,
            "agent_name": self.agent_name,
            "role": self.role,
            "type": self.type,
            "payload": self.payload,
            "evidence": [e.to_dict() for e in self.evidence],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "tags": self.tags,
            "ts": self.ts.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def make_event(
    *,
    run_id: str,
    trace_id: str,
    workflow: str,
    stage: str,
    attempt: int,
    agent_name: str,
    role: str,
    type: str,
    payload: Optional[Dict[str, Any]] = None,
    parent_span_id: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> AgentEventEnvelope:
    return AgentEventEnvelope(
        run_id=run_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        workflow=workflow,
        stage=stage,
        attempt=attempt,
        agent_name=agent_name,
        role=role,
        type=type,
        payload=payload or {},
        metrics=metrics or {},
        tags=tags or {},
    )


