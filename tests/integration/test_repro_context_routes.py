"""Integration tests for P2C repro_context API endpoints.

Covers:
  - GET    /api/research/repro/context           (list)
  - GET    /api/research/repro/context/{pack_id} (get)
  - POST   /api/research/repro/context/{pack_id}/session (create session)
  - DELETE /api/research/repro/context/{pack_id} (soft-delete)
  - POST   /api/research/repro/context/generate  (SSE streaming)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import repro_context as repro_context_module
from paperbot.infrastructure.stores.models import Base
from paperbot.infrastructure.stores.repro_context_store import SqlAlchemyReproContextStore


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _parse_sse_events(text: str) -> List[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except Exception:
                pass
    return events


def _make_store(tmp_path: Path) -> SqlAlchemyReproContextStore:
    db_url = f"sqlite:///{tmp_path / 'test_repro.db'}"
    store = SqlAlchemyReproContextStore(db_url=db_url)
    Base.metadata.create_all(store._provider.engine)
    return store


def _seed_pack(store: SqlAlchemyReproContextStore, pack_id: str = "pack_abc123", **kwargs):
    defaults = dict(
        pack_id=pack_id,
        user_id="test_user",
        paper_id="arxiv:2401.00001",
        depth="fast",
        pack_data={"paper_id": "arxiv:2401.00001", "objective": "Reproduce method X"},
        project_id="proj_1",
        confidence_overall=0.85,
        warning_count=1,
    )
    defaults.update(kwargs)
    store.save(**defaults)
    # Mark as completed so get() returns pack data
    store.update_status(
        pack_id,
        status="completed",
        pack_data=defaults["pack_data"],
        confidence_overall=defaults["confidence_overall"],
        warning_count=defaults["warning_count"],
    )
    return defaults


@pytest.fixture()
def client_and_store(tmp_path: Path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(repro_context_module, "_store", store)
    with TestClient(api_main.app) as client:
        yield client, store


# ------------------------------------------------------------------ #
# GET /api/research/repro/context  (list)                              #
# ------------------------------------------------------------------ #


def test_list_packs_empty(client_and_store):
    client, _ = client_and_store
    resp = client.get("/api/research/repro/context", params={"user_id": "nobody"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_list_packs_returns_seeded(client_and_store):
    client, store = client_and_store
    _seed_pack(store, pack_id="pack_1", user_id="alice")
    _seed_pack(store, pack_id="pack_2", user_id="alice", paper_id="arxiv:2401.99999")

    resp = client.get("/api/research/repro/context", params={"user_id": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


def test_list_packs_filters_by_paper_id(client_and_store):
    client, store = client_and_store
    _seed_pack(store, pack_id="pack_a", user_id="bob", paper_id="arxiv:1111.11111")
    _seed_pack(store, pack_id="pack_b", user_id="bob", paper_id="arxiv:2222.22222")

    resp = client.get(
        "/api/research/repro/context",
        params={"user_id": "bob", "paper_id": "arxiv:1111.11111"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1


# ------------------------------------------------------------------ #
# GET /api/research/repro/context/{pack_id}  (get detail)              #
# ------------------------------------------------------------------ #


def test_get_pack_found(client_and_store):
    client, store = client_and_store
    _seed_pack(store, pack_id="pack_detail")

    resp = client.get("/api/research/repro/context/pack_detail")
    assert resp.status_code == 200
    body = resp.json()
    assert body["context_pack_id"] == "pack_detail"


def test_get_pack_not_found(client_and_store):
    client, _ = client_and_store
    resp = client.get("/api/research/repro/context/nonexistent")
    assert resp.status_code == 404


# ------------------------------------------------------------------ #
# POST /api/research/repro/context/{pack_id}/session                   #
# ------------------------------------------------------------------ #


def test_create_session_success(client_and_store):
    client, store = client_and_store
    _seed_pack(
        store,
        pack_id="pack_sess",
        pack_data={
            "paper_id": "arxiv:2401.00001",
            "task_roadmap": [
                {"title": "Setup env"},
                {"title": "Implement model"},
            ],
        },
    )

    resp = client.post(
        "/api/research/repro/context/pack_sess/session",
        json={"executor_preference": "auto"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"].startswith("sess_")
    assert body["runbook_id"].startswith("rb_")
    assert len(body["initial_steps"]) == 2
    assert body["initial_steps"][0]["title"] == "Setup env"
    assert "initial_prompt" in body


def test_create_session_pack_not_found(client_and_store):
    client, _ = client_and_store
    resp = client.post(
        "/api/research/repro/context/nonexistent/session",
        json={"executor_preference": "auto"},
    )
    assert resp.status_code == 404


# ------------------------------------------------------------------ #
# DELETE /api/research/repro/context/{pack_id}                         #
# ------------------------------------------------------------------ #


def test_delete_pack_success(client_and_store):
    client, store = client_and_store
    _seed_pack(store, pack_id="pack_del")

    resp = client.delete("/api/research/repro/context/pack_del")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Should no longer be retrievable
    resp2 = client.get("/api/research/repro/context/pack_del")
    assert resp2.status_code == 404


def test_delete_pack_not_found(client_and_store):
    client, _ = client_and_store
    resp = client.delete("/api/research/repro/context/nonexistent")
    assert resp.status_code == 404


# ------------------------------------------------------------------ #
# POST /api/research/repro/context/generate  (SSE)                     #
# ------------------------------------------------------------------ #


def test_generate_stream_returns_sse(client_and_store, monkeypatch):
    """Verify the generate endpoint returns a valid SSE stream.

    We monkeypatch the ExtractionOrchestrator to avoid real LLM calls.
    """
    client, store = client_and_store

    from dataclasses import dataclass, field

    @dataclass
    class _FakeConfidence:
        overall: float = 0.9

    @dataclass
    class _FakeObservation:
        id: str = "obs_1"
        type: str = "method"
        title: str = "Fake observation"
        confidence: float = 0.9

        def to_full(self):
            return {"id": self.id, "type": self.type, "title": self.title, "confidence": self.confidence}

    @dataclass
    class _FakePack:
        paper_id: str = "arxiv:2401.00001"
        objective: str = "Reproduce"
        observations: list = field(default_factory=lambda: [_FakeObservation()])
        warnings: list = field(default_factory=list)
        confidence: _FakeConfidence = field(default_factory=_FakeConfidence)
        task_roadmap: list = field(default_factory=list)

    class _FakeOrchestrator:
        async def run(self, request, on_stage_complete=None):
            if on_stage_complete:
                await on_stage_complete("extraction", [_FakeObservation()], [])
            return _FakePack()

    monkeypatch.setattr(repro_context_module, "ExtractionOrchestrator", _FakeOrchestrator)

    resp = client.post(
        "/api/research/repro/context/generate",
        json={"paper_id": "arxiv:2401.00001", "user_id": "test", "depth": "fast"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    events = _parse_sse_events(resp.text)
    assert len(events) >= 1

    event_types = [e.get("type") for e in events]
    assert "status" in event_types or "result" in event_types
    # Final event should be a result
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["status"] == "completed"
