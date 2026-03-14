from __future__ import annotations

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.auth import dependencies as auth_deps
from paperbot.api.routes import research as research_route


class _FakeWorkflowMetricStore:
    def __init__(self) -> None:
        self.track_ids: list[int | None] = []

    def record_metric(self, *, track_id=None, **kwargs) -> None:
        self.track_ids.append(track_id)


def _override_user_id(user_id: str):
    def _dep_override():
        return user_id

    return _dep_override


def test_context_route_uses_explicit_track_id_without_activation(monkeypatch):
    captured: dict[str, object] = {}
    metric_store = _FakeWorkflowMetricStore()

    class _FakeContextEngine:
        def __init__(self, **kwargs) -> None:
            captured["init_kwargs"] = kwargs

        async def build_context_pack(
            self,
            *,
            user_id: str,
            query: str,
            track_id: int | None = None,
            include_cross_track: bool = False,
        ):
            captured["user_id"] = user_id
            captured["query"] = query
            captured["track_id"] = track_id
            captured["include_cross_track"] = include_cross_track
            return {
                "routing": {"track_id": track_id},
                "paper_recommendations": [],
                "paper_recommendation_reasons": {},
            }

        async def close(self) -> None:
            return None

    monkeypatch.setattr(research_route, "_workflow_metric_store", metric_store)
    monkeypatch.setattr(research_route, "ContextEngine", _FakeContextEngine)

    app = api_main.app
    app.dependency_overrides[auth_deps.get_required_user_id] = _override_user_id("u-explicit")
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/research/context",
                json={
                    "query": "agentic retrieval",
                    "track_id": 42,
                    "paper_limit": 0,
                    "offline": True,
                    "include_cross_track": False,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["track_id"] == 42
    assert captured["user_id"] == "u-explicit"
    assert response.json()["context_pack"]["routing"]["track_id"] == 42
    assert metric_store.track_ids[-1] == 42


def test_router_suggest_uses_grounded_query(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResearchStore:
        def get_active_track(self, *, user_id: str):
            return {"id": 7, "name": "Retrieval Systems"}

    class _FakeTrackRouter:
        def suggest_track(self, *, user_id: str, query: str, active_track_id: int):
            captured["user_id"] = user_id
            captured["query"] = query
            captured["active_track_id"] = active_track_id
            return {"track_id": active_track_id, "score": 0.9}

    class _FakeGrounder:
        def ground_query(self, *, user_id: str, query: str, limit: int = 3):
            return type(
                "_Grounded",
                (),
                {
                    "canonical_query": "retrieval augmented generation latency",
                    "concepts": [object()],
                    "to_dict": lambda self: {
                        "original_query": query,
                        "canonical_query": "retrieval augmented generation latency",
                        "search_queries": [
                            query,
                            "retrieval augmented generation latency",
                        ],
                        "concepts": [{"id": "rag"}],
                    },
                },
            )()

    monkeypatch.setattr(research_route, "_research_store", _FakeResearchStore())
    monkeypatch.setattr(research_route, "_track_router", _FakeTrackRouter())
    monkeypatch.setattr(research_route, "_workflow_query_grounder", _FakeGrounder())

    app = api_main.app
    app.dependency_overrides[auth_deps.get_required_user_id] = _override_user_id("u1")
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/research/router/suggest",
                json={"query": "rag latency"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["query"] == "rag latency retrieval augmented generation latency"
    assert response.json()["suggestion"]["query_grounding"]["canonical_query"] == (
        "retrieval augmented generation latency"
    )
