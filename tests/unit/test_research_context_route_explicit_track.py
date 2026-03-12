from __future__ import annotations

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import research as research_route


class _FakeWorkflowMetricStore:
    def __init__(self) -> None:
        self.track_ids: list[int | None] = []

    def record_metric(self, *, track_id=None, **kwargs) -> None:
        self.track_ids.append(track_id)


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

    with TestClient(api_main.app) as client:
        response = client.post(
            "/api/research/context",
            json={
                "user_id": "u-explicit",
                "query": "agentic retrieval",
                "track_id": 42,
                "paper_limit": 0,
                "offline": True,
                "include_cross_track": False,
            },
        )

    assert response.status_code == 200
    assert captured["track_id"] == 42
    assert captured["user_id"] == "u-explicit"
    assert response.json()["context_pack"]["routing"]["track_id"] == 42
    assert metric_store.track_ids[-1] == 42
