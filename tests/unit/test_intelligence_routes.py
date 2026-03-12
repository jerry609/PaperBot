from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.auth import dependencies as auth_deps
from paperbot.api.routes import intelligence as intelligence_route
from paperbot.infrastructure.services.intelligence_radar_service import RadarProfile


class _FakeIntelligenceService:
    def __init__(self):
        self.list_feed_calls = []

    def needs_refresh(self, *, user_id: str = "default", max_age_minutes: int = 45) -> bool:
        return False

    def refresh(self, *, user_id: str = "default"):
        return {"refreshed_at": "2026-03-10T11:00:00+00:00"}

    def list_feed(
        self,
        *,
        user_id: str = "default",
        limit: int = 8,
        source=None,
        keyword=None,
        repo=None,
        sort_by: str = "score",
        sort_order: str = "desc",
    ):
        self.list_feed_calls.append(
            {
                "user_id": user_id,
                "limit": limit,
                "source": source,
                "keyword": keyword,
                "repo": repo,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }
        )
        return [
            {
                "external_id": "reddit:keyword:rag",
                "source": "reddit",
                "source_label": "Reddit Search",
                "kind": "keyword_spike",
                "title": "Reddit spike: rag",
                "summary": "24h mentions: 12 across r/MachineLearning. Top post: RAG agents are back.",
                "url": "https://reddit.example/rag",
                "keyword_hits": ["rag"],
                "author_matches": ["Alice Zhang"],
                "repo_matches": ["org/rag-agent"],
                "match_reasons": [
                    "keyword: rag",
                    "delta: +5",
                    "author: Alice Zhang",
                    "repo: org/rag-agent",
                ],
                "score": 91.0,
                "metric_name": "mentions/24h",
                "metric_value": 12,
                "metric_delta": 5,
                "published_at": "2026-03-10T10:00:00+00:00",
                "detected_at": "2026-03-10T11:00:00+00:00",
                "payload": {"subreddits": ["MachineLearning"]},
            }
        ][:limit]

    def latest_refresh(self, *, user_id: str = "default"):
        return "2026-03-10T11:00:00+00:00"

    def build_profile(self, *, user_id: str = "default") -> RadarProfile:
        return RadarProfile(
            keywords=["rag", "agents"],
            scholar_names=["Alice Zhang"],
            watch_repos=["org/rag-agent"],
            subreddits=["MachineLearning"],
        )


class _FakeResearchStore:
    def list_tracks(self, *, user_id: str, include_archived: bool, limit: int):
        return [
            {
                "id": 7,
                "name": "RAG Agents",
                "keywords": ["rag", "agents"],
                "methods": ["retrieval"],
            }
        ]


def _override_user_id(user_id: str):
    def _dep_override():
        return user_id

    return _dep_override


def test_intelligence_feed_route_returns_external_signal_payload(monkeypatch):
    service = _FakeIntelligenceService()
    monkeypatch.setattr(intelligence_route, "_service", service)
    monkeypatch.setattr(intelligence_route, "_research_store", _FakeResearchStore())

    app = api_main.app
    app.dependency_overrides[auth_deps.get_required_user_id] = _override_user_id("default")
    try:
        with TestClient(app) as client:
            resp = client.get(
                "/api/intelligence/feed",
                params={
                    "limit": 1,
                    "source": "reddit",
                    "keyword": "rag",
                    "repo": "org/rag-agent",
                    "sort_by": "delta",
                    "sort_order": "desc",
                    "track_id": 7,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()

    assert service.list_feed_calls == [
        {
            "user_id": "default",
            "limit": 50,
            "source": "reddit",
            "keyword": "rag",
            "repo": "org/rag-agent",
            "sort_by": "delta",
            "sort_order": "desc",
        }
    ]

    assert payload["refreshed_at"] == "2026-03-10T11:00:00+00:00"
    assert payload["keywords"] == ["rag", "agents"]
    assert payload["watch_repos"] == ["org/rag-agent"]
    assert payload["subreddits"] == ["MachineLearning"]
    assert len(payload["items"]) == 1

    item = payload["items"][0]
    assert item["title"] == "Reddit spike: rag"
    assert item["keyword_hits"] == ["rag"]
    assert item["author_matches"] == ["Alice Zhang"]
    assert item["repo_matches"] == ["org/rag-agent"]
    assert item["match_reasons"] == [
        "keyword: rag",
        "delta: +5",
        "author: Alice Zhang",
        "repo: org/rag-agent",
    ]
    assert item["metric"] == {"name": "mentions/24h", "value": 12, "delta": 5}
    assert item["matched_tracks"] == [
        {
            "track_id": 7,
            "track_name": "RAG Agents",
            "matched_keywords": ["rag"],
        }
    ]
    assert "rag" in item["research_query"]


def test_intelligence_feed_route_uses_authenticated_user_id(monkeypatch):
    service = _FakeIntelligenceService()
    monkeypatch.setattr(intelligence_route, "_service", service)
    monkeypatch.setattr(intelligence_route, "_research_store", _FakeResearchStore())

    app = api_main.app
    app.dependency_overrides[auth_deps.get_required_user_id] = _override_user_id("u-radar")
    try:
        with TestClient(app) as client:
            resp = client.get(
                "/api/intelligence/feed",
                params={
                    "limit": 1,
                    "source": "reddit",
                    "keyword": "rag",
                    "repo": "org/rag-agent",
                    "sort_by": "delta",
                    "sort_order": "desc",
                    "track_id": 7,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert service.list_feed_calls[0]["user_id"] == "u-radar"
