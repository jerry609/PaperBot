from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import research as research_route


class _FakeSemanticScholarClient:
    def __init__(self, api_key=None):
        self.api_key = api_key

    async def search_authors(self, query, limit=10, fields=None):
        return [
            {
                "authorId": "1001",
                "name": "Alice",
                "affiliations": ["Lab"],
                "paperCount": 10,
                "citationCount": 100,
                "hIndex": 12,
            },
            {
                "authorId": "1002",
                "name": "Alice B",
                "affiliations": ["Institute"],
                "paperCount": 8,
                "citationCount": 50,
                "hIndex": 8,
            },
        ]

    async def get_author(self, author_id, fields=None):
        return {
            "authorId": author_id,
            "name": "Alice",
            "affiliations": ["Lab"],
            "paperCount": 10,
            "citationCount": 100,
            "hIndex": 12,
        }

    async def get_author_papers(self, author_id, limit=100, fields=None):
        return [
            {
                "title": "Paper A",
                "year": 2025,
                "citationCount": 10,
                "venue": "NeurIPS",
                "fieldsOfStudy": ["Machine Learning"],
                "authors": [
                    {"authorId": author_id, "name": "Alice"},
                    {"authorId": "c1", "name": "Bob"},
                ],
            },
            {
                "title": "Paper B",
                "year": 2024,
                "citationCount": 4,
                "venue": "ICLR",
                "fieldsOfStudy": ["Machine Learning", "Optimization"],
                "authors": [
                    {"authorId": author_id, "name": "Alice"},
                    {"authorId": "c1", "name": "Bob"},
                    {"authorId": "c2", "name": "Carol"},
                ],
            },
        ]

    async def close(self):
        return None


def test_scholar_network_route(monkeypatch):
    monkeypatch.setattr(research_route, "SemanticScholarClient", _FakeSemanticScholarClient)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/scholar/network",
            json={
                "scholar_id": "s1",
                "max_papers": 20,
                "recent_years": 10,
                "max_nodes": 10,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scholar"]["name"] == "Alice"
    assert payload["stats"]["coauthor_count"] == 2
    assert len(payload["edges"]) == 2


def test_scholar_trends_route(monkeypatch):
    monkeypatch.setattr(research_route, "SemanticScholarClient", _FakeSemanticScholarClient)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/scholar/trends",
            json={
                "scholar_id": "s1",
                "max_papers": 20,
                "year_window": 10,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scholar"]["name"] == "Alice"
    assert len(payload["publication_velocity"]) >= 1
    assert payload["trend_summary"]["publication_trend"] in {"up", "down", "flat"}


def test_scholar_list_route(monkeypatch):
    from paperbot.domain.scholar import Scholar

    class _FakeScholarProfileAgent:
        def list_tracked_scholars(self):
            return [
                Scholar(
                    scholar_id="s1",
                    semantic_scholar_id="1001",
                    name="Alice",
                    affiliations=["Lab A"],
                    keywords=["llm", "agents"],
                    h_index=20,
                    citation_count=2000,
                    paper_count=80,
                ),
                Scholar(
                    scholar_id="s2",
                    semantic_scholar_id="1002",
                    name="Bob",
                    affiliations=["Lab B"],
                    keywords=["security"],
                    h_index=10,
                    citation_count=500,
                    paper_count=40,
                ),
            ]

        def get_cache_stats(self, scholar_id: str):
            if scholar_id == "1001":
                return {
                    "paper_count": 12,
                    "history_length": 3,
                    "last_updated": "2026-02-12T08:00:00+00:00",
                }
            return {
                "paper_count": 2,
                "history_length": 1,
                "last_updated": None,
            }

    monkeypatch.setattr(
        "paperbot.agents.scholar_tracking.scholar_profile_agent.ScholarProfileAgent",
        _FakeScholarProfileAgent,
    )

    with TestClient(api_main.app) as client:
        resp = client.get("/api/research/scholars?limit=10")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 2
    assert len(payload["items"]) == 2
    assert payload["items"][0]["name"] == "Alice"
    assert payload["items"][0]["status"] == "active"
    assert payload["items"][1]["name"] == "Bob"


def test_scholar_create_route(monkeypatch):
    class _FakeSubscriptionService:
        def add_scholar(self, scholar):
            if scholar.get("semantic_scholar_id") == "dup":
                raise ValueError("semantic_scholar_id already exists")
            return scholar

    monkeypatch.setattr(research_route, "_subscription_service", _FakeSubscriptionService())

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/scholars",
            json={
                "name": "Carol",
                "semantic_scholar_id": "1003",
                "keywords": ["rag", "agents"],
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scholar"]["name"] == "Carol"
    assert payload["scholar"]["semantic_scholar_id"] == "1003"


def test_scholar_delete_route(monkeypatch):
    class _FakeSubscriptionService:
        def remove_scholar(self, scholar_ref):
            if scholar_ref == "missing":
                return None
            return {"name": "Carol", "semantic_scholar_id": scholar_ref}

    monkeypatch.setattr(research_route, "_subscription_service", _FakeSubscriptionService())

    with TestClient(api_main.app) as client:
        resp = client.delete("/api/research/scholars/1003")
        missing = client.delete("/api/research/scholars/missing")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["removed"] is True
    assert payload["scholar"]["semantic_scholar_id"] == "1003"
    assert missing.status_code == 404


def test_scholar_update_route(monkeypatch):
    class _FakeSubscriptionService:
        def update_scholar(self, scholar_ref, updates):
            if scholar_ref == "missing":
                return None
            return {
                "name": updates.get("name") or "Alice",
                "semantic_scholar_id": updates.get("semantic_scholar_id") or "1001",
                "keywords": updates.get("keywords") or [],
            }

    monkeypatch.setattr(research_route, "_subscription_service", _FakeSubscriptionService())

    with TestClient(api_main.app) as client:
        resp = client.patch(
            "/api/research/scholars/1001",
            json={"name": "Alice Updated", "keywords": ["rag", "safety"]},
        )
        missing = client.patch("/api/research/scholars/missing", json={"name": "x"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scholar"]["name"] == "Alice Updated"
    assert payload["scholar"]["keywords"] == ["rag", "safety"]
    assert missing.status_code == 404


def test_scholar_search_route(monkeypatch):
    monkeypatch.setattr(research_route, "SemanticScholarClient", _FakeSemanticScholarClient)

    with TestClient(api_main.app) as client:
        resp = client.get("/api/research/scholars/search?query=alice&limit=5")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["query"] == "alice"
    assert payload["total"] == 2
    assert payload["items"][0]["author_id"] == "1001"
