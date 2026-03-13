from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import research as research_route
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore


def _prepare_stores(tmp_path: Path):
    db_path = tmp_path / "research-discovery-collections.db"
    db_url = f"sqlite:///{db_path}"
    research_store = SqlAlchemyResearchStore(db_url=db_url)
    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    return research_store, paper_store


def test_discovery_seed_route_returns_graph_and_items(tmp_path, monkeypatch):
    research_store, paper_store = _prepare_stores(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_paper_store", paper_store)

    track = research_store.create_track(user_id="u-discovery", name="Graph Track", activate=True)
    saved = paper_store.upsert_paper(
        paper={"title": "Graph Neural Networks in Practice", "authors": ["Alice"], "year": 2024}
    )
    research_store.add_paper_feedback(
        user_id="u-discovery",
        track_id=int(track["id"]),
        paper_id=str(saved["id"]),
        action="save",
        metadata={},
    )

    class _FakeS2Client:
        def __init__(self, *args, **kwargs):
            return None

        async def get_paper(self, paper_id, fields=None):
            return {
                "paperId": "seed",
                "title": "Seed",
                "year": 2025,
                "references": [
                    {
                        "citedPaper": {
                            "paperId": "p1",
                            "title": "Graph Retrieval",
                            "year": 2024,
                            "citationCount": 12,
                            "authors": [{"name": "A"}],
                        }
                    }
                ],
                "citations": [
                    {
                        "citingPaper": {
                            "paperId": "p2",
                            "title": "Neural Systems",
                            "year": 2026,
                            "citationCount": 9,
                            "authors": [{"name": "B"}],
                        }
                    }
                ],
            }

        async def get_author(self, author_id, fields=None):
            return {"name": "Author", "paperCount": 10}

        async def get_author_papers(self, author_id, limit=10, fields=None):
            return []

        async def close(self):
            return None

    class _FakeOpenAlexConnector:
        async def resolve_work(self, **kwargs):
            return {"id": "https://openalex.org/W1", "title": "Seed OA", "related_works": []}

        async def get_related_works(self, work, limit=20):
            return [
                {
                    "id": "https://openalex.org/W3",
                    "title": "Graph Reasoning",
                    "publication_year": 2025,
                    "cited_by_count": 30,
                    "authorships": [{"author": {"display_name": "C"}}],
                }
            ]

        async def get_referenced_works(self, work, limit=20):
            return []

        async def get_citing_works(self, work, limit=20):
            return []

        async def close(self):
            return None

    monkeypatch.setattr(research_route, "SemanticScholarClient", _FakeS2Client)
    monkeypatch.setattr(
        "paperbot.infrastructure.connectors.openalex_connector.OpenAlexConnector",
        _FakeOpenAlexConnector,
    )

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/discovery/seed",
            json={
                "user_id": "u-discovery",
                "track_id": int(track["id"]),
                "seed_type": "doi",
                "seed_id": "10.1000/seed.1",
                "limit": 10,
                "personalized": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["seed"]["seed_type"] == "doi"
    assert len(payload["items"]) >= 2
    assert payload["nodes"]
    assert payload["edges"]
    assert "why_this_paper" in payload["items"][0]


def test_collections_routes_crud(tmp_path, monkeypatch):
    research_store, paper_store = _prepare_stores(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_paper_store", paper_store)

    track = research_store.create_track(user_id="u-col", name="Collections Track", activate=True)
    paper = paper_store.upsert_paper(
        paper={"title": "Collection Paper", "authors": ["Alice"], "year": 2026}
    )

    with TestClient(api_main.app) as client:
        create_resp = client.post(
            "/api/research/collections",
            json={
                "user_id": "u-col",
                "name": "Must Read",
                "description": "focus papers",
                "track_id": int(track["id"]),
            },
        )
        assert create_resp.status_code == 200
        collection_id = int(create_resp.json()["collection"]["id"])

        list_resp = client.get("/api/research/collections", params={"user_id": "u-col"})
        assert list_resp.status_code == 200
        assert len(list_resp.json()["items"]) == 1

        add_resp = client.post(
            f"/api/research/collections/{collection_id}/items",
            json={
                "user_id": "u-col",
                "paper_id": str(paper["id"]),
                "note": "read this week",
                "tags": ["rr", "seed"],
            },
        )
        assert add_resp.status_code == 200
        assert len(add_resp.json()["items"]) == 1

        patch_resp = client.patch(
            f"/api/research/collections/{collection_id}/items/{paper['id']}",
            json={
                "user_id": "u-col",
                "note": "updated note",
                "tags": ["priority"],
            },
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["items"][0]["note"] == "updated note"
        assert patch_resp.json()["items"][0]["tags"] == ["priority"]

        del_resp = client.delete(
            f"/api/research/collections/{collection_id}/items/{paper['id']}",
            params={"user_id": "u-col"},
        )
        assert del_resp.status_code == 200
        assert del_resp.json()["ok"] is True

        final_items = client.get(
            f"/api/research/collections/{collection_id}/items", params={"user_id": "u-col"}
        )
        assert final_items.status_code == 200
        assert final_items.json()["items"] == []
