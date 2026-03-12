from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import research as research_route
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore


def _prepare_db(tmp_path: Path):
    db_path = tmp_path / "paper-routes.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    paper = paper_store.upsert_paper(
        paper={
            "title": "UniICL",
            "url": "https://arxiv.org/abs/2501.12345",
            "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
        }
    )
    track = research_store.create_track(user_id="u1", name="u1-track", activate=True)
    research_store.add_paper_feedback(
        user_id="u1",
        track_id=int(track["id"]),
        paper_id=str(paper["id"]),
        action="save",
        metadata={"title": "UniICL"},
    )
    research_store.ingest_repo_enrichment_rows(
        rows=[
            {
                "title": "UniICL",
                "paper_url": "https://arxiv.org/abs/2501.12345",
                "repo_url": "https://github.com/example/unicicl",
                "query": "icl compression",
                "github": {
                    "full_name": "example/unicicl",
                    "stars": 321,
                    "forks": 12,
                    "open_issues": 1,
                    "watchers": 18,
                    "language": "Python",
                    "license": "MIT",
                    "topics": ["icl", "llm"],
                    "html_url": "https://github.com/example/unicicl",
                },
            }
        ]
    )
    return research_store, int(paper["id"])


def test_saved_and_detail_routes(tmp_path, monkeypatch):
    store, paper_id = _prepare_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", store)

    with TestClient(api_main.app) as client:
        saved = client.get("/api/research/papers/saved", params={"user_id": "u1"})
        detail = client.get(f"/api/research/papers/{paper_id}", params={"user_id": "u1"})

    assert saved.status_code == 200
    assert len(saved.json()["items"]) == 1

    assert detail.status_code == 200
    payload = detail.json()["detail"]
    assert payload["paper"]["id"] == paper_id
    assert payload["paper"]["title"] == "UniICL"
    assert len(payload["repos"]) == 1
    assert payload["repos"][0]["repo_url"] == "https://github.com/example/unicicl"


def test_update_status_route(tmp_path, monkeypatch):
    store, paper_id = _prepare_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", store)

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/research/papers/{paper_id}/status",
            json={"user_id": "u1", "status": "reading", "mark_saved": True},
        )

    assert resp.status_code == 200
    payload = resp.json()["status"]
    assert payload["paper_id"] == paper_id
    assert payload["status"] == "reading"


def test_paper_repos_route(tmp_path, monkeypatch):
    store, paper_id = _prepare_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", store)

    with TestClient(api_main.app) as client:
        resp = client.get(f"/api/research/papers/{paper_id}/repos")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["paper_id"] == str(paper_id)
    assert len(payload["repos"]) == 1
    assert payload["repos"][0]["full_name"] == "example/unicicl"


def test_track_feed_route_with_pagination_and_feedback_boost(tmp_path, monkeypatch):
    db_path = tmp_path / "track-feed.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    p1 = paper_store.upsert_paper(
        paper={
            "title": "Retrieval-Augmented Generation in Practice",
            "abstract": "rag retrieval pipeline",
            "url": "https://example.com/p1",
        }
    )
    p2 = paper_store.upsert_paper(
        paper={
            "title": "General Foundation Models",
            "abstract": "broad model overview",
            "url": "https://example.com/p2",
        }
    )
    paper_store.upsert_paper(
        paper={
            "title": "Unrelated Database Benchmark",
            "abstract": "oltp benchmark",
            "url": "https://example.com/p3",
        }
    )

    track = research_store.create_track(
        user_id="u-feed",
        name="rag-track",
        keywords=["rag", "retrieval"],
        activate=True,
    )
    research_store.add_paper_feedback(
        user_id="u-feed",
        track_id=int(track["id"]),
        paper_id=str(p2["id"]),
        action="save",
        metadata={"title": "General Foundation Models"},
    )

    monkeypatch.setattr(research_route, "_research_store", research_store)

    with TestClient(api_main.app) as client:
        page1 = client.get(
            f"/api/research/tracks/{int(track['id'])}/feed",
            params={"user_id": "u-feed", "limit": 1, "offset": 0},
        )
        page2 = client.get(
            f"/api/research/tracks/{int(track['id'])}/feed",
            params={"user_id": "u-feed", "limit": 1, "offset": 1},
        )

    assert page1.status_code == 200
    assert page2.status_code == 200

    payload1 = page1.json()
    payload2 = page2.json()

    assert payload1["total"] >= 2
    assert len(payload1["items"]) == 1
    assert len(payload2["items"]) == 1
    assert payload1["items"][0]["paper"]["id"] != payload2["items"][0]["paper"]["id"]

    ids = {payload1["items"][0]["paper"]["id"], payload2["items"][0]["paper"]["id"]}
    assert int(p1["id"]) in ids
    assert int(p2["id"]) in ids


def test_deadline_radar_route_returns_workflow_query_and_track_match(tmp_path, monkeypatch):
    db_path = tmp_path / "deadline-radar.db"
    db_url = f"sqlite:///{db_path}"
    research_store = SqlAlchemyResearchStore(db_url=db_url)
    research_store.create_track(
        user_id="u-deadline",
        name="nlp-track",
        keywords=["llm", "retrieval"],
        activate=True,
    )
    research_store.create_track(
        user_id="u-deadline",
        name="acl-track",
        keywords=[],
        venues=["ACL"],
        methods=["retrieval"],
        activate=False,
    )

    monkeypatch.setattr(research_route, "_research_store", research_store)

    with TestClient(api_main.app) as client:
        resp = client.get(
            "/api/research/deadlines/radar",
            params={"user_id": "u-deadline", "days": 365, "ccf_levels": "A"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["items"]

    first = payload["items"][0]
    assert isinstance(first.get("workflow_query"), str)
    assert first["workflow_query"]

    matched_any = any(item.get("matched_tracks") for item in payload["items"])
    assert matched_any

    acl_item = next(item for item in payload["items"] if item["name"] == "ACL 2026")
    acl_match = next(
        match for match in acl_item["matched_tracks"] if match["track_name"] == "acl-track"
    )
    assert "acl" in acl_match["matched_terms"]
