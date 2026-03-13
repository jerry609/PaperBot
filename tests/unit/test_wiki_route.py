from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import wiki as wiki_route
from paperbot.application.services.wiki_concept_service import WikiConceptService
from paperbot.infrastructure.stores.paper_store import PaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.infrastructure.stores.wiki_concept_store import WikiConceptStore


def test_wiki_route_returns_grounded_concepts(tmp_path: Path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'wiki-route.db'}"
    paper_store = PaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)
    saved_paper = paper_store.upsert_paper(
        paper={
            "title": "Retrieval-Augmented Generation for Long Context QA",
            "abstract": "A practical RAG pipeline with retrieval and answer synthesis.",
            "keywords": ["rag", "retrieval-augmented generation"],
            "fields_of_study": ["Method"],
            "citation_count": 42,
            "year": 2025,
        }
    )
    track = research_store.create_track(
        user_id="default",
        name="RAG Systems",
        description="Track retrieval-augmented generation and context routing papers.",
        keywords=["rag"],
        methods=["retrieval-augmented generation"],
        activate=True,
    )
    research_store.add_paper_feedback(
        user_id="default",
        track_id=int(track["id"]),
        paper_id=str(saved_paper["id"]),
        action="save",
    )

    monkeypatch.setattr(
        wiki_route,
        "_service",
        WikiConceptService(WikiConceptStore(db_url=db_url)),
    )

    with TestClient(api_main.app) as client:
        response = client.get("/api/wiki/concepts?q=rag")

    assert response.status_code == 200
    payload = response.json()
    assert "All" in payload["categories"]
    rag_item = next(item for item in payload["items"] if item["id"] == "rag")
    assert rag_item["paper_count"] >= 1
    assert rag_item["track_count"] >= 1
    assert rag_item["related_papers"] == ["Retrieval-Augmented Generation for Long Context QA"]


def test_wiki_route_rejects_cross_user_grounding(monkeypatch):
    monkeypatch.setattr(
        wiki_route,
        "_service",
        WikiConceptService(WikiConceptStore()),
    )

    with TestClient(api_main.app) as client:
        response = client.get("/api/wiki/concepts?user_id=someone-else&q=rag")

    assert response.status_code == 403
    assert "authenticated user context" in response.json()["detail"]
