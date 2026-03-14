from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.auth import dependencies as auth_deps
from paperbot.api.routes import wiki as wiki_route
from paperbot.application.services.wiki_concept_service import WikiConceptService
from paperbot.infrastructure.stores.paper_store import PaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.infrastructure.stores.wiki_concept_store import WikiConceptStore


def test_wiki_route_returns_grounded_concepts(tmp_path: Path, monkeypatch):
    user_id = "wiki-user"
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
        user_id=user_id,
        name="RAG Systems",
        description="Track retrieval-augmented generation and context routing papers.",
        keywords=["rag"],
        methods=["retrieval-augmented generation"],
        activate=True,
    )
    research_store.add_paper_feedback(
        user_id=user_id,
        track_id=int(track["id"]),
        paper_id=str(saved_paper["id"]),
        action="save",
    )

    monkeypatch.setattr(
        wiki_route,
        "_service",
        WikiConceptService(WikiConceptStore(db_url=db_url)),
    )

    app = api_main.app
    app.dependency_overrides[auth_deps.get_required_user_id] = lambda: user_id
    try:
        with TestClient(app) as client:
            response = client.get("/api/wiki/concepts?q=rag")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert "All" in payload["categories"]
    rag_item = next(item for item in payload["items"] if item["id"] == "rag")
    assert rag_item["paper_count"] >= 1
    assert rag_item["track_count"] >= 1
    assert rag_item["related_papers"] == ["Retrieval-Augmented Generation for Long Context QA"]


def test_wiki_route_uses_authenticated_user_context(monkeypatch, tmp_path: Path):
    user_id = "wiki-auth-user"
    db_url = f"sqlite:///{tmp_path / 'wiki-route-auth.db'}"
    paper_store = PaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)
    saved_paper = paper_store.upsert_paper(
        paper={
            "title": "Grounded RAG",
            "abstract": "RAG systems grounded by user context.",
            "keywords": ["rag"],
            "citation_count": 7,
            "year": 2026,
        }
    )
    track = research_store.create_track(
        user_id=user_id,
        name="Authenticated RAG",
        keywords=["rag"],
        activate=True,
    )
    research_store.add_paper_feedback(
        user_id=user_id,
        track_id=int(track["id"]),
        paper_id=str(saved_paper["id"]),
        action="save",
    )
    monkeypatch.setattr(
        wiki_route,
        "_service",
        WikiConceptService(WikiConceptStore(db_url=db_url)),
    )

    app = api_main.app
    app.dependency_overrides[auth_deps.get_required_user_id] = lambda: user_id
    try:
        with TestClient(app) as client:
            response = client.get("/api/wiki/concepts?user_id=someone-else&q=rag")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert any(item["id"] == "rag" for item in payload["items"])
