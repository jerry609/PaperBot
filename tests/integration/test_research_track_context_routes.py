from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import research as research_route
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.memory.schema import MemoryCandidate


def _prepare_context_route_db(tmp_path: Path):
    db_path = tmp_path / "track-context.db"
    db_url = f"sqlite:///{db_path}"

    paper_store = SqlAlchemyPaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url, auto_create_schema=True)
    memory_store = SqlAlchemyMemoryStore(db_url=db_url)

    track = research_store.create_track(
        user_id="u-context",
        name="Agentic Retrieval",
        description="Focus on retrieval pipelines.",
        keywords=["rag", "retrieval"],
        activate=True,
    )
    track_id = int(track["id"])

    research_store.add_task(
        user_id="u-context",
        track_id=track_id,
        title="Validate reranker",
        status="todo",
        priority=5,
    )
    research_store.add_milestone(
        user_id="u-context",
        track_id=track_id,
        name="Freeze eval set",
        status="doing",
        notes="Keep it small and stable.",
    )

    paper = paper_store.upsert_paper(
        paper={
            "title": "Context-Routed Retrieval",
            "abstract": "Track context for agentic retrieval.",
            "url": "https://example.com/context-routed-retrieval",
        }
    )
    research_store.add_paper_feedback(
        user_id="u-context",
        track_id=track_id,
        paper_id=str(paper["id"]),
        action="save",
        metadata={"title": "Context-Routed Retrieval"},
    )
    research_store.add_paper_feedback(
        user_id="u-context",
        track_id=track_id,
        paper_id=str(paper["id"]),
        action="like",
        metadata={"title": "Context-Routed Retrieval"},
    )

    memory_store.add_memories(
        user_id="u-context",
        memories=[
            MemoryCandidate(
                kind="fact",
                content="RAG baselines should keep retrieval latency below 200ms.",
                confidence=0.9,
                tags=["retrieval", "latency"],
                scope_type="track",
                scope_id=str(track_id),
                status="approved",
            ),
            MemoryCandidate(
                kind="note",
                content="Compare reranking against OpenAlex-only recall.",
                confidence=0.5,
                tags=["retrieval", "openalex"],
                scope_type="track",
                scope_id=str(track_id),
                status="pending",
            ),
        ],
    )

    return research_store, memory_store, track_id


def test_track_context_route_returns_consolidated_snapshot(tmp_path, monkeypatch):
    research_store, memory_store, track_id = _prepare_context_route_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_memory_store", memory_store)

    with TestClient(api_main.app) as client:
        response = client.get(
            f"/api/research/tracks/{track_id}/context",
            params={"user_id": "u-context"},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["track_id"] == track_id
    assert payload["track"]["name"] == "Agentic Retrieval"
    assert [task["title"] for task in payload["tasks"]] == ["Validate reranker"]
    assert [milestone["name"] for milestone in payload["milestones"]] == ["Freeze eval set"]
    assert payload["memory"]["approved_items"] == 1
    assert payload["memory"]["pending_items"] == 1
    assert payload["memory"]["total_items"] == 2
    assert "retrieval" in payload["memory"]["top_tags"]
    assert payload["feedback"]["actions"]["save"] == 1
    assert payload["feedback"]["actions"]["like"] == 1
    assert payload["saved_papers"]["total_items"] == 1
    assert payload["saved_papers"]["recent_items"][0]["paper"]["title"] == "Context-Routed Retrieval"
    assert "feedback_coverage" in payload["eval_summary"]


def test_track_context_route_returns_404_for_missing_or_inaccessible_track(tmp_path, monkeypatch):
    research_store, memory_store, track_id = _prepare_context_route_db(tmp_path)
    monkeypatch.setattr(research_route, "_research_store", research_store)
    monkeypatch.setattr(research_route, "_memory_store", memory_store)

    with TestClient(api_main.app) as client:
        missing = client.get(
            "/api/research/tracks/999999/context",
            params={"user_id": "u-context"},
        )
        wrong_user = client.get(
            f"/api/research/tracks/{track_id}/context",
            params={"user_id": "other-user"},
        )

    assert missing.status_code == 404
    assert wrong_user.status_code == 404
