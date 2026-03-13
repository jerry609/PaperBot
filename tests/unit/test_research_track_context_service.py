from __future__ import annotations

from typing import Any, Dict, List, Optional

from paperbot.application.services.research_track_context_service import (
    ResearchTrackContextService,
    TrackContextQuery,
)


class _FakeTrackReader:
    def __init__(self) -> None:
        self._track = {
            "id": 7,
            "user_id": "u1",
            "name": "Agentic Retrieval",
            "description": "Focus on retrieval pipelines.",
            "keywords": ["rag", "retrieval"],
            "is_active": True,
        }
        self.last_eval_days: Optional[int] = None

    def get_track(self, *, user_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        if user_id == "u1" and track_id == 7:
            return dict(self._track)
        return None

    def get_active_track(self, *, user_id: str) -> Optional[Dict[str, Any]]:
        if user_id == "u1":
            return dict(self._track)
        return None

    def list_tasks(
        self,
        *,
        user_id: str,
        track_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return [
            {"id": 1, "track_id": track_id, "title": "Evaluate reranker", "status": "todo"},
            {"id": 2, "track_id": track_id, "title": "Compare OpenAlex", "status": "doing"},
        ][:limit]

    def list_milestones(
        self,
        *,
        user_id: str,
        track_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": 11,
                "track_id": track_id,
                "name": "Collect benchmark set",
                "status": "todo",
                "updated_at": "2026-03-10T09:00:00+00:00",
            }
        ][:limit]

    def list_effective_paper_feedback(
        self,
        *,
        user_id: str,
        track_id: int,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return [
            {"id": 91, "track_id": track_id, "paper_id": "p1", "action": "save", "ts": "2026-03-11T08:30:00+00:00"},
            {"id": 92, "track_id": track_id, "paper_id": "p2", "action": "like", "ts": "2026-03-09T12:00:00+00:00"},
            {"id": 93, "track_id": track_id, "paper_id": "p3", "action": "save", "ts": "2026-03-08T12:00:00+00:00"},
        ][:limit]

    def list_saved_papers(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        collection_id: Optional[int] = None,
        limit: int = 200,
        sort_by: str = "saved_at",
    ) -> List[Dict[str, Any]]:
        return [
            {"paper": {"id": 101, "title": "RAG from First Principles"}, "saved_at": "2026-03-11T08:30:00+00:00"},
            {"paper": {"id": 102, "title": "Context Routers"}, "saved_at": "2026-03-07T08:30:00+00:00"},
        ][:limit]

    def summarize_eval(
        self,
        *,
        user_id: str,
        track_id: Optional[int] = None,
        days: int = 30,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        self.last_eval_days = days
        return {"total_runs": 4, "feedback_coverage": 0.75}


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_memories(
        self,
        *,
        user_id: str,
        limit: int = 100,
        kind: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        include_pending: bool = False,
        include_deleted: bool = False,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "user_id": user_id,
                "limit": limit,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "status": status,
                "include_pending": include_pending,
                "include_deleted": include_deleted,
            }
        )
        if status == "approved":
            return [
                {
                    "id": 1,
                    "scope_type": "track",
                    "scope_id": scope_id,
                    "status": "approved",
                    "tags": ["retrieval", "benchmark"],
                    "updated_at": "2026-03-10T12:00:00+00:00",
                },
                {
                    "id": 2,
                    "scope_type": "track",
                    "scope_id": scope_id,
                    "status": "approved",
                    "tags": ["retrieval"],
                    "updated_at": "2026-03-11T15:00:00+00:00",
                },
            ]
        if status == "pending":
            return [
                {
                    "id": 3,
                    "scope_type": "track",
                    "scope_id": scope_id,
                    "status": "pending",
                    "tags": ["followup"],
                    "created_at": "2026-03-11T09:00:00+00:00",
                }
            ]
        return []


def test_get_track_context_aggregates_track_scoped_reads() -> None:
    track_reader = _FakeTrackReader()
    memory_store = _FakeMemoryStore()
    service = ResearchTrackContextService(track_reader=track_reader, memory_store=memory_store)

    snapshot = service.get_track_context(
        user_id="u1",
        query=TrackContextQuery(
            task_limit=2,
            milestone_limit=1,
            feedback_limit=2,
            saved_preview_limit=1,
            eval_days=14,
        ),
    )

    assert snapshot is not None
    payload = snapshot.to_dict()

    assert payload["track"]["id"] == 7
    assert [task["title"] for task in payload["tasks"]] == [
        "Evaluate reranker",
        "Compare OpenAlex",
    ]
    assert [item["action"] for item in payload["feedback"]["recent_items"]] == ["save", "like"]
    assert payload["feedback"]["actions"] == {"save": 2, "like": 1}
    assert payload["memory"] == {
        "total_items": 3,
        "approved_items": 2,
        "pending_items": 1,
        "top_tags": ["retrieval", "benchmark", "followup"],
        "latest_memory_at": "2026-03-11T15:00:00+00:00",
    }
    assert payload["saved_papers"]["total_items"] == 2
    assert len(payload["saved_papers"]["recent_items"]) == 1
    assert payload["eval_summary"]["feedback_coverage"] == 0.75
    assert track_reader.last_eval_days == 14
    assert memory_store.calls == [
        {
            "user_id": "u1",
            "limit": 500,
            "scope_type": "track",
            "scope_id": "7",
            "status": "approved",
            "include_pending": True,
            "include_deleted": False,
        },
        {
            "user_id": "u1",
            "limit": 500,
            "scope_type": "track",
            "scope_id": "7",
            "status": "pending",
            "include_pending": True,
            "include_deleted": False,
        },
    ]


def test_get_track_context_returns_none_when_track_is_missing() -> None:
    service = ResearchTrackContextService(
        track_reader=_FakeTrackReader(),
        memory_store=_FakeMemoryStore(),
    )

    snapshot = service.get_track_context(user_id="missing-user", track_id=999)

    assert snapshot is None
