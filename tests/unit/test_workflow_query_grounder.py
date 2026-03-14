from __future__ import annotations

from paperbot.application.ports.wiki_concept_port import GroundingSnapshot
from paperbot.application.services.wiki_concept_service import WikiConceptService
from paperbot.application.services.workflow_query_grounder import WorkflowQueryGrounder


class _FakeWikiConceptStore:
    def load_grounding_snapshot(
        self,
        *,
        user_id: str,
        paper_limit: int = 250,
        track_limit: int = 100,
    ) -> GroundingSnapshot:
        return {
            "papers": [
                {
                    "title": "Scaling Laws for Retrieval-Augmented Generation",
                    "abstract": "RAG systems improve retrieval-grounded generation quality.",
                    "keywords": ["rag", "retrieval"],
                    "fields_of_study": ["Method"],
                    "citation_count": 12,
                    "year": 2026,
                }
            ],
            "tracks": [
                {
                    "name": "Retrieval Systems",
                    "description": "RAG latency and retriever optimization",
                    "keywords": ["rag"],
                    "methods": ["retrieval augmented generation"],
                }
            ],
        }


def test_workflow_query_grounder_expands_short_concept_ids():
    grounder = WorkflowQueryGrounder(WikiConceptService(_FakeWikiConceptStore()))

    grounded = grounder.ground_query(user_id="wiki-user", query="rag latency")

    assert grounded.original_query == "rag latency"
    assert grounded.canonical_query == "retrieval augmented generation latency"
    assert grounded.search_queries == [
        "rag latency",
        "retrieval augmented generation latency",
    ]
    assert grounded.concepts[0].id == "rag"


def test_workflow_query_grounder_keeps_broader_aliases_without_overwriting_query():
    grounder = WorkflowQueryGrounder(WikiConceptService(_FakeWikiConceptStore()))

    grounded = grounder.ground_query(user_id="wiki-user", query="alignment roadmap")

    assert grounded.canonical_query == "alignment roadmap"
    assert grounded.search_queries == ["alignment roadmap"]
