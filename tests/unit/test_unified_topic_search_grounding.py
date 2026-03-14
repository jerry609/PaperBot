from __future__ import annotations

import pytest

from paperbot.application.services.paper_search_service import SearchResult
from paperbot.application.services.wiki_concept_service import WikiConceptService
from paperbot.application.services.workflow_query_grounder import WorkflowQueryGrounder
from paperbot.application.workflows import unified_topic_search as uts
from paperbot.domain.paper import PaperCandidate


class _FakeWikiConceptStore:
    def load_grounding_snapshot(
        self,
        *,
        user_id: str,
        paper_limit: int = 250,
        track_limit: int = 100,
    ):
        return {
            "papers": [
                {
                    "title": "Scaling Laws for Retrieval-Augmented Generation",
                    "abstract": "Retrieval-augmented generation systems scale with corpus quality.",
                    "keywords": ["rag"],
                    "fields_of_study": ["Method"],
                    "citation_count": 24,
                    "year": 2026,
                }
            ],
            "tracks": [],
        }


@pytest.mark.asyncio
async def test_run_unified_topic_search_adds_grounded_variants(monkeypatch):
    captured_queries: list[str] = []

    async def _fake_search_candidate_papers(service, *, query, sources, max_results, persist):
        captured_queries.append(query)
        candidate = PaperCandidate(
            title="Scaling Laws for Retrieval-Augmented Generation",
            abstract=(
                "Retrieval-augmented generation systems scale with corpus size and retriever "
                "quality in enough detail to pass the academic paper filter."
            ),
            year=2026,
            citation_count=24,
            url="https://example.com/rag-scaling",
            keywords=["rag", "retrieval"],
            fields_of_study=["Method"],
        )
        return SearchResult(
            papers=[candidate],
            provenance={candidate.title_hash: ["papers_cool"]},
        )

    monkeypatch.setattr(uts, "search_candidate_papers", _fake_search_candidate_papers)
    grounder = WorkflowQueryGrounder(WikiConceptService(_FakeWikiConceptStore()))

    result = await uts.run_unified_topic_search(
        user_id="ground-user",
        queries=["rag latency"],
        search_service=object(),
        query_grounder=grounder,
        persist=False,
    )

    assert captured_queries == [
        "rag latency",
        "retrieval augmented generation latency",
    ]
    assert result["queries"][0]["canonical_query"] == "retrieval augmented generation latency"
    assert result["queries"][0]["search_queries"] == captured_queries
    assert result["queries"][0]["grounded_concepts"][0]["id"] == "rag"
    assert result["queries"][0]["items"][0]["matched_queries"] == captured_queries
