from __future__ import annotations

import asyncio

import pytest

from paperbot.infrastructure.api_clients.citation_graph import CitationGraphClient


class _FakeSemanticScholarClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.max_concurrency = 0
        self.current_concurrency = 0

    async def get_paper(self, paper_id, fields=None):
        self.current_concurrency += 1
        self.max_concurrency = max(self.max_concurrency, self.current_concurrency)
        await asyncio.sleep(0)
        try:
            payload = self.payloads.get(paper_id)
            return dict(payload) if isinstance(payload, dict) else payload
        finally:
            self.current_concurrency -= 1

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_traverse_references_builds_directed_graph():
    client = _FakeSemanticScholarClient(
        {
            "seed": {
                "paperId": "seed",
                "title": "Seed",
                "year": 2025,
                "references": [
                    {"citedPaper": {"paperId": "r1", "title": "Ref One", "year": 2024, "citationCount": 12}},
                    {"citedPaper": {"paperId": "r2", "title": "Ref Two", "year": 2023, "citationCount": 5}},
                ],
                "citations": [
                    {"citingPaper": {"paperId": "c1", "title": "Ignored Citation", "year": 2026}},
                ],
            }
        }
    )

    graph_client = CitationGraphClient(semantic_scholar_client=client, max_concurrency=2)
    graph = await graph_client.traverse("seed", direction="references", max_hops=1, max_papers_per_hop=10)

    assert set(graph.nodes) == {"seed", "r1", "r2"}
    assert {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges} == {
        ("seed", "r1", "references"),
        ("seed", "r2", "references"),
    }


@pytest.mark.asyncio
async def test_traverse_both_deduplicates_cycles_and_preserves_hops():
    client = _FakeSemanticScholarClient(
        {
            "seed": {
                "paperId": "seed",
                "title": "Seed",
                "references": [{"citedPaper": {"paperId": "a", "title": "Paper A"}}],
                "citations": [{"citingPaper": {"paperId": "b", "title": "Paper B"}}],
            },
            "a": {
                "paperId": "a",
                "title": "Paper A",
                "references": [{"citedPaper": {"paperId": "seed", "title": "Seed"}}],
                "citations": [],
            },
            "b": {
                "paperId": "b",
                "title": "Paper B",
                "references": [],
                "citations": [{"citingPaper": {"paperId": "seed", "title": "Seed"}}],
            },
        }
    )

    graph_client = CitationGraphClient(semantic_scholar_client=client, max_concurrency=4)
    graph = await graph_client.traverse("seed", direction="both", max_hops=2, max_papers_per_hop=10)

    assert graph.nodes["seed"].hop == 0
    assert graph.nodes["a"].hop == 1
    assert graph.nodes["b"].hop == 1
    assert len([edge for edge in graph.edges if edge.source_id == "seed" and edge.target_id == "a"]) == 1
    assert len([edge for edge in graph.edges if edge.source_id == "b" and edge.target_id == "seed"]) == 1


@pytest.mark.asyncio
async def test_traverse_applies_relevance_filter_and_limit_per_hop():
    client = _FakeSemanticScholarClient(
        {
            "seed": {
                "paperId": "seed",
                "title": "Seed",
                "references": [
                    {"citedPaper": {"paperId": "keep", "title": "Keep Me", "citationCount": 1}},
                    {"citedPaper": {"paperId": "drop", "title": "Drop Me", "citationCount": 99}},
                ],
                "citations": [],
            }
        }
    )

    graph_client = CitationGraphClient(semantic_scholar_client=client)
    graph = await graph_client.traverse(
        "seed",
        direction="references",
        max_hops=1,
        max_papers_per_hop=1,
        relevance_filter=lambda paper: 10.0 if paper.get("title") == "Keep Me" else 0.5,
    )

    assert set(graph.nodes) == {"seed", "keep"}
    assert [(edge.source_id, edge.target_id) for edge in graph.edges] == [("seed", "keep")]


@pytest.mark.asyncio
async def test_traverse_fetches_same_hop_in_parallel():
    client = _FakeSemanticScholarClient(
        {
            "seed": {
                "paperId": "seed",
                "title": "Seed",
                "references": [
                    {"citedPaper": {"paperId": "a", "title": "Paper A"}},
                    {"citedPaper": {"paperId": "b", "title": "Paper B"}},
                ],
                "citations": [],
            },
            "a": {
                "paperId": "a",
                "title": "Paper A",
                "references": [],
                "citations": [],
            },
            "b": {
                "paperId": "b",
                "title": "Paper B",
                "references": [],
                "citations": [],
            },
        }
    )

    graph_client = CitationGraphClient(semantic_scholar_client=client, max_concurrency=4)
    await graph_client.traverse("seed", direction="references", max_hops=2, max_papers_per_hop=10)

    assert client.max_concurrency >= 2


@pytest.mark.asyncio
async def test_traverse_rejects_invalid_arguments():
    client = _FakeSemanticScholarClient({})
    graph_client = CitationGraphClient(semantic_scholar_client=client)

    with pytest.raises(ValueError, match="seed_paper_id"):
        await graph_client.traverse("", direction="references")

    with pytest.raises(ValueError, match="direction"):
        await graph_client.traverse("seed", direction="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_hops"):
        await graph_client.traverse("seed", direction="references", max_hops=0)

    with pytest.raises(ValueError, match="max_papers_per_hop"):
        await graph_client.traverse("seed", direction="references", max_papers_per_hop=0)
