from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import paperscool as paperscool_route


class _FakeWorkflow:
    def run(self, *, queries, branches, top_k_per_query, show_per_branch):
        return {
            "source": "papers.cool",
            "fetched_at": "2026-02-09T00:00:00+00:00",
            "sources": ["papers_cool"],
            "queries": [
                {
                    "raw_query": queries[0],
                    "normalized_query": "icl compression",
                    "tokens": ["icl", "compression"],
                    "total_hits": 1,
                    "items": [
                        {
                            "paper_id": "2025.acl-long.24@ACL",
                            "title": "UniICL",
                            "url": "https://papers.cool/venue/2025.acl-long.24@ACL",
                            "external_url": "",
                            "pdf_url": "",
                            "authors": ["A"],
                            "subject_or_venue": "ACL.2025 - Long Papers",
                            "published_at": "",
                            "snippet": "",
                            "keywords": ["icl", "compression"],
                            "branches": branches,
                            "matched_keywords": ["icl", "compression"],
                            "matched_queries": ["icl compression"],
                            "score": 10.0,
                            "pdf_stars": 30,
                            "kimi_stars": 30,
                            "alternative_urls": [],
                        }
                    ],
                }
            ],
            "items": [],
            "summary": {
                "unique_items": 1,
                "total_query_hits": 1,
                "top_titles": ["UniICL"],
                "query_highlights": [
                    {
                        "raw_query": queries[0],
                        "normalized_query": "icl compression",
                        "hit_count": 1,
                        "top_title": "UniICL",
                        "top_keywords": ["icl", "compression"],
                    }
                ],
            },
        }


def test_paperscool_search_route_success(monkeypatch):
    monkeypatch.setattr(paperscool_route, "PapersCoolTopicSearchWorkflow", _FakeWorkflow)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/search",
            json={
                "queries": ["ICL压缩"],
                "sources": ["papers_cool"],
                "branches": ["arxiv", "venue"],
                "top_k_per_query": 5,
                "show_per_branch": 25,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["source"] == "papers.cool"
    assert payload["summary"]["unique_items"] == 1


def test_paperscool_search_route_requires_queries():
    with TestClient(api_main.app) as client:
        resp = client.post("/api/research/paperscool/search", json={"queries": []})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "queries is required"
