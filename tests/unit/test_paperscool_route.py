from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import paperscool as paperscool_route


class _FakeWorkflow:
    def run(self, *, queries, sources, branches, top_k_per_query, show_per_branch):
        return {
            "source": "papers.cool",
            "fetched_at": "2026-02-09T00:00:00+00:00",
            "sources": sources,
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
                "source_breakdown": {sources[0]: 1},
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


def test_paperscool_daily_route_success(monkeypatch, tmp_path):
    monkeypatch.setattr(paperscool_route, "PapersCoolTopicSearchWorkflow", _FakeWorkflow)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "sources": ["papers_cool"],
                "branches": ["arxiv", "venue"],
                "save": True,
                "formats": ["both"],
                "output_dir": str(tmp_path / "daily"),
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report"]["stats"]["unique_items"] == 1
    assert payload["markdown_path"] is not None
    assert payload["json_path"] is not None


def test_paperscool_daily_route_with_llm_enrichment(monkeypatch):
    monkeypatch.setattr(paperscool_route, "PapersCoolTopicSearchWorkflow", _FakeWorkflow)

    called = {"value": False}

    def _fake_enrich(report, *, llm_features, llm_service=None, max_items_per_query=3):
        called["value"] = True
        report = dict(report)
        report["llm_analysis"] = {"enabled": True, "features": llm_features}
        return report

    monkeypatch.setattr(paperscool_route, "enrich_daily_paper_report", _fake_enrich)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_llm_analysis": True,
                "llm_features": ["summary", "trends"],
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert called["value"] is True
    assert payload["report"]["llm_analysis"]["enabled"] is True


def test_paperscool_daily_route_with_judge(monkeypatch):
    monkeypatch.setattr(paperscool_route, "PapersCoolTopicSearchWorkflow", _FakeWorkflow)

    called = {"value": False}

    def _fake_judge(report, *, llm_service=None, max_items_per_query=5, n_runs=1):
        called["value"] = True
        report = dict(report)
        report["judge"] = {
            "enabled": True,
            "max_items_per_query": max_items_per_query,
            "n_runs": n_runs,
            "recommendation_count": {"must_read": 1, "worth_reading": 0, "skim": 0, "skip": 0},
        }
        return report

    monkeypatch.setattr(paperscool_route, "apply_judge_scores_to_report", _fake_judge)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_judge": True,
                "judge_runs": 2,
                "judge_max_items_per_query": 4,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert called["value"] is True
    assert payload["report"]["judge"]["enabled"] is True
