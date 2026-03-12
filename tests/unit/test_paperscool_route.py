from fastapi.testclient import TestClient
from types import SimpleNamespace

from paperbot.api import main as api_main
from paperbot.api.routes import paperscool as paperscool_route
from paperbot.infrastructure.stores.pipeline_session_store import PipelineSessionStore


def _parse_sse_events(text: str):
    """Parse SSE text into a list of event dicts."""
    import json

    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except Exception:
                pass
    return events


class _FakeWorkflow:
    def run(self, *, queries, sources, branches, top_k_per_query, show_per_branch, min_score=0.0):
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


async def _fake_run_topic_search(
    *,
    user_id="default",
    queries,
    sources,
    branches,
    top_k_per_query,
    show_per_branch,
    min_score=0.0,
):
    """Async fake that replaces _run_topic_search for tests."""
    return _FakeWorkflow().run(
        queries=queries,
        sources=sources,
        branches=branches,
        top_k_per_query=top_k_per_query,
        show_per_branch=show_per_branch,
        min_score=min_score,
    )


async def _fake_run_topic_search_multi(
    *,
    user_id="default",
    queries,
    sources,
    branches,
    top_k_per_query,
    show_per_branch,
    min_score=0.0,
):
    """Async fake returning multiple papers for filter testing."""
    return _FakeWorkflowMultiPaper().run(
        queries=queries,
        sources=sources,
        branches=branches,
        top_k_per_query=top_k_per_query,
        show_per_branch=show_per_branch,
        min_score=min_score,
    )


def test_paperscool_search_route_success(monkeypatch):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

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


def test_paperscool_search_route_passes_user_id(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_search(**kwargs):
        captured.update(kwargs)
        return _FakeWorkflow().run(
            queries=kwargs["queries"],
            sources=kwargs["sources"],
            branches=kwargs["branches"],
            top_k_per_query=kwargs["top_k_per_query"],
            show_per_branch=kwargs["show_per_branch"],
            min_score=kwargs["min_score"],
        )

    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_search)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/search",
            json={
                "user_id": "u-search",
                "queries": ["rag latency"],
            },
        )

    assert resp.status_code == 200
    assert captured["user_id"] == "u-search"


def test_paperscool_search_route_requires_queries():
    with TestClient(api_main.app) as client:
        resp = client.post("/api/research/paperscool/search", json={"queries": []})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "queries is required"


def test_paperscool_curate_route_returns_report_without_ingest():
    search_result = _FakeWorkflow().run(
        queries=["ICL压缩"],
        sources=["papers_cool"],
        branches=["arxiv", "venue"],
        top_k_per_query=5,
        show_per_branch=25,
    )

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/curate",
            json={"search_result": search_result, "title": "Curated Digest", "top_n": 5},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report"]["title"] == "Curated Digest"
    assert "registry_ingest" not in payload["report"]
    assert "UniICL" in payload["markdown"]


def test_paperscool_ingest_route_wires_explicit_ingest(monkeypatch):
    captured: list[str] = []
    captured_index_triggers: list[str] = []

    def _fake_ingest(*, report, persist_judge_scores=False):
        payload = {
            **report,
            "registry_ingest": {"total": 1, "created": 1, "updated": 0},
        }
        if persist_judge_scores:
            payload["judge_registry_ingest"] = {"total": 1, "created": 1, "updated": 0}
        return SimpleNamespace(report=payload)

    monkeypatch.setattr(paperscool_route, "ingest_curated_report", _fake_ingest)
    monkeypatch.setattr(
        paperscool_route,
        "_enqueue_repo_enrichment_async",
        lambda report: captured.append(str(report.get("title") or "")),
    )
    monkeypatch.setattr(
        paperscool_route,
        "_schedule_document_indexing_for_report",
        lambda report, *, trigger_source: (
            captured_index_triggers.append(trigger_source),
            report.__setitem__(
                "document_index_ingest",
                {"total": 1, "queued": 1, "skipped": 0, "trigger_source": trigger_source},
            ),
            report["document_index_ingest"],
        )[2],
    )

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/ingest",
            json={
                "report": {
                    "title": "Curated Digest",
                    "generated_at": "2026-03-12T00:00:00+00:00",
                    "queries": [],
                    "global_top": [],
                },
                "persist_judge_scores": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["registry_ingest"]["total"] == 1
    assert payload["judge_registry_ingest"]["created"] == 1
    assert payload["document_index_ingest"]["queued"] == 1
    assert captured == ["Curated Digest"]
    assert captured_index_triggers == ["paperscool_ingest_api"]


def test_paperscool_daily_route_success(monkeypatch, tmp_path):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)
    captured_index_triggers: list[str] = []
    monkeypatch.setattr(
        paperscool_route,
        "_schedule_document_indexing_for_report",
        lambda report, *, trigger_source: (
            captured_index_triggers.append(trigger_source),
            report.__setitem__(
                "document_index_ingest",
                {"total": 1, "queued": 1, "skipped": 0, "trigger_source": trigger_source},
            ),
            report["document_index_ingest"],
        )[2],
    )

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
    assert payload["report"]["document_index_ingest"]["queued"] == 1
    assert captured_index_triggers == ["paperscool_daily_sync"]


def test_paperscool_daily_route_with_llm_enrichment(monkeypatch):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

    class _FakeLLMService:
        def summarize_paper(self, *, title, abstract):
            return f"summary of {title}"

        def assess_relevance(self, *, paper, query):
            return {"score": 4, "reason": "relevant"}

        def analyze_trends(self, *, topic, papers):
            return f"trend:{topic}:{len(papers)}"

        def generate_daily_insight(self, report):
            return "daily insight"

    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: _FakeLLMService())

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
    # SSE stream response
    events = _parse_sse_events(resp.text)
    types = [e.get("type") for e in events]
    assert "llm_done" in types
    result_event = next(e for e in events if e.get("type") == "result")
    assert result_event["data"]["report"]["llm_analysis"]["enabled"] is True


def test_paperscool_daily_route_with_judge(monkeypatch):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

    class _FakeJudgment:
        def to_dict(self):
            return {
                "relevance": {"score": 5, "rationale": ""},
                "novelty": {"score": 4, "rationale": ""},
                "rigor": {"score": 4, "rationale": ""},
                "impact": {"score": 4, "rationale": ""},
                "clarity": {"score": 4, "rationale": ""},
                "overall": 4.2,
                "one_line_summary": "good",
                "recommendation": "must_read",
                "judge_model": "fake",
                "judge_cost_tier": 1,
            }

    class _FakeJudge:
        def __init__(self, llm_service=None):
            pass

        def judge_single(self, *, paper, query):
            return _FakeJudgment()

        def judge_with_calibration(self, *, paper, query, n_runs=1):
            return _FakeJudgment()

    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: object())
    monkeypatch.setattr(paperscool_route, "PaperJudge", _FakeJudge)

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
    events = _parse_sse_events(resp.text)
    types = [e.get("type") for e in events]
    assert "judge_done" in types
    result_event = next(e for e in events if e.get("type") == "result")
    assert result_event["data"]["report"]["judge"]["enabled"] is True


def test_paperscool_analyze_route_stream(monkeypatch):
    class _FakeLLM:
        def analyze_trends(self, *, topic, papers):
            return f"trend:{topic}:{len(papers)}"

    class _FakeJudgment:
        def to_dict(self):
            return {
                "relevance": {"score": 5, "rationale": ""},
                "novelty": {"score": 4, "rationale": ""},
                "rigor": {"score": 4, "rationale": ""},
                "impact": {"score": 4, "rationale": ""},
                "clarity": {"score": 4, "rationale": ""},
                "overall": 4.2,
                "one_line_summary": "good",
                "recommendation": "must_read",
                "judge_model": "fake",
                "judge_cost_tier": 1,
            }

    class _FakeJudge:
        def __init__(self, llm_service=None):
            pass

        def judge_single(self, *, paper, query):
            return _FakeJudgment()

        def judge_with_calibration(self, *, paper, query, n_runs=1):
            return _FakeJudgment()

    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: _FakeLLM())
    monkeypatch.setattr(paperscool_route, "PaperJudge", _FakeJudge)

    report = {
        "title": "Daily",
        "date": "2026-02-09",
        "generated_at": "2026-02-09T00:00:00+00:00",
        "source": "papers.cool",
        "sources": ["papers_cool"],
        "stats": {"unique_items": 1, "total_query_hits": 1, "query_count": 1},
        "queries": [
            {
                "raw_query": "ICL压缩",
                "normalized_query": "icl compression",
                "total_hits": 1,
                "top_items": [
                    {
                        "title": "UniICL",
                        "score": 10.0,
                        "snippet": "compress context",
                        "keywords": ["icl", "compression"],
                    }
                ],
            }
        ],
        "global_top": [],
    }

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/analyze",
            json={
                "report": report,
                "run_judge": True,
                "run_trends": True,
                "judge_token_budget": 5000,
            },
        )

    assert resp.status_code == 200
    text = resp.text
    assert '"type": "trend"' in text
    assert '"type": "judge"' in text
    assert "[DONE]" in text


def test_paperscool_repos_route_extracts_and_enriches(monkeypatch):
    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "full_name": "owner/repo",
                "stargazers_count": 42,
                "forks_count": 7,
                "open_issues_count": 1,
                "watchers_count": 5,
                "language": "Python",
                "license": {"spdx_id": "MIT"},
                "updated_at": "2026-02-01T00:00:00Z",
                "pushed_at": "2026-02-02T00:00:00Z",
                "archived": False,
                "topics": ["llm"],
                "html_url": "https://github.com/owner/repo",
            }

    monkeypatch.setattr(paperscool_route.requests, "get", lambda *args, **kwargs: _FakeResp())

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/repos",
            json={
                "papers": [
                    {
                        "title": "Repo Paper",
                        "url": "https://papers.cool/arxiv/1234",
                        "external_url": "https://github.com/owner/repo",
                    }
                ],
                "include_github_api": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["matched_repos"] == 1
    assert payload["repos"][0]["repo_url"] == "https://github.com/owner/repo"
    assert payload["repos"][0]["github"]["stars"] == 42


def test_paperscool_daily_route_persists_judge_scores(monkeypatch):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

    class _FakeJudgment:
        def to_dict(self):
            return {
                "relevance": {"score": 5, "rationale": ""},
                "novelty": {"score": 4, "rationale": ""},
                "rigor": {"score": 4, "rationale": ""},
                "impact": {"score": 4, "rationale": ""},
                "clarity": {"score": 4, "rationale": ""},
                "overall": 4.2,
                "one_line_summary": "good",
                "recommendation": "must_read",
                "judge_model": "fake",
                "judge_cost_tier": 1,
            }

    class _FakeJudge:
        def __init__(self, llm_service=None):
            pass

        def judge_single(self, *, paper, query):
            return _FakeJudgment()

        def judge_with_calibration(self, *, paper, query, n_runs=1):
            return _FakeJudgment()

    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: object())
    monkeypatch.setattr(paperscool_route, "PaperJudge", _FakeJudge)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_judge": True,
            },
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    result_event = next(e for e in events if e.get("type") == "result")
    report = result_event["data"]["report"]
    # Judge registry ingest should have been attempted
    assert "judge_registry_ingest" in report


class _FakeWorkflowMultiPaper:
    """Workflow returning multiple papers for filter testing."""

    def run(self, *, queries, sources, branches, top_k_per_query, show_per_branch, min_score=0.0):
        return {
            "source": "papers.cool",
            "fetched_at": "2026-02-10T00:00:00+00:00",
            "sources": sources,
            "queries": [
                {
                    "raw_query": queries[0],
                    "normalized_query": "icl compression",
                    "tokens": ["icl", "compression"],
                    "total_hits": 3,
                    "items": [
                        {
                            "paper_id": "p1",
                            "title": "GoodPaper",
                            "url": "https://papers.cool/venue/p1",
                            "score": 10.0,
                            "snippet": "excellent work",
                            "keywords": ["icl"],
                            "branches": branches,
                            "matched_queries": ["icl compression"],
                        },
                        {
                            "paper_id": "p2",
                            "title": "MediocreWork",
                            "url": "https://papers.cool/venue/p2",
                            "score": 5.0,
                            "snippet": "average",
                            "keywords": ["icl"],
                            "branches": branches,
                            "matched_queries": ["icl compression"],
                        },
                        {
                            "paper_id": "p3",
                            "title": "WeakPaper",
                            "url": "https://papers.cool/venue/p3",
                            "score": 2.0,
                            "snippet": "not great",
                            "keywords": ["icl"],
                            "branches": branches,
                            "matched_queries": ["icl compression"],
                        },
                    ],
                }
            ],
            "items": [],
            "summary": {
                "unique_items": 3,
                "total_query_hits": 3,
            },
        }


def test_dailypaper_sse_filter_removes_low_papers(monkeypatch):
    """End-to-end: judge scores papers, filter removes 'skip' and 'skim'."""
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search_multi)

    # Judge returns different recommendations per paper title
    class _VaryingJudgment:
        def __init__(self, title):
            self._title = title

        def to_dict(self):
            rec_map = {
                "GoodPaper": ("must_read", 4.5),
                "MediocreWork": ("skim", 2.9),
                "WeakPaper": ("skip", 1.8),
            }
            rec, overall = rec_map.get(self._title, ("skip", 1.0))
            return {
                "relevance": {"score": 4, "rationale": ""},
                "novelty": {"score": 3, "rationale": ""},
                "rigor": {"score": 3, "rationale": ""},
                "impact": {"score": 3, "rationale": ""},
                "clarity": {"score": 3, "rationale": ""},
                "overall": overall,
                "one_line_summary": f"summary of {self._title}",
                "recommendation": rec,
                "judge_model": "fake",
                "judge_cost_tier": 1,
            }

    class _FakeJudge:
        def __init__(self, llm_service=None):
            pass

        def judge_single(self, *, paper, query):
            return _VaryingJudgment(paper.get("title", ""))

        def judge_with_calibration(self, *, paper, query, n_runs=1):
            return _VaryingJudgment(paper.get("title", ""))

    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: object())
    monkeypatch.setattr(paperscool_route, "PaperJudge", _FakeJudge)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_judge": True,
                "judge_max_items_per_query": 10,
            },
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    types = [e.get("type") for e in events]

    # All expected phases present
    assert "judge_done" in types
    assert "filter_done" in types
    assert "result" in types

    # Check filter_done event
    filter_event = next(e for e in events if e.get("type") == "filter_done")
    assert filter_event["data"]["total_before"] == 3
    assert filter_event["data"]["total_after"] == 1  # only GoodPaper kept
    assert filter_event["data"]["removed_count"] == 2

    # Check filter log has details for removed papers
    filter_log = filter_event["data"]["log"]
    removed_titles = {entry["title"] for entry in filter_log}
    assert "MediocreWork" in removed_titles
    assert "WeakPaper" in removed_titles
    assert "GoodPaper" not in removed_titles

    # Check final result only has the kept paper
    result_event = next(e for e in events if e.get("type") == "result")
    final_report = result_event["data"]["report"]
    final_items = final_report["queries"][0]["top_items"]
    assert len(final_items) == 1
    assert final_items[0]["title"] == "GoodPaper"


def test_dailypaper_sse_full_pipeline_llm_judge_filter(monkeypatch):
    """End-to-end: LLM enrichment + Judge + Filter in one SSE stream."""
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search_multi)

    class _FakeLLMService:
        def summarize_paper(self, *, title, abstract):
            return f"summary of {title}"

        def assess_relevance(self, *, paper, query):
            return {"score": 4, "reason": "relevant"}

        def analyze_trends(self, *, topic, papers):
            return f"trend:{topic}:{len(papers)}"

        def generate_daily_insight(self, report):
            return "daily insight"

        def complete(self, **kwargs):
            return "{}"

        def describe_task_provider(self, task_type):
            return {"model_name": "fake", "cost_tier": 1}

    class _VaryingJudgment:
        def __init__(self, title):
            self._title = title

        def to_dict(self):
            rec_map = {
                "GoodPaper": ("must_read", 4.5),
                "MediocreWork": ("worth_reading", 3.7),
                "WeakPaper": ("skip", 1.8),
            }
            rec, overall = rec_map.get(self._title, ("skip", 1.0))
            return {
                "relevance": {"score": 4, "rationale": ""},
                "novelty": {"score": 3, "rationale": ""},
                "rigor": {"score": 3, "rationale": ""},
                "impact": {"score": 3, "rationale": ""},
                "clarity": {"score": 3, "rationale": ""},
                "overall": overall,
                "one_line_summary": f"summary of {self._title}",
                "recommendation": rec,
                "judge_model": "fake",
                "judge_cost_tier": 1,
            }

    class _FakeJudge:
        def __init__(self, llm_service=None):
            pass

        def judge_single(self, *, paper, query):
            return _VaryingJudgment(paper.get("title", ""))

        def judge_with_calibration(self, *, paper, query, n_runs=1):
            return _VaryingJudgment(paper.get("title", ""))

    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: _FakeLLMService())
    monkeypatch.setattr(paperscool_route, "PaperJudge", _FakeJudge)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_llm_analysis": True,
                "llm_features": ["summary", "trends"],
                "enable_judge": True,
                "judge_max_items_per_query": 10,
            },
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    types = [e.get("type") for e in events]

    # Full pipeline phases
    assert "search_done" in types
    assert "report_built" in types
    assert "trend" in types
    assert "llm_done" in types
    assert "judge_done" in types
    assert "filter_done" in types
    assert "result" in types

    # Filter keeps must_read + worth_reading, removes skip
    filter_event = next(e for e in events if e.get("type") == "filter_done")
    assert filter_event["data"]["total_after"] == 2  # GoodPaper + MediocreWork
    assert filter_event["data"]["removed_count"] == 1  # WeakPaper

    # Final report has 2 papers
    result_event = next(e for e in events if e.get("type") == "result")
    final_items = result_event["data"]["report"]["queries"][0]["top_items"]
    assert len(final_items) == 2
    final_titles = {item["title"] for item in final_items}
    assert final_titles == {"GoodPaper", "MediocreWork"}

    # LLM analysis present
    assert result_event["data"]["report"]["llm_analysis"]["enabled"] is True

    # Filter metadata in report
    assert result_event["data"]["report"]["filter"]["enabled"] is True


def test_dailypaper_sync_path_no_llm_no_judge(monkeypatch):
    """When no LLM/Judge, endpoint returns sync JSON (not SSE)."""
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "sources": ["papers_cool"],
                "branches": ["arxiv", "venue"],
            },
        )

    assert resp.status_code == 200
    # Should be JSON, not SSE
    payload = resp.json()
    assert "report" in payload
    assert payload["report"]["stats"]["unique_items"] == 1
    # No filter block in sync path
    assert "filter" not in payload["report"]


def test_paperscool_repos_route_can_persist(monkeypatch):
    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "full_name": "owner/repo",
                "stargazers_count": 42,
                "forks_count": 7,
                "open_issues_count": 1,
                "watchers_count": 5,
                "language": "Python",
                "license": {"spdx_id": "MIT"},
                "updated_at": "2026-02-01T00:00:00Z",
                "pushed_at": "2026-02-02T00:00:00Z",
                "archived": False,
                "topics": ["llm"],
                "html_url": "https://github.com/owner/repo",
            }

    class _FakeStore:
        def __init__(self):
            self.rows = []

        def ingest_repo_enrichment_rows(self, *, rows, source):
            self.rows.extend(rows)
            return {
                "total": len(rows),
                "created": len(rows),
                "updated": 0,
                "skipped": 0,
                "unresolved_paper": 0,
            }

    fake_store = _FakeStore()
    monkeypatch.setattr(paperscool_route.requests, "get", lambda *args, **kwargs: _FakeResp())
    monkeypatch.setattr(paperscool_route, "SqlAlchemyResearchStore", lambda: fake_store)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/repos",
            json={
                "papers": [
                    {
                        "title": "Repo Paper",
                        "url": "https://papers.cool/arxiv/1234",
                        "external_url": "https://github.com/owner/repo",
                    }
                ],
                "include_github_api": True,
                "persist": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["persist_summary"]["total"] == 1
    assert len(fake_store.rows) == 1


def test_paperscool_daily_route_enqueues_repo_enrichment(monkeypatch):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

    called = {"count": 0}

    def _fake_enqueue(report):
        called["count"] += 1

    monkeypatch.setattr(paperscool_route, "_enqueue_repo_enrichment_async", _fake_enqueue)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_llm_analysis": False,
                "enable_judge": False,
            },
        )

    assert resp.status_code == 200
    assert called["count"] == 1


def test_paperscool_daily_resume_session(monkeypatch, tmp_path):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)
    paperscool_route._pipeline_session_store = PipelineSessionStore(
        db_url=f"sqlite:///{tmp_path / 'daily-session.db'}"
    )

    class _FakeJudgment:
        def to_dict(self):
            return {
                "relevance": {"score": 5, "rationale": ""},
                "novelty": {"score": 4, "rationale": ""},
                "rigor": {"score": 4, "rationale": ""},
                "impact": {"score": 4, "rationale": ""},
                "clarity": {"score": 4, "rationale": ""},
                "overall": 4.2,
                "one_line_summary": "good",
                "recommendation": "must_read",
                "judge_model": "fake",
                "judge_cost_tier": 1,
            }

    class _FakeJudge:
        def __init__(self, llm_service=None):
            pass

        def judge_single(self, *, paper, query):
            return _FakeJudgment()

        def judge_with_calibration(self, *, paper, query, n_runs=1):
            return _FakeJudgment()

    monkeypatch.setattr(paperscool_route, "PaperJudge", _FakeJudge)
    monkeypatch.setattr(paperscool_route, "get_llm_service", lambda: object())

    with TestClient(api_main.app) as client:
        first = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_judge": True,
                "judge_runs": 1,
                "judge_max_items_per_query": 2,
            },
        )
        assert first.status_code == 200
        first_events = _parse_sse_events(first.text)
        first_result = next(e for e in first_events if e.get("type") == "result")
        session_id = first_result["data"].get("session_id")
        assert session_id

        session_resp = client.get(f"/api/research/paperscool/sessions/{session_id}")
        assert session_resp.status_code == 200
        assert session_resp.json()["session"]["status"] == "completed"

        resumed = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "enable_judge": True,
                "session_id": session_id,
                "resume": True,
            },
        )
        assert resumed.status_code == 200
        resumed_events = _parse_sse_events(resumed.text)
        resumed_result = next(e for e in resumed_events if e.get("type") == "result")
        assert resumed_result["data"].get("resumed") is True


def test_paperscool_daily_route_pending_approval_and_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)

    calls = {"ingest": 0}

    def _fake_ingest(*, report, persist_judge_scores=False):
        calls["ingest"] += 1
        payload = dict(report)
        payload["registry_ingest"] = {"saved": 1}
        return SimpleNamespace(report=payload)

    monkeypatch.setattr(paperscool_route, "ingest_curated_report", _fake_ingest)
    monkeypatch.setattr(
        paperscool_route,
        "_pipeline_session_store",
        PipelineSessionStore(db_url=f"sqlite:///{tmp_path / 'daily-approval.db'}"),
    )

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/research/paperscool/daily",
            json={
                "queries": ["ICL压缩"],
                "require_approval": True,
            },
        )
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        types = [e.get("type") for e in events]
        assert "approval_required" in types
        result_event = next(e for e in events if e.get("type") == "result")
        assert result_event["data"].get("approval_status") == "pending_approval"
        session_id = result_event["data"].get("session_id")
        assert session_id

        session_resp = client.get(f"/api/research/paperscool/sessions/{session_id}")
        assert session_resp.status_code == 200
        assert session_resp.json()["session"]["status"] == "pending_approval"

        queue_resp = client.get("/api/research/paperscool/approvals?limit=10")
        assert queue_resp.status_code == 200
        ids = [item["session_id"] for item in queue_resp.json().get("items", [])]
        assert session_id in ids

    # Ingest is gated until explicit approve
    assert calls["ingest"] == 0


def test_paperscool_daily_approval_decisions(monkeypatch, tmp_path):
    monkeypatch.setattr(paperscool_route, "_run_topic_search", _fake_run_topic_search)
    monkeypatch.setattr(
        paperscool_route,
        "_pipeline_session_store",
        PipelineSessionStore(db_url=f"sqlite:///{tmp_path / 'daily-approval-decision.db'}"),
    )
    monkeypatch.setattr(
        paperscool_route,
        "ingest_curated_report",
        lambda *, report, persist_judge_scores=False: SimpleNamespace(
            report={
                **report,
                "registry_ingest": {"saved": len(report.get("queries") or [])},
                **({"judge_registry_ingest": {"saved": 0}} if persist_judge_scores else {}),
            }
        ),
    )

    repo_calls = {"count": 0}

    def _fake_enqueue(_report):
        repo_calls["count"] += 1

    monkeypatch.setattr(paperscool_route, "_enqueue_repo_enrichment_async", _fake_enqueue)

    with TestClient(api_main.app) as client:
        # Session A -> approve
        pending_a = client.post(
            "/api/research/paperscool/daily",
            json={"queries": ["ICL压缩"], "require_approval": True},
        )
        assert pending_a.status_code == 200
        events_a = _parse_sse_events(pending_a.text)
        result_a = next(e for e in events_a if e.get("type") == "result")
        session_a = result_a["data"]["session_id"]

        approve = client.post(f"/api/research/paperscool/sessions/{session_a}/approve", json={})
        assert approve.status_code == 200
        approved_session = approve.json()["session"]
        assert approved_session["status"] == "completed"
        assert approved_session["result"]["approval_status"] == "approved"
        assert "registry_ingest" in approved_session["result"]["report"]

        # Session B -> reject
        pending_b = client.post(
            "/api/research/paperscool/daily",
            json={"queries": ["RAG"], "require_approval": True},
        )
        assert pending_b.status_code == 200
        events_b = _parse_sse_events(pending_b.text)
        result_b = next(e for e in events_b if e.get("type") == "result")
        session_b = result_b["data"]["session_id"]

        reject = client.post(
            f"/api/research/paperscool/sessions/{session_b}/reject",
            json={"reason": "Not ready"},
        )
        assert reject.status_code == 200
        rejected_session = reject.json()["session"]
        assert rejected_session["status"] == "rejected"
        assert rejected_session["state"].get("reject_reason") == "Not ready"
        assert rejected_session["result"].get("approval_status") == "rejected"

        queue_resp = client.get("/api/research/paperscool/approvals?limit=10")
        assert queue_resp.status_code == 200
        ids = [item["session_id"] for item in queue_resp.json().get("items", [])]
        assert session_a not in ids
        assert session_b not in ids

    assert repo_calls["count"] == 1
