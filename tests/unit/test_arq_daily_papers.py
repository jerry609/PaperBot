from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arq", reason="arq not installed")


def test_build_daily_paper_cron_jobs_disabled(monkeypatch):
    monkeypatch.setenv("PAPERBOT_DAILYPAPER_ENABLED", "false")
    from paperbot.infrastructure.queue import arq_worker

    jobs = arq_worker._build_daily_paper_cron_jobs()
    assert jobs == []


def test_build_daily_paper_cron_jobs_enabled(monkeypatch):
    monkeypatch.setenv("PAPERBOT_DAILYPAPER_ENABLED", "true")
    monkeypatch.setenv("PAPERBOT_DAILYPAPER_CRON_HOUR", "7")
    monkeypatch.setenv("PAPERBOT_DAILYPAPER_CRON_MINUTE", "45")

    from paperbot.infrastructure.queue import arq_worker

    jobs = arq_worker._build_daily_paper_cron_jobs()
    assert len(jobs) == 1


@pytest.mark.asyncio
async def test_cron_daily_papers_enqueues_figure_flags(monkeypatch):
    from paperbot.infrastructure.queue import arq_worker

    class _FakeRedis:
        def __init__(self):
            self.kwargs = None

        async def enqueue_job(self, name, **kwargs):
            self.kwargs = {"name": name, **kwargs}

            class _FakeJob:
                job_id = "job-123"

            return _FakeJob()

    class _NoopEventLog:
        def append(self, event):
            return None

    monkeypatch.setattr(arq_worker, "_event_log", lambda: _NoopEventLog())
    monkeypatch.setenv("PAPERBOT_DAILYPAPER_ENABLE_FIGURES", "true")
    monkeypatch.setenv("PAPERBOT_DAILYPAPER_FIGURES_MAX_ITEMS", "7")
    monkeypatch.setenv("MINERU_API_BASE_URL", "https://mineru.net/api/v4")
    monkeypatch.setenv("MINERU_MODEL_VERSION", "vlm")
    monkeypatch.setenv("MINERU_MAX_WAIT_SECONDS", "120")

    redis = _FakeRedis()
    result = await arq_worker.cron_daily_papers({"redis": redis})

    assert result["status"] == "ok"
    assert redis.kwargs is not None
    assert redis.kwargs["name"] == "daily_papers_job"
    assert redis.kwargs["enable_figures"] is True
    assert redis.kwargs["figures_max_items"] == 7
    assert redis.kwargs["mineru_api_base_url"] == "https://mineru.net/api/v4"
    assert redis.kwargs["mineru_model_version"] == "vlm"
    assert redis.kwargs["mineru_max_wait_seconds"] == 120.0


@pytest.mark.asyncio
async def test_daily_papers_job_generates_report_and_feed(tmp_path, monkeypatch):
    _fake_search_result = {
        "source": "papers.cool",
        "sources": ["papers_cool"],
        "queries": [
            {
                "raw_query": "ICL压缩",
                "normalized_query": "icl compression",
                "total_hits": 1,
                "items": [
                    {
                        "title": "UniICL",
                        "url": "https://papers.cool/venue/2025.acl-long.24@ACL",
                        "score": 9.9,
                        "matched_queries": ["icl compression"],
                    }
                ],
            }
        ],
        "items": [
            {
                "title": "UniICL",
                "url": "https://papers.cool/venue/2025.acl-long.24@ACL",
                "score": 9.9,
                "matched_queries": ["icl compression"],
            }
        ],
        "summary": {
            "unique_items": 1,
            "total_query_hits": 1,
        },
    }

    import paperbot.application.workflows.unified_topic_search as uts_mod
    from paperbot.infrastructure.queue import arq_worker

    async def _fake_run_unified(**kwargs):
        return _fake_search_result

    monkeypatch.setattr(uts_mod, "run_unified_topic_search", _fake_run_unified)

    result = await arq_worker.daily_papers_job(
        ctx={},
        queries=["ICL压缩"],
        sources=["papers_cool"],
        branches=["arxiv", "venue"],
        output_dir=str(tmp_path / "daily"),
        save=True,
    )

    assert result["status"] == "ok"
    assert result["report"]["stats"]["unique_items"] == 1
    assert result["markdown_path"] is not None
    assert result["json_path"] is not None
    assert Path(result["markdown_path"]).exists()
    assert Path(result["json_path"]).exists()
    assert len(result["feed_events"]) >= 1
