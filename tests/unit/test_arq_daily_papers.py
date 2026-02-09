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
async def test_daily_papers_job_generates_report_and_feed(tmp_path, monkeypatch):
    class _FakeWorkflow:
        def run(self, *, queries, sources, branches, top_k_per_query, show_per_branch):
            return {
                "source": "papers.cool",
                "sources": sources,
                "queries": [
                    {
                        "raw_query": queries[0],
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

    import paperbot.application.workflows.paperscool_topic_search as topic_mod
    from paperbot.infrastructure.queue import arq_worker

    monkeypatch.setattr(topic_mod, "PapersCoolTopicSearchWorkflow", _FakeWorkflow)

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
