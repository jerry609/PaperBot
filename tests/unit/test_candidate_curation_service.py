from __future__ import annotations

from types import SimpleNamespace

import pytest

from paperbot.application.services import candidate_curation
from paperbot.application.services.candidate_curation import (
    curate_search_result,
    ingest_curated_report,
)
from paperbot.infrastructure.stores.paper_store import SqlAlchemyPaperStore


def _sample_search_result() -> dict:
    return {
        "source": "papers.cool",
        "fetched_at": "2026-03-12T00:00:00+00:00",
        "sources": ["papers_cool"],
        "queries": [
            {
                "raw_query": "ICL压缩",
                "normalized_query": "icl compression",
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
                        "snippet": "A paper about ICL compression.",
                        "keywords": ["icl", "compression"],
                        "branches": ["arxiv", "venue"],
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
            "source_breakdown": {"papers_cool": 1},
            "query_highlights": [],
        },
    }


@pytest.mark.asyncio
async def test_curate_search_result_builds_report_without_registry_ingest() -> None:
    curated = await curate_search_result(
        search_result=_sample_search_result(),
        title="Curated Digest",
        top_n=5,
    )

    assert curated.report["title"] == "Curated Digest"
    assert "registry_ingest" not in curated.report
    assert curated.events[0].type == "report_built"
    assert curated.events[0].data["report"]["title"] == "Curated Digest"


@pytest.mark.asyncio
async def test_ingest_curated_report_is_explicit_and_can_include_judge_scores(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curated = await curate_search_result(search_result=_sample_search_result(), title="Curated", top_n=5)
    store = SqlAlchemyPaperStore(db_url=f"sqlite:///{tmp_path / 'curated.db'}")

    monkeypatch.setattr(
        candidate_curation,
        "persist_judge_scores_to_registry",
        lambda report, paper_store=None: {"total": 1, "created": 1, "updated": 0},
    )

    ingested = ingest_curated_report(
        report=curated.report,
        persist_judge_scores=True,
        paper_store=store,
    )

    assert ingested.report["registry_ingest"]["total"] == 1
    assert ingested.report["judge_registry_ingest"] == {"total": 1, "created": 1, "updated": 0}
