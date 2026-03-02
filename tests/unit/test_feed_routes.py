"""Tests for RSS/Atom feed API routes."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _sample_report():
    return {
        "title": "DailyPaper Digest",
        "date": "2026-03-02",
        "generated_at": "2026-03-02T08:30:00+00:00",
        "stats": {"unique_items": 2, "total_query_hits": 5},
        "queries": [
            {
                "normalized_query": "KV cache",
                "total_hits": 2,
                "top_items": [
                    {
                        "title": "FlashKV: Efficient KV Cache",
                        "url": "https://arxiv.org/abs/2601.00001",
                        "snippet": "Novel KV cache compression method",
                        "score": 9.0,
                        "judge": {
                            "overall": 4.5,
                            "recommendation": "must_read",
                            "one_line_summary": "Great paper on KV cache",
                        },
                        "digest_card": {
                            "highlight": "2x inference speedup",
                            "tags": ["KV Cache", "LLM"],
                        },
                    },
                    {
                        "title": "PagedAttention v2",
                        "url": "https://arxiv.org/abs/2601.00002",
                        "snippet": "Improved paged attention",
                        "score": 7.5,
                    },
                ],
            }
        ],
        "global_top": [],
    }


# Import feed module directly to avoid importing the full FastAPI app
# which triggers multipart checks in CI environments.
from paperbot.api.routes.feed import (
    _build_atom_xml,
    _build_rss_xml,
    _collect_papers_from_report,
    _load_latest_reports,
)


# ── Unit tests for helper functions ──────────────────────────

def test_collect_papers_from_report():
    report = _sample_report()
    papers = _collect_papers_from_report(report)
    assert len(papers) == 2
    assert papers[0]["title"] == "FlashKV: Efficient KV Cache"


def test_load_latest_reports(tmp_path):
    # Write sample reports
    for i in range(3):
        path = tmp_path / f"2026-03-0{i+1}-digest.json"
        path.write_text(json.dumps({"title": f"Report {i}", "date": f"2026-03-0{i+1}"}))

    reports = _load_latest_reports(tmp_path, limit=2)
    assert len(reports) == 2


def test_load_latest_reports_empty_dir(tmp_path):
    reports = _load_latest_reports(tmp_path)
    assert reports == []


def test_load_latest_reports_nonexistent_dir(tmp_path):
    reports = _load_latest_reports(tmp_path / "nonexistent")
    assert reports == []


# ── RSS/Atom XML generation ──────────────────────────────────

def test_build_rss_xml():
    reports = [_sample_report()]
    xml = _build_rss_xml(reports)

    assert "<?xml" in xml
    assert "rss" in xml.lower()
    assert "FlashKV" in xml
    assert "PagedAttention" in xml


def test_build_rss_xml_empty():
    xml = _build_rss_xml([])
    assert "<?xml" in xml or "<rss" in xml


def test_build_atom_xml():
    reports = [_sample_report()]
    xml = _build_atom_xml(reports)

    assert "FlashKV" in xml


def test_build_atom_xml_empty():
    xml = _build_atom_xml([])
    assert isinstance(xml, str)


# ── API endpoint tests ───────────────────────────────────────
# These test the XML generation functions directly since importing
# the full FastAPI app requires python-multipart which may not be
# installed in CI.

def test_daily_rss_with_reports(tmp_path):
    report_path = tmp_path / "2026-03-02-digest.json"
    report_path.write_text(json.dumps(_sample_report()))

    reports = _load_latest_reports(tmp_path)
    xml = _build_rss_xml(reports)
    assert "FlashKV" in xml
    assert "PagedAttention" in xml


def test_daily_atom_with_reports(tmp_path):
    report_path = tmp_path / "2026-03-02-digest.json"
    report_path.write_text(json.dumps(_sample_report()))

    reports = _load_latest_reports(tmp_path)
    xml = _build_atom_xml(reports)
    assert "FlashKV" in xml


def test_track_rss_filters_by_query(tmp_path):
    report_path = tmp_path / "2026-03-02-digest.json"
    report_path.write_text(json.dumps(_sample_report()))

    reports = _load_latest_reports(tmp_path)

    # Filter to matching track
    filtered = []
    for report in reports:
        track_report = {
            **report,
            "queries": [
                q for q in (report.get("queries") or [])
                if "kv cache" in (q.get("normalized_query") or "").lower()
            ],
        }
        if track_report["queries"]:
            filtered.append(track_report)

    xml = _build_rss_xml(filtered, title="PaperBot · KV cache")
    assert "FlashKV" in xml

    # Non-matching track
    filtered_none = []
    for report in reports:
        track_report = {
            **report,
            "queries": [
                q for q in (report.get("queries") or [])
                if "nonexistent" in (q.get("normalized_query") or "").lower()
            ],
        }
        if track_report["queries"]:
            filtered_none.append(track_report)

    xml_empty = _build_rss_xml(filtered_none)
    assert "FlashKV" not in xml_empty


# ── Feed service export test ─────────────────────────────────

def test_scholar_feed_service_export():
    from paperbot.workflows.feed import ScholarFeedService

    svc = ScholarFeedService()
    svc.process_daily_paper_report(_sample_report())
    entries = svc.export_feed_entries(limit=10)

    assert len(entries) > 0
    assert "title" in entries[0]
    assert "description" in entries[0]
