"""Tests for daily digest card prompt, LLM extraction, and pipeline integration."""
import json

from paperbot.application.prompts.paper_analysis import (
    DAILY_DIGEST_CARD_SYSTEM,
    DAILY_DIGEST_CARD_USER,
)
from paperbot.application.prompts.registry import PromptRegistry
from paperbot.application.workflows.dailypaper import (
    SUPPORTED_LLM_FEATURES,
    build_daily_paper_report,
    enrich_daily_paper_report,
    normalize_llm_features,
    render_daily_paper_markdown,
)
from paperbot.application.services.email_template import build_digest_html, build_digest_text


# ── Prompt template tests ────────────────────────────────────

def test_daily_digest_card_prompt_has_required_fields():
    assert "highlight" in DAILY_DIGEST_CARD_USER
    assert "method" in DAILY_DIGEST_CARD_USER
    assert "finding" in DAILY_DIGEST_CARD_USER
    assert "tags" in DAILY_DIGEST_CARD_USER


def test_daily_digest_card_registered_in_registry():
    registry = PromptRegistry()
    tmpl = registry.get("daily_digest_card")
    assert tmpl.name == "daily_digest_card"
    assert tmpl.system == DAILY_DIGEST_CARD_SYSTEM


# ── LLM feature normalization ────────────────────────────────

def test_digest_card_in_supported_features():
    assert "digest_card" in SUPPORTED_LLM_FEATURES


def test_normalize_llm_features_includes_digest_card():
    result = normalize_llm_features(["digest_card", "summary"])
    assert "digest_card" in result
    assert "summary" in result


# ── Pipeline integration (fake LLM) ──────────────────────────

class _FakeLLMService:
    def summarize_paper(self, title: str, abstract: str) -> str:
        return f"summary:{title}"

    def analyze_trends(self, *, topic: str, papers):
        return f"trend:{topic}"

    def assess_relevance(self, *, paper, query: str):
        return {"score": 80, "reason": "relevant"}

    def generate_daily_insight(self, report):
        return "insight"

    def extract_daily_digest_card(self, title: str, abstract: str):
        return {
            "highlight": f"Key finding from {title}",
            "method": "Novel approach",
            "finding": "Significant improvement",
            "tags": ["LLM", "efficiency"],
        }


def _sample_search_result():
    return {
        "source": "papers.cool",
        "sources": ["papers_cool"],
        "queries": [
            {
                "raw_query": "test",
                "normalized_query": "test query",
                "total_hits": 1,
                "items": [
                    {
                        "title": "TestPaper",
                        "url": "https://example.com/paper",
                        "score": 8.5,
                        "snippet": "A test abstract",
                        "matched_queries": ["test query"],
                    }
                ],
            }
        ],
        "items": [],
        "summary": {"unique_items": 1, "total_query_hits": 1},
    }


def test_enrich_with_digest_card():
    report = build_daily_paper_report(
        search_result=_sample_search_result(), title="Digest Test", top_n=5,
    )
    enriched = enrich_daily_paper_report(
        report,
        llm_service=_FakeLLMService(),
        llm_features=["digest_card"],
    )
    item = enriched["queries"][0]["top_items"][0]

    assert "digest_card" in item
    assert item["digest_card"]["highlight"] == "Key finding from TestPaper"
    assert item["digest_card"]["tags"] == ["LLM", "efficiency"]


def test_digest_card_in_markdown_output():
    report = build_daily_paper_report(
        search_result=_sample_search_result(), title="Digest Test", top_n=5,
    )
    enriched = enrich_daily_paper_report(
        report,
        llm_service=_FakeLLMService(),
        llm_features=["digest_card"],
    )
    md = render_daily_paper_markdown(enriched)

    assert "Highlight:" in md
    assert "Key finding from TestPaper" in md
    assert "Tags: LLM, efficiency" in md


def test_digest_card_in_email_html():
    report = build_daily_paper_report(
        search_result=_sample_search_result(), title="Digest Test", top_n=5,
    )
    # Manually add digest_card to an item (simulating enrichment)
    item = report["queries"][0]["top_items"][0]
    item["digest_card"] = {
        "highlight": "Major breakthrough",
        "method": "New method",
        "finding": "Better results",
        "tags": ["AI", "NLP"],
    }
    # Also add judge so tier grouping works
    item["judge"] = {
        "overall": 4.0,
        "recommendation": "must_read",
        "one_line_summary": "good",
        "relevance": {"score": 5, "rationale": ""},
        "novelty": {"score": 4, "rationale": ""},
        "rigor": {"score": 4, "rationale": ""},
        "impact": {"score": 4, "rationale": ""},
        "clarity": {"score": 4, "rationale": ""},
    }
    html = build_digest_html(report)

    assert "Major breakthrough" in html
    assert "AI" in html


def test_main_figure_inline_in_email_html():
    report = build_daily_paper_report(
        search_result=_sample_search_result(), title="Digest Test", top_n=5,
    )
    item = report["queries"][0]["top_items"][0]
    item["judge"] = {
        "overall": 4.0,
        "recommendation": "must_read",
        "one_line_summary": "good",
        "relevance": {"score": 5, "rationale": ""},
        "novelty": {"score": 4, "rationale": ""},
        "rigor": {"score": 4, "rationale": ""},
        "impact": {"score": 4, "rationale": ""},
        "clarity": {"score": 4, "rationale": ""},
    }
    item["main_figure"] = {
        "caption": "Figure 1: Overview",
        "inline_data_url": "data:image/png;base64,QUFBQQ==",
    }

    html = build_digest_html(report)
    assert "主方法图" in html
    assert "data:image/png;base64,QUFBQQ==" in html
    assert "Figure 1: Overview" in html


def test_digest_card_in_email_text():
    report = build_daily_paper_report(
        search_result=_sample_search_result(), title="Digest Test", top_n=5,
    )
    item = report["queries"][0]["top_items"][0]
    item["digest_card"] = {
        "highlight": "Big discovery",
        "method": "Fancy method",
        "finding": "Great results",
        "tags": ["ML", "CV"],
    }
    item["judge"] = {
        "overall": 4.0,
        "recommendation": "must_read",
        "one_line_summary": "good paper",
    }
    text = build_digest_text(report)

    assert "Big discovery" in text
    assert "ML, CV" in text
