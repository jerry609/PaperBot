from paperbot.application.workflows.dailypaper import (
    DailyPaperReporter,
    build_daily_paper_report,
    enrich_daily_paper_report,
    normalize_llm_features,
    normalize_output_formats,
    render_daily_paper_markdown,
)


class _FakeLLMService:
    def summarize_paper(self, title: str, abstract: str) -> str:
        return f"summary:{title}"

    def analyze_trends(self, *, topic: str, papers):
        return f"trend:{topic}:{len(papers)}"

    def assess_relevance(self, *, paper, query: str):
        return {"score": 88, "reason": f"match:{query}"}

    def generate_daily_insight(self, report):
        return "daily insight"


def _sample_search_result():
    return {
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
                        "score": 10.2,
                        "snippet": "compress in-context learning",
                        "matched_queries": ["icl compression"],
                    }
                ],
            }
        ],
        "items": [
            {
                "title": "UniICL",
                "url": "https://papers.cool/venue/2025.acl-long.24@ACL",
                "score": 10.2,
                "snippet": "compress in-context learning",
                "matched_queries": ["icl compression"],
            }
        ],
        "summary": {
            "unique_items": 1,
            "total_query_hits": 1,
        },
    }


def test_build_and_render_daily_report(tmp_path):
    report = build_daily_paper_report(
        search_result=_sample_search_result(), title="My Daily", top_n=5
    )
    markdown = render_daily_paper_markdown(report)

    assert report["title"] == "My Daily"
    assert report["stats"]["unique_items"] == 1
    assert "# My Daily" in markdown
    assert "UniICL" in markdown

    reporter = DailyPaperReporter(output_dir=str(tmp_path / "daily"))
    artifacts = reporter.write(
        report=report, markdown=markdown, formats=["markdown", "json"], slug="my-daily"
    )

    assert artifacts.markdown_path is not None
    assert artifacts.json_path is not None


def test_enrich_daily_report_with_llm_features():
    report = build_daily_paper_report(search_result=_sample_search_result(), title="My Daily", top_n=5)
    enriched = enrich_daily_paper_report(
        report,
        llm_service=_FakeLLMService(),
        llm_features=["summary", "trends", "insight", "relevance"],
    )
    top_item = enriched["queries"][0]["top_items"][0]

    assert top_item["ai_summary"] == "summary:UniICL"
    assert top_item["relevance"]["score"] == 88
    assert enriched["llm_analysis"]["daily_insight"] == "daily insight"
    assert "LLM Insights" in render_daily_paper_markdown(enriched)


def test_normalize_output_formats_supports_both_alias():
    assert normalize_output_formats(["both"]) == ["markdown", "json"]
    assert normalize_output_formats(["json", "markdown"]) == ["json", "markdown"]


def test_normalize_llm_features_filters_unknown_items():
    assert normalize_llm_features(["summary", "foo", "trends", "summary"]) == ["summary", "trends"]
