from paperbot.application.workflows.dailypaper import (
    DailyPaperReporter,
    build_daily_paper_report,
    normalize_output_formats,
    render_daily_paper_markdown,
)


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


def test_normalize_output_formats_supports_both_alias():
    assert normalize_output_formats(["both"]) == ["markdown", "json"]
    assert normalize_output_formats(["json", "markdown"]) == ["json", "markdown"]
