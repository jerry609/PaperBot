from paperbot.application.workflows.paperscool_topic_search import PapersCoolTopicSearchWorkflow
from paperbot.infrastructure.connectors.paperscool_connector import PapersCoolRecord


class _FakeConnector:
    def __init__(self):
        self.calls = []

    def search(self, *, branch: str, query: str, highlight: bool = True, show=None):
        self.calls.append((branch, query, highlight, show))
        if query == "icl compression":
            if branch == "arxiv":
                return [
                    PapersCoolRecord(
                        paper_id="2510.00001",
                        title="UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation",
                        url="https://papers.cool/arxiv/2510.00001",
                        source_branch="arxiv",
                        external_url="https://arxiv.org/abs/2510.00001",
                        pdf_url="https://arxiv.org/pdf/2510.00001",
                        authors=["Author A"],
                        subject_or_venue="Artificial Intelligence",
                        published_at="2025-10-01 00:00:00 UTC",
                        snippet="ICL compression with efficient selection and generation.",
                        keywords=["icl", "compression", "llms"],
                        pdf_stars=10,
                        kimi_stars=8,
                    )
                ]
            if branch == "venue":
                return [
                    PapersCoolRecord(
                        paper_id="2025.acl-long.24@ACL",
                        title="UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation",
                        url="https://papers.cool/venue/2025.acl-long.24@ACL",
                        source_branch="venue",
                        external_url="https://aclanthology.org/2025.acl-long.24/",
                        pdf_url="https://aclanthology.org/2025.acl-long.24.pdf",
                        authors=["Author B"],
                        subject_or_venue="ACL.2025 - Long Papers",
                        published_at="",
                        snippet="ICL compression in venue paper.",
                        keywords=["icl", "compression", "selection"],
                        pdf_stars=30,
                        kimi_stars=30,
                    )
                ]

        if query == "kv cache acceleration" and branch == "arxiv":
            return [
                PapersCoolRecord(
                    paper_id="2412.19442",
                    title="A Survey on Large Language Model Acceleration based on KV Cache Management",
                    url="https://papers.cool/arxiv/2412.19442",
                    source_branch="arxiv",
                    external_url="https://arxiv.org/abs/2412.19442",
                    pdf_url="https://arxiv.org/pdf/2412.19442",
                    authors=["Author C"],
                    subject_or_venue="Computation and Language",
                    published_at="2024-12-15 03:21:00 UTC",
                    snippet="KV cache acceleration survey.",
                    keywords=["kv", "cache", "acceleration"],
                    pdf_stars=24,
                    kimi_stars=27,
                )
            ]
        return []


def test_topic_search_normalizes_and_deduplicates_results():
    workflow = PapersCoolTopicSearchWorkflow(connector=_FakeConnector())

    result = workflow.run(
        queries=["ICL压缩", " kv cache加速 ", "ICL 压缩"],
        branches=["arxiv", "venue"],
        top_k_per_query=3,
    )

    assert result["source"] == "papers.cool"
    assert len(result["queries"]) == 2
    assert result["queries"][0]["normalized_query"] == "icl compression"
    assert result["queries"][1]["normalized_query"] == "kv cache acceleration"

    # UniICL should be merged across arxiv+venue via title fallback dedupe.
    assert len(result["items"]) == 2
    uniicl_item = next(it for it in result["items"] if it["title"].startswith("UniICL"))
    assert sorted(uniicl_item["branches"]) == ["arxiv", "venue"]
    assert "https://papers.cool/venue/2025.acl-long.24@ACL" in uniicl_item["alternative_urls"]

    query_items = {row["normalized_query"]: row["items"] for row in result["queries"]}
    assert len(query_items["icl compression"]) == 1
    assert query_items["icl compression"][0]["matched_keywords"] == ["compression", "icl"]
    assert len(query_items["kv cache acceleration"]) == 1
    assert query_items["kv cache acceleration"][0]["score"] > 0

    assert result["summary"]["unique_items"] == 2
    assert result["summary"]["total_query_hits"] == 2
    assert len(result["summary"]["top_titles"]) == 2
    highlight = {h["normalized_query"]: h for h in result["summary"]["query_highlights"]}
    assert highlight["icl compression"]["top_title"].startswith("UniICL")
