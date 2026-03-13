"""Tests for PaperDeduplicator -- three-tier dedup (DOI / arxiv_id / rapidfuzz title)."""

from __future__ import annotations

from typing import List, Optional

from paperbot.application.services.paper_dedup import PaperDeduplicator, normalize_title
from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper import PaperCandidate


def _make_paper(
    title: str,
    *,
    abstract: str = "",
    authors: Optional[List[str]] = None,
    citation_count: int = 0,
    identities: Optional[List[PaperIdentity]] = None,
    year: Optional[int] = None,
    venue: Optional[str] = None,
    url: Optional[str] = None,
    pdf_url: Optional[str] = None,
    publication_date: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> PaperCandidate:
    return PaperCandidate(
        title=title,
        abstract=abstract,
        authors=list(authors or []),
        citation_count=citation_count,
        identities=list(identities or []),
        year=year,
        venue=venue,
        url=url,
        pdf_url=pdf_url,
        publication_date=publication_date,
        keywords=list(keywords or []),
        fields_of_study=list(fields_of_study or []),
    )


class TestNormalizeTitle:
    def test_lowercases_and_strips_punctuation(self) -> None:
        assert normalize_title("Hello, World!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert normalize_title("  foo   bar  ") == "foo bar"

    def test_empty_string(self) -> None:
        assert normalize_title("") == ""


class TestDOIExactMatchDedup:
    def test_same_doi_from_two_sources_merges(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Attention Is All You Need",
            citation_count=50000,
            identities=[PaperIdentity(source="doi", external_id="10.5555/attention")],
        )
        p2 = _make_paper(
            "Attention Is All You Need",
            citation_count=60000,
            abstract="A longer abstract from the second source.",
            identities=[PaperIdentity(source="doi", external_id="10.5555/attention")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is False

        results = dedup.results()
        assert len(results) == 1
        assert results[0].citation_count == 60000
        assert results[0].abstract == "A longer abstract from the second source."

    def test_doi_case_insensitive(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Paper A",
            identities=[PaperIdentity(source="doi", external_id="10.1234/ABC")],
        )
        p2 = _make_paper(
            "Paper A",
            identities=[PaperIdentity(source="doi", external_id="10.1234/abc")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is False
        assert len(dedup.results()) == 1


class TestArxivIdMatchDedup:
    def test_same_arxiv_id_merges(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Diffusion Models Beat GANs",
            citation_count=100,
            identities=[PaperIdentity(source="arxiv", external_id="2301.12345")],
        )
        p2 = _make_paper(
            "Diffusion Models Beat GANs",
            citation_count=200,
            identities=[PaperIdentity(source="arxiv", external_id="2301.12345")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is False

        results = dedup.results()
        assert len(results) == 1
        assert results[0].citation_count == 200

    def test_arxiv_version_stripping(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Some Paper",
            identities=[PaperIdentity(source="arxiv", external_id="2301.12345v1")],
        )
        p2 = _make_paper(
            "Some Paper",
            identities=[PaperIdentity(source="arxiv", external_id="2301.12345v3")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is False
        assert len(dedup.results()) == 1


class TestFuzzyTitleMatchDedup:
    def test_minor_casing_difference_merges(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper("Attention Is All You Need", citation_count=100)
        p2 = _make_paper("Attention is All you Need", citation_count=200)

        assert dedup.add(p1) is True
        assert dedup.add(p2) is False

        results = dedup.results()
        assert len(results) == 1
        assert results[0].citation_count == 200

    def test_punctuation_difference_merges(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper("BERT: Pre-training of Deep Bidirectional Transformers")
        p2 = _make_paper("BERT Pre-training of Deep Bidirectional Transformers")

        assert dedup.add(p1) is True
        assert dedup.add(p2) is False
        assert len(dedup.results()) == 1


class TestDifferentPapersNotMerged:
    def test_genuinely_different_titles_both_kept(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper("Attention Is All You Need")
        p2 = _make_paper("ImageNet Classification with Deep Convolutional Neural Networks")

        assert dedup.add(p1) is True
        assert dedup.add(p2) is True
        assert len(dedup.results()) == 2

    def test_same_title_with_conflicting_dois_stays_separate(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Same Title",
            identities=[PaperIdentity(source="doi", external_id="10.1234/foo")],
        )
        p2 = _make_paper(
            "Same Title",
            identities=[PaperIdentity(source="doi", external_id="10.1234/bar")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is True
        assert len(dedup.results()) == 2

    def test_same_title_with_conflicting_arxiv_ids_stays_separate(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Same Title",
            identities=[PaperIdentity(source="arxiv", external_id="2501.12345")],
        )
        p2 = _make_paper(
            "Same Title",
            identities=[PaperIdentity(source="arxiv", external_id="2501.67890")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is True
        assert len(dedup.results()) == 2

    def test_different_dois_different_titles_both_kept(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Paper Alpha",
            identities=[PaperIdentity(source="doi", external_id="10.1234/alpha")],
        )
        p2 = _make_paper(
            "Paper Beta",
            identities=[PaperIdentity(source="doi", external_id="10.1234/beta")],
        )

        assert dedup.add(p1) is True
        assert dedup.add(p2) is True
        assert len(dedup.results()) == 2


class TestMergePicksBestMetadata:
    def test_merged_result_keeps_highest_citations_and_longest_abstract(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Attention Is All You Need",
            abstract="Short.",
            citation_count=100,
            year=2017,
            identities=[
                PaperIdentity(source="doi", external_id="10.5555/attention"),
                PaperIdentity(source="semantic_scholar", external_id="s2-123"),
            ],
        )
        p2 = _make_paper(
            "Attention Is All You Need",
            abstract="A much longer and more detailed abstract describing the paper.",
            citation_count=50000,
            identities=[
                PaperIdentity(source="doi", external_id="10.5555/attention"),
                PaperIdentity(source="arxiv", external_id="1706.03762"),
            ],
        )

        dedup.add(p1)
        dedup.add(p2)

        results = dedup.results()
        assert len(results) == 1
        merged = results[0]

        # Best citation count
        assert merged.citation_count == 50000
        # Longest abstract
        assert "much longer" in merged.abstract
        # Merged identities: doi + semantic_scholar + arxiv (doi not doubled)
        sources = {i.source for i in merged.identities}
        assert sources == {"doi", "semantic_scholar", "arxiv"}
        assert len(merged.identities) == 3
        # Year from first paper preserved
        assert merged.year == 2017

    def test_merge_fills_missing_year_and_venue(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Some Paper",
            identities=[PaperIdentity(source="doi", external_id="10.1234/x")],
        )
        p2 = _make_paper(
            "Some Paper",
            year=2023,
            venue="NeurIPS",
            identities=[PaperIdentity(source="doi", external_id="10.1234/x")],
        )

        dedup.add(p1)
        dedup.add(p2)

        merged = dedup.results()[0]
        assert merged.year == 2023
        assert merged.venue == "NeurIPS"

    def test_merge_preserves_richer_metadata_when_second_copy_is_more_complete(self) -> None:
        dedup = PaperDeduplicator()
        p1 = _make_paper(
            "Unified Paper",
            abstract="Short abstract.",
            identities=[PaperIdentity(source="doi", external_id="10.1234/unified")],
        )
        p2 = _make_paper(
            "Unified Paper",
            abstract="A much longer abstract for the same paper.",
            authors=["Alice", "Bob"],
            identities=[PaperIdentity(source="doi", external_id="10.1234/unified")],
            url="https://example.com/paper",
            pdf_url="https://example.com/paper.pdf",
            publication_date="2026-03-01",
            keywords=["retrieval", "agents"],
            fields_of_study=["NLP"],
        )

        dedup.add(p1)
        dedup.add(p2)

        merged = dedup.results()[0]
        assert merged.authors == ["Alice", "Bob"]
        assert merged.url == "https://example.com/paper"
        assert merged.pdf_url == "https://example.com/paper.pdf"
        assert merged.publication_date == "2026-03-01"
        assert merged.keywords == ["retrieval", "agents"]
        assert merged.fields_of_study == ["NLP"]
        assert "much longer" in merged.abstract

    def test_results_include_identity_free_empty_title_papers(self) -> None:
        dedup = PaperDeduplicator()
        paper = _make_paper("")

        assert dedup.add(paper) is True
        assert dedup.results() == [paper]


class TestThresholdEdgeCases:
    def test_custom_threshold_high_rejects_partial_match(self) -> None:
        dedup = PaperDeduplicator(title_threshold=0.99)
        p1 = _make_paper("Attention Is All You Need")
        p2 = _make_paper("Attention Is Almost All You Need")

        dedup.add(p1)
        dedup.add(p2)

        # With a very high threshold, these should NOT be merged
        assert len(dedup.results()) == 2

    def test_custom_threshold_low_merges_partial_match(self) -> None:
        dedup = PaperDeduplicator(title_threshold=0.5)
        p1 = _make_paper("Attention Is All You Need")
        p2 = _make_paper("Attention Is Almost All You Need")

        dedup.add(p1)
        dedup.add(p2)

        # With a low threshold, these should be merged
        assert len(dedup.results()) == 1
