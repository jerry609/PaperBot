from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.crawling.parsers.usenix import parse_usenix_security_html
from paperbot.infrastructure.crawling.parsers.ndss import parse_ndss_html


def test_usenix_parser_contract_fixture():
    html = Path("evals/fixtures/crawler_parsing/usenix_security_2023.html").read_text(encoding="utf-8")
    papers = parse_usenix_security_html(html, year="23")
    assert len(papers) >= 1
    for p in papers:
        assert p.get("title")
        assert p.get("url")
        assert p["conference"] == "USENIX"


def test_ndss_parser_contract_fixture():
    html = Path("evals/fixtures/crawler_parsing/ndss_2023.html").read_text(encoding="utf-8")
    papers = parse_ndss_html(html, year="23")
    assert len(papers) >= 1
    for p in papers:
        assert p.get("title")
        assert p.get("url")
        assert p["conference"] == "NDSS"


