from __future__ import annotations

from typing import Any, Dict, List
from bs4 import BeautifulSoup
import re


def _complete_usenix_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"https://www.usenix.org{url}"
    return f"https://www.usenix.org/{url}"


def parse_usenix_security_html(html: str, year: str) -> List[Dict[str, Any]]:
    """
    Parse USENIX Security technical sessions HTML.

    Contract:
    - return non-empty list when HTML contains paper nodes
    - each item contains title + url (pdf or presentation link) + conference/year
    """
    soup = BeautifulSoup(html, "html.parser")
    papers: List[Dict[str, Any]] = []

    paper_nodes = soup.find_all(["article", "div"], class_=["node-paper", "paper-item"])
    if not paper_nodes:
        paper_nodes = soup.find_all(["div", "article"], class_=["paper", "technical-paper"])

    for node in paper_nodes:
        title_elem = (
            node.find(["h2", "h3"], class_=["node-title", "paper-title"])
            or node.find("div", class_="field-title")
        )
        if not title_elem:
            continue
        title = title_elem.get_text(strip=True)

        pdf_link = node.find("a", href=re.compile(r"\.pdf($|\?)", re.I))
        href = pdf_link.get("href") if pdf_link else None
        if not href:
            pres_link = node.find("a", href=re.compile(r"/presentation/", re.I))
            href = pres_link.get("href") if pres_link else None
        if not href:
            continue

        papers.append(
            {
                "title": title,
                "url": _complete_usenix_url(href),
                "conference": "USENIX",
                "year": year,
            }
        )

    return papers


