from __future__ import annotations

from typing import Any, Dict, List
from bs4 import BeautifulSoup
import re


def _complete_ndss_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"https://www.ndss-symposium.org{url}"
    return f"https://www.ndss-symposium.org/{url}"


def parse_ndss_html(html: str, year: str) -> List[Dict[str, Any]]:
    """
    Parse NDSS HTML snapshot.

    Contract:
    - return non-empty list when HTML contains links to PDF
    - each item contains title + url + conference/year
    """
    soup = BeautifulSoup(html, "html.parser")
    papers: List[Dict[str, Any]] = []

    # Look for entries that contain PDF links.
    entries = soup.find_all(["article", "div"], class_=re.compile(r"(paper|entry|node)", re.I))
    if not entries:
        entries = soup.find_all(["li", "div"])

    for node in entries:
        pdf = node.find("a", href=re.compile(r"\.pdf($|\?)", re.I))
        if not pdf:
            continue
        href = pdf.get("href") or ""

        title_elem = node.find(["h2", "h3", "a"], class_=re.compile(r"(title|paper-title)", re.I))
        title = title_elem.get_text(strip=True) if title_elem else ""
        if not title:
            # fallback: use link text
            title = pdf.get_text(strip=True) or "NDSS Paper"

        papers.append(
            {
                "title": title,
                "url": _complete_ndss_url(href),
                "conference": "NDSS",
                "year": year,
            }
        )

    return papers


