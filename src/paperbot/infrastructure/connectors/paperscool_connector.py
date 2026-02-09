from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup, Tag

from paperbot.application.collaboration.message_schema import make_event
from paperbot.application.ports.event_log_port import EventLogPort


@dataclass
class PapersCoolRecord:
    paper_id: str
    title: str
    url: str
    source_branch: str
    external_url: str
    pdf_url: str
    authors: List[str]
    subject_or_venue: str
    published_at: str
    snippet: str
    keywords: List[str]
    pdf_stars: int
    kimi_stars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "url": self.url,
            "source_branch": self.source_branch,
            "external_url": self.external_url,
            "pdf_url": self.pdf_url,
            "authors": self.authors,
            "subject_or_venue": self.subject_or_venue,
            "published_at": self.published_at,
            "snippet": self.snippet,
            "keywords": self.keywords,
            "pdf_stars": self.pdf_stars,
            "kimi_stars": self.kimi_stars,
        }


class PapersCoolConnector:
    """Minimal papers.cool connector for branch search pages."""

    def __init__(self, *, base_url: str = "https://papers.cool", timeout_s: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._headers = {"User-Agent": "PaperBot/2.0"}

    def build_search_url(
        self,
        *,
        branch: str,
        query: str,
        highlight: bool = True,
        show: Optional[int] = None,
    ) -> str:
        normalized_branch = (branch or "").strip().lower()
        if normalized_branch not in {"arxiv", "venue"}:
            raise ValueError(f"Unsupported branch: {branch}")

        params: Dict[str, Any] = {"query": query.strip()}
        if highlight:
            params["highlight"] = "1"
        if show is not None:
            params["show"] = int(show)

        return f"{self.base_url}/{normalized_branch}/search?{urlencode(params)}"

    def fetch_search_html(
        self,
        *,
        branch: str,
        query: str,
        highlight: bool = True,
        show: Optional[int] = None,
    ) -> str:
        url = self.build_search_url(branch=branch, query=query, highlight=highlight, show=show)
        resp = requests.get(url, headers=self._headers, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.text

    def search(
        self,
        *,
        branch: str,
        query: str,
        highlight: bool = True,
        show: Optional[int] = None,
    ) -> List[PapersCoolRecord]:
        html = self.fetch_search_html(branch=branch, query=query, highlight=highlight, show=show)
        return self.parse_search_html(html, branch=branch)

    def parse_search_html(self, html: str, *, branch: str) -> List[PapersCoolRecord]:
        normalized_branch = (branch or "").strip().lower()
        if normalized_branch not in {"arxiv", "venue"}:
            raise ValueError(f"Unsupported branch: {branch}")

        soup = BeautifulSoup(html, "html.parser")
        records: List[PapersCoolRecord] = []
        for card in soup.select(".paper"):
            record = self._parse_card(card, branch=normalized_branch)
            if record:
                records.append(record)
        return records

    def emit_events(
        self,
        records: List[PapersCoolRecord],
        *,
        event_log: EventLogPort,
        run_id: str,
        trace_id: Optional[str] = None,
    ) -> None:
        trace_id = trace_id or run_id
        for record in records:
            event_log.append(
                make_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    workflow="feeds",
                    stage="paperscool_ingest",
                    attempt=0,
                    agent_name="PapersCoolConnector",
                    role="system",
                    type="source_record",
                    payload={"source": "papers_cool", "record": record.to_dict()},
                    tags={
                        "source": "papers_cool",
                        "branch": record.source_branch,
                        "paper_id": record.paper_id,
                    },
                )
            )

    def _parse_card(self, card: Tag, *, branch: str) -> Optional[PapersCoolRecord]:
        paper_id = (card.get("id") or "").strip()
        title_link = card.select_one("a.title-link")
        if not paper_id or title_link is None:
            return None

        title = title_link.get_text(" ", strip=True)
        if not title:
            return None

        title_href = title_link.get("href") or ""
        url = urljoin(self.base_url + "/", title_href)

        external_url = self._extract_external_url(card)
        pdf_link = card.select_one("a.title-pdf")
        pdf_url = ""
        if pdf_link is not None:
            pdf_url = (pdf_link.get("data") or pdf_link.get("href") or "").strip()
            if pdf_url:
                pdf_url = urljoin(self.base_url + "/", pdf_url)

        authors = [
            anchor.get_text(" ", strip=True)
            for anchor in card.select("p.authors a.author")
            if anchor.get_text(" ", strip=True)
        ]

        subject_or_venue = self._extract_meta_text(card, "subjects")
        published_at = self._extract_meta_text(card, "date")
        summary = card.select_one(".summary")
        snippet = summary.get_text(" ", strip=True) if summary is not None else ""

        keywords = [kw.strip() for kw in (card.get("keywords") or "").split(",") if kw.strip()]
        pdf_stars = self._extract_sup_number(card.select_one("a.title-pdf sup"))
        kimi_stars = self._extract_sup_number(card.select_one("a.title-kimi sup"))

        return PapersCoolRecord(
            paper_id=paper_id,
            title=title,
            url=url,
            source_branch=branch,
            external_url=external_url,
            pdf_url=pdf_url,
            authors=authors,
            subject_or_venue=subject_or_venue,
            published_at=published_at,
            snippet=snippet,
            keywords=keywords,
            pdf_stars=pdf_stars,
            kimi_stars=kimi_stars,
        )

    def _extract_external_url(self, card: Tag) -> str:
        title_node = card.select_one("h2.title")
        if title_node is None:
            return ""
        for anchor in title_node.select("a[href]"):
            classes = set(anchor.get("class") or [])
            if "title-link" in classes:
                continue
            href = (anchor.get("href") or "").strip()
            if not href:
                continue
            return urljoin(self.base_url + "/", href)
        return ""

    def _extract_meta_text(self, card: Tag, class_name: str) -> str:
        node = card.select_one(f"p.metainfo.{class_name}")
        if node is None:
            return ""
        text = node.get_text(" ", strip=True)
        return re.sub(r"^[^:：]+[:：]\s*", "", text)

    def _extract_sup_number(self, node: Optional[Tag]) -> int:
        if node is None:
            return 0
        text = node.get_text(" ", strip=True)
        if not text:
            return 0
        match = re.search(r"\d+", text)
        if not match:
            return 0
        return int(match.group())
