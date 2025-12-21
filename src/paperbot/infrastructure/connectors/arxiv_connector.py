from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from paperbot.application.collaboration.message_schema import make_event
from paperbot.application.ports.event_log_port import EventLogPort


ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArxivRecord:
    arxiv_id: str
    title: str
    summary: str
    published: str
    updated: str
    authors: List[str]
    abs_url: str
    pdf_url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "summary": self.summary,
            "published": self.published,
            "updated": self.updated,
            "authors": self.authors,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
        }


class ArxivConnector:
    """
    Minimal arXiv Atom feed connector.

    Phase: fixtures-first for offline E2E/IT.
    """

    def parse_atom(self, xml_text: str) -> List[ArxivRecord]:
        root = ET.fromstring(xml_text)
        records: List[ArxivRecord] = []

        for entry in root.findall("atom:entry", ARXIV_NS):
            arxiv_id = (entry.findtext("atom:id", default="", namespaces=ARXIV_NS) or "").strip()
            title = (entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ARXIV_NS) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ARXIV_NS) or "").strip()
            updated = (entry.findtext("atom:updated", default="", namespaces=ARXIV_NS) or "").strip()

            authors = []
            for a in entry.findall("atom:author/atom:name", ARXIV_NS):
                if a.text:
                    authors.append(a.text.strip())

            abs_url = ""
            pdf_url = ""
            for link in entry.findall("atom:link", ARXIV_NS):
                rel = (link.attrib.get("rel") or "").lower()
                typ = (link.attrib.get("type") or "").lower()
                href = link.attrib.get("href") or ""
                if rel == "alternate" and href:
                    abs_url = href
                if (typ == "application/pdf" or "pdf" in rel) and href:
                    pdf_url = href

            # Fallback: build PDF from id
            if abs_url and not pdf_url and abs_url.startswith("http"):
                pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

            if not (arxiv_id and title):
                continue

            records.append(
                ArxivRecord(
                    arxiv_id=arxiv_id,
                    title=title,
                    summary=summary,
                    published=published,
                    updated=updated,
                    authors=authors,
                    abs_url=abs_url,
                    pdf_url=pdf_url,
                )
            )

        return records

    def emit_events(
        self,
        records: List[ArxivRecord],
        *,
        event_log: EventLogPort,
        run_id: str,
        trace_id: Optional[str] = None,
    ) -> None:
        # One trace for the ingestion batch; record-level info goes into payload.
        trace_id = trace_id or run_id
        for r in records:
            event_log.append(
                make_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    workflow="feeds",
                    stage="arxiv_ingest",
                    attempt=0,
                    agent_name="ArxivConnector",
                    role="system",
                    type="source_record",
                    payload={"source": "arxiv", "record": r.to_dict()},
                    tags={"source": "arxiv", "arxiv_id": r.arxiv_id},
                )
            )


