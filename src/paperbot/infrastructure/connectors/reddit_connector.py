from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from paperbot.application.collaboration.message_schema import make_event
from paperbot.application.ports.event_log_port import EventLogPort


@dataclass
class RedditRecord:
    title: str
    link: str
    guid: str
    published: str
    subreddit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            "guid": self.guid,
            "published": self.published,
            "subreddit": self.subreddit,
        }


class RedditConnector:
    """
    Minimal Reddit RSS connector (no OAuth required).
    """

    def parse_rss(self, xml_text: str, *, subreddit: str) -> List[RedditRecord]:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return []

        records: List[RedditRecord] = []
        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            guid = (item.findtext("guid") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if not (title and (link or guid)):
                continue
            records.append(
                RedditRecord(
                    title=title,
                    link=link,
                    guid=guid or link,
                    published=pub,
                    subreddit=subreddit,
                )
            )
        return records

    def emit_events(
        self,
        records: List[RedditRecord],
        *,
        event_log: EventLogPort,
        run_id: str,
        trace_id: Optional[str] = None,
    ) -> None:
        trace_id = trace_id or run_id
        for r in records:
            event_log.append(
                make_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    workflow="feeds",
                    stage="reddit_ingest",
                    attempt=0,
                    agent_name="RedditConnector",
                    role="system",
                    type="source_record",
                    payload={"source": "reddit", "record": r.to_dict()},
                    tags={"source": "reddit", "guid": r.guid, "subreddit": r.subreddit},
                )
            )


