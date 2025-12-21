from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from paperbot.application.collaboration.message_schema import make_event
from paperbot.application.ports.event_log_port import EventLogPort


@dataclass
class XRecord:
    post_id: str
    text: str
    author: str = ""
    created_at: str = ""
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "text": self.text,
            "author": self.author,
            "created_at": self.created_at,
            "url": self.url,
        }


class XImporter:
    """
    Best-effort import-only connector for Twitter/X.

    Does NOT crawl. Accepts external exports (JSONL/CSV) and emits normalized events.
    """

    def parse_jsonl(self, lines: Iterable[str]) -> List[XRecord]:
        records: List[XRecord] = []
        for line in lines:
            line = (line or "").strip()
            if not line:
                continue
            obj = json.loads(line)
            post_id = str(obj.get("id") or obj.get("post_id") or obj.get("tweet_id") or "")
            text = str(obj.get("text") or obj.get("content") or "")
            if not (post_id and text):
                continue
            records.append(
                XRecord(
                    post_id=post_id,
                    text=text,
                    author=str(obj.get("author") or obj.get("user") or obj.get("username") or ""),
                    created_at=str(obj.get("created_at") or obj.get("time") or ""),
                    url=str(obj.get("url") or ""),
                )
            )
        return records

    def parse_csv(self, csv_text: str) -> List[XRecord]:
        reader = csv.DictReader(csv_text.splitlines())
        records: List[XRecord] = []
        for row in reader:
            post_id = str(row.get("id") or row.get("post_id") or row.get("tweet_id") or "")
            text = str(row.get("text") or row.get("content") or "")
            if not (post_id and text):
                continue
            records.append(
                XRecord(
                    post_id=post_id,
                    text=text,
                    author=str(row.get("author") or row.get("username") or ""),
                    created_at=str(row.get("created_at") or ""),
                    url=str(row.get("url") or ""),
                )
            )
        return records

    def emit_events(
        self,
        records: List[XRecord],
        *,
        event_log: EventLogPort,
        run_id: str,
        trace_id: Optional[str] = None,
        source: str = "twitter_x",
    ) -> None:
        trace_id = trace_id or run_id
        for r in records:
            event_log.append(
                make_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    workflow="feeds",
                    stage="x_import",
                    attempt=0,
                    agent_name="XImporter",
                    role="system",
                    type="source_record",
                    payload={"source": source, "record": r.to_dict()},
                    tags={"source": source, "post_id": r.post_id},
                )
            )


