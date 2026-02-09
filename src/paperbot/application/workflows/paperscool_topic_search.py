from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from paperbot.infrastructure.connectors.paperscool_connector import (
    PapersCoolConnector,
    PapersCoolRecord,
)


_QUERY_ALIASES = {
    "icl压缩": "icl compression",
    "icl 压缩": "icl compression",
    "icl 隐式偏置": "icl implicit bias",
    "icl隐式偏置": "icl implicit bias",
    "kv cache加速": "kv cache acceleration",
    "kv cache 加速": "kv cache acceleration",
}


@dataclass(frozen=True)
class QuerySpec:
    raw_query: str
    normalized_query: str
    tokens: List[str]


class PapersCoolTopicSearchWorkflow:
    def __init__(self, connector: Optional[PapersCoolConnector] = None):
        self.connector = connector or PapersCoolConnector()

    def normalize_queries(self, queries: Sequence[str]) -> List[QuerySpec]:
        specs: List[QuerySpec] = []
        seen = set()
        for raw in queries:
            raw_query = (raw or "").strip()
            if not raw_query:
                continue
            normalized = _normalize_query(raw_query)
            if normalized in seen:
                continue
            seen.add(normalized)
            specs.append(
                QuerySpec(
                    raw_query=raw_query,
                    normalized_query=normalized,
                    tokens=_tokenize_query(normalized),
                )
            )
        return specs

    def run(
        self,
        *,
        queries: Sequence[str],
        branches: Sequence[str] = ("arxiv", "venue"),
        top_k_per_query: int = 5,
        show_per_branch: int = 25,
    ) -> Dict[str, Any]:
        query_specs = self.normalize_queries(queries)
        if not query_specs:
            return {
                "source": "papers.cool",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "queries": [],
                "items": [],
            }

        aggregated_items: List[Dict[str, Any]] = []
        by_url: Dict[str, Dict[str, Any]] = {}
        by_title: Dict[str, Dict[str, Any]] = {}

        for spec in query_specs:
            for branch in branches:
                records = self.connector.search(
                    branch=branch,
                    query=spec.normalized_query,
                    highlight=True,
                    show=show_per_branch,
                )
                for record in records:
                    item = self._build_item(record=record, query_spec=spec)
                    current = self._find_existing_item(item, by_url=by_url, by_title=by_title)
                    if current is None:
                        aggregated_items.append(item)
                        self._index_item(item, by_url=by_url, by_title=by_title)
                    else:
                        self._merge_item(current, item)

        aggregated_items.sort(key=lambda it: it["score"], reverse=True)

        query_views: List[Dict[str, Any]] = []
        for spec in query_specs:
            matched = [
                self._serialize_item(item)
                for item in aggregated_items
                if spec.normalized_query in item["matched_queries"]
            ]
            query_views.append(
                {
                    "raw_query": spec.raw_query,
                    "normalized_query": spec.normalized_query,
                    "tokens": spec.tokens,
                    "total_hits": len(matched),
                    "items": matched[: max(int(top_k_per_query), 0)],
                }
            )

        return {
            "source": "papers.cool",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "queries": query_views,
            "items": [self._serialize_item(item) for item in aggregated_items],
        }

    def _build_item(self, *, record: PapersCoolRecord, query_spec: QuerySpec) -> Dict[str, Any]:
        matched_keywords = _matched_keywords(record=record, tokens=query_spec.tokens)
        score = _score_record(
            record=record, token_count=len(query_spec.tokens), matched=matched_keywords
        )
        return {
            "paper_id": record.paper_id,
            "title": record.title,
            "url": record.url,
            "external_url": record.external_url,
            "pdf_url": record.pdf_url,
            "authors": record.authors,
            "subject_or_venue": record.subject_or_venue,
            "published_at": record.published_at,
            "snippet": record.snippet,
            "keywords": record.keywords,
            "branches": [record.source_branch],
            "pdf_stars": record.pdf_stars,
            "kimi_stars": record.kimi_stars,
            "matched_keywords": matched_keywords,
            "matched_queries": [query_spec.normalized_query],
            "score": score,
            "alternative_urls": [],
        }

    def _find_existing_item(
        self,
        item: Dict[str, Any],
        *,
        by_url: Dict[str, Dict[str, Any]],
        by_title: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        url = (item.get("url") or "").strip()
        if url and url in by_url:
            return by_url[url]

        title_key = _normalize_title(item.get("title") or "")
        if title_key and title_key in by_title:
            return by_title[title_key]
        return None

    def _index_item(
        self,
        item: Dict[str, Any],
        *,
        by_url: Dict[str, Dict[str, Any]],
        by_title: Dict[str, Dict[str, Any]],
    ) -> None:
        url = (item.get("url") or "").strip()
        if url:
            by_url[url] = item
        title_key = _normalize_title(item.get("title") or "")
        if title_key:
            by_title[title_key] = item

    def _merge_item(self, target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        target["matched_keywords"] = sorted(
            set(target["matched_keywords"]).union(incoming["matched_keywords"])
        )
        target["matched_queries"] = sorted(
            set(target["matched_queries"]).union(incoming["matched_queries"])
        )
        target["branches"] = sorted(set(target["branches"]).union(incoming["branches"]))
        target["keywords"] = sorted(set(target["keywords"]).union(incoming["keywords"]))
        target["authors"] = sorted(set(target["authors"]).union(incoming["authors"]))

        incoming_url = (incoming.get("url") or "").strip()
        target_url = (target.get("url") or "").strip()
        if incoming_url and incoming_url != target_url:
            alt = set(target.get("alternative_urls") or [])
            alt.add(incoming_url)
            target["alternative_urls"] = sorted(alt)

        target["pdf_stars"] = max(int(target["pdf_stars"]), int(incoming["pdf_stars"]))
        target["kimi_stars"] = max(int(target["kimi_stars"]), int(incoming["kimi_stars"]))
        target["score"] = max(float(target["score"]), float(incoming["score"]))

    def _serialize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "paper_id": item["paper_id"],
            "title": item["title"],
            "url": item["url"],
            "external_url": item["external_url"],
            "pdf_url": item["pdf_url"],
            "authors": item["authors"],
            "subject_or_venue": item["subject_or_venue"],
            "published_at": item["published_at"],
            "snippet": item["snippet"],
            "keywords": item["keywords"],
            "branches": item["branches"],
            "matched_keywords": item["matched_keywords"],
            "matched_queries": item["matched_queries"],
            "score": round(float(item["score"]), 4),
            "pdf_stars": item["pdf_stars"],
            "kimi_stars": item["kimi_stars"],
            "alternative_urls": item["alternative_urls"],
        }


def _normalize_query(query: str) -> str:
    base = re.sub(r"\s+", " ", query.strip()).lower()
    if base in _QUERY_ALIASES:
        return _QUERY_ALIASES[base]
    return base


def _tokenize_query(query: str) -> List[str]:
    seen = set()
    tokens: List[str] = []
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _normalize_title(title: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", title.lower()))


def _matched_keywords(*, record: PapersCoolRecord, tokens: Iterable[str]) -> List[str]:
    haystack = " ".join([record.title, record.snippet, " ".join(record.keywords)]).lower()
    matched = []
    for token in tokens:
        if token and token in haystack:
            matched.append(token)
    return sorted(set(matched))


def _extract_year(record: PapersCoolRecord) -> Optional[int]:
    text = " ".join([record.published_at, record.subject_or_venue, record.paper_id])
    year_match = re.search(r"(20\d{2})", text)
    if not year_match:
        return None
    year = int(year_match.group(1))
    if year < 1990 or year > 2100:
        return None
    return year


def _score_record(*, record: PapersCoolRecord, token_count: int, matched: Sequence[str]) -> float:
    hit_count = len(matched)
    coverage = hit_count / max(token_count, 1)
    popularity = math.log1p(max(record.pdf_stars, 0) + max(record.kimi_stars, 0))

    freshness = 0.0
    year = _extract_year(record)
    if year is not None:
        freshness = max(0.0, min((year - 2018) * 0.15, 2.0))

    return (3.0 * hit_count) + (2.0 * coverage) + popularity + freshness
