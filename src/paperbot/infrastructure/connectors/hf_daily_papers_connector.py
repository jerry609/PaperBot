from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import time
from typing import Any, Dict, List, Optional, Tuple

import requests


_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_TRENDING_MODES = {"hot", "rising", "new"}


@dataclass
class HFDailyPaperRecord:
    paper_id: str
    title: str
    summary: str
    published_at: str
    submitted_on_daily_at: str
    authors: List[str]
    ai_keywords: List[str]
    upvotes: int
    paper_url: str
    external_url: str
    pdf_url: str


@dataclass
class _CacheEntry:
    payload: List[Dict[str, Any]]
    expires_at_epoch_s: float


class HFDailyPapersConnector:
    """Connector for Hugging Face Daily Papers API."""

    def __init__(
        self,
        *,
        base_url: str = "https://huggingface.co",
        timeout_s: float = 20.0,
        cache_ttl_s: float = 300.0,
        cache_max_entries: int = 32,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._headers = {"User-Agent": "PaperBot/2.0"}
        self._cache_ttl_s = max(float(cache_ttl_s or 0), 0.0)
        self._cache_max_entries = max(1, int(cache_max_entries or 1))
        self._cache: Dict[str, _CacheEntry] = {}
        self._metrics: Dict[str, int] = {
            "requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "degraded": 0,
        }

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def fetch_daily_papers(
        self, *, limit: int = 100, page: int = 0, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch one page of daily papers.

        Hugging Face currently caps `limit` at 100.
        """
        safe_limit = max(1, min(int(limit), 100))
        safe_page = max(int(page), 0)
        cache_key = f"daily:{safe_limit}:{safe_page}"
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        params = {"limit": safe_limit, "p": safe_page}
        url = f"{self.base_url}/api/daily_papers"
        self._metrics["requests"] += 1
        try:
            resp = requests.get(url, params=params, headers=self._headers, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception:
            self._metrics["errors"] += 1
            raise

        payload = resp.json()
        if not isinstance(payload, list):
            return []
        if use_cache:
            self._cache_set(cache_key, payload)
        return payload

    def get_daily(
        self,
        *,
        limit: int = 100,
        page_size: int = 100,
        max_pages: int = 5,
    ) -> List[HFDailyPaperRecord]:
        """Fetch parsed daily papers with graceful degradation on transient errors."""
        target_count = max(1, int(limit))
        safe_page_size = max(1, min(int(page_size), 100))
        safe_max_pages = max(1, int(max_pages))
        rows: List[HFDailyPaperRecord] = []
        seen: set[str] = set()

        for page in range(safe_max_pages):
            try:
                payload = self.fetch_daily_papers(limit=safe_page_size, page=page)
            except Exception:
                self._metrics["degraded"] += 1
                break
            if not payload:
                break

            for raw in payload:
                record = self._parse_record(raw)
                if record is None:
                    continue
                dedupe = (record.paper_id or "").strip().lower()
                if dedupe and dedupe in seen:
                    continue
                if dedupe:
                    seen.add(dedupe)
                rows.append(record)
                if len(rows) >= target_count:
                    return rows[:target_count]
        return rows[:target_count]

    def get_trending(
        self,
        *,
        mode: str = "hot",
        limit: int = 30,
        page_size: int = 100,
        max_pages: int = 3,
    ) -> List[HFDailyPaperRecord]:
        """Return HF daily papers in hot/rising/new ordering."""
        safe_limit = max(1, int(limit))
        mode_key = _normalize_trending_mode(mode)
        pool_limit = max(100, safe_limit)
        pool = self.get_daily(
            limit=pool_limit,
            page_size=max(1, min(int(page_size), 100)),
            max_pages=max_pages,
        )
        ranked = _rank_records(pool, mode=mode_key)
        return ranked[:safe_limit]

    def search(
        self,
        *,
        query: str,
        max_results: int = 25,
        page_size: int = 100,
        max_pages: int = 5,
        sort: str = "hot",
    ) -> List[HFDailyPaperRecord]:
        tokens = _tokenize_query(query)
        if not tokens:
            return []

        max_pages = max(1, int(max_pages))
        candidates: List[tuple[float, HFDailyPaperRecord]] = []
        seen: set[str] = set()

        for page in range(max_pages):
            try:
                rows = self.fetch_daily_papers(limit=page_size, page=page)
            except Exception:
                self._metrics["degraded"] += 1
                break
            if not rows:
                break

            for raw in rows:
                record = self._parse_record(raw)
                if record is None:
                    continue

                dedupe_key = f"{record.paper_id}|{record.title.lower()}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                text = " ".join(
                    [record.title, record.summary, " ".join(record.ai_keywords)]
                ).lower()
                hit_count = sum(1 for token in tokens if token and token in text)
                if hit_count <= 0:
                    continue

                score = float(hit_count) * 10.0 + float(record.upvotes) * 0.2
                candidates.append((score, record))

            # Early stop when we already have enough matched papers.
            if len(candidates) >= max_results:
                break

        mode = _normalize_trending_mode(sort)
        candidates.sort(
            key=lambda pair: _candidate_sort_key(pair, mode=mode),
            reverse=True,
        )
        return [record for _, record in candidates[: max(int(max_results), 0)]]

    def _cache_get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if self._cache_ttl_s <= 0:
            return None
        row = self._cache.get(key)
        if row is None:
            return None
        if row.expires_at_epoch_s <= time():
            self._cache.pop(key, None)
            return None
        self._metrics["cache_hits"] += 1
        return copy.deepcopy(row.payload)

    def _cache_set(self, key: str, payload: List[Dict[str, Any]]) -> None:
        if self._cache_ttl_s <= 0:
            return
        if len(self._cache) >= self._cache_max_entries:
            oldest_key = next(iter(self._cache.keys()))
            self._cache.pop(oldest_key, None)
        self._cache[key] = _CacheEntry(
            payload=copy.deepcopy(payload),
            expires_at_epoch_s=time() + self._cache_ttl_s,
        )

    def _parse_record(self, raw: Dict[str, Any]) -> Optional[HFDailyPaperRecord]:
        if not isinstance(raw, dict):
            return None

        paper_obj = raw.get("paper")
        if not isinstance(paper_obj, dict):
            paper_obj = raw

        paper_id = str(paper_obj.get("id") or "").strip()
        title = str(paper_obj.get("title") or raw.get("title") or "").strip()
        if not paper_id or not title:
            return None

        summary = str(paper_obj.get("summary") or raw.get("summary") or "").strip()
        published_at = str(paper_obj.get("publishedAt") or raw.get("publishedAt") or "").strip()
        submitted_on_daily_at = str(paper_obj.get("submittedOnDailyAt") or "").strip()
        upvotes = int(paper_obj.get("upvotes") or 0)

        authors: List[str] = []
        for author in paper_obj.get("authors") or []:
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
            else:
                name = str(author or "").strip()
            if name:
                authors.append(name)

        ai_keywords = [
            str(keyword).strip()
            for keyword in (paper_obj.get("ai_keywords") or [])
            if str(keyword).strip()
        ]

        paper_url = f"{self.base_url}/papers/{paper_id}"
        external_url, pdf_url = _build_external_links(paper_id)

        return HFDailyPaperRecord(
            paper_id=paper_id,
            title=title,
            summary=summary,
            published_at=published_at,
            submitted_on_daily_at=submitted_on_daily_at,
            authors=authors,
            ai_keywords=ai_keywords,
            upvotes=upvotes,
            paper_url=paper_url,
            external_url=external_url,
            pdf_url=pdf_url,
        )


def _build_external_links(paper_id: str) -> tuple[str, str]:
    compact_id = (paper_id or "").strip()
    match = _ARXIV_ID_RE.match(compact_id)
    if not match:
        return "", ""

    normalized = match.group(1) + (match.group(2) or "")
    return (
        f"https://arxiv.org/abs/{normalized}",
        f"https://arxiv.org/pdf/{normalized}.pdf",
    )


def _tokenize_query(query: str) -> List[str]:
    seen = set()
    tokens: List[str] = []
    for token in re.findall(r"[a-z0-9]+", (query or "").lower()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _date_sort_key(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _record_date(record: HFDailyPaperRecord) -> datetime:
    return _date_sort_key(record.submitted_on_daily_at or record.published_at)


def _rising_score(record: HFDailyPaperRecord) -> float:
    now = datetime.now(timezone.utc)
    age_hours = max((now - _record_date(record)).total_seconds() / 3600.0, 1.0)
    return float(record.upvotes) / (age_hours ** 0.5)


def _candidate_sort_key(
    candidate: Tuple[float, HFDailyPaperRecord],
    *,
    mode: str,
) -> Tuple[float, datetime, int]:
    score, record = candidate
    date_key = _record_date(record)
    if mode == "new":
        return (date_key.timestamp(), date_key, record.upvotes)
    if mode == "rising":
        return (score + _rising_score(record), date_key, record.upvotes)
    return (score + record.upvotes, date_key, record.upvotes)


def _normalize_trending_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    if key in _TRENDING_MODES:
        return key
    return "hot"


def _rank_records(records: List[HFDailyPaperRecord], *, mode: str) -> List[HFDailyPaperRecord]:
    mode_key = _normalize_trending_mode(mode)
    if mode_key == "new":
        return sorted(records, key=lambda r: (_record_date(r), r.upvotes), reverse=True)
    if mode_key == "rising":
        return sorted(records, key=lambda r: (_rising_score(r), _record_date(r)), reverse=True)
    return sorted(records, key=lambda r: (r.upvotes, _record_date(r)), reverse=True)
