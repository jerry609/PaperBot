"""RSS 2.0 and Atom feed endpoints for DailyPaper digests."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Query
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()

_REPORTS_DIR = Path("./reports/dailypaper")
_FEED_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_latest_reports(
    reports_dir: Path, *, limit: int = 20
) -> List[Dict[str, Any]]:
    """Load the most recent DailyPaper JSON reports from disk."""
    if not reports_dir.exists():
        return []

    json_files = sorted(reports_dir.glob("*.json"), reverse=True)[:limit]
    reports: List[Dict[str, Any]] = []
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                reports.append(data)
        except Exception:
            continue
    return reports


def _collect_papers_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect deduplicated papers from a single report."""
    seen: set = set()
    papers: List[Dict[str, Any]] = []
    for q in report.get("queries") or []:
        for item in q.get("top_items") or []:
            key = item.get("title") or id(item)
            if key not in seen:
                seen.add(key)
                papers.append(item)
    return papers


def _filter_reports_by_track(
    reports: List[Dict[str, Any]],
    *,
    track_name: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    needle = (track_name or "").strip().lower()
    if not needle:
        return filtered

    for report in reports:
        track_report = {
            **report,
            "queries": [
                q
                for q in (report.get("queries") or [])
                if needle in (q.get("normalized_query") or q.get("raw_query") or "").lower()
            ],
        }
        if track_report["queries"]:
            filtered.append(track_report)
    return filtered


def _paper_matches_keyword(item: Dict[str, Any], keyword: str) -> bool:
    needle = (keyword or "").strip().lower()
    if not needle:
        return False
    fields = [
        str(item.get("title") or ""),
        str(item.get("snippet") or item.get("abstract") or ""),
        str((item.get("judge") or {}).get("one_line_summary") or ""),
        str((item.get("digest_card") or {}).get("highlight") or ""),
        str((item.get("digest_card") or {}).get("method") or ""),
        str((item.get("digest_card") or {}).get("finding") or ""),
    ]
    tags = (item.get("digest_card") or {}).get("tags") or []
    fields.extend(str(tag) for tag in tags)
    text = " ".join(fields).lower()
    return needle in text


def _filter_reports_by_keyword(
    reports: List[Dict[str, Any]],
    *,
    keyword: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for report in reports:
        queries = []
        for query in report.get("queries") or []:
            top_items = [
                item
                for item in (query.get("top_items") or [])
                if _paper_matches_keyword(item, keyword)
            ]
            if top_items:
                queries.append({**query, "top_items": top_items})
        if queries:
            filtered.append({**report, "queries": queries})
    return filtered


def _reports_signature(reports_dir: Path) -> Tuple[int, int]:
    if not reports_dir.exists():
        return (0, 0)
    json_files = list(reports_dir.glob("*.json"))
    if not json_files:
        return (0, 0)
    latest_mtime_ns = max(int(p.stat().st_mtime_ns) for p in json_files)
    return (len(json_files), latest_mtime_ns)


def _cached_xml(cache_key: str, reports_dir: Path) -> str:
    signature = _reports_signature(reports_dir)
    row = _FEED_CACHE.get(cache_key)
    if row and row.get("signature") == signature:
        return str(row.get("xml") or "")
    return ""


def _store_cached_xml(cache_key: str, reports_dir: Path, xml: str) -> str:
    _FEED_CACHE[cache_key] = {"signature": _reports_signature(reports_dir), "xml": xml}
    return xml


def _build_rss_xml(
    reports: List[Dict[str, Any]],
    *,
    title: str = "PaperBot DailyPaper",
    description: str = "Daily curated research papers from PaperBot",
    link: str = "",
) -> str:
    """Build RSS 2.0 XML from DailyPaper reports."""
    try:
        from feedgen.feed import FeedGenerator

        fg = FeedGenerator()
        fg.title(title)
        fg.description(description)
        fg.link(href=link or "https://paperbot.local", rel="alternate")
        fg.language("en")
        fg.lastBuildDate(datetime.now(timezone.utc))

        for report in reports:
            date_str = report.get("date") or ""
            generated_at = report.get("generated_at") or ""

            papers = _collect_papers_from_report(report)
            for item in papers:
                paper_title = item.get("title") or "Untitled"
                url = item.get("url") or item.get("external_url") or ""
                snippet = item.get("snippet") or item.get("abstract") or ""

                judge = item.get("judge") or {}
                one_line = str(judge.get("one_line_summary") or "")
                rec = judge.get("recommendation", "")

                dc = item.get("digest_card") or {}
                highlight = str(dc.get("highlight") or "")
                tags = dc.get("tags") or []

                # Build description
                desc_parts: List[str] = []
                if highlight:
                    desc_parts.append(f"💎 {highlight}")
                elif one_line:
                    desc_parts.append(f"💬 {one_line}")
                if snippet:
                    desc_parts.append(snippet[:500])
                if tags:
                    desc_parts.append("Tags: " + ", ".join(tags))

                fe = fg.add_entry()
                fe.title(paper_title)
                if url:
                    fe.link(href=url)
                    fe.guid(url, permalink=True)
                else:
                    fe.guid(f"paperbot:{date_str}:{paper_title}", permalink=False)
                fe.description("\n".join(desc_parts) or snippet[:500] or paper_title)

                if rec:
                    fe.category(term=rec)
                for tag in tags[:5]:
                    fe.category(term=tag)
                main_figure = item.get("main_figure") or {}
                if isinstance(main_figure, dict):
                    fig_url = str(main_figure.get("url") or "").strip()
                    if fig_url:
                        try:
                            fe.enclosure(url=fig_url, length=0, type="image/jpeg")
                        except Exception:
                            pass

                # Publication date
                if generated_at:
                    try:
                        dt = datetime.fromisoformat(
                            generated_at.replace("Z", "+00:00")
                        )
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        fe.pubDate(dt)
                    except Exception:
                        pass

        return fg.rss_str(pretty=True).decode("utf-8")

    except ImportError:
        return _fallback_rss_xml(reports, title=title, description=description)


def _build_atom_xml(
    reports: List[Dict[str, Any]],
    *,
    title: str = "PaperBot DailyPaper",
    link: str = "",
) -> str:
    """Build Atom XML from DailyPaper reports."""
    try:
        from feedgen.feed import FeedGenerator

        fg = FeedGenerator()
        fg.title(title)
        fg.link(href=link or "https://paperbot.local", rel="alternate")
        fg.id(link or "https://paperbot.local/feed")
        fg.updated(datetime.now(timezone.utc))

        for report in reports:
            date_str = report.get("date") or ""
            generated_at = report.get("generated_at") or ""
            papers = _collect_papers_from_report(report)

            for item in papers:
                paper_title = item.get("title") or "Untitled"
                url = item.get("url") or ""
                snippet = item.get("snippet") or ""
                dc = item.get("digest_card") or {}
                highlight = str(dc.get("highlight") or "")

                fe = fg.add_entry()
                fe.title(paper_title)
                if url:
                    fe.link(href=url)
                    fe.id(url)
                else:
                    fe.id(f"paperbot:{date_str}:{paper_title}")
                fe.summary(highlight or snippet[:500] or paper_title)

                if generated_at:
                    try:
                        dt = datetime.fromisoformat(
                            generated_at.replace("Z", "+00:00")
                        )
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        fe.updated(dt)
                    except Exception:
                        pass

        return fg.atom_str(pretty=True).decode("utf-8")

    except ImportError:
        return "<feed>feedgen not installed</feed>"


def _fallback_rss_xml(
    reports: List[Dict[str, Any]],
    *,
    title: str = "PaperBot DailyPaper",
    description: str = "",
) -> str:
    """Minimal RSS 2.0 XML without feedgen dependency."""
    items: List[str] = []
    for report in reports:
        for item in _collect_papers_from_report(report):
            paper_title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            import xml.sax.saxutils as saxutils

            t = saxutils.escape(paper_title)
            d = saxutils.escape(snippet[:500])
            u = saxutils.escape(url)
            items.append(
                f"<item><title>{t}</title>"
                f"<link>{u}</link>"
                f"<description>{d}</description>"
                f"</item>"
            )

    import xml.sax.saxutils as saxutils

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0">'
        f"<channel><title>{saxutils.escape(title)}</title>"
        f"<description>{saxutils.escape(description)}</description>"
        + "".join(items)
        + "</channel></rss>"
    )


@router.get("/feed/daily.xml")
async def daily_rss(
    limit: int = Query(default=20, ge=1, le=100),
):
    """RSS 2.0 feed of recent DailyPaper digest papers."""
    cache_key = f"daily_rss:{limit}"
    xml = _cached_xml(cache_key, _REPORTS_DIR)
    if not xml:
        reports = _load_latest_reports(_REPORTS_DIR, limit=limit)
        xml = _store_cached_xml(cache_key, _REPORTS_DIR, _build_rss_xml(reports))
    return Response(content=xml, media_type="application/rss+xml; charset=utf-8")


@router.get("/feed/daily.atom")
async def daily_atom(
    limit: int = Query(default=20, ge=1, le=100),
):
    """Atom feed of recent DailyPaper digest papers."""
    cache_key = f"daily_atom:{limit}"
    xml = _cached_xml(cache_key, _REPORTS_DIR)
    if not xml:
        reports = _load_latest_reports(_REPORTS_DIR, limit=limit)
        xml = _store_cached_xml(cache_key, _REPORTS_DIR, _build_atom_xml(reports))
    return Response(content=xml, media_type="application/atom+xml; charset=utf-8")


@router.get("/feed/track/{track_name}.xml")
async def track_rss(
    track_name: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """RSS 2.0 feed filtered to a specific query/track."""
    cache_key = f"track_rss:{track_name}:{limit}"
    xml = _cached_xml(cache_key, _REPORTS_DIR)
    if not xml:
        reports = _load_latest_reports(_REPORTS_DIR, limit=limit)
        filtered = _filter_reports_by_track(reports, track_name=track_name)
        xml = _store_cached_xml(
            cache_key,
            _REPORTS_DIR,
            _build_rss_xml(
                filtered,
                title=f"PaperBot · {track_name}",
                description=f"Papers matching '{track_name}'",
            ),
        )
    return Response(content=xml, media_type="application/rss+xml; charset=utf-8")


@router.get("/feed/keyword/{keyword}.xml")
async def keyword_rss(
    keyword: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """RSS 2.0 feed filtered by keyword over digest-card fields and summaries."""
    cache_key = f"keyword_rss:{keyword}:{limit}"
    xml = _cached_xml(cache_key, _REPORTS_DIR)
    if not xml:
        reports = _load_latest_reports(_REPORTS_DIR, limit=limit)
        filtered = _filter_reports_by_keyword(reports, keyword=keyword)
        xml = _store_cached_xml(
            cache_key,
            _REPORTS_DIR,
            _build_rss_xml(
                filtered,
                title=f"PaperBot · keyword:{keyword}",
                description=f"Papers matching keyword '{keyword}'",
            ),
        )
    return Response(content=xml, media_type="application/rss+xml; charset=utf-8")
