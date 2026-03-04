"""Shared HTML / plain-text email template for DailyPaper digest.

Used by both SMTP (daily_push_service) and Resend (resend_service) channels.

Layout (BestBlogs-inspired):
  1. Header — title, date, stats
  2. 本期导读 — trend narrative from llm_analysis
  3. 三步精选流程 — static methodology blurb
  4. 分层推荐 — papers grouped by recommendation tier
     - Must Read  → full 方法大框
     - Worth Reading → full 方法大框
     - Skim → compact card (title + one-liner)
  5. Footer — unsubscribe
"""
from __future__ import annotations

import html as _html
from typing import Any, Dict, List, Optional, Tuple

# ── colour palette ──────────────────────────────────────────────
_BLUE = "#2563eb"
_DARK_BLUE = "#1e40af"
_ORANGE = "#f59e0b"
_GREEN = "#16a34a"
_TEAL = "#0d9488"
_GRAY_50 = "#f9fafb"
_GRAY_100 = "#f3f4f6"
_GRAY_200 = "#e5e7eb"
_GRAY_400 = "#9ca3af"
_GRAY_500 = "#6b7280"
_GRAY_900 = "#111827"

_TIER_STYLES: Dict[str, Dict[str, str]] = {
    "must_read": {"color": _GREEN, "bg": "#f0fdf4", "label": "🔥 Must Read", "border": _GREEN},
    "worth_reading": {"color": _BLUE, "bg": "#eff6ff", "label": "👍 Worth Reading", "border": _BLUE},
    "skim": {"color": _GRAY_500, "bg": _GRAY_50, "label": "📋 Skim", "border": _GRAY_400},
}

DEFAULT_MAX_PER_TIER = 15


# ── helpers ─────────────────────────────────────────────────────

def _esc(val: Any) -> str:
    return _html.escape(str(val)) if val else ""


def _truncate(text: str, limit: int = 300) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def _query_name(q: Dict[str, Any], fallback: str = "Query") -> str:
    """Extract display name from a query dict (handles multiple key conventions)."""
    return (
        q.get("query") or q.get("raw_query") or q.get("normalized_query")
        or q.get("name") or fallback
    )


def _collect_all_papers(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deduplicate papers from queries + global_top, preserving order."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in report.get("queries") or []:
        for item in q.get("top_items") or []:
            key = item.get("title") or id(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
    for item in report.get("global_top") or []:
        key = item.get("title") or id(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _group_by_tier(
    papers: List[Dict[str, Any]], *, max_per_tier: int = DEFAULT_MAX_PER_TIER
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Group papers into (must_read, worth_reading, skim) tiers."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "must_read": [], "worth_reading": [], "skim": [],
    }
    for p in papers:
        judge = p.get("judge") or {}
        rec = judge.get("recommendation", "skim")
        if rec == "skip":
            continue
        bucket = buckets.get(rec, buckets["skim"])
        bucket.append(p)
    # Sort each tier by score descending
    for items in buckets.values():
        items.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return [
        (tier, buckets[tier][:max_per_tier])
        for tier in ("must_read", "worth_reading", "skim")
        if buckets.get(tier)
    ]


# ── HTML components ─────────────────────────────────────────────

def _method_framework_html(item: Dict[str, Any]) -> str:
    """Build the 方法大框 from judge dimension rationales + snippet."""
    judge: Dict[str, Any] = item.get("judge") or {}
    snippet = str(item.get("snippet") or "")

    rows: List[Tuple[str, str]] = []

    # 研究问题 ← relevance rationale
    rel = (judge.get("relevance") or {}).get("rationale", "")
    if rel:
        rows.append(("🎯 研究问题", rel))

    # 核心方法 ← snippet (abstract) truncated
    if snippet:
        rows.append(("🔬 核心方法", _truncate(snippet, 250)))

    # 关键证据 ← rigor rationale
    rig = (judge.get("rigor") or {}).get("rationale", "")
    if rig:
        rows.append(("📊 关键证据", rig))

    # 适用场景 ← impact rationale
    imp = (judge.get("impact") or {}).get("rationale", "")
    if imp:
        rows.append(("🏷️ 适用场景", imp))

    # 创新点 ← novelty rationale
    nov = (judge.get("novelty") or {}).get("rationale", "")
    if nov:
        rows.append(("💡 创新点", nov))

    if not rows:
        return ""

    inner = "".join(
        f'<tr><td style="padding:3px 8px 3px 0;vertical-align:top;white-space:nowrap;'
        f'font-size:12px;color:{_GRAY_500};font-weight:600;">{_esc(label)}</td>'
        f'<td style="padding:3px 0;font-size:12px;color:{_GRAY_500};line-height:1.5;">{_esc(val)}</td></tr>'
        for label, val in rows
    )
    return (
        f'<div style="background:{_GRAY_50};border-left:3px solid {_ORANGE};'
        f'padding:10px 12px;margin-top:10px;border-radius:0 6px 6px 0;">'
        f'<table style="border-collapse:collapse;width:100%;">{inner}</table>'
        f'</div>'
    )


def _main_figure_html(item: Dict[str, Any]) -> str:
    """Render method figure block when image source is displayable."""
    mf = item.get("main_figure") or {}
    if not isinstance(mf, dict):
        return ""

    image_src = ""
    url = str(mf.get("url") or "").strip()
    inline_data_url = str(mf.get("inline_data_url") or "").strip()

    if inline_data_url.startswith("data:image/"):
        image_src = inline_data_url
    elif url.startswith(("http://", "https://")) and ".zip#" not in url and not url.endswith(".zip"):
        image_src = url

    if not image_src:
        return ""

    caption = _esc(str(mf.get("caption") or ""))
    caption_html = (
        f'<div style="margin-top:6px;font-size:12px;color:{_GRAY_500};line-height:1.5;">{caption}</div>'
        if caption
        else ""
    )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {_GRAY_200};'
        f'border-radius:8px;background:{_GRAY_50};">'
        f'<div style="font-size:12px;color:{_GRAY_500};margin-bottom:6px;">🖼️ 主方法图</div>'
        f'<img src="{_esc(image_src)}" alt="main-figure" '
        f'style="max-width:100%;border-radius:6px;display:block;" />'
        f'{caption_html}'
        f'</div>'
    )


def _paper_card_full_html(idx: int, item: Dict[str, Any]) -> str:
    """Full paper card with method framework (for must_read / worth_reading)."""
    title = _esc(item.get("title") or "Untitled")
    url = _esc(item.get("url") or item.get("external_url") or "")
    score = float(item.get("score") or 0)
    venue = _esc(item.get("subject_or_venue") or "")
    authors: List[str] = list(item.get("authors") or [])
    judge: Dict[str, Any] = item.get("judge") or {}
    one_line = str(judge.get("one_line_summary") or "")
    overall = judge.get("overall", 0)

    # title link
    if url:
        title_html = f'<a href="{url}" style="color:{_DARK_BLUE};text-decoration:none;font-weight:600;font-size:15px;">{title}</a>'
    else:
        title_html = f'<span style="color:{_DARK_BLUE};font-weight:600;font-size:15px;">{title}</span>'

    # metadata pills
    pills: List[str] = []
    if venue:
        pills.append(f'<span style="background:{_GRAY_100};color:{_GRAY_500};padding:2px 8px;border-radius:4px;font-size:11px;">📍 {venue}</span>')
    pills.append(f'<span style="background:#eff6ff;color:{_BLUE};padding:2px 8px;border-radius:4px;font-size:11px;">⭐ {score:.2f}</span>')
    if overall:
        pills.append(f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-size:11px;">Judge {overall:.1f}/5</span>')
    if authors:
        author_str = _esc(", ".join(authors[:3]))
        if len(authors) > 3:
            author_str += " et al."
        pills.append(f'<span style="color:{_GRAY_400};font-size:11px;">👤 {author_str}</span>')
    meta_html = " ".join(pills)

    # one-line summary
    summary_html = ""
    if one_line:
        summary_html = f'<div style="margin-top:6px;font-size:13px;color:{_GRAY_900};font-style:italic;">💬 {_esc(one_line)}</div>'

    framework = _method_framework_html(item)
    main_figure = _main_figure_html(item)

    # Digest card (highlight + tags)
    digest_card = item.get("digest_card") or {}
    digest_html = ""
    if digest_card:
        parts: List[str] = []
        highlight = _esc(str(digest_card.get("highlight") or ""))
        if highlight:
            parts.append(
                f'<div style="font-size:13px;color:{_GRAY_900};font-weight:600;margin-bottom:4px;">'
                f'💎 {highlight}</div>'
            )
        dc_method = _esc(str(digest_card.get("method") or ""))
        dc_finding = _esc(str(digest_card.get("finding") or ""))
        if dc_method:
            parts.append(f'<div style="font-size:12px;color:{_GRAY_500};margin-bottom:2px;">🔬 {dc_method}</div>')
        if dc_finding:
            parts.append(f'<div style="font-size:12px;color:{_GRAY_500};margin-bottom:2px;">📌 {dc_finding}</div>')
        tags = digest_card.get("tags") or []
        if tags:
            tag_pills = " ".join(
                f'<span style="background:#e0e7ff;color:#3730a3;padding:1px 6px;border-radius:3px;'
                f'font-size:10px;margin-right:3px;">{_esc(t)}</span>'
                for t in tags[:6]
            )
            parts.append(f'<div style="margin-top:4px;">{tag_pills}</div>')
        if parts:
            digest_html = (
                f'<div style="margin-top:8px;padding:8px 10px;background:#f5f3ff;'
                f'border-radius:6px;">{"".join(parts)}</div>'
            )

    return (
        f'<div style="background:#fff;border:1px solid {_GRAY_200};border-radius:8px;'
        f'padding:14px 16px;margin-bottom:12px;">'
        f'<div><span style="color:{_GRAY_400};font-size:13px;margin-right:6px;">{idx}.</span>{title_html}</div>'
        f'<div style="margin-top:6px;">{meta_html}</div>'
        f'{summary_html}'
        f'{main_figure}'
        f'{digest_html}'
        f'{framework}'
        f'</div>'
    )


def _paper_card_compact_html(idx: int, item: Dict[str, Any]) -> str:
    """Compact card for skim-tier papers (title + score + one-liner)."""
    title = _esc(item.get("title") or "Untitled")
    url = _esc(item.get("url") or item.get("external_url") or "")
    score = float(item.get("score") or 0)
    judge: Dict[str, Any] = item.get("judge") or {}
    one_line = _esc(str(judge.get("one_line_summary") or ""))

    if url:
        title_html = f'<a href="{url}" style="color:{_DARK_BLUE};text-decoration:none;">{title}</a>'
    else:
        title_html = title

    summary = f' — <span style="color:{_GRAY_500};">{one_line}</span>' if one_line else ""
    return (
        f'<div style="padding:6px 0;border-bottom:1px solid {_GRAY_100};font-size:13px;">'
        f'<span style="color:{_GRAY_400};margin-right:4px;">{idx}.</span> {title_html}'
        f' <span style="color:{_GRAY_400};font-size:11px;">(⭐ {score:.2f})</span>'
        f'{summary}'
        f'</div>'
    )


def _tier_section_html(tier: str, items: List[Dict[str, Any]]) -> str:
    """Render a recommendation tier section."""
    style = _TIER_STYLES.get(tier, _TIER_STYLES["skim"])
    use_full = tier in ("must_read", "worth_reading")

    if use_full:
        cards = "\n".join(_paper_card_full_html(i, it) for i, it in enumerate(items, 1))
    else:
        cards = "\n".join(_paper_card_compact_html(i, it) for i, it in enumerate(items, 1))

    return (
        f'<div style="margin-bottom:32px;">'
        f'<div style="border-left:4px solid {style["border"]};padding:8px 14px;'
        f'background:{style["bg"]};border-radius:0 6px 6px 0;margin-bottom:14px;">'
        f'<span style="font-size:17px;font-weight:700;color:{style["color"]};">{style["label"]}</span>'
        f' <span style="color:{_GRAY_400};font-size:14px;">({len(items)})</span>'
        f'</div>'
        f'{cards}'
        f'</div>'
    )


def _intro_section_html(report: Dict[str, Any]) -> str:
    """本期导读 — narrative intro from llm_analysis."""
    llm = report.get("llm_analysis") or {}
    daily_insight = str(llm.get("daily_insight") or "").strip()
    query_trends = llm.get("query_trends") or []

    parts: List[str] = []
    if daily_insight:
        parts.append(f'<p style="margin:0 0 8px;font-size:14px;color:{_GRAY_900};line-height:1.6;">{_esc(daily_insight)}</p>')

    if query_trends:
        trend_items = "".join(
            f'<li style="margin-bottom:4px;"><strong>{_esc(t.get("query", ""))}</strong>: '
            f'{_esc(_truncate(str(t.get("analysis", "")), 200))}</li>'
            for t in query_trends[:5]
        )
        parts.append(f'<ul style="margin:0;padding-left:20px;font-size:13px;color:{_GRAY_500};line-height:1.6;">{trend_items}</ul>')

    if not parts:
        # Fallback: summarize from queries
        queries = report.get("queries") or []
        if queries:
            topics = [_query_name(q) for q in queries if q.get("top_items")]
            topics = [t for t in topics if t and t != "Query"]
            if topics:
                topic_str = "、".join(_esc(t) for t in topics[:5])
                parts.append(
                    f'<p style="margin:0;font-size:14px;color:{_GRAY_900};line-height:1.6;">'
                    f'本期主要关注方向：{topic_str}。</p>'
                )

    if not parts:
        return ""

    return (
        f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;'
        f'padding:16px;margin-bottom:24px;">'
        f'<h2 style="margin:0 0 10px;font-size:16px;color:#92400e;">📖 本期导读</h2>'
        f'{"".join(parts)}'
        f'</div>'
    )


def _process_section_html(stats: Dict[str, Any]) -> str:
    """三步精选流程 — static methodology blurb."""
    unique = stats.get("unique_items", 0)
    hits = stats.get("total_query_hits", 0)
    return (
        f'<div style="background:{_GRAY_50};border-radius:8px;padding:14px 16px;margin-bottom:24px;'
        f'font-size:12px;color:{_GRAY_500};line-height:1.6;">'
        f'<strong style="color:{_GRAY_900};">🔍 三步精选流程</strong><br>'
        f'① 聚合收集 — 从 Papers.cool / arXiv 抓取 {unique} 篇，匹配 {hits} 次命中<br>'
        f'② AI 智能分析 — Judge 多维评分 + Trend 趋势洞察 + Insight 综合研判<br>'
        f'③ 分层推荐 — Must Read / Worth Reading / Skim 三级分流，聚焦高价值论文'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def build_digest_html(
    report: Dict[str, Any],
    *,
    unsub_link: Optional[str] = None,
    max_per_tier: int = DEFAULT_MAX_PER_TIER,
) -> str:
    """Build a rich HTML email body from a DailyPaper *report* dict."""
    date_str = _esc(report.get("date") or "")
    stats = report.get("stats") or {}
    unique = stats.get("unique_items", 0)
    hits = stats.get("total_query_hits", 0)

    # ── header ──
    header = (
        f'<div style="border-top:4px solid {_BLUE};padding:20px 0 16px;">'
        f'<h1 style="margin:0;font-size:24px;color:{_BLUE};font-weight:700;">📄 PaperBot DailyPaper</h1>'
        f'<p style="margin:6px 0 0;color:{_GRAY_500};font-size:14px;">'
        f'{date_str} · {unique} papers · {hits} hits</p>'
        f'</div>'
    )

    # ── 本期导读 ──
    intro = _intro_section_html(report)

    # ── 三步精选流程 ──
    process = _process_section_html(stats)

    # ── 分层推荐 ──
    all_papers = _collect_all_papers(report)
    tiers = _group_by_tier(all_papers, max_per_tier=max_per_tier)
    tier_html = "\n".join(_tier_section_html(t, items) for t, items in tiers)

    if not tier_html:
        # Fallback: no judge data, show flat list by query
        tier_html = _fallback_query_sections_html(report, max_per_tier)

    # ── footer ──
    unsub_html = ""
    if unsub_link:
        unsub_html = f' <a href="{_esc(unsub_link)}" style="color:{_GRAY_400};">Unsubscribe</a>'
    footer = (
        f'<div style="margin-top:32px;padding-top:16px;border-top:1px solid {_GRAY_200};'
        f'font-size:12px;color:{_GRAY_400};">'
        f'You received this from PaperBot DailyPaper.{unsub_html}'
        f'</div>'
    )

    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        f'<body style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;'
        f'max-width:680px;margin:0 auto;padding:24px;color:{_GRAY_900};background:#fff;">'
        f'{header}{intro}{process}{tier_html}{footer}'
        f'</body></html>'
    )


def _fallback_query_sections_html(report: Dict[str, Any], max_items: int) -> str:
    """Fallback when no judge data: group by query like before."""
    section_colors = ["#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ef4444"]
    queries = list(report.get("queries") or [])
    sections: List[str] = []
    if queries:
        for qi, q in enumerate(queries):
            color = section_colors[qi % len(section_colors)]
            q_name = _query_name(q, fallback=f"Query {qi + 1}")
            top_items = list(q.get("top_items") or [])[:max_items]
            if not top_items:
                continue
            cards = "\n".join(_paper_card_full_html(i, it) for i, it in enumerate(top_items, 1))
            sections.append(
                f'<div style="border-left:4px solid {color};padding-left:16px;margin-bottom:28px;">'
                f'<h2 style="margin:0 0 12px;font-size:17px;color:{_GRAY_900};">'
                f'{_esc(q_name)} <span style="font-weight:400;color:{_GRAY_400};font-size:14px;">'
                f'({len(top_items)} hits)</span></h2>'
                f'{cards}</div>'
            )
    else:
        global_top = list(report.get("global_top") or [])[:max_items]
        if global_top:
            cards = "\n".join(_paper_card_full_html(i, it) for i, it in enumerate(global_top, 1))
            sections.append(
                f'<div style="margin-bottom:28px;">'
                f'<h2 style="margin:0 0 12px;font-size:17px;color:{_GRAY_900};">Top Papers</h2>'
                f'{cards}</div>'
            )
    return "\n".join(sections)


# ── plain-text version ──────────────────────────────────────────

def build_digest_text(
    report: Dict[str, Any],
    *,
    unsub_link: Optional[str] = None,
    max_per_tier: int = DEFAULT_MAX_PER_TIER,
) -> str:
    """Build a plain-text fallback from a DailyPaper *report* dict."""
    lines: List[str] = []
    lines.append("📄 PaperBot DailyPaper")
    title = str(report.get("title") or "DailyPaper Digest")
    lines.append(title)
    lines.append(f"Date: {report.get('date', '-')}")
    stats = report.get("stats") or {}
    lines.append(f"Papers: {stats.get('unique_items', 0)} · Hits: {stats.get('total_query_hits', 0)}")
    lines.append("")

    # 导读
    llm = report.get("llm_analysis") or {}
    daily_insight = str(llm.get("daily_insight") or "").strip()
    if daily_insight:
        lines.append("📖 本期导读")
        lines.append(daily_insight)
        lines.append("")
    query_trends = llm.get("query_trends") or []
    if query_trends:
        for t in query_trends[:5]:
            lines.append(f"  · {t.get('query', '')}: {_truncate(str(t.get('analysis', '')), 200)}")
        lines.append("")

    # 三步精选
    lines.append(f"🔍 三步精选: 聚合 {stats.get('unique_items', 0)} 篇 → AI Judge 评分 → 分层推荐")
    lines.append("")

    # 分层推荐
    all_papers = _collect_all_papers(report)
    tiers = _group_by_tier(all_papers, max_per_tier=max_per_tier)

    tier_labels = {"must_read": "🔥 Must Read", "worth_reading": "👍 Worth Reading", "skim": "📋 Skim"}

    if tiers:
        for tier, items in tiers:
            label = tier_labels.get(tier, tier)
            lines.append(f"{'='*50}")
            lines.append(f"{label} ({len(items)})")
            lines.append(f"{'='*50}")
            lines.append("")
            use_full = tier in ("must_read", "worth_reading")
            for idx, item in enumerate(items, 1):
                _append_paper_text(lines, idx, item, full=use_full)
            lines.append("")
    else:
        # Fallback: by query
        for q in report.get("queries") or []:
            q_name = _query_name(q)
            top_items = list(q.get("top_items") or [])[:max_per_tier]
            if not top_items:
                continue
            lines.append(f"▌ {q_name} ({len(top_items)} hits)")
            lines.append("")
            for idx, item in enumerate(top_items, 1):
                _append_paper_text(lines, idx, item, full=True)
            lines.append("")

    lines.append("---")
    if unsub_link:
        lines.append(f"Unsubscribe: {unsub_link}")
    else:
        lines.append("You received this from PaperBot DailyPaper.")
    return "\n".join(lines)


def _append_paper_text(
    lines: List[str], idx: int, item: Dict[str, Any], *, full: bool = True
) -> None:
    title = item.get("title") or "Untitled"
    url = item.get("url") or ""
    score = float(item.get("score") or 0)
    venue = item.get("subject_or_venue") or ""
    authors = list(item.get("authors") or [])
    judge: Dict[str, Any] = item.get("judge") or {}
    one_line = str(judge.get("one_line_summary") or "")
    rec = judge.get("recommendation", "")

    badge = ""
    if rec == "must_read":
        badge = "[Must Read] "
    elif rec == "worth_reading":
        badge = "[Worth Reading] "

    lines.append(f"  {idx}. {badge}{title} (⭐ {score:.2f})")
    meta_parts: List[str] = []
    if venue:
        meta_parts.append(venue)
    if authors:
        meta_parts.append(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
    if meta_parts:
        lines.append(f"     {' | '.join(meta_parts)}")
    if url:
        lines.append(f"     {url}")
    if one_line:
        lines.append(f"     💬 {one_line}")

    if full:
        # Digest card text version
        digest_card = item.get("digest_card") or {}
        if digest_card:
            dc_highlight = str(digest_card.get("highlight") or "")
            dc_method = str(digest_card.get("method") or "")
            dc_finding = str(digest_card.get("finding") or "")
            dc_tags = digest_card.get("tags") or []
            if dc_highlight:
                lines.append(f"     💎 {dc_highlight}")
            if dc_method:
                lines.append(f"     🔬 方法: {dc_method}")
            if dc_finding:
                lines.append(f"     📌 发现: {dc_finding}")
            if dc_tags:
                lines.append(f"     🏷️ {', '.join(dc_tags)}")

        # 方法大框 text version
        snippet = str(item.get("snippet") or "")
        rel = str((judge.get("relevance") or {}).get("rationale", ""))
        rig = str((judge.get("rigor") or {}).get("rationale", ""))
        imp = str((judge.get("impact") or {}).get("rationale", ""))
        nov = str((judge.get("novelty") or {}).get("rationale", ""))

        framework: List[Tuple[str, str]] = []
        if rel:
            framework.append(("🎯 研究问题", rel))
        if snippet:
            framework.append(("🔬 核心方法", _truncate(snippet, 200)))
        if rig:
            framework.append(("📊 关键证据", rig))
        if imp:
            framework.append(("🏷️ 适用场景", imp))
        if nov:
            framework.append(("💡 创新点", nov))

        if framework:
            for label, val in framework:
                lines.append(f"     {label}: {val}")

    lines.append("")
