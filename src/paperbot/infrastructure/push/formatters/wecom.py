"""WeCom (企业微信) push formatter."""
from __future__ import annotations

from typing import Any, Dict, List

from paperbot.infrastructure.push.formatters.base import PushFormatter


class WeComFormatter(PushFormatter):
    """Format DailyPaper digest for WeCom webhook.

    Supports both markdown message and news card format.
    """

    @property
    def channel_type(self) -> str:
        return "wecom"

    def format_digest(
        self,
        report: Dict[str, Any],
        *,
        max_papers: int = 10,
    ) -> Dict[str, Any]:
        papers = self._collect_top_papers(report, max_papers)
        title = report.get("title") or "DailyPaper Digest"
        date = report.get("date") or ""
        stats = report.get("stats") or {}

        # Build markdown message
        lines = [
            f"# 📄 {title}",
            f"> {date} · {stats.get('unique_items', 0)} papers",
            "",
        ]

        # Daily insight
        llm = report.get("llm_analysis") or {}
        insight = str(llm.get("daily_insight") or "").strip()
        if insight:
            lines.append(f"**📖 本期导读**")
            lines.append(insight[:500])
            lines.append("")

        for idx, item in enumerate(papers, 1):
            paper_title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            judge = item.get("judge") or {}
            rec = judge.get("recommendation", "")
            overall = judge.get("overall")
            one_line = str(judge.get("one_line_summary") or "")

            dc = item.get("digest_card") or {}
            highlight = str(dc.get("highlight") or "")
            tags = dc.get("tags") or []

            badge = ""
            if rec == "must_read":
                badge = "🔥"
            elif rec == "worth_reading":
                badge = "👍"

            if url:
                lines.append(f"{idx}. {badge} [{paper_title}]({url})")
            else:
                lines.append(f"{idx}. {badge} {paper_title}")

            meta: List[str] = []
            if overall:
                meta.append(f"⭐{overall:.1f}")
            if highlight:
                meta.append(f"💎 {highlight[:150]}")
            elif one_line:
                meta.append(f"💬 {one_line[:150]}")
            if tags:
                meta.append(" ".join(f"`{t}`" for t in tags[:4]))

            if meta:
                lines.append(f"   {'  |  '.join(meta)}")
            lines.append("")

        markdown_text = "\n".join(lines)

        # Also build news card articles for richer rendering
        articles = []
        for item in papers[:8]:
            paper_title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            dc = item.get("digest_card") or {}
            desc = str(dc.get("highlight") or (item.get("judge") or {}).get("one_line_summary") or "")
            mf = item.get("main_figure") or {}

            article: Dict[str, Any] = {
                "title": paper_title[:128],
                "description": desc[:512],
                "url": url,
            }
            if mf.get("url"):
                article["picurl"] = mf["url"]
            articles.append(article)

        return {
            "markdown": {"content": markdown_text[:4096]},
            "news": {"articles": articles},
        }
