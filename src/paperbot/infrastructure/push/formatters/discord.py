"""Discord Rich Embed push formatter."""
from __future__ import annotations

from typing import Any, Dict, List

from paperbot.infrastructure.push.formatters.base import PushFormatter

# Discord embed color (blue)
_EMBED_COLOR = 0x2563EB


class DiscordFormatter(PushFormatter):
    """Format DailyPaper digest as Discord Rich Embed JSON."""

    @property
    def channel_type(self) -> str:
        return "discord"

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

        # Build embed fields for top papers
        fields: List[Dict[str, Any]] = []
        for idx, item in enumerate(papers[:10], 1):
            paper_title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            judge = item.get("judge") or {}
            rec = judge.get("recommendation", "")
            overall = judge.get("overall")
            one_line = str(judge.get("one_line_summary") or "")

            # Digest card
            dc = item.get("digest_card") or {}
            highlight = str(dc.get("highlight") or "")
            tags = dc.get("tags") or []

            badge = ""
            if rec == "must_read":
                badge = "🔥 "
            elif rec == "worth_reading":
                badge = "👍 "

            name = f"{idx}. {badge}{paper_title[:100]}"
            if url:
                name = f"{idx}. {badge}[{paper_title[:80]}]({url})"

            value_parts: List[str] = []
            if overall:
                value_parts.append(f"⭐ {overall:.1f}/5")
            if highlight:
                value_parts.append(f"💎 {highlight[:200]}")
            elif one_line:
                value_parts.append(f"💬 {one_line[:200]}")
            if tags:
                value_parts.append(" ".join(f"`{t}`" for t in tags[:4]))

            value = "\n".join(value_parts) or "—"

            fields.append({
                "name": name[:256],
                "value": value[:1024],
                "inline": False,
            })

        # Description from daily insight
        llm = report.get("llm_analysis") or {}
        description = str(llm.get("daily_insight") or "").strip()
        if not description:
            description = f"{stats.get('unique_items', 0)} papers collected"

        embed: Dict[str, Any] = {
            "title": f"📄 {title}",
            "description": description[:4096],
            "color": _EMBED_COLOR,
            "fields": fields[:25],  # Discord max 25 fields
            "footer": {"text": f"PaperBot · {date}"},
        }

        # Main figure as thumbnail
        for item in papers:
            mf = item.get("main_figure")
            if mf and mf.get("url"):
                embed["thumbnail"] = {"url": mf["url"]}
                break

        return {"embeds": [embed]}
