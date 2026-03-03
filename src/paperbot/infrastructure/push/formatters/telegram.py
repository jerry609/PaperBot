"""Telegram MarkdownV2 push formatter."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from paperbot.infrastructure.push.formatters.base import PushFormatter


def _escape_mdv2(text: str) -> str:
    """Escape special chars for Telegram MarkdownV2."""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", str(text or ""))


class TelegramFormatter(PushFormatter):
    """Format DailyPaper digest for Telegram MarkdownV2."""

    @property
    def channel_type(self) -> str:
        return "telegram"

    def format_digest(
        self,
        report: Dict[str, Any],
        *,
        max_papers: int = 10,
    ) -> Dict[str, Any]:
        papers = self._collect_top_papers(report, max_papers)
        title = _escape_mdv2(report.get("title") or "DailyPaper Digest")
        date = _escape_mdv2(report.get("date") or "")
        stats = report.get("stats") or {}

        lines = [
            f"*{title}*",
            f"_{date}_  \\|  {stats.get('unique_items', 0)} papers",
            "",
        ]

        # Daily insight
        llm = report.get("llm_analysis") or {}
        insight = str(llm.get("daily_insight") or "").strip()
        if insight:
            lines.append(f"_{_escape_mdv2(insight[:300])}_")
            lines.append("")

        # Papers
        for idx, item in enumerate(papers, 1):
            title_text = _escape_mdv2(item.get("title") or "Untitled")
            url = item.get("url") or ""
            judge = item.get("judge") or {}
            rec = judge.get("recommendation", "")
            overall = judge.get("overall")
            one_line = _escape_mdv2(str(judge.get("one_line_summary") or ""))

            # Digest card
            dc = item.get("digest_card") or {}
            highlight = _escape_mdv2(str(dc.get("highlight") or ""))
            tags = dc.get("tags") or []

            # Badge
            badge = ""
            if rec == "must_read":
                badge = "🔥 "
            elif rec == "worth_reading":
                badge = "👍 "

            # Title with link
            if url:
                escaped_url = url.replace(")", "\\)")
                lines.append(f"{idx}\\. {badge}[{title_text}]({escaped_url})")
            else:
                lines.append(f"{idx}\\. {badge}{title_text}")

            # Score
            if overall:
                lines.append(f"   ⭐ {_escape_mdv2(f'{overall:.1f}')}/5")

            # Highlight or one-liner
            if highlight:
                lines.append(f"   💎 {highlight}")
            elif one_line:
                lines.append(f"   💬 {one_line}")

            # Tags
            if tags:
                tag_str = " ".join(f"\\#{_escape_mdv2(t)}" for t in tags[:4])
                lines.append(f"   {tag_str}")

            lines.append("")

        # Main figure if available
        main_fig = None
        for item in papers:
            mf = item.get("main_figure")
            if mf and mf.get("url"):
                main_fig = mf
                break

        text = "\n".join(lines)

        result: Dict[str, Any] = {
            "text": text,
            "parse_mode": "MarkdownV2",
        }

        if main_fig:
            result["photo_url"] = main_fig["url"]
            result["photo_caption"] = _escape_mdv2(main_fig.get("caption", ""))

        # Inline keyboard for paper links
        buttons = []
        for item in papers[:5]:
            url = item.get("url") or ""
            title_short = (item.get("title") or "Paper")[:30]
            if url:
                buttons.append({"text": title_short, "url": url})

        if buttons:
            result["inline_keyboard"] = [buttons[:3], buttons[3:5]] if len(buttons) > 3 else [buttons]

        return result
