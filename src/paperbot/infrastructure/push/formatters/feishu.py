"""Feishu / Lark (飞书) interactive card push formatter."""
from __future__ import annotations

from typing import Any, Dict, List

from paperbot.infrastructure.push.formatters.base import PushFormatter


class FeishuFormatter(PushFormatter):
    """Format DailyPaper digest as Feishu/Lark interactive card.

    Supports both interactive card (msg_type: interactive) and
    post (msg_type: post) formats.
    Registered under both 'feishu' and 'lark' keys.
    """

    @property
    def channel_type(self) -> str:
        return "feishu"

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

        # Build interactive card
        elements: List[Dict[str, Any]] = []

        # Header note
        llm = report.get("llm_analysis") or {}
        insight = str(llm.get("daily_insight") or "").strip()
        if insight:
            elements.append({
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": insight[:300]}],
            })
            elements.append({"tag": "hr"})

        # Paper cards
        for idx, item in enumerate(papers, 1):
            paper_title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            judge = item.get("judge") or {}
            rec = judge.get("recommendation", "")
            overall = judge.get("overall")

            dc = item.get("digest_card") or {}
            highlight = str(dc.get("highlight") or "")
            one_line = str(judge.get("one_line_summary") or "")
            tags = dc.get("tags") or []

            badge = ""
            if rec == "must_read":
                badge = "🔥 "
            elif rec == "worth_reading":
                badge = "👍 "

            # Title element
            title_text = f"{idx}. {badge}{paper_title}"
            if overall:
                title_text += f" ⭐{overall:.1f}"

            content_parts: List[str] = []
            if highlight:
                content_parts.append(f"💎 {highlight}")
            elif one_line:
                content_parts.append(f"💬 {one_line}")
            if tags:
                content_parts.append(" | ".join(tags[:4]))

            md_content = f"**{title_text}**\n" + "\n".join(content_parts)

            element: Dict[str, Any] = {
                "tag": "div",
                "text": {"tag": "lark_md", "content": md_content},
            }

            # Add action button with link
            if url:
                element["extra"] = {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "阅读"},
                    "type": "primary",
                    "url": url,
                }

            elements.append(element)

        # Footer stats
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "note",
            "elements": [{
                "tag": "plain_text",
                "content": f"PaperBot · {date} · {stats.get('unique_items', 0)} papers",
            }],
        })

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"📄 {title}"},
                "template": "blue",
            },
            "elements": elements,
        }

        # Also build a simpler post format
        post_lines: List[List[Dict[str, Any]]] = []
        for item in papers[:8]:
            paper_title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            line: List[Dict[str, Any]] = []
            if url:
                line.append({"tag": "a", "text": paper_title[:100], "href": url})
            else:
                line.append({"tag": "text", "text": paper_title[:100]})
            dc = item.get("digest_card") or {}
            hl = str(dc.get("highlight") or "")
            if hl:
                line.append({"tag": "text", "text": f"\n  💎 {hl[:150]}"})
            post_lines.append(line)

        post = {
            "zh_cn": {
                "title": f"📄 {title} - {date}",
                "content": post_lines,
            }
        }

        return {
            "interactive": {"card": card},
            "post": post,
        }
