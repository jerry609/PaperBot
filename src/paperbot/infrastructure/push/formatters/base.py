"""Push formatter base class and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PushFormatter(ABC):
    """Abstract base class for channel-specific push formatters.

    Each formatter converts a DailyPaper report into a format suitable
    for a specific push channel (Telegram, Discord, WeCom, Feishu, etc.).
    """

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """Channel type identifier (e.g., 'telegram', 'discord')."""
        ...

    @abstractmethod
    def format_digest(
        self,
        report: Dict[str, Any],
        *,
        max_papers: int = 10,
    ) -> Dict[str, Any]:
        """Format a DailyPaper report for this channel.

        Returns a dict with channel-specific payload structure.
        The exact keys depend on the channel type.
        """
        ...

    def _collect_top_papers(
        self, report: Dict[str, Any], max_papers: int = 10
    ) -> List[Dict[str, Any]]:
        """Helper to collect deduplicated top papers from report."""
        seen: set = set()
        papers: List[Dict[str, Any]] = []
        for q in report.get("queries") or []:
            for item in q.get("top_items") or []:
                key = item.get("title") or id(item)
                if key not in seen:
                    seen.add(key)
                    papers.append(item)
        for item in report.get("global_top") or []:
            key = item.get("title") or id(item)
            if key not in seen:
                seen.add(key)
                papers.append(item)
        # Sort by judge overall or score
        papers.sort(
            key=lambda p: float(
                (p.get("judge") or {}).get("overall") or p.get("score") or 0
            ),
            reverse=True,
        )
        return papers[:max_papers]

    def _paper_title_line(self, item: Dict[str, Any]) -> str:
        """Helper to build a title line with score."""
        title = item.get("title") or "Untitled"
        score = item.get("score")
        judge = item.get("judge") or {}
        overall = judge.get("overall")
        rec = judge.get("recommendation", "")

        parts = [title]
        if overall:
            parts.append(f"[{overall:.1f}/5]")
        elif score:
            parts.append(f"[{score}]")
        if rec == "must_read":
            parts.append("*")
        return " ".join(parts)
