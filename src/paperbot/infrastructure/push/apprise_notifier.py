"""Apprise-based multi-channel push notifier.

Wraps the Apprise library to provide unified push across
Telegram, Discord, WeCom, Feishu, Slack, Email, DingTalk, and more.
Falls back gracefully when Apprise is not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_HAS_APPRISE = False
try:
    import apprise

    _HAS_APPRISE = True
except ImportError:
    apprise = None  # type: ignore[assignment]


class AppriseNotifier:
    """Unified push notifier wrapping Apprise.

    Channel configuration is loaded from a YAML file with tagged groups.
    """

    def __init__(
        self,
        urls: Optional[List[str]] = None,
        *,
        tags: Optional[Dict[str, List[str]]] = None,
    ):
        self._urls = urls or []
        self._tags = tags or {}
        self._apprise: Optional[Any] = None

        if _HAS_APPRISE and self._urls:
            self._apprise = apprise.Apprise()
            for url in self._urls:
                tag_set = self._resolve_tags(url)
                self._apprise.add(url, tag=tag_set or None)

    @classmethod
    def from_yaml(cls, path: str) -> "AppriseNotifier":
        """Load channel config from a YAML file.

        Expected format:
        ```yaml
        channels:
          - url: "tgram://bot_token/chat_id"
            tags: ["daily", "telegram"]
          - url: "discord://webhook_id/webhook_token"
            tags: ["daily", "discord"]
        ```
        """
        config_path = Path(path)
        if not config_path.exists():
            logger.warning("Push config not found: %s", path)
            return cls(urls=[])

        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        channels = data.get("channels") or []
        urls: List[str] = []
        tags: Dict[str, List[str]] = {}

        for ch in channels:
            if not isinstance(ch, dict):
                continue
            url = str(ch.get("url") or "").strip()
            if not url:
                continue
            urls.append(url)
            ch_tags = ch.get("tags") or []
            if ch_tags:
                tags[url] = [str(t).strip() for t in ch_tags if str(t).strip()]

        return cls(urls=urls, tags=tags)

    def _resolve_tags(self, url: str) -> List[str]:
        return self._tags.get(url, [])

    @property
    def available(self) -> bool:
        return _HAS_APPRISE and self._apprise is not None

    @property
    def channel_count(self) -> int:
        return len(self._urls)

    def push(
        self,
        *,
        title: str = "",
        body: str,
        body_format: str = "text",
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push a notification to configured channels.

        Args:
            title: Notification title.
            body: Notification body content.
            body_format: One of "text", "html", "markdown".
            tag: Optional tag to filter channels (only notify channels with this tag).

        Returns:
            Dict with "ok" boolean and optional "error" message.
        """
        if not self.available:
            return {"ok": False, "error": "apprise not available or no channels configured"}

        notify_type = apprise.NotifyType.INFO

        # Map format
        fmt_map = {
            "text": apprise.NotifyFormat.TEXT,
            "html": apprise.NotifyFormat.HTML,
            "markdown": apprise.NotifyFormat.MARKDOWN,
        }
        notify_format = fmt_map.get(body_format, apprise.NotifyFormat.TEXT)

        try:
            result = self._apprise.notify(
                title=title or "",
                body=body,
                notify_type=notify_type,
                body_format=notify_format,
                tag=tag or apprise.common.MATCH_ALL_TAG,
            )
            return {"ok": bool(result)}
        except Exception as exc:
            logger.warning("Apprise push failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def push_daily_digest(
        self,
        *,
        report: Dict[str, Any],
        markdown: str = "",
        html: str = "",
        tag: str = "daily",
    ) -> Dict[str, Any]:
        """Push a DailyPaper digest to channels tagged with 'daily'.

        Prefers HTML body if available, falls back to markdown, then plain text.
        """
        title_str = str(report.get("title") or "DailyPaper Digest")
        date_str = str(report.get("date") or "")
        subject = f"{title_str} - {date_str}" if date_str else title_str

        if html:
            return self.push(title=subject, body=html, body_format="html", tag=tag)
        if markdown:
            return self.push(title=subject, body=markdown, body_format="markdown", tag=tag)
        return self.push(title=subject, body=f"New daily digest: {subject}", tag=tag)
