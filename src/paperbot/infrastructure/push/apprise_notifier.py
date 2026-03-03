"""Apprise-based multi-channel push notifier.

Wraps the Apprise library to provide unified push across
Telegram, Discord, WeCom, Feishu, Slack, Email, DingTalk, and more.
Falls back gracefully when Apprise is not installed.
"""
from __future__ import annotations

import logging
import os
import hashlib
import time
from dataclasses import dataclass
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


_SCHEME_TO_FORMATTER = {
    "tgram": "telegram",
    "telegram": "telegram",
    "discord": "discord",
    "wecom": "wecom",
    "feishu": "feishu",
    "lark": "lark",
}

_RETRYABLE_ERROR_HINTS = (
    "429",
    "too many requests",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "connection reset",
    "network is unreachable",
    "503",
    "502",
    "504",
)


@dataclass
class PushChannel:
    url: str
    tags: List[str]


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
        self._channels = [PushChannel(url=url, tags=self._resolve_tags(url)) for url in self._urls]
        self._apprise: Optional[Any] = None
        self._sent_fingerprints: Dict[str, float] = {}
        self._idempotency_ttl_s = max(
            0.0, float(os.getenv("PAPERBOT_PUSH_IDEMPOTENCY_TTL_S", "600"))
        )

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
        return _HAS_APPRISE and bool(self._urls)

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
        """Push a DailyPaper digest to channels tagged with `tag`.

        Uses channel-specific formatter payloads when available, and falls back
        to plain HTML/markdown body otherwise.
        """
        if not self.available:
            return {"ok": False, "error": "apprise not available or no channels configured"}

        title_str = str(report.get("title") or "DailyPaper Digest")
        date_str = str(report.get("date") or "")
        subject = f"{title_str} - {date_str}" if date_str else title_str

        matched = [ch for ch in self._channels if not tag or tag in ch.tags]
        if not matched:
            return {"ok": False, "error": f"no channels match tag '{tag}'"}

        details: List[Dict[str, Any]] = []
        all_ok = True
        retry_attempts = max(1, int(os.getenv("PAPERBOT_PUSH_RETRY_ATTEMPTS", "3")))
        retry_backoff_s = max(0.0, float(os.getenv("PAPERBOT_PUSH_RETRY_BACKOFF_S", "0.8")))
        for channel in matched:
            body, body_format = self._build_channel_body(
                channel.url,
                report=report,
                fallback_html=html,
                fallback_markdown=markdown,
                subject=subject,
            )
            fingerprint = self._delivery_fingerprint(channel.url, subject, body)
            if self._is_recent_duplicate(fingerprint):
                details.append(
                    {
                        "url": channel.url,
                        "ok": True,
                        "attempts": 0,
                        "skipped": True,
                        "error_code": "idempotent_replay",
                    }
                )
                continue
            try:
                channel_apprise = apprise.Apprise()
                channel_apprise.add(channel.url, tag=channel.tags or None)
                fmt_map = {
                    "text": apprise.NotifyFormat.TEXT,
                    "html": apprise.NotifyFormat.HTML,
                    "markdown": apprise.NotifyFormat.MARKDOWN,
                }
                ok, attempts, error = self._notify_with_retry(
                    channel_apprise=channel_apprise,
                    title=subject,
                    body=body,
                    body_format=fmt_map.get(body_format, apprise.NotifyFormat.TEXT),
                    retry_attempts=retry_attempts,
                    retry_backoff_s=retry_backoff_s,
                )
                row = {"url": channel.url, "ok": ok, "attempts": attempts}
                if error:
                    row["error"] = error
                    row["error_code"] = self._map_error_code(error)
                else:
                    row["error_code"] = ""
                if ok:
                    self._mark_sent(fingerprint)
                details.append(row)
                all_ok = all_ok and ok
            except Exception as exc:
                logger.warning("Apprise digest push failed url=%s err=%s", channel.url, exc)
                error_text = str(exc)
                details.append(
                    {
                        "url": channel.url,
                        "ok": False,
                        "error": error_text,
                        "error_code": self._map_error_code(error_text),
                    }
                )
                all_ok = False

        result: Dict[str, Any] = {"ok": all_ok, "channels": details}
        if not all_ok:
            result["error"] = "one or more channel pushes failed"
        return result

    @staticmethod
    def _is_retryable_error(error_text: str) -> bool:
        normalized = str(error_text or "").strip().lower()
        return any(hint in normalized for hint in _RETRYABLE_ERROR_HINTS)

    @staticmethod
    def _map_error_code(error_text: str) -> str:
        normalized = str(error_text or "").strip().lower()
        if not normalized:
            return ""
        if "429" in normalized or "too many requests" in normalized:
            return "rate_limited"
        if "timed out" in normalized or "timeout" in normalized:
            return "timeout"
        if "401" in normalized or "403" in normalized or "invalid webhook" in normalized:
            return "auth_failed"
        if any(code in normalized for code in ("502", "503", "504", "temporarily unavailable")):
            return "downstream_unavailable"
        if "network" in normalized or "connection" in normalized:
            return "network_error"
        return "delivery_failed"

    def _notify_with_retry(
        self,
        *,
        channel_apprise: Any,
        title: str,
        body: str,
        body_format: Any,
        retry_attempts: int,
        retry_backoff_s: float,
    ) -> tuple[bool, int, str]:
        last_error = ""
        for attempt in range(1, max(1, retry_attempts) + 1):
            try:
                ok = bool(
                    channel_apprise.notify(
                        title=title,
                        body=body,
                        notify_type=apprise.NotifyType.INFO,
                        body_format=body_format,
                        tag=apprise.common.MATCH_ALL_TAG,
                    )
                )
            except Exception as exc:
                ok = False
                last_error = str(exc)
                retryable = self._is_retryable_error(last_error)
                if not retryable or attempt >= retry_attempts:
                    return False, attempt, last_error
                delay = retry_backoff_s * (2 ** (attempt - 1))
                logger.info(
                    "Apprise push retryable failure attempt=%s/%s delay=%.2fs err=%s",
                    attempt,
                    retry_attempts,
                    delay,
                    last_error,
                )
                if delay > 0:
                    time.sleep(delay)
                continue

            if ok:
                return True, attempt, ""

            last_error = "notification rejected by downstream channel"
            if attempt >= retry_attempts:
                return False, attempt, last_error
            delay = retry_backoff_s * (2 ** (attempt - 1))
            logger.info(
                "Apprise push returned not-ok attempt=%s/%s delay=%.2fs",
                attempt,
                retry_attempts,
                delay,
            )
            if delay > 0:
                time.sleep(delay)

        return False, max(1, retry_attempts), last_error

    @staticmethod
    def _delivery_fingerprint(url: str, title: str, body: str) -> str:
        digest = hashlib.sha1(f"{url}|{title}|{body}".encode("utf-8")).hexdigest()
        return digest

    def _is_recent_duplicate(self, fingerprint: str) -> bool:
        if self._idempotency_ttl_s <= 0:
            return False
        now = time.time()
        for key, expires_at in list(self._sent_fingerprints.items()):
            if expires_at <= now:
                self._sent_fingerprints.pop(key, None)
        expiry = self._sent_fingerprints.get(fingerprint)
        return bool(expiry and expiry > now)

    def _mark_sent(self, fingerprint: str) -> None:
        if self._idempotency_ttl_s <= 0:
            return
        self._sent_fingerprints[fingerprint] = time.time() + self._idempotency_ttl_s

    def _build_channel_body(
        self,
        url: str,
        *,
        report: Dict[str, Any],
        fallback_html: str,
        fallback_markdown: str,
        subject: str,
    ) -> tuple[str, str]:
        channel_type = self._channel_type_from_url(url)
        payload = self._format_payload(channel_type, report)

        if payload:
            body, body_format = self._payload_to_body(channel_type, payload)
            if body:
                return body, body_format

        if fallback_html:
            return fallback_html, "html"
        if fallback_markdown:
            return fallback_markdown, "markdown"
        return f"New daily digest: {subject}", "text"

    @staticmethod
    def _channel_type_from_url(url: str) -> str:
        scheme = str(url or "").split("://", 1)[0].strip().lower()
        return _SCHEME_TO_FORMATTER.get(scheme, "")

    @staticmethod
    def _payload_to_body(channel_type: str, payload: Dict[str, Any]) -> tuple[str, str]:
        if channel_type == "telegram":
            return str(payload.get("text") or ""), "markdown"
        if channel_type == "wecom":
            return str((payload.get("markdown") or {}).get("content") or ""), "markdown"
        if channel_type in {"feishu", "lark"}:
            interactive = (payload.get("interactive") or {}).get("card") or {}
            elements = interactive.get("elements") or []
            lines: List[str] = []
            for e in elements:
                text_obj = e.get("text") if isinstance(e, dict) else None
                if isinstance(text_obj, dict):
                    content = str(text_obj.get("content") or "").strip()
                    if content:
                        lines.append(content)
            return "\n".join(lines), "markdown"
        if channel_type == "discord":
            embeds = payload.get("embeds") or []
            if not embeds:
                return "", "text"
            embed = embeds[0] if isinstance(embeds[0], dict) else {}
            chunks: List[str] = []
            title = str(embed.get("title") or "").strip()
            desc = str(embed.get("description") or "").strip()
            if title:
                chunks.append(f"## {title}")
            if desc:
                chunks.append(desc)
            for field in embed.get("fields") or []:
                if not isinstance(field, dict):
                    continue
                name = str(field.get("name") or "").strip()
                value = str(field.get("value") or "").strip()
                if name or value:
                    chunks.append(f"- {name}: {value}")
            return "\n".join(chunks), "markdown"
        return "", "text"

    @staticmethod
    def _format_payload(channel_type: str, report: Dict[str, Any]) -> Dict[str, Any]:
        if not channel_type:
            return {}
        try:
            from paperbot.infrastructure.push.formatters import get_formatter

            formatter = get_formatter(channel_type)
            if formatter is None:
                return {}
            return formatter.format_digest(report, max_papers=10)
        except Exception:
            return {}
