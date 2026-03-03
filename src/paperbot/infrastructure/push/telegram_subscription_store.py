"""Lightweight persistence for Telegram digest subscriptions."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelegramSubscriptionStore:
    """Persist subscriptions keyed by Telegram chat id."""

    def __init__(self, path: str | None = None):
        configured = path or os.getenv("PAPERBOT_TELEGRAM_SUBS_PATH", "data/telegram_subscriptions.json")
        self._path = Path(configured).expanduser()

    def _load(self) -> Dict[str, Dict[str, List[str] | str]]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def _save(self, payload: Dict[str, Dict[str, List[str] | str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        key = (scope or "").strip().lower()
        if key not in {"keyword", "track"}:
            return "keyword"
        return key

    @staticmethod
    def _normalize_value(value: str) -> str:
        return " ".join((value or "").strip().split())

    def subscribe(self, *, chat_id: str, value: str, scope: str = "keyword") -> Dict[str, List[str]]:
        normalized_chat = str(chat_id).strip()
        normalized_scope = self._normalize_scope(scope)
        normalized_value = self._normalize_value(value)
        if not normalized_chat or not normalized_value:
            raise ValueError("chat_id and value are required")

        payload = self._load()
        row = payload.setdefault(
            normalized_chat,
            {"keyword": [], "track": [], "updated_at": _utc_now_iso()},
        )
        values = row.setdefault(normalized_scope, [])
        if isinstance(values, list) and normalized_value not in values:
            values.append(normalized_value)
            values.sort()
        row["updated_at"] = _utc_now_iso()
        self._save(payload)
        return self.list_subscriptions(chat_id=normalized_chat)

    def unsubscribe(self, *, chat_id: str, value: str = "", scope: str = "keyword") -> Dict[str, List[str]]:
        normalized_chat = str(chat_id).strip()
        normalized_scope = self._normalize_scope(scope)
        normalized_value = self._normalize_value(value)
        if not normalized_chat:
            raise ValueError("chat_id is required")

        payload = self._load()
        row = payload.get(normalized_chat) or {}
        values = row.get(normalized_scope)
        if not isinstance(values, list):
            values = []

        if normalized_value:
            values = [item for item in values if item != normalized_value]
        else:
            values = []
        row[normalized_scope] = values
        row["updated_at"] = _utc_now_iso()
        payload[normalized_chat] = row
        self._save(payload)
        return self.list_subscriptions(chat_id=normalized_chat)

    def list_subscriptions(self, *, chat_id: str) -> Dict[str, List[str]]:
        normalized_chat = str(chat_id).strip()
        payload = self._load()
        row = payload.get(normalized_chat) or {}
        keywords = row.get("keyword")
        tracks = row.get("track")
        return {
            "keyword": list(keywords) if isinstance(keywords, list) else [],
            "track": list(tracks) if isinstance(tracks, list) else [],
        }

