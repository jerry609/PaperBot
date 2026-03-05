"""Telegram command webhook endpoint for digest subscriptions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel

from paperbot.infrastructure.push.telegram_subscription_store import TelegramSubscriptionStore

router = APIRouter()

_REPORTS_DIR = Path("./reports/dailypaper")


class TelegramCommandRequest(BaseModel):
    chat_id: str
    text: str
    username: Optional[str] = None


class TelegramCommandResponse(BaseModel):
    ok: bool
    command: str
    reply: str
    subscriptions: Dict[str, List[str]]


def _parse_telegram_command(text: str) -> Tuple[str, str, str]:
    raw = (text or "").strip()
    if not raw:
        return "help", "keyword", ""

    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    command_alias = {
        "/subscribe": "subscribe",
        "/unsubscribe": "unsubscribe",
        "/list": "list",
        "/today": "today",
        "/help": "help",
    }
    command = command_alias.get(cmd, "help")

    scope = "keyword"
    value = arg
    if ":" in arg:
        maybe_scope, rest = arg.split(":", 1)
        if maybe_scope.strip().lower() in {"keyword", "track"}:
            scope = maybe_scope.strip().lower()
            value = rest.strip()
    return command, scope, value


def _latest_digest_titles(limit: int = 5) -> List[str]:
    if not _REPORTS_DIR.exists():
        return []
    json_files = sorted(_REPORTS_DIR.glob("*.json"), reverse=True)
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            titles: List[str] = []
            seen: set[str] = set()
            for query in payload.get("queries") or []:
                for item in query.get("top_items") or []:
                    title = str(item.get("title") or "").strip()
                    if title and title not in seen:
                        seen.add(title)
                        titles.append(title)
                    if len(titles) >= limit:
                        return titles
            if titles:
                return titles
        except Exception:
            continue
    return []


@router.post("/push/telegram/command", response_model=TelegramCommandResponse)
async def telegram_command(body: TelegramCommandRequest):
    store = TelegramSubscriptionStore()
    command, scope, value = _parse_telegram_command(body.text)
    subscriptions = store.list_subscriptions(chat_id=body.chat_id)

    if command == "subscribe":
        if not value:
            return TelegramCommandResponse(
                ok=False,
                command=command,
                reply="Usage: /subscribe <keyword> or /subscribe track:<track-name>",
                subscriptions=subscriptions,
            )
        subscriptions = store.subscribe(chat_id=body.chat_id, value=value, scope=scope)
        return TelegramCommandResponse(
            ok=True,
            command=command,
            reply=f"Subscribed {scope}: {value}",
            subscriptions=subscriptions,
        )

    if command == "unsubscribe":
        if not value:
            return TelegramCommandResponse(
                ok=False,
                command=command,
                reply="Usage: /unsubscribe <keyword> or /unsubscribe track:<track-name>",
                subscriptions=subscriptions,
            )
        subscriptions = store.unsubscribe(chat_id=body.chat_id, value=value, scope=scope)
        return TelegramCommandResponse(
            ok=True,
            command=command,
            reply=f"Unsubscribed {scope}: {value}",
            subscriptions=subscriptions,
        )

    if command == "list":
        rows = []
        if subscriptions["keyword"]:
            rows.append("Keywords: " + ", ".join(subscriptions["keyword"]))
        if subscriptions["track"]:
            rows.append("Tracks: " + ", ".join(subscriptions["track"]))
        reply = "\n".join(rows) if rows else "No subscriptions yet."
        return TelegramCommandResponse(
            ok=True,
            command=command,
            reply=reply,
            subscriptions=subscriptions,
        )

    if command == "today":
        titles = _latest_digest_titles(limit=5)
        if not titles:
            reply = "No daily digest available yet."
        else:
            reply = "Today's picks:\n" + "\n".join(f"- {title}" for title in titles)
        return TelegramCommandResponse(
            ok=True,
            command=command,
            reply=reply,
            subscriptions=subscriptions,
        )

    return TelegramCommandResponse(
        ok=True,
        command="help",
        reply=(
            "Commands:\n"
            "/subscribe <keyword>\n"
            "/subscribe track:<track-name>\n"
            "/unsubscribe <keyword>\n"
            "/unsubscribe track:<track-name>\n"
            "/list\n"
            "/today"
        ),
        subscriptions=subscriptions,
    )

