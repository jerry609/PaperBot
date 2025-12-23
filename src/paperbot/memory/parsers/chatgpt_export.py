from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..schema import NormalizedMessage
from .types import ParsedChatLog


def _parse_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(ts, str):
        try:
            # Some exports use ISO strings.
            return datetime.fromisoformat(ts)
        except Exception:
            return None
    return None


def _msg_text(message_obj: Dict[str, Any]) -> str:
    content = message_obj.get("content") or {}
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if p is not None).strip()
        if "text" in content:
            return str(content.get("text") or "").strip()
    if isinstance(content, str):
        return content.strip()
    # Some exports store text at top-level
    if "text" in message_obj:
        return str(message_obj.get("text") or "").strip()
    return ""


def parse_chatgpt_export(data: bytes, filename: Optional[str] = None) -> Optional[ParsedChatLog]:
    """
    Parse ChatGPT "conversations.json" style exports.

    Best-effort and branch-tolerant: messages are sorted by create_time when available.
    """
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        return None

    # ChatGPT export is typically: List[conversation]
    if not isinstance(obj, list):
        return None

    messages: List[NormalizedMessage] = []
    conversation_count = 0
    for conv in obj:
        if not isinstance(conv, dict):
            continue
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            continue
        conversation_count += 1
        conversation_id = str(conv.get("id") or conv.get("conversation_id") or "")

        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author") or {}
            role = str((author.get("role") if isinstance(author, dict) else author) or "unknown").lower()
            content = _msg_text(message)
            if not content:
                continue
            ts = _parse_ts(message.get("create_time") or node.get("create_time"))
            messages.append(
                NormalizedMessage(
                    role=role if role in {"system", "user", "assistant", "tool"} else "unknown",
                    content=content,
                    ts=ts,
                    platform="chatgpt",
                    conversation_id=conversation_id or None,
                    message_id=str(message.get("id") or node_id),
                )
            )

    if not messages:
        return ParsedChatLog(platform="chatgpt", messages=[], metadata={"filename": filename, "conversation_count": 0})

    # Sort by timestamp when available; otherwise keep stable insertion order.
    messages.sort(key=lambda m: (m.ts is None, m.ts or datetime(1970, 1, 1, tzinfo=timezone.utc)))

    return ParsedChatLog(
        platform="chatgpt",
        messages=messages,
        metadata={"filename": filename, "conversation_count": conversation_count, "source": "chatgpt_export"},
    )
