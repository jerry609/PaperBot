from __future__ import annotations

import re
from typing import List, Optional

from ..schema import NormalizedMessage
from .types import ParsedChatLog


_ROLE_PREFIXES = [
    ("user", re.compile(r"^(user|human|me|我|用户)\s*[:：]\s*", re.IGNORECASE)),
    ("assistant", re.compile(r"^(assistant|ai|bot|助手|模型)\s*[:：]\s*", re.IGNORECASE)),
    ("system", re.compile(r"^(system|系统)\s*[:：]\s*", re.IGNORECASE)),
]


def parse_plaintext_chat(data: bytes, filename: Optional[str] = None, *, platform_hint: Optional[str] = None) -> ParsedChatLog:
    try:
        text = data.decode("utf-8")
    except Exception:
        # fallback; best-effort
        text = data.decode("utf-8", errors="ignore")

    lines = [ln.rstrip() for ln in text.splitlines()]
    messages: List[NormalizedMessage] = []

    current_role = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_role
        content = "\n".join(buffer).strip()
        if content:
            messages.append(
                NormalizedMessage(
                    role=current_role or "unknown",
                    content=content,
                    platform=(platform_hint or "unknown"),
                    conversation_id=None,
                    message_id=None,
                )
            )
        buffer = []

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            buffer.append("")
            continue

        matched_role = None
        payload = stripped
        for role, rx in _ROLE_PREFIXES:
            m = rx.match(stripped)
            if m:
                matched_role = role
                payload = stripped[m.end():]
                break

        if matched_role:
            if current_role is not None:
                flush()
            current_role = matched_role
            buffer.append(payload)
        else:
            buffer.append(stripped)

    flush()

    # If no role structure detected, treat the entire text as one "user" message for extraction.
    if not messages and text.strip():
        messages = [NormalizedMessage(role="user", content=text.strip(), platform=(platform_hint or "unknown"))]

    return ParsedChatLog(
        platform=(platform_hint or "unknown"),
        messages=messages,
        metadata={"filename": filename, "source": "plaintext"},
    )
