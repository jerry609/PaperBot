from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Tuple

from ..schema import NormalizedMessage
from .chatgpt_export import parse_chatgpt_export
from .loose_json import parse_loose_json
from .plaintext import parse_plaintext_chat
from .types import ParsedChatLog

Parser = Callable[[bytes, Optional[str]], Optional[ParsedChatLog]]


def _try_json(data: bytes) -> Optional[Any]:
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def parse_chat_log(data: bytes, *, filename: Optional[str] = None, platform_hint: Optional[str] = None) -> ParsedChatLog:
    """
    Parse an uploaded chat log (JSON/TXT/MD) into a canonical message list.

    The parser is best-effort: it tries known formats first, then falls back to a plaintext parser.
    """
    candidates: List[Tuple[str, Parser]] = []

    hint = (platform_hint or "").strip().lower()
    if hint:
        if "chatgpt" in hint or "openai" in hint or "gpt" in hint:
            candidates.append(("chatgpt", parse_chatgpt_export))
        # Gemini formats vary widely; we currently rely on heuristics.

    # Default order
    candidates.extend(
        [
            ("chatgpt", parse_chatgpt_export),
            ("loose_json", parse_loose_json),
        ]
    )

    for _, parser in candidates:
        parsed = parser(data, filename)
        if parsed and parsed.messages:
            return parsed

    # If it's JSON but not a known export, attempt a generic "messages" shape.
    obj = _try_json(data)
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        msgs: List[NormalizedMessage] = []
        for m in obj["messages"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "unknown").lower()
            content = str(m.get("content") or m.get("text") or "")
            if content.strip():
                msgs.append(NormalizedMessage(role=role, content=content, platform=hint or "unknown"))
        if msgs:
            return ParsedChatLog(platform=hint or "unknown", messages=msgs, metadata={"source": "generic_json"})

    # Plaintext fallback
    parsed_txt = parse_plaintext_chat(data, filename, platform_hint=platform_hint)
    if parsed_txt.messages:
        return parsed_txt

    return ParsedChatLog(platform=platform_hint or "unknown", messages=[], metadata={"error": "unrecognized_format"})
