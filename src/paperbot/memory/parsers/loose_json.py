from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..schema import NormalizedMessage
from .types import ParsedChatLog


def _as_text_parts(parts: Any) -> str:
    if isinstance(parts, list):
        texts: List[str] = []
        for p in parts:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict):
                if "text" in p:
                    texts.append(str(p.get("text") or ""))
                elif "content" in p:
                    texts.append(str(p.get("content") or ""))
        return "\n".join(t.strip() for t in texts if t and str(t).strip()).strip()
    if isinstance(parts, dict) and "text" in parts:
        return str(parts.get("text") or "").strip()
    if isinstance(parts, str):
        return parts.strip()
    return ""


def _map_role(role: Any) -> str:
    r = str(role or "unknown").lower()
    if r in {"assistant", "system", "tool", "user"}:
        return r
    if r in {"model", "bot", "ai"}:
        return "assistant"
    if r in {"human"}:
        return "user"
    return "unknown"


def _extract_message_like(obj: Any, platform: str) -> Optional[NormalizedMessage]:
    if not isinstance(obj, dict):
        return None

    role = None
    if "role" in obj:
        role = obj.get("role")
    elif "author" in obj:
        author = obj.get("author")
        if isinstance(author, dict) and "role" in author:
            role = author.get("role")
        else:
            role = author
    elif "sender" in obj:
        role = obj.get("sender")

    content = ""
    if "content" in obj:
        c = obj.get("content")
        if isinstance(c, dict) and "parts" in c:
            content = _as_text_parts(c.get("parts"))
        else:
            content = _as_text_parts(c)
    elif "parts" in obj:
        content = _as_text_parts(obj.get("parts"))
    elif "text" in obj:
        content = str(obj.get("text") or "").strip()
    elif "message" in obj and isinstance(obj.get("message"), dict):
        return _extract_message_like(obj.get("message"), platform)

    if not content:
        return None

    return NormalizedMessage(role=_map_role(role), content=content, platform=platform)


def _walk(obj: Any, *, max_nodes: int = 50_000, max_depth: int = 10) -> Iterable[Any]:
    stack: List[Tuple[Any, int]] = [(obj, 0)]
    seen = 0
    while stack and seen < max_nodes:
        cur, depth = stack.pop()
        seen += 1
        yield cur
        if depth >= max_depth:
            continue
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))


def parse_loose_json(data: bytes, filename: Optional[str] = None, platform_hint: Optional[str] = None) -> Optional[ParsedChatLog]:
    """
    Best-effort parser for "various LLM chat logs" stored as JSON.

    Supports common shapes found in:
    - Gemini API logs (request.contents + response.candidates)
    - Generic {"messages": [...]} exports
    - Ad-hoc objects containing role/content/text/parts fields
    """
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        return None

    platform = (platform_hint or "unknown").strip().lower() or "unknown"
    messages: List[NormalizedMessage] = []

    def add(msg: Optional[NormalizedMessage]):
        if msg is None:
            return
        if not msg.content.strip():
            return
        messages.append(msg)

    # Gemini API-ish request/response wrapper
    if isinstance(obj, dict) and ("request" in obj or "response" in obj):
        req = obj.get("request")
        resp = obj.get("response")
        if isinstance(req, dict):
            contents = req.get("contents")
            if isinstance(contents, list):
                for c in contents:
                    add(_extract_message_like(c, platform))
        if isinstance(resp, dict):
            cands = resp.get("candidates")
            if isinstance(cands, list):
                for cand in cands:
                    if isinstance(cand, dict):
                        add(_extract_message_like(cand.get("content"), platform))
                        # Some SDKs store text directly.
                        add(_extract_message_like(cand, platform))

    # Gemini API direct response shape
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        for cand in obj.get("candidates") or []:
            if isinstance(cand, dict):
                add(_extract_message_like(cand.get("content"), platform))
                add(_extract_message_like(cand, platform))

    # Common message containers
    if isinstance(obj, dict):
        for key in ("messages", "contents", "chat_history", "history"):
            v = obj.get(key)
            if isinstance(v, list):
                for item in v:
                    add(_extract_message_like(item, platform))

    # prompt/response pairs used by some export tools
    if isinstance(obj, dict) and ("prompt" in obj or "response" in obj) and isinstance(obj.get("prompt"), (str, dict)):
        prompt = obj.get("prompt")
        response = obj.get("response")
        prompt_text = _as_text_parts(prompt.get("parts") if isinstance(prompt, dict) else prompt)
        response_text = _as_text_parts(response.get("parts") if isinstance(response, dict) else response)
        if prompt_text:
            add(NormalizedMessage(role="user", content=prompt_text, platform=platform))
        if response_text:
            add(NormalizedMessage(role="assistant", content=response_text, platform=platform))

    # General recursive scan
    if not messages:
        for node in _walk(obj):
            add(_extract_message_like(node, platform))

    # Deduplicate (role+content)
    dedup: Dict[str, NormalizedMessage] = {}
    for m in messages:
        k = f"{m.role}:{m.content.strip()}"
        if k not in dedup:
            dedup[k] = m

    final = list(dedup.values())
    if not final:
        return None

    return ParsedChatLog(platform=platform, messages=final, metadata={"filename": filename, "source": "loose_json"})

