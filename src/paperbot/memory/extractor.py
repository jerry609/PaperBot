from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    def repair_json(s: str) -> str:  # type: ignore[no-redef]
        return s

from paperbot.infrastructure.llm import ModelRouter, TaskType

from .schema import MemoryCandidate, NormalizedMessage


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


_PII_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(\+?\d[\d -]{7,}\d)\b"), "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> str:
    out = text
    for rx, repl in _PII_PATTERNS:
        out = rx.sub(repl, out)
    return out


def _heuristic_extract_from_user_text(text: str) -> List[MemoryCandidate]:
    t = _normalize_text(text)
    if not t:
        return []

    out: List[MemoryCandidate] = []

    # Name / identity
    m = re.search(r"(我叫|我的名字是|请叫我)\s*([^\s，,。.!！？]{1,20})", t)
    if m:
        out.append(
            MemoryCandidate(
                kind="profile",
                content=f"用户希望被称呼为：{m.group(2)}",
                confidence=0.85,
                tags=["name", "profile"],
                evidence={"pattern": m.group(1)},
            )
        )

    # Preferences
    for pref_rx, polarity in [
        (re.compile(r"我(喜欢|偏好|更喜欢|爱)\s*([^。.!！？]{1,60})"), "like"),
        (re.compile(r"我(不喜欢|讨厌|反感)\s*([^。.!！？]{1,60})"), "dislike"),
    ]:
        for m in pref_rx.finditer(t):
            obj = _normalize_text(m.group(2))
            if obj:
                out.append(
                    MemoryCandidate(
                        kind="preference",
                        content=f"用户{polarity}：{obj}",
                        confidence=0.72,
                        tags=["preference", polarity],
                        evidence={"pattern": m.group(1)},
                    )
                )

    # Goals / plans
    for m in re.finditer(r"(我想|我要|我计划|我准备)\s*([^。.!！？]{1,80})", t):
        goal = _normalize_text(m.group(2))
        if goal:
            out.append(
                MemoryCandidate(
                    kind="goal",
                    content=f"用户目标/计划：{goal}",
                    confidence=0.68,
                    tags=["goal"],
                    evidence={"pattern": m.group(1)},
                )
            )

    # Constraints
    for m in re.finditer(r"(必须|不能|不要|请不要|限制是)\s*([^。.!！？]{1,80})", t):
        c = _normalize_text(m.group(2))
        if c:
            out.append(
                MemoryCandidate(
                    kind="constraint",
                    content=f"约束/禁忌：{c}",
                    confidence=0.62,
                    tags=["constraint"],
                    evidence={"pattern": m.group(1)},
                )
            )

    # TODOs
    for m in re.finditer(r"(帮我|请帮我|需要你)\s*([^。.!！？]{1,100})", t):
        todo = _normalize_text(m.group(2))
        if todo:
            out.append(
                MemoryCandidate(
                    kind="todo",
                    content=f"待办/请求：{todo}",
                    confidence=0.58,
                    tags=["todo"],
                    evidence={"pattern": m.group(1)},
                )
            )

    # Deduplicate within batch
    uniq: Dict[str, MemoryCandidate] = {}
    for cand in out:
        key = _sha256_hex(f"{cand.kind}:{_normalize_text(cand.content)}")
        if key not in uniq:
            uniq[key] = cand
    return list(uniq.values())


def _llm_extract(messages: List[NormalizedMessage], *, language_hint: Optional[str] = None) -> List[MemoryCandidate]:
    user_text = "\n".join(m.content for m in messages if m.role == "user")
    user_text = user_text.strip()
    if not user_text:
        return []

    system = (
        "你是一个“长期记忆抽取器”。从用户对话中提炼可复用、稳定、不会过期的事实/偏好/目标/约束。"
        "输出必须是 JSON 数组，每个元素包含字段：kind, content, confidence, tags, evidence。"
        "kind 只能取：profile, preference, goal, project, constraint, todo, fact。"
        "content 用一句话描述；不要包含隐私（手机号/邮箱/精确地址）。confidence 0~1。tags 为字符串数组。"
        "evidence 包含引用依据（例如原句摘要/对话片段）。不要输出其它文本。"
    )
    if language_hint:
        system += f"\n对话语言偏好：{language_hint}"

    # Keep prompt bounded.
    prompt = redact_pii(user_text)
    if len(prompt) > 24000:
        prompt = prompt[-24000:]

    router = ModelRouter.from_env()
    provider = router.get_provider(TaskType.EXTRACTION)
    raw = provider.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    repaired = repair_json(raw)
    data = json.loads(repaired)
    if not isinstance(data, list):
        return []

    out: List[MemoryCandidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        content = str(item.get("content") or "").strip()
        if not kind or not content:
            continue
        confidence = float(item.get("confidence") or 0.6)
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        evidence = item.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {"raw": str(evidence)}
        out.append(
            MemoryCandidate(
                kind=kind,  # type: ignore[arg-type]
                content=content,
                confidence=max(0.0, min(1.0, confidence)),
                tags=[str(t) for t in tags if str(t).strip()][:12],
                evidence=evidence,
            )
        )

    # Dedup
    uniq: Dict[str, MemoryCandidate] = {}
    for cand in out:
        key = _sha256_hex(f"{cand.kind}:{_normalize_text(cand.content)}")
        if key not in uniq:
            uniq[key] = cand
    return list(uniq.values())


def extract_memories(
    messages: List[NormalizedMessage],
    *,
    use_llm: bool = False,
    redact: bool = True,
    language_hint: Optional[str] = None,
) -> List[MemoryCandidate]:
    """
    Extract long-term memories from normalized messages.

    If `use_llm` is True and a provider is configured, extraction uses an LLM and falls back to heuristics on failure.
    """
    msgs = messages
    if redact:
        msgs = [
            NormalizedMessage(
                role=m.role,
                content=redact_pii(m.content),
                ts=m.ts,
                platform=m.platform,
                conversation_id=m.conversation_id,
                message_id=m.message_id,
                metadata=m.metadata,
            )
            for m in messages
        ]

    if use_llm:
        try:
            extracted = _llm_extract(msgs, language_hint=language_hint)
            if extracted:
                return extracted
        except Exception:
            pass

    out: List[MemoryCandidate] = []
    for idx, m in enumerate(msgs):
        if m.role != "user":
            continue
        for cand in _heuristic_extract_from_user_text(m.content):
            out.append(
                MemoryCandidate(
                    kind=cand.kind,
                    content=cand.content,
                    confidence=cand.confidence,
                    tags=cand.tags,
                    evidence={**cand.evidence, "conversation_id": m.conversation_id, "message_index": idx},
                )
            )

    # Batch dedup
    uniq: Dict[str, MemoryCandidate] = {}
    for cand in out:
        key = _sha256_hex(f"{cand.kind}:{_normalize_text(cand.content)}")
        if key not in uniq:
            uniq[key] = cand
    return list(uniq.values())


def build_memory_context(memories: Iterable[MemoryCandidate], *, max_items: int = 8) -> str:
    """
    Build a prompt-ready memory block.
    """
    items = list(memories)[:max_items]
    if not items:
        return ""
    lines = ["# User Long-term Memory (best-effort)"]
    for m in items:
        tag_str = ""
        if m.tags:
            tag_str = f" [{', '.join(m.tags[:6])}]"
        lines.append(f"- ({m.kind}, {m.confidence:.2f}) {m.content}{tag_str}")
    return "\n".join(lines).strip() + "\n"


def memory_candidate_to_dict(m: MemoryCandidate) -> Dict[str, Any]:
    return asdict(m)
