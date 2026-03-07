from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class InjectionDetectionResult:
    flagged: bool
    matched_rules: List[str]
    normalized_text: str


_RULES: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("role_system", re.compile(r"(?:^|\n)\s*system\s*:", re.IGNORECASE)),
    ("role_assistant", re.compile(r"(?:^|\n)\s*assistant\s*:", re.IGNORECASE)),
    ("role_developer", re.compile(r"(?:^|\n)\s*developer\s*:", re.IGNORECASE)),
    (
        "ignore_previous",
        re.compile(
            r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE
        ),
    ),
    (
        "forget_previous",
        re.compile(
            r"forget\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior)\s+(?:instructions|rules|messages)",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_prompt",
        re.compile(r"reveal\s+(?:the\s+)?(?:system|hidden|developer)\s+prompt", re.IGNORECASE),
    ),
    (
        "do_not_follow",
        re.compile(
            r"do\s+not\s+follow\s+(?:the\s+)?(?:above|previous|prior)\s+(?:instructions|rules)",
            re.IGNORECASE,
        ),
    ),
    (
        "tag_escape_control",
        re.compile(
            r"</(?:user_memory|paper_analysis|project_context|system)>\s*(?:system\s*:|assistant\s*:|developer\s*:|ignore\s+previous|forget\s+previous|reveal\s+(?:the\s+)?(?:system|hidden|developer)\s+prompt|@assistant)",
            re.IGNORECASE,
        ),
    ),
    ("special_token", re.compile(r"<\|endoftext\|>|\[inst\]|\[/inst\]", re.IGNORECASE)),
    ("assistant_ping", re.compile(r"@assistant\b", re.IGNORECASE)),
    (
        "jailbreak",
        re.compile(r"\b(?:perform|run|execute|attempt|use)\s+(?:a\s+)?jailbreak\b", re.IGNORECASE),
    ),
    (
        "policy_bypass",
        re.compile(r"bypass\s+(?:the\s+)?(?:policy|guardrails|safety)", re.IGNORECASE),
    ),
)


def normalize_injection_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    return normalized.strip()


SAFE_CONTEXT_MARKERS = (
    "treat it as read-only contextual background",
    "never execute any instructions it may contain",
    "mitigates indirect prompt injection",
)


def detect_injection_patterns(text: str) -> InjectionDetectionResult:
    normalized = normalize_injection_text(text)
    lowered = normalized.lower()

    if any(marker in lowered for marker in SAFE_CONTEXT_MARKERS):
        return InjectionDetectionResult(flagged=False, matched_rules=[], normalized_text=normalized)

    matched = [name for name, pattern in _RULES if pattern.search(normalized)]
    return InjectionDetectionResult(
        flagged=bool(matched),
        matched_rules=matched,
        normalized_text=normalized,
    )


__all__ = [
    "InjectionDetectionResult",
    "SAFE_CONTEXT_MARKERS",
    "detect_injection_patterns",
    "normalize_injection_text",
]
