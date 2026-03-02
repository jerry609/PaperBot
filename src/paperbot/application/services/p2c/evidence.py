from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .models import EvidenceLink


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


class EvidenceLinker:
    """Utility for binding extracted claims to text spans."""

    def find_keyword_evidence(
        self,
        text: str,
        *,
        keywords: Sequence[str],
        section: str,
        supports: Sequence[str],
        max_links: int = 3,
    ) -> List[EvidenceLink]:
        evidence: List[EvidenceLink] = []
        if not text:
            return evidence

        for keyword in keywords:
            pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                evidence.append(
                    EvidenceLink(
                        type="paper_span",
                        ref=f"{section}#char:{match.start()}-{match.end()}",
                        supports=list(supports),
                        confidence=0.82,
                    )
                )
                if len(evidence) >= max_links:
                    return evidence
        return evidence

    def from_match(
        self,
        *,
        section: str,
        supports: Sequence[str],
        start: int,
        end: int,
        confidence: float = 0.84,
    ) -> EvidenceLink:
        return EvidenceLink(
            type="paper_span",
            ref=f"{section}#char:{start}-{end}",
            supports=list(supports),
            confidence=_clamp(confidence),
        )

    def section_anchor(
        self,
        text: str,
        *,
        section: str,
        supports: Sequence[str],
        max_chars: int = 180,
    ) -> List[EvidenceLink]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        end = min(len(cleaned), max_chars)
        return [
            EvidenceLink(
                type="paper_span",
                ref=f"{section}#char:0-{end}",
                supports=list(supports),
                confidence=0.75,
            )
        ]


def calibrate_confidence(
    base_confidence: float,
    evidence: Iterable[EvidenceLink],
    *,
    required: bool = False,
) -> float:
    links = list(evidence)
    base = _clamp(base_confidence)
    if not links:
        penalty = 0.55 if required else 0.8
        return _clamp(base * penalty)

    quality = sum(_clamp(link.confidence) for link in links) / len(links)
    multiplier = 0.75 + (0.25 * quality)
    if required and quality < 0.6:
        multiplier *= 0.9
    return _clamp(base * multiplier)
