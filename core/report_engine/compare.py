from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class CompareItem:
    id: str
    title: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    code_stats: Dict[str, Any] = field(default_factory=dict)
    repro: Optional[Dict[str, Any]] = None


def normalize_compare_items(items: Optional[List[Dict[str, Any]]]) -> List[CompareItem]:
    if not items:
        return []
    normalized: List[CompareItem] = []
    for idx, raw in enumerate(items):
        normalized.append(
            CompareItem(
                id=str(raw.get("id") or raw.get("paper_id") or f"item-{idx+1}"),
                title=raw.get("title") or raw.get("paper_title") or f"Item {idx+1}",
                metrics=raw.get("metrics", {}),
                code_stats=raw.get("code_stats", {}),
                repro=raw.get("repro"),
            )
        )
    return normalized

