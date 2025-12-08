"""
对比项数据模型与规范化工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class CompareItem:
    """对比项数据模型。"""
    id: str
    title: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    code_stats: Dict[str, Any] = field(default_factory=dict)
    repro: Optional[Dict[str, Any]] = None


def normalize_compare_items(items: Optional[List[Dict[str, Any]]]) -> List[CompareItem]:
    """
    将原始对比数据规范化为 CompareItem 列表。
    
    Args:
        items: 原始对比数据列表
        
    Returns:
        规范化后的 CompareItem 列表
    """
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

