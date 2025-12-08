from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SectionView:
    title: str
    slug: str
    content: Any


@dataclass
class ViewModel:
    title: str
    subtitle: str
    toc: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    compare_items: List[Dict[str, Any]]
    env_info: str
    data_time: str


class ViewModelBuilder:
    """
    将原始上下文转换为模板友好的视图模型。
    目标：模板层少逻辑，主要展示。
    """

    def build(
        self,
        layout: Dict[str, Any],
        chapters: List[Dict[str, Any]],
        compare_items: List[Any],
        ctx: Dict[str, Any],
    ) -> ViewModel:
        return ViewModel(
            title=layout.get("title", ""),
            subtitle=layout.get("subtitle", ""),
            toc=layout.get("toc", []),
            chapters=chapters,
            compare_items=[ci if isinstance(ci, dict) else ci.__dict__ for ci in compare_items],
            env_info=ctx.get("env_info", ""),
            data_time=ctx.get("data_time", ""),
        )

