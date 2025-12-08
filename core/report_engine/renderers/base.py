from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Protocol


@dataclass
class RenderContext:
    title: str
    subtitle: str
    toc: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    model_info: str = ""
    data_time: str = ""
    env_info: str = ""


class Renderer(Protocol):
    def render(self, ctx: RenderContext, **kwargs) -> Any:  # pragma: no cover - interface
        ...

