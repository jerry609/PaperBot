from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable


@dataclass
class SourceDescriptor:
    name: str
    version: str = "v1"
    reliability: float = 0.5  # 0..1
    rate_limit_rps: Optional[float] = None
    auth: str = "none"  # none/api_key/oauth/cookie/...
    metadata: Dict[str, Any] = field(default_factory=dict)


class SourceRegistry:
    """
    Minimal in-process registry for data sources.

    Phase-0: supports explicit registration; later can be config/entrypoints driven.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, SourceDescriptor] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register(self, desc: SourceDescriptor, factory: Optional[Callable[[], Any]] = None) -> None:
        self._sources[desc.name] = desc
        if factory:
            self._factories[desc.name] = factory

    def get(self, name: str) -> Optional[SourceDescriptor]:
        return self._sources.get(name)

    def create(self, name: str) -> Any:
        if name not in self._factories:
            raise KeyError(f"Source factory not registered: {name}")
        return self._factories[name]()

    def all(self) -> Dict[str, SourceDescriptor]:
        return dict(self._sources)


