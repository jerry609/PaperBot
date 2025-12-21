from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Callable


class AcquisitionMode(str, Enum):
    api_first = "api_first"
    http_static = "http_static"
    restricted = "restricted"
    high_risk_dynamic = "high_risk_dynamic"
    import_only = "import_only"


@dataclass
class SourceDescriptor:
    name: str
    version: str = "v1"
    reliability: float = 0.5  # 0..1
    rate_limit_rps: Optional[float] = None
    auth: str = "none"  # none/api_key/oauth/cookie/...
    acquisition_mode: AcquisitionMode = AcquisitionMode.api_first
    enabled_by_default: bool = True
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


