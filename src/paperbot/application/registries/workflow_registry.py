from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable


@dataclass
class WorkflowDescriptor:
    name: str
    version: str = "v1"
    stages: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowRegistry:
    """
    Minimal workflow registry.

    Phase-0: just metadata + factory; later supports declarative graphs and gating policies.
    """

    def __init__(self) -> None:
        self._workflows: Dict[str, WorkflowDescriptor] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register(self, desc: WorkflowDescriptor, factory: Optional[Callable[[], Any]] = None) -> None:
        self._workflows[desc.name] = desc
        if factory:
            self._factories[desc.name] = factory

    def get(self, name: str) -> Optional[WorkflowDescriptor]:
        return self._workflows.get(name)

    def create(self, name: str) -> Any:
        if name not in self._factories:
            raise KeyError(f"Workflow factory not registered: {name}")
        return self._factories[name]()

    def all(self) -> Dict[str, WorkflowDescriptor]:
        return dict(self._workflows)


