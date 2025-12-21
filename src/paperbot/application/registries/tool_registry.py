from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable


@dataclass
class ToolDescriptor:
    name: str
    version: str = "v1"
    schema: Dict[str, Any] = field(default_factory=dict)  # JSON schema-ish
    auth_scope: str = ""
    rate_limit_rps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Minimal in-process tool registry.

    Phase-0: centralize tool metadata and invocation to prepare for MCP/tool governance.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(self, desc: ToolDescriptor, handler: Optional[Callable[[Dict[str, Any]], Any]] = None) -> None:
        self._tools[desc.name] = desc
        if handler:
            self._handlers[desc.name] = handler

    def get(self, name: str) -> Optional[ToolDescriptor]:
        return self._tools.get(name)

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self._handlers:
            raise KeyError(f"Tool handler not registered: {name}")
        return self._handlers[name](args)

    def all(self) -> Dict[str, ToolDescriptor]:
        return dict(self._tools)


