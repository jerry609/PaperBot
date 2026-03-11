"""ModelEndpointPort — LLM endpoint management interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ModelEndpointPort(Protocol):
    """Abstract interface for model endpoint operations."""

    def list_endpoints(
        self, *, enabled_only: bool = False, include_secrets: bool = False
    ) -> List[Dict[str, Any]]: ...

    def get_endpoint(
        self, endpoint_id: int, include_secrets: bool = False
    ) -> Optional[Dict[str, Any]]: ...

    def upsert_endpoint(
        self, *, payload: Dict[str, Any], endpoint_id: Optional[int] = None
    ) -> Dict[str, Any]: ...

    def delete_endpoint(self, endpoint_id: int) -> bool: ...
