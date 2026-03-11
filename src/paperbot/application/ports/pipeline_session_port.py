"""PipelineSessionPort — workflow session/checkpoint persistence interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class PipelineSessionPort(Protocol):
    """Abstract interface for pipeline session operations."""

    def start_session(
        self,
        *,
        workflow: str,
        payload: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]: ...

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    def list_sessions(
        self, *, workflow: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]: ...

    def save_checkpoint(
        self, session_id: str, checkpoint_data: Dict[str, Any]
    ) -> None: ...

    def update_status(self, session_id: str, status: str) -> None: ...
