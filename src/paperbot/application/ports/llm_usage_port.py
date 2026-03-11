"""LLMUsagePort — LLM token/cost usage tracking interface."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMUsagePort(Protocol):
    """Abstract interface for LLM usage tracking operations."""

    def record_usage(
        self,
        *,
        task_type: str,
        provider_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_cost: float = 0.0,
    ) -> None: ...

    def summarize(
        self, *, days: int = 7, task_type: Optional[str] = None
    ) -> Dict[str, Any]: ...
