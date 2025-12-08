"""
声明式流水线抽象，替代硬编码顺序逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, List, Optional, Dict

try:
    from src.paperbot.core.abstractions import ExecutionResult, ensure_execution_result
except ImportError:
    from core.abstractions import ExecutionResult, ensure_execution_result


@dataclass
class StageResult:
    name: str
    status: str
    output: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    stages: List[StageResult] = field(default_factory=list)
    status: str = "success"

    def failed(self) -> bool:
        return self.status == "failed"


class PipelineStage:
    def __init__(
        self,
        name: str,
        run_fn: Callable[[Any], Any],
        *,
        is_critical: bool = True,
        skip_if: Optional[Callable[[Any], bool]] = None,
        update_context: Optional[Callable[[Any, Any], Any]] = None,
    ):
        self.name = name
        self.run_fn = run_fn
        self.is_critical = is_critical
        self.skip_if = skip_if
        self.update_context_fn = update_context

    def should_skip(self, ctx: Any) -> bool:
        return bool(self.skip_if and self.skip_if(ctx))

    async def run(self, ctx: Any) -> StageResult:
        if self.should_skip(ctx):
            return StageResult(name=self.name, status="skipped")

        start = perf_counter()
        try:
            raw = await self.run_fn(ctx)
            res = ensure_execution_result(raw)
            duration_ms = (perf_counter() - start) * 1000
            status = "success" if res.success else "error"
            return StageResult(
                name=self.name,
                status=status,
                output=res.data if res.success else raw,
                error=res.error,
                duration_ms=duration_ms,
                metadata=res.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (perf_counter() - start) * 1000
            return StageResult(
                name=self.name,
                status="error",
                error=str(exc),
                duration_ms=duration_ms,
            )

    def update_context(self, ctx: Any, stage_output: StageResult) -> Any:
        if self.update_context_fn:
            return self.update_context_fn(ctx, stage_output)
        return ctx


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        self.stages.append(stage)
        return self

    async def run(self, ctx: Any) -> PipelineResult:
        results: List[StageResult] = []
        current_ctx = ctx

        for stage in self.stages:
            stage_result = await stage.run(current_ctx)
            results.append(stage_result)

            if stage_result.status == "error" and stage.is_critical:
                return PipelineResult(stages=results, status="failed")

            current_ctx = stage.update_context(current_ctx, stage_result)

        return PipelineResult(stages=results, status="success")

