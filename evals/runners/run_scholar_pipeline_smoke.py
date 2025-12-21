from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure local imports work without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.domain.paper import PaperMeta
from paperbot.domain.influence.result import InfluenceResult
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator, PipelineContext


class SmokeCoordinator(ScholarWorkflowCoordinator):
    """
    Offline smoke coordinator: overrides stages to avoid network/LLM access.
    This is only for eval smoke and should stay deterministic.
    """

    async def _run_research_stage(self, ctx: PipelineContext) -> PipelineContext:
        ctx.research_result = {"venue_tier": 2}
        ctx.stages["research"] = {"result": ctx.research_result}
        return ctx

    async def _run_code_analysis_stage(self, ctx: PipelineContext) -> PipelineContext:
        ctx.code_analysis_result = {"health_score": 0.0}
        ctx.stages["code_analysis"] = {"result": ctx.code_analysis_result}
        return ctx

    async def _run_quality_stage(self, ctx: PipelineContext) -> PipelineContext:
        ctx.quality_result = {"quality_score": 0.0}
        ctx.stages["quality"] = {"result": ctx.quality_result}
        return ctx

    async def _run_influence_stage(self, ctx: PipelineContext) -> PipelineContext:
        # deterministic, minimal result
        ctx.influence_result = InfluenceResult(
            total_score=10.0,
            academic_score=10.0,
            engineering_score=0.0,
            explanation="smoke",
            metrics_breakdown={},
        )
        return ctx

    async def _run_report_stage(self, ctx: PipelineContext):
        # Do not write files in smoke eval
        return None


async def run_case(case_path: Path) -> Dict[str, Any]:
    data = json.loads(case_path.read_text(encoding="utf-8"))
    paper = PaperMeta.from_dict(data["paper"])
    scholar_name = data.get("scholar_name")

    event_log = InMemoryEventLog()
    run_id = new_run_id()
    trace_id = new_trace_id()

    coordinator = SmokeCoordinator({"enable_fail_fast": True})
    report_path, influence, pipeline_data = await coordinator.run_paper_pipeline(
        paper=paper,
        scholar_name=scholar_name,
        persist_report=False,
        event_log=event_log,
        run_id=run_id,
        trace_id=trace_id,
    )

    # Assertions (minimal)
    assert pipeline_data.get("status") in ("success", "error")
    assert influence is not None

    # Ensure at least one score_update event was emitted
    has_score = any(e.get("type") == "score_update" for e in event_log.events)

    return {
        "case": str(case_path),
        "status": pipeline_data.get("status"),
        "report_path": str(report_path) if report_path else None,
        "influence": influence.to_dict() if hasattr(influence, "to_dict") else {},
        "events_emitted": len(event_log.events),
        "has_score_update": has_score,
        "run_id": run_id,
        "trace_id": trace_id,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    case_path = root / "cases" / "scholar_pipeline" / "smoke_basic.json"
    result = asyncio.run(run_case(case_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


