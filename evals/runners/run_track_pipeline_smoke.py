from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Ensure local imports work without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # type: ignore

from paperbot.application.collaboration.message_schema import new_run_id, new_trace_id
from paperbot.application.workflows.scholar_pipeline import ScholarPipeline
from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator, PipelineContext
from paperbot.domain.influence.result import InfluenceResult
from paperbot.domain.paper import PaperMeta
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.infrastructure.services.data_source import BaseDataSource
from paperbot.agents.scholar_tracking.paper_tracker_agent import PaperTrackerAgent


class StubDataSource(BaseDataSource):
    async def fetch_papers_by_author(self, scholar, limit: int = 20) -> List[PaperMeta]:
        # Return deterministic papers for any scholar.
        papers = [
            PaperMeta(
                paper_id="stub_paper_001",
                title="Stub Paper One",
                authors=["Alice"],
                year=2025,
                venue="StubConf",
                citation_count=3,
                has_code=False,
            ),
            PaperMeta(
                paper_id="stub_paper_002",
                title="Stub Paper Two",
                authors=["Bob"],
                year=2025,
                venue="StubConf",
                citation_count=7,
                has_code=False,
            ),
        ]
        return papers[:limit]


class SmokeCoordinator(ScholarWorkflowCoordinator):
    """
    Offline smoke coordinator: overrides stages to avoid network/LLM access.
    """

    async def _run_research_stage(self, ctx: PipelineContext) -> PipelineContext:
        ctx.research_result = {"venue_tier": 2}
        ctx.stages["research"] = {"result": ctx.research_result}
        return ctx

    async def _run_quality_stage(self, ctx: PipelineContext) -> PipelineContext:
        ctx.quality_result = {"quality_score": 0.0}
        ctx.stages["quality"] = {"result": ctx.quality_result}
        return ctx

    async def _run_influence_stage(self, ctx: PipelineContext) -> PipelineContext:
        ctx.influence_result = InfluenceResult(
            total_score=10.0,
            academic_score=10.0,
            engineering_score=0.0,
            explanation="smoke",
            metrics_breakdown={},
        )
        return ctx

    async def _run_report_stage(self, ctx: PipelineContext):
        return None


async def main() -> Dict[str, Any]:
    tmp_root = Path(tempfile.mkdtemp(prefix="paperbot-track-smoke-"))
    try:
        # Build a minimal subscriptions YAML, with absolute cache/output dirs for isolation.
        subs_path = tmp_root / "subscriptions.yaml"
        cache_dir = tmp_root / "cache"
        out_dir = tmp_root / "output"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        subs = {
            "subscriptions": {
                "scholars": [
                    {
                        "name": "SmokeScholar",
                        "semantic_scholar_id": "smoke_author_001",
                        "keywords": ["smoke"],
                    }
                ],
                "settings": {
                    "check_interval": "weekly",
                    "papers_per_scholar": 20,
                    "min_influence_score": 0,
                    "output_dir": str(out_dir),
                    "cache_dir": str(cache_dir),
                    "reporting": {"template": "paper_report.md.j2", "persist_history": False},
                },
            }
        }
        subs_path.write_text(yaml.safe_dump(subs, allow_unicode=True), encoding="utf-8")

        # Tracking trunk (offline, deterministic)
        tracker = PaperTrackerAgent(
            {
                "subscriptions_config_path": str(subs_path),
                "offline": True,
                # Avoid any accidental API calls; datasource will supply papers.
                "data_source": {"type": "api"},
            }
        )
        tracker.data_source = StubDataSource()

        scholar = tracker.profile_agent.get_scholar_by_id("smoke_author_001")
        assert scholar is not None, "failed to resolve scholar from subscriptions"

        tracking = await tracker.track_scholar(scholar, dry_run=True)
        assert tracking.get("new_papers_count", 0) > 0, "expected new papers > 0"

        # Analysis trunk: use ScholarPipeline API, but swap coordinator to deterministic one.
        pipeline = ScholarPipeline({"output_dir": str(out_dir), "enable_fail_fast": True})
        pipeline._coordinator = SmokeCoordinator({"output_dir": str(out_dir), "enable_fail_fast": True})  # type: ignore[attr-defined]

        ev = InMemoryEventLog()
        run_id = new_run_id()
        trace_id = new_trace_id()

        paper = PaperMeta.from_dict(tracking["new_papers"][0])
        _, influence, _ = await pipeline.analyze_paper(
            paper=paper,
            scholar_name=scholar.name,
            persist_report=False,
            event_log=ev,
            run_id=run_id,
            trace_id=trace_id,
        )

        has_score_update = any(e.get("type") == "score_update" for e in ev.events)
        assert has_score_update, "expected score_update event"

        return {
            "status": "ok",
            "run_id": run_id,
            "trace_id": trace_id,
            "tracking_status": tracking.get("status"),
            "new_papers_count": tracking.get("new_papers_count"),
            "events_emitted": len(ev.events),
            "has_score_update": has_score_update,
            "influence_total": getattr(influence, "total_score", None),
            "tmp_root": str(tmp_root),
        }
    finally:
        # Best-effort cleanup
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    print(json.dumps(asyncio.run(main()), ensure_ascii=False, indent=2))


