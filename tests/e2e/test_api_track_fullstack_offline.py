from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient


class _StubSemanticScholarAgent:
    async def fetch_papers_by_author(self, author_id: str, limit: int = 20, offset: int = 0):
        # Return 1 deterministic paper in Semantic Scholar style -> PaperMeta.from_semantic_scholar will parse.
        return [
            # PaperMeta is built inside SemanticScholarAgent; for our route we need PaperMeta objects.
            # Easiest: return dicts that PaperMeta.from_dict can accept later (track_scholar stores p.to_dict()).
        ]

    async def close(self):
        return None


def _stub_papers() -> List[Dict[str, Any]]:
    return [
        {
            "paper_id": "e2e_paper_001",
            "title": "E2E Offline Paper",
            "authors": ["Alice"],
            "abstract": "offline",
            "year": 2025,
            "venue": "E2EConf",
            "citation_count": 5,
            "github_url": None,
            "has_code": False,
            "url": None,
            "doi": None,
        }
    ]


@pytest.mark.asyncio
async def test_api_track_fullstack_offline_emits_db_events(monkeypatch, tmp_path):
    # Import inside test after monkeypatch hooks are ready.
    from paperbot.api import main as api_main

    # Force DB to temp file so we can query events
    monkeypatch.setenv("PAPERBOT_DB_URL", f"sqlite:///{tmp_path / 'paperbot_e2e.db'}")

    # Patch PaperTrackerAgent.track_scholar to avoid external API calls.
    from paperbot.agents.scholar_tracking import paper_tracker_agent as pta

    async def _track_scholar(self, scholar, dry_run: bool = False):
        return {
            "scholar_id": scholar.semantic_scholar_id,
            "scholar_name": scholar.name,
            "status": "success",
            "total_papers": 1,
            "cached_papers": 0,
            "new_papers_count": 1,
            "new_papers": _stub_papers(),
            "source": "stub",
            "fetched_at": "2025-01-01T00:00:00",
        }

    monkeypatch.setattr(pta.PaperTrackerAgent, "track_scholar", _track_scholar)

    # Patch ScholarPipeline to use deterministic coordinator stages (avoid LLM/network)
    from paperbot.application.workflows import scholar_pipeline as sp
    from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator, PipelineContext
    from paperbot.domain.influence.result import InfluenceResult

    class SmokeCoordinator(ScholarWorkflowCoordinator):
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
                explanation="e2e",
                metrics_breakdown={},
            )
            return ctx

        async def _run_report_stage(self, ctx: PipelineContext):
            return None

    orig_init = sp.ScholarPipeline.__init__

    def _init(self, config=None):
        orig_init(self, config=config)
        self._coordinator = SmokeCoordinator(config or {})

    monkeypatch.setattr(sp.ScholarPipeline, "__init__", _init)

    with TestClient(api_main.app) as client:
        # Use scholar_id from default config, no external calls due to patches.
        with client.stream("GET", "/api/track", params={"scholar_id": "1741101", "max_new_papers": 1}) as resp:
            assert resp.status_code == 200
            lines = []
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    lines.append(payload)
                if len(lines) >= 3:
                    break

        # First progress should include run_id/trace_id
        first = lines[0]
        assert first["type"] in ("progress", "error")
        run_id = (first.get("data") or {}).get("run_id")
        assert run_id

        # Verify DB-backed event log recorded at least one event for this run_id
        elog = client.app.state.event_log
        events = elog.list_events(run_id, limit=100)
        assert len(events) >= 1


