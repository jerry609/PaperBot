from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator


class _FakeEventLog:
    def __init__(self):
        self.events = []

    def append(self, event):
        self.events.append(event)


class _FakeInfluence:
    def __init__(self, total=77.0, academic=70.0, engineering=84.0, momentum=12.0):
        self.total_score = total
        self.academic_score = academic
        self.engineering_score = engineering
        self.metrics_breakdown = {"academic": {"momentum_score": momentum}}


@pytest.mark.asyncio
async def test_workflow_coordinator_publishes_all_four_stage_scores(tmp_path):
    coordinator = ScholarWorkflowCoordinator({"output_dir": str(tmp_path), "enable_fail_fast": False})
    coordinator._research_agent = SimpleNamespace(process=AsyncMock(return_value={"venue_tier": 1}))
    coordinator._code_analysis_agent = SimpleNamespace(
        process=AsyncMock(return_value={"health_score": 91.0, "is_empty_repo": False})
    )
    coordinator._quality_agent = SimpleNamespace(
        process=AsyncMock(return_value={"quality_scores": {"repo": {"overall_score": 0.82}}})
    )
    coordinator._influence_calculator = SimpleNamespace(calculate=lambda paper, code_meta: _FakeInfluence())
    coordinator._report_writer = None

    paper = SimpleNamespace(
        paper_id="paper-1",
        title="Test Paper",
        abstract="Abstract",
        citation_count=120,
        github_url="https://github.com/example/repo",
        has_code=True,
    )
    event_log = _FakeEventLog()

    report_path, influence, pipeline = await coordinator.run_paper_pipeline(
        paper,
        event_log=event_log,
        run_id="run-1",
        trace_id="trace-1",
    )

    assert report_path is None
    assert influence.total_score == 77.0
    assert pipeline["status"] == "success"

    score_events = [event for event in event_log.events if event.type == "score_update"]
    assert [event.stage for event in score_events] == ["research", "code", "quality", "influence"]


@pytest.mark.asyncio
async def test_workflow_coordinator_early_exit_skips_remaining_stages(tmp_path):
    coordinator = ScholarWorkflowCoordinator(
        {
            "output_dir": str(tmp_path),
            "enable_fail_fast": True,
            "fail_fast": {"early_exit_threshold": 10.0},
        }
    )
    coordinator._research_agent = SimpleNamespace(process=AsyncMock(return_value={}))
    coordinator._code_analysis_agent = SimpleNamespace(process=AsyncMock())
    coordinator._quality_agent = SimpleNamespace(process=AsyncMock())
    coordinator._influence_calculator = SimpleNamespace(calculate=AsyncMock())
    coordinator._report_writer = None

    paper = SimpleNamespace(
        paper_id="paper-low",
        title="Low Impact Paper",
        abstract="Abstract",
        citation_count=0,
        github_url="https://github.com/example/repo",
        has_code=True,
    )

    report_path, influence, pipeline = await coordinator.run_paper_pipeline(paper)

    assert report_path is None
    assert influence.total_score == 0.0
    assert pipeline["status"] == "success"
    assert pipeline["stages"]["code"]["status"] == "skipped"
    assert pipeline["stages"]["quality"]["status"] == "skipped"
    assert pipeline["stages"]["influence"]["status"] == "skipped"
    assert pipeline["stages"]["report"]["status"] == "skipped"
    coordinator._code_analysis_agent.process.assert_not_awaited()
    coordinator._quality_agent.process.assert_not_awaited()
