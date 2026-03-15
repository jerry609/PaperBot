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


class _CaptureInfluenceCalculator:
    def __init__(self):
        self.code_meta = None

    def calculate(self, paper, code_meta):
        self.code_meta = code_meta
        return _FakeInfluence()


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
async def test_workflow_coordinator_uses_reproducibility_score_for_code_stage(tmp_path):
    coordinator = ScholarWorkflowCoordinator({"output_dir": str(tmp_path), "enable_fail_fast": False})
    coordinator._research_agent = SimpleNamespace(process=AsyncMock(return_value={"venue_tier": 1}))
    coordinator._code_analysis_agent = SimpleNamespace(
        process=AsyncMock(
            return_value={
                "reproducibility_score": 82.0,
                "updated_at": "2026-03-01T00:00:00+00:00",
                "last_commit_date": "2026-03-02T00:00:00+00:00",
                "has_readme": True,
                "stars": 12,
                "forks": 3,
            }
        )
    )
    coordinator._quality_agent = SimpleNamespace(
        process=AsyncMock(return_value={"quality_score": 0.71, "quality_scores": {"repo": {"overall_score": 0.71}}})
    )
    coordinator._influence_calculator = SimpleNamespace(calculate=lambda paper, code_meta: _FakeInfluence())
    coordinator._report_writer = None

    paper = SimpleNamespace(
        paper_id="paper-repro",
        title="Repro Paper",
        abstract="Abstract",
        citation_count=30,
        github_url="https://github.com/example/repo",
        has_code=True,
    )
    event_log = _FakeEventLog()

    await coordinator.run_paper_pipeline(
        paper,
        event_log=event_log,
        run_id="run-repro",
        trace_id="trace-repro",
    )

    score_events = [event for event in event_log.events if event.type == "score_update"]
    code_event = next(event for event in score_events if event.stage == "code")
    assert code_event.payload["score"]["score"] == pytest.approx(82.0)


@pytest.mark.asyncio
async def test_workflow_coordinator_passes_aligned_code_meta_to_influence(tmp_path):
    coordinator = ScholarWorkflowCoordinator({"output_dir": str(tmp_path), "enable_fail_fast": False})
    coordinator._research_agent = SimpleNamespace(process=AsyncMock(return_value={"venue_tier": 1}))
    coordinator._code_analysis_agent = SimpleNamespace(
        process=AsyncMock(
            return_value={
                "reproducibility_score": 82.0,
                "updated_at": "2026-03-01T00:00:00+00:00",
                "last_commit_date": "2026-03-02T00:00:00+00:00",
                "has_readme": True,
                "stars": 12,
                "forks": 3,
            }
        )
    )
    coordinator._quality_agent = SimpleNamespace(
        process=AsyncMock(return_value={"quality_score": 0.71, "quality_scores": {"repo": {"overall_score": 0.71}}})
    )
    influence_calculator = _CaptureInfluenceCalculator()
    coordinator._influence_calculator = influence_calculator
    coordinator._report_writer = None

    paper = SimpleNamespace(
        paper_id="paper-meta",
        title="Meta Paper",
        abstract="Abstract",
        citation_count=30,
        github_url="https://github.com/example/repo",
        has_code=True,
    )

    await coordinator.run_paper_pipeline(paper)

    assert influence_calculator.code_meta is not None
    assert influence_calculator.code_meta.updated_at == "2026-03-01T00:00:00+00:00"
    assert influence_calculator.code_meta.last_commit_date == "2026-03-02T00:00:00+00:00"
    assert influence_calculator.code_meta.reproducibility_score == pytest.approx(82.0)


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
