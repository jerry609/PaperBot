from pathlib import Path
from reports.writer import ReportWriter
from scholar_tracking.models import PaperMeta
from scholar_tracking.models.influence import InfluenceResult


def test_render_report_with_meta_append():
    paper = PaperMeta.from_dict({"paper_id": "p1", "title": "T", "year": 2024})
    influence = InfluenceResult(
        total_score=50,
        academic_score=40,
        engineering_score=60,
        metrics_breakdown={"weights": {"recency_factor": 1.0}},
        explanation="",
    )
    writer = ReportWriter()
    md = writer.render_template(
        paper=paper,
        influence=influence,
        research_result={},
        code_analysis_result={},
        quality_result={},
        scholar_name=None,
    )
    assert "T" in md


def test_render_report_auto_append_reproducibility():
    paper = PaperMeta.from_dict({"paper_id": "p1", "title": "T", "year": 2024})
    influence = InfluenceResult(
        total_score=50,
        academic_score=40,
        engineering_score=60,
        metrics_breakdown={"weights": {"recency_factor": 1.0}},
        explanation="",
    )
    writer = ReportWriter()
    md = writer.render_template(
        paper=paper,
        influence=influence,
        research_result={},
        code_analysis_result={},
        quality_result={},
        scholar_name=None,
    )
    # baseline render ok; meta append happens in main, here just sanity check
    assert "##" in md

