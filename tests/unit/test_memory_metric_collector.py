"""
Unit tests for MemoryMetricCollector (P0 Acceptance Criteria).
"""
from __future__ import annotations

import pytest

from paperbot.memory.eval.collector import MemoryMetricCollector


def test_collector_record_extraction_precision(tmp_path):
    """Test recording extraction precision metric."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    collector.record_extraction_precision(
        correct_count=85,
        total_count=100,
        evaluator_id="test:unit",
        detail={"test": True},
    )

    latest = collector.get_latest_metrics()
    assert "extraction_precision" in latest
    assert latest["extraction_precision"]["value"] == 0.85
    assert latest["extraction_precision"]["meets_target"] is True  # target is 0.85


def test_collector_record_false_positive_rate(tmp_path):
    """Test recording false positive rate metric."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    collector.record_false_positive_rate(
        false_positive_count=3,
        total_approved_count=100,
        evaluator_id="test:unit",
    )

    latest = collector.get_latest_metrics()
    assert "false_positive_rate" in latest
    assert latest["false_positive_rate"]["value"] == 0.03
    assert latest["false_positive_rate"]["meets_target"] is True  # target is <= 0.05


def test_collector_record_false_positive_rate_fails_target(tmp_path):
    """Test false positive rate exceeding target."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    collector.record_false_positive_rate(
        false_positive_count=10,
        total_approved_count=100,
        evaluator_id="test:unit",
    )

    latest = collector.get_latest_metrics()
    assert latest["false_positive_rate"]["value"] == 0.10
    assert latest["false_positive_rate"]["meets_target"] is False  # 10% > 5%


def test_collector_record_retrieval_hit_rate(tmp_path):
    """Test recording retrieval hit rate metric."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    collector.record_retrieval_hit_rate(
        hits=8,
        expected=10,
        evaluator_id="test:unit",
    )

    latest = collector.get_latest_metrics()
    assert "retrieval_hit_rate" in latest
    assert latest["retrieval_hit_rate"]["value"] == 0.80
    assert latest["retrieval_hit_rate"]["meets_target"] is True  # target is >= 0.80


def test_collector_record_injection_pollution_rate(tmp_path):
    """Test recording injection pollution rate metric."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    collector.record_injection_pollution_rate(
        polluted_count=1,
        total_injections=100,
        evaluator_id="test:unit",
    )

    latest = collector.get_latest_metrics()
    assert "injection_pollution_rate" in latest
    assert latest["injection_pollution_rate"]["value"] == 0.01
    assert latest["injection_pollution_rate"]["meets_target"] is True  # target is <= 0.02


def test_collector_record_deletion_compliance(tmp_path):
    """Test recording deletion compliance metric."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    # Perfect compliance: no deleted items retrieved
    collector.record_deletion_compliance(
        deleted_retrieved_count=0,
        deleted_total_count=10,
        evaluator_id="test:unit",
    )

    latest = collector.get_latest_metrics()
    assert "deletion_compliance" in latest
    assert latest["deletion_compliance"]["value"] == 1.0
    assert latest["deletion_compliance"]["meets_target"] is True


def test_collector_record_deletion_compliance_fails(tmp_path):
    """Test deletion compliance failure when deleted items are retrieved."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    # Bad compliance: 2 deleted items were retrieved
    collector.record_deletion_compliance(
        deleted_retrieved_count=2,
        deleted_total_count=10,
        evaluator_id="test:unit",
    )

    latest = collector.get_latest_metrics()
    assert latest["deletion_compliance"]["value"] == 0.8  # 1 - 2/10
    assert latest["deletion_compliance"]["meets_target"] is False


def test_collector_get_metrics_summary(tmp_path):
    """Test getting metrics summary with pass/fail status."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    # Record all metrics meeting targets
    collector.record_extraction_precision(correct_count=90, total_count=100)
    collector.record_false_positive_rate(false_positive_count=2, total_approved_count=100)
    collector.record_retrieval_hit_rate(hits=85, expected=100)
    collector.record_injection_pollution_rate(polluted_count=1, total_injections=100)
    collector.record_deletion_compliance(deleted_retrieved_count=0, deleted_total_count=10)

    summary = collector.get_metrics_summary()
    assert summary["status"] == "pass"
    assert len(summary["metrics"]) == 5


def test_collector_get_metrics_summary_fails(tmp_path):
    """Test metrics summary fails when any metric doesn't meet target."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    # One metric fails
    collector.record_extraction_precision(correct_count=50, total_count=100)  # 50% < 85%

    summary = collector.get_metrics_summary()
    assert summary["status"] == "fail"


def test_collector_get_metric_history(tmp_path):
    """Test getting metric history."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    # Record multiple values
    collector.record_extraction_precision(correct_count=80, total_count=100)
    collector.record_extraction_precision(correct_count=85, total_count=100)
    collector.record_extraction_precision(correct_count=90, total_count=100)

    history = collector.get_metric_history("extraction_precision", limit=10)
    assert len(history) == 3
    # Most recent first
    assert history[0]["value"] == 0.90
    assert history[1]["value"] == 0.85
    assert history[2]["value"] == 0.80


def test_collector_skips_zero_totals(tmp_path):
    """Test that collector skips recording when total is zero."""
    db_url = f"sqlite:///{tmp_path / 'metrics.db'}"
    collector = MemoryMetricCollector(db_url=db_url)

    collector.record_extraction_precision(correct_count=0, total_count=0)
    collector.record_false_positive_rate(false_positive_count=0, total_approved_count=0)
    collector.record_retrieval_hit_rate(hits=0, expected=0)
    collector.record_deletion_compliance(deleted_retrieved_count=0, deleted_total_count=0)

    latest = collector.get_latest_metrics()
    assert len(latest) == 0  # No metrics recorded
