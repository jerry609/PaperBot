"""
Integration tests for Scope and Acceptance Criteria 
memory hooks in API routes.

Tests the Scope and Acceptance criteria hooks:
- false_positive_rate: recorded when user rejects high-confidence items
- deletion_compliance: recorded when clearing track memory
- retrieval_hit_rate: recorded when user provides memory feedback
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Create test client with isolated database."""
    monkeypatch.setenv("PAPERBOT_DB_URL", f"sqlite:///{tmp_path / 'test.db'}")

    from paperbot.api import main as api_main

    with TestClient(api_main.app) as client:
        yield client


def test_memory_feedback_records_hit_rate(test_client):
    """Test that memory feedback endpoint records retrieval_hit_rate metric."""
    # Record memory feedback
    response = test_client.post(
        "/api/research/memory/feedback",
        json={
            "user_id": "test_user",
            "memory_ids": [1, 2, 3, 4, 5],
            "helpful_ids": [1, 2, 3, 4],
            "not_helpful_ids": [5],
            "query": "test query",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_rated"] == 5
    assert data["helpful_count"] == 4
    assert data["hit_rate"] == 0.8

    # Verify metric was recorded
    metrics_response = test_client.get("/api/memory/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert "retrieval_hit_rate" in metrics["metrics"]
    assert metrics["metrics"]["retrieval_hit_rate"]["value"] == 0.8


def test_bulk_moderate_reject_records_false_positive(test_client):
    """Test that rejecting high-confidence items records false_positive_rate."""
    # First create a track
    track_response = test_client.post(
        "/api/research/tracks",
        json={"user_id": "test_user", "name": "Test Track", "activate": True},
    )
    assert track_response.status_code == 200
    track_id = track_response.json()["track"]["id"]

    # Create a high-confidence approved memory
    mem_response = test_client.post(
        "/api/research/memory/items",
        json={
            "user_id": "test_user",
            "kind": "preference",
            "content": "Test preference",
            "confidence": 0.85,
            "scope_type": "track",
            "scope_id": str(track_id),
            "status": "approved",
        },
    )
    assert mem_response.status_code == 200
    item_id = mem_response.json()["item"]["id"]

    # Reject the high-confidence item (this should record false_positive_rate)
    reject_response = test_client.post(
        "/api/research/memory/bulk_moderate",
        json={
            "user_id": "test_user",
            "item_ids": [item_id],
            "status": "rejected",
        },
    )
    assert reject_response.status_code == 200

    # Verify false_positive_rate metric was recorded
    metrics_response = test_client.get("/api/memory/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert "false_positive_rate" in metrics["metrics"]


def test_clear_track_memory_records_deletion_compliance(test_client):
    """Test that clearing track memory records deletion_compliance metric."""
    # Create a track
    track_response = test_client.post(
        "/api/research/tracks",
        json={"user_id": "test_user", "name": "Delete Test Track", "activate": True},
    )
    assert track_response.status_code == 200
    track_id = track_response.json()["track"]["id"]

    # Create some memories in the track
    for i in range(3):
        test_client.post(
            "/api/research/memory/items",
            json={
                "user_id": "test_user",
                "kind": "fact",
                "content": f"Test fact {i}",
                "confidence": 0.7,
                "scope_type": "track",
                "scope_id": str(track_id),
            },
        )

    # Clear track memory (triggers deletion_compliance check)
    clear_response = test_client.post(
        f"/api/research/tracks/{track_id}/memory/clear",
        params={"user_id": "test_user", "confirm": True},
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["deleted_count"] == 3

    # Verify deletion_compliance metric was recorded
    metrics_response = test_client.get("/api/memory/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert "deletion_compliance" in metrics["metrics"]
    # Should be 1.0 (100% compliant) since no deleted items should be retrievable
    assert metrics["metrics"]["deletion_compliance"]["value"] == 1.0


def test_metrics_endpoints(test_client):
    """Test the /memory/metrics endpoints."""
    # Get initial metrics (should be empty)
    response = test_client.get("/api/memory/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "metrics" in data
    assert "targets" in data

    # Record a metric
    test_client.post(
        "/api/research/memory/feedback",
        json={
            "user_id": "test_user",
            "memory_ids": [1, 2],
            "helpful_ids": [1, 2],
            "not_helpful_ids": [],
        },
    )

    # Get metric history
    history_response = test_client.get("/api/memory/metrics/retrieval_hit_rate")
    assert history_response.status_code == 200
    data = history_response.json()
    assert "history" in data
    assert len(data["history"]) >= 1
    assert data["history"][0]["value"] == 1.0  # 100% hit rate


def test_metrics_summary_pass_fail(test_client):
    """Test that metrics summary correctly reports pass/fail status."""
    # Record a metric that meets target
    test_client.post(
        "/api/research/memory/feedback",
        json={
            "user_id": "test_user",
            "memory_ids": [1, 2, 3, 4, 5],
            "helpful_ids": [1, 2, 3, 4, 5],  # 100% hit rate > 80% target
            "not_helpful_ids": [],
        },
    )

    response = test_client.get("/api/memory/metrics")
    data = response.json()

    # With only one metric recorded that meets target, status should be pass
    assert data["metrics"]["retrieval_hit_rate"]["meets_target"] is True
