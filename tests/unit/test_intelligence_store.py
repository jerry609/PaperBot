from datetime import datetime, timezone
from pathlib import Path

from paperbot.infrastructure.stores.intelligence_store import IntelligenceStore


def test_intelligence_store_serializes_datetime_payload(tmp_path: Path):
    store = IntelligenceStore(
        db_url=f"sqlite:///{tmp_path / 'intelligence-store.db'}",
        auto_create_schema=True,
    )
    observed_at = datetime(2026, 3, 11, 12, 30, tzinfo=timezone.utc)

    row = store.upsert_event(
        user_id="default",
        external_id="signal-1",
        source="github",
        source_label="GitHub",
        kind="repo_release",
        title="Test signal",
        summary="Serializable payload roundtrip",
        payload={
            "observed_at": observed_at,
            "nested": {
                "latest_seen": observed_at,
                "series": [observed_at],
            },
        },
    )

    assert row["payload"]["observed_at"] == observed_at.isoformat()
    assert row["payload"]["nested"]["latest_seen"] == observed_at.isoformat()
    assert row["payload"]["nested"]["series"] == [observed_at.isoformat()]


def test_intelligence_store_latest_detected_at_restores_utc_timezone(tmp_path: Path):
    store = IntelligenceStore(
        db_url=f"sqlite:///{tmp_path / 'intelligence-store-latest.db'}",
        auto_create_schema=True,
    )
    observed_at = datetime(2026, 3, 11, 12, 30, tzinfo=timezone.utc)

    store.upsert_event(
        user_id="default",
        external_id="signal-latest",
        source="reddit",
        source_label="Reddit",
        kind="keyword_spike",
        title="Latest signal",
        summary="Roundtrip latest_detected_at",
        detected_at=observed_at,
    )

    latest = store.latest_detected_at(user_id="default")

    assert latest == observed_at
    assert latest.tzinfo == timezone.utc
