from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from paperbot.infrastructure.stores.intelligence_store import IntelligenceStore
from paperbot.infrastructure.stores.models import IntelligenceEventModel


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


def test_intelligence_store_upsert_event_retries_after_integrity_error(tmp_path: Path, monkeypatch):
    store = IntelligenceStore(
        db_url=f"sqlite:///{tmp_path / 'intelligence-store-race.db'}",
        auto_create_schema=True,
    )
    original_session_factory = store._provider.session
    observed_at = datetime(2026, 3, 11, 12, 45, tzinfo=timezone.utc)
    injected_race = False

    def session_factory():
        session = original_session_factory()
        original_commit = session.commit

        def commit():
            nonlocal injected_race
            if not injected_race:
                injected_race = True
                with original_session_factory() as competing:
                    competing.add(
                        IntelligenceEventModel(
                            user_id="default",
                            external_id="signal-race",
                            created_at=observed_at,
                            detected_at=observed_at,
                            updated_at=observed_at,
                            source="reddit",
                            source_label="Reddit",
                            kind="keyword_spike",
                            title="Competing row",
                            summary="Created by competing transaction",
                        )
                    )
                    competing.commit()
                raise IntegrityError("insert", params={}, orig=Exception("duplicate"))
            return original_commit()

        session.commit = commit
        return session

    monkeypatch.setattr(store._provider, "session", session_factory)

    row = store.upsert_event(
        user_id="default",
        external_id="signal-race",
        source="github",
        source_label="GitHub",
        kind="repo_release",
        title="Race-safe signal",
        summary="Should win after retry",
        metric_value=7,
        detected_at=observed_at,
    )

    assert injected_race is True
    assert row["title"] == "Race-safe signal"
    assert row["summary"] == "Should win after retry"
    assert row["metric_value"] == 7
