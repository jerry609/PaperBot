from __future__ import annotations

from pathlib import Path

import yaml


def test_build_cron_jobs_from_subscriptions_daily(tmp_path, monkeypatch):
    # Create a minimal subscriptions config
    cfg = {
        "subscriptions": {
            "scholars": [{"name": "X", "semantic_scholar_id": "123"}],
            "settings": {"check_interval": "daily"},
        }
    }
    p = tmp_path / "subs.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("PAPERBOT_SUBSCRIPTIONS_PATH", str(p))
    monkeypatch.setenv("PAPERBOT_TRACK_CRON_HOUR", "3")
    monkeypatch.setenv("PAPERBOT_TRACK_CRON_MINUTE", "15")
    monkeypatch.setenv("PAPERBOT_TRACK_CRON_WEEKDAY", "mon")

    from paperbot.infrastructure.queue import arq_worker

    jobs = arq_worker._build_cron_jobs_from_subscriptions()
    assert len(jobs) == 1


def test_build_cron_jobs_from_subscriptions_disabled(tmp_path, monkeypatch):
    cfg = {
        "subscriptions": {
            "scholars": [{"name": "X", "semantic_scholar_id": "123"}],
            "settings": {"check_interval": None},
        }
    }
    p = tmp_path / "subs.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("PAPERBOT_SUBSCRIPTIONS_PATH", str(p))

    from paperbot.infrastructure.queue import arq_worker

    jobs = arq_worker._build_cron_jobs_from_subscriptions()
    assert jobs == []


