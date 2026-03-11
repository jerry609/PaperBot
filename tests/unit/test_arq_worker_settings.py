import pytest

pytest.importorskip("arq", reason="arq not installed")

from paperbot.infrastructure.queue import arq_worker
from paperbot.infrastructure.queue.arq_worker import WorkerSettings


def _functions_by_name():
    return {fn.name: fn for fn in WorkerSettings.functions}


def test_arq_worker_settings_has_bounded_functions():
    functions = _functions_by_name()

    assert set(functions) >= {
        "track_scholar_job",
        "analyze_paper_job",
        "cron_track_subscriptions",
        "cron_daily_papers",
        "daily_papers_job",
    }
    assert functions["track_scholar_job"].timeout_s == 600
    assert functions["track_scholar_job"].max_tries == 3
    assert functions["daily_papers_job"].timeout_s == 1200
    assert functions["daily_papers_job"].max_tries == 2
    assert functions["cron_daily_papers"].timeout_s == 60
    assert functions["cron_daily_papers"].max_tries == 1


def test_event_log_helper_reuses_singleton(monkeypatch):
    created = []

    class _FakeEventLog:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(arq_worker, "_EVENT_LOG", None)
    monkeypatch.setattr(arq_worker, "SqlAlchemyEventLog", _FakeEventLog)

    first = arq_worker._event_log()
    second = arq_worker._event_log()

    assert first is second
    assert len(created) == 1
