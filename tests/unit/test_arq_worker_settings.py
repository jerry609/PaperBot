import pytest

pytest.importorskip("arq", reason="arq not installed")

from paperbot.infrastructure.queue.arq_worker import WorkerSettings


def test_arq_worker_settings_has_functions():
    assert hasattr(WorkerSettings, "functions")
    assert len(WorkerSettings.functions) >= 2


