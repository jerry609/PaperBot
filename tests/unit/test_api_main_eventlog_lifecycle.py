from __future__ import annotations

from unittest.mock import Mock

import pytest

from paperbot.api import main as api_main


@pytest.mark.asyncio
async def test_shutdown_closes_event_log_and_obsidian_runtime(monkeypatch):
    event_log = Mock()
    api_main.app.state.event_log = event_log
    shutdown_obsidian = Mock()
    monkeypatch.setattr(api_main.obsidian, "shutdown_obsidian_runtime", shutdown_obsidian)

    await api_main._shutdown_runtime()

    event_log.close.assert_called_once_with()
    shutdown_obsidian.assert_called_once_with(api_main.app)


@pytest.mark.asyncio
async def test_startup_keeps_event_bus_when_sqlalchemy_backend_fails(monkeypatch):
    def _raise_sqlalchemy():
        raise RuntimeError("db unavailable")

    initialize_obsidian = Mock()
    monkeypatch.setattr(api_main, "SqlAlchemyEventLog", _raise_sqlalchemy)
    monkeypatch.setattr(api_main.obsidian, "initialize_obsidian_runtime", initialize_obsidian)

    await api_main._startup_eventlog()

    event_log = api_main.app.state.event_log
    assert isinstance(event_log, api_main.CompositeEventLog)
    backend_types = {type(backend) for backend in event_log._backends}
    assert api_main.LoggingEventLog in backend_types
    assert api_main.EventBusEventLog in backend_types
    assert len(event_log._backends) == 2
    initialize_obsidian.assert_called_once_with(api_main.app)
