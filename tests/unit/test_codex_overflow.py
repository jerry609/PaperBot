"""Unit tests for _should_overflow_to_codex overflow routing stub.

TDD: These tests define the required API. They will fail (RED) until
_should_overflow_to_codex is implemented in repro/orchestrator.py.
"""

from __future__ import annotations


def test_overflow_returns_false_when_unset(monkeypatch):
    """_should_overflow_to_codex returns False when env var is not set."""
    monkeypatch.delenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", raising=False)
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is False


def test_overflow_returns_true_when_true(monkeypatch):
    """_should_overflow_to_codex returns True when env var is 'true'."""
    monkeypatch.setenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "true")
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is True


def test_overflow_returns_true_when_one(monkeypatch):
    """_should_overflow_to_codex returns True when env var is '1'."""
    monkeypatch.setenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "1")
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is True


def test_overflow_returns_false_when_zero(monkeypatch):
    """_should_overflow_to_codex returns False when env var is '0'."""
    monkeypatch.setenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "0")
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is False


def test_overflow_returns_false_when_empty(monkeypatch):
    """_should_overflow_to_codex returns False when env var is empty string."""
    monkeypatch.setenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "")
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is False


def test_overflow_returns_true_when_yes(monkeypatch):
    """_should_overflow_to_codex returns True when env var is 'yes'."""
    monkeypatch.setenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "yes")
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is True


def test_overflow_returns_true_when_on(monkeypatch):
    """_should_overflow_to_codex returns True when env var is 'on'."""
    monkeypatch.setenv("PAPERBOT_CODEX_OVERFLOW_THRESHOLD", "on")
    from paperbot.repro.orchestrator import _should_overflow_to_codex
    assert _should_overflow_to_codex() is True
