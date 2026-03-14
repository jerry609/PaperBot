"""
Integration test stubs for SSE delivery via the /api/events endpoint.

These stubs exist for VALIDATION.md tracking but remain SKIPPED until Plan 07-02
wires the FastAPI SSE endpoint.

Mark with @pytest.mark.integration so they are excluded from the standard
unit test run.
"""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_event_delivered_within_1s():
    """SSE client receives an event within 1 second of it being appended."""
    pytest.skip("endpoint not yet wired — Plan 07-02")


@pytest.mark.integration
def test_heartbeat_on_idle():
    """SSE stream emits a keepalive comment when no events arrive for > N seconds."""
    pytest.skip("endpoint not yet wired — Plan 07-02")
