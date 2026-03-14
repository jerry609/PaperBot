"""Unit tests for the MCP audit helper (log_tool_call)."""

from paperbot.core.di import Container
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog


class TestLogToolCall:
    def setup_method(self):
        Container._instance = None

    def _register_event_log(self):
        """Helper: register InMemoryEventLog in the DI container."""
        log = InMemoryEventLog()
        container = Container.instance()
        container.register(EventLogPort, lambda: log)
        return log

    def test_creates_event_with_correct_fields(self):
        """log_tool_call() creates an AgentEventEnvelope with workflow='mcp',
        stage='tool_call', agent_name='paperbot-mcp'."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        log_tool_call(
            tool_name="test_tool",
            arguments={"query": "hello"},
            result_summary="found 3 results",
            duration_ms=42.0,
        )

        assert len(log.events) == 1
        event = log.events[0]
        assert event["workflow"] == "mcp"
        assert event["stage"] == "tool_call"
        assert event["agent_name"] == "paperbot-mcp"
        assert event["role"] == "system"
        assert event["type"] == "tool_result"

    def test_generates_run_id_when_none(self):
        """log_tool_call() with run_id=None generates a new run_id (non-empty string)."""
        from paperbot.mcp.tools._audit import log_tool_call

        self._register_event_log()

        returned_run_id = log_tool_call(
            tool_name="test_tool",
            arguments={},
            result_summary="ok",
            duration_ms=1.0,
            run_id=None,
        )

        assert isinstance(returned_run_id, str)
        assert len(returned_run_id) > 0

    def test_uses_provided_run_id(self):
        """log_tool_call() with run_id='abc123' uses that run_id in the event."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        returned_run_id = log_tool_call(
            tool_name="test_tool",
            arguments={},
            result_summary="ok",
            duration_ms=1.0,
            run_id="abc123",
        )

        assert returned_run_id == "abc123"
        assert log.events[0]["run_id"] == "abc123"

    def test_stores_event_in_event_log(self):
        """log_tool_call() stores event in InMemoryEventLog (.events list)."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        log_tool_call(
            tool_name="store_test",
            arguments={"q": "papers"},
            result_summary="5 papers",
            duration_ms=100.0,
        )

        assert len(log.events) == 1
        event = log.events[0]
        assert event["payload"]["tool"] == "store_test"
        assert event["payload"]["arguments"] == {"q": "papers"}
        assert event["payload"]["result_summary"] == "5 papers"

    def test_degrades_silently_without_event_log(self):
        """log_tool_call() with no EventLogPort registered degrades silently."""
        from paperbot.mcp.tools._audit import log_tool_call

        # Do NOT register any EventLogPort -- container is fresh
        Container.instance()  # ensure container exists but empty

        returned_run_id = log_tool_call(
            tool_name="test_tool",
            arguments={},
            result_summary="ok",
            duration_ms=1.0,
        )

        # Should return a valid run_id, no exception raised
        assert isinstance(returned_run_id, str)
        assert len(returned_run_id) > 0

    def test_records_duration_ms_in_metrics(self):
        """log_tool_call() records duration_ms in event metrics."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        log_tool_call(
            tool_name="test_tool",
            arguments={},
            result_summary="ok",
            duration_ms=123.45,
        )

        assert log.events[0]["metrics"]["duration_ms"] == 123.45

    def test_records_error_field(self):
        """log_tool_call() records error field when error is provided."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        log_tool_call(
            tool_name="test_tool",
            arguments={},
            result_summary="",
            duration_ms=5.0,
            error="Connection timeout",
        )

        event = log.events[0]
        assert event["type"] == "error"
        assert event["payload"]["error"] == "Connection timeout"

    def test_accepts_structured_summary_and_redacts_sensitive_arguments(self):
        """log_tool_call() accepts structured summaries and redacts sensitive argument keys."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        log_tool_call(
            tool_name="test_tool",
            arguments={
                "query": "hello",
                "api_key": "secret-value",
                "nested": {"token": "nested-secret"},
            },
            result_summary={"count": 3, "status": "ok"},
            duration_ms=10.0,
        )

        payload = log.events[0]["payload"]
        assert payload["arguments"]["query"] == "hello"
        assert payload["arguments"]["api_key"] == "***redacted***"
        assert payload["arguments"]["nested"]["token"] == "***redacted***"
        assert payload["result_summary"] == {"count": 3, "status": "ok"}

    def test_truncates_oversized_audit_fields(self):
        """log_tool_call() truncates oversized text fields before persistence."""
        from paperbot.mcp.tools._audit import log_tool_call

        log = self._register_event_log()

        oversized = "x" * 1200
        log_tool_call(
            tool_name="test_tool",
            arguments={"query": oversized},
            result_summary=oversized,
            duration_ms=1.0,
        )

        payload = log.events[0]["payload"]
        assert payload["arguments"]["query"].endswith("...[truncated]")
        assert len(payload["arguments"]["query"]) > 1000
        assert payload["result_summary"].endswith("...[truncated]")
